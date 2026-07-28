"""Tkinter desktop application for standalone two-camera calibration."""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import open3d as o3d

try:
    import ttkbootstrap as ttkb
except ImportError:  # The core calibration package remains usable without it.
    ttkb = None

from .realsense import RealSenseDevice, capture_pair, list_devices
from .registration import RegistrationConfig, RegistrationResult, calibrate
from .transforms import (
    load_transform,
    matrix_from_translation_euler,
    matrix_from_translation_quaternion,
    quaternion_from_matrix,
    save_transform,
)

LOGGER = logging.getLogger(__name__)


class _OrbitPointCloudView(ttk.Frame):
    """Fast software 3-D orbit view with a stable, robust rotation center."""

    def __init__(
        self,
        master: tk.Misc,
        points: np.ndarray,
        colors: np.ndarray,
        title: str,
    ) -> None:
        super().__init__(master)
        self._points = np.asarray(points, dtype=np.float64)
        self._colors = np.asarray(colors, dtype=np.float64)
        finite = np.all(np.isfinite(self._points), axis=1)
        self._points = self._points[finite]
        self._colors = self._colors[finite]
        if not len(self._points):
            raise ValueError(f"{title} cloud is empty")

        # Exclude extreme flying pixels when selecting the orbit center and
        # scale. Points are not removed from the underlying calibration cloud.
        low = np.percentile(self._points, 1.0, axis=0)
        high = np.percentile(self._points, 99.0, axis=0)
        central = np.all(
            (self._points >= low) & (self._points <= high), axis=1
        )
        robust_points = self._points[central]
        self._center = np.median(robust_points, axis=0)
        extent = np.maximum(
            np.percentile(robust_points, 99.0, axis=0)
            - np.percentile(robust_points, 1.0, axis=0),
            1.0e-4,
        )
        self._base_scale = 0.82 / float(np.max(extent))
        self._yaw = 0.0
        self._pitch = 0.0
        self._zoom = 1.0
        self._pan = np.zeros(2, dtype=np.float64)
        self._drag_button = 0
        self._last_mouse = (0, 0)
        self._render_pending = False
        self._photo: tk.PhotoImage | None = None

        ttk.Label(self, text=title).pack(pady=(2, 3))
        self.canvas = tk.Canvas(
            self,
            width=580,
            height=570,
            background="#161616",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._schedule_render())
        self.canvas.bind("<ButtonPress-1>", lambda event: self._start_drag(event, 1))
        self.canvas.bind("<ButtonPress-2>", lambda event: self._start_drag(event, 2))
        self.canvas.bind("<ButtonPress-3>", lambda event: self._start_drag(event, 3))
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<B2-Motion>", self._drag)
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drag)
        self.canvas.bind("<ButtonRelease-2>", self._stop_drag)
        self.canvas.bind("<ButtonRelease-3>", self._stop_drag)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom_by(1.12))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom_by(1 / 1.12))
        self.canvas.bind("<Double-Button-1>", lambda _event: self.reset())
        self.after_idle(self._render)

    def _start_drag(self, event: tk.Event, button: int) -> None:
        self._drag_button = button
        self._last_mouse = (event.x, event.y)

    def _stop_drag(self, _event: tk.Event) -> None:
        self._drag_button = 0

    def _drag(self, event: tk.Event) -> None:
        dx = event.x - self._last_mouse[0]
        dy = event.y - self._last_mouse[1]
        self._last_mouse = (event.x, event.y)
        if self._drag_button == 1:
            self._yaw -= dx * 0.009
            self._pitch = float(
                np.clip(self._pitch + dy * 0.009, -1.52, 1.52)
            )
        else:
            self._pan += [dx, dy]
        self._schedule_render()

    def _mouse_wheel(self, event: tk.Event) -> None:
        self._zoom_by(1.12 if event.delta > 0 else 1 / 1.12)

    def _zoom_by(self, factor: float) -> None:
        self._zoom = float(np.clip(self._zoom * factor, 0.1, 20.0))
        self._schedule_render()

    def reset(self) -> None:
        self._yaw = self._pitch = 0.0
        self._zoom = 1.0
        self._pan[:] = 0.0
        self._schedule_render()

    def _schedule_render(self) -> None:
        if not self._render_pending:
            self._render_pending = True
            self.after_idle(self._render)

    def _render(self) -> None:
        self._render_pending = False
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rotation_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rotated = (self._points - self._center) @ (rotation_x @ rotation_y).T
        pixel_scale = min(width, height) * self._base_scale * self._zoom
        x = np.rint(
            width / 2 + rotated[:, 0] * pixel_scale + self._pan[0]
        ).astype(np.int64)
        y = np.rint(
            height / 2 + rotated[:, 1] * pixel_scale + self._pan[1]
        ).astype(np.int64)
        visible = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x[visible], y[visible]
        colors = self._colors[visible]
        depth = rotated[visible, 2]
        order = np.argsort(depth)[::-1]
        x, y, colors = x[order], y[order], colors[order]
        image = np.full((height, width, 3), 22, dtype=np.uint8)
        rgb = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
        # Large 5x5 splats remain visible after aggressive viewer downsampling.
        for offset_y in range(-2, 3):
            yy = np.clip(y + offset_y, 0, height - 1)
            for offset_x in range(-2, 3):
                xx = np.clip(x + offset_x, 0, width - 1)
                image[yy, xx] = rgb
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + image.tobytes()
        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        self.canvas.delete("cloud")
        self.canvas.create_image(
            width // 2,
            height // 2,
            image=self._photo,
            anchor="center",
            tags="cloud",
        )


class CalibrationApp(ttk.Frame):
    """Small CPU-only calibration application."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=18, style="App.TFrame")
        self.master = master
        self.grid(sticky="nsew")
        master.title("Multicam Calibration Studio")
        master.geometry("1100x760")
        master.minsize(940, 680)
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._devices: dict[str, RealSenseDevice] = {}
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._result: RegistrationResult | None = None
        self._T_base_cam2: np.ndarray | None = None
        self._pcd_cam1: o3d.geometry.PointCloud | None = None
        self._pcd_cam2: o3d.geometry.PointCloud | None = None
        self._preview_photos: tuple[tk.PhotoImage, tk.PhotoImage] | None = None
        self._ros_tf: object | None = None
        self._captured_from_ros_pair = False
        self._result_child_frame = "camera_2_depth_optical_frame"

        self.camera1_var = tk.StringVar()
        self.camera2_var = tk.StringVar()
        self.base_frame_var = tk.StringVar(value="base")
        self.cam1_frame_var = tk.StringVar(value="camera_1_depth_optical_frame")
        self.cam2_frame_var = tk.StringVar(value="camera_2_depth_optical_frame")
        self.transform_source_var = tk.StringVar(value="Manual / file")
        self.voxel_var = tk.StringVar(value="0.008")
        self.global_voxel_var = tk.StringVar(value="0.025")
        self.refinement_var = tk.StringVar(value="point_to_plane")
        self.registration_mode_var = tk.StringVar(value="FPFH global + ICP")
        self.warmup_var = tk.StringVar(value="30")
        self.accumulate_var = tk.StringVar(value="3")
        self.cam1_min_depth_var = tk.StringVar(value="0.10")
        self.cam1_max_depth_var = tk.StringVar(value="3.00")
        self.cam2_min_depth_var = tk.StringVar(value="0.07")
        self.cam2_max_depth_var = tk.StringVar(value="1.00")
        self.cam1_acquisition_var = tk.StringVar(value="Direct RealSense")
        self.cam1_pointcloud_topic_var = tk.StringVar(
            value="/camera_1/depth/color/points"
        )
        self.cam2_pointcloud_topic_var = tk.StringVar(
            value="/camera_2/depth/color/points"
        )
        self.cam2_link_frame_var = tk.StringVar(value="camera_2_link")
        self.status_var = tk.StringVar(value="Select two cameras and enter camera 1 TF.")
        self.transform_vars = [
            tk.StringVar(value=value)
            for value in ("0", "0", "0", "0", "0", "0", "1")
        ]
        self.initial_pose_vars = [
            tk.StringVar(value="0") for _ in range(6)
        ]
        self._manual_transform_expanded = False

        self._configure_styles()
        self._build_header()
        self._build_tabs()
        self._build_camera_section()
        self._build_transform_section()
        self._build_capture_settings()
        self._build_registration_section()
        self._build_actions()
        self._build_previews()
        self._build_output()
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_events)
        self.after(50, self.refresh_devices)

    def _configure_styles(self) -> None:
        """Apply a restrained visual system shared by native and themed Tk."""
        style = ttk.Style(self.master)
        if ttkb is None:
            style.theme_use("clam")
        style.configure("App.TFrame", background="#10151d")
        style.configure(
            "Header.TLabel",
            font=("TkDefaultFont", 22, "bold"),
            foreground="#f4f7fb",
            background="#10151d",
        )
        style.configure(
            "Subtitle.TLabel",
            font=("TkDefaultFont", 10),
            foreground="#94a3b8",
            background="#10151d",
        )
        style.configure("Section.TLabelframe", padding=18)
        style.configure(
            "Section.TLabelframe.Label", font=("TkDefaultFont", 11, "bold")
        )
        style.configure("Primary.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("Status.TLabel", font=("TkDefaultFont", 9))
        style.configure("TNotebook", tabmargins=(3, 6, 3, 0))
        style.configure("TNotebook.Tab", padding=(18, 10))

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, text="Multicam Calibration Studio", style="Header.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=(
                "Targetless RGB-D alignment  •  FPFH global registration  •  "
                "Multiscale ICP refinement"
            ),
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self._tab_sources = ttk.Frame(self.notebook, padding=18)
        self._tab_transform = ttk.Frame(self.notebook, padding=18)
        self._tab_capture = ttk.Frame(self.notebook, padding=18)
        self._tab_registration = ttk.Frame(self.notebook, padding=18)
        self._tab_result = ttk.Frame(self.notebook, padding=18)
        tabs = (
            (self._tab_sources, "1  Sources"),
            (self._tab_transform, "2  ROS & TF"),
            (self._tab_capture, "3  Capture & Inspect"),
            (self._tab_registration, "4  Registration"),
            (self._tab_result, "5  Result"),
        )
        for tab, label in tabs:
            tab.columnconfigure(0, weight=1)
            self.notebook.add(tab, text=label)
        self._tab_result.rowconfigure(1, weight=1)

    @staticmethod
    def _tab_intro(parent: tk.Misc, title: str, description: str) -> None:
        intro = ttk.Frame(parent)
        intro.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(
            intro, text=title, font=("TkDefaultFont", 15, "bold")
        ).pack(anchor="w")
        ttk.Label(
            intro, text=description, foreground="#8795a8"
        ).pack(anchor="w", pady=(3, 0))

    def _build_camera_section(self) -> None:
        self._tab_intro(
            self._tab_sources,
            "Choose camera inputs",
            "Use two direct RealSense devices, or receive both clouds from ROS.",
        )
        frame = ttk.LabelFrame(
            self._tab_sources,
            text="Camera streams",
            padding=18,
            style="Section.TLabelframe",
        )
        frame.grid(row=1, column=0, sticky="new")
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Camera 1 (calibrated)").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.camera1_combo = ttk.Combobox(
            frame, textvariable=self.camera1_var, state="readonly"
        )
        self.camera1_combo.grid(row=0, column=1, sticky="ew")
        ttk.Label(frame, text="Camera 2 (to calibrate)").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(7, 0)
        )
        self.camera2_combo = ttk.Combobox(
            frame, textvariable=self.camera2_var, state="readonly"
        )
        self.camera2_combo.grid(row=1, column=1, sticky="ew", pady=(7, 0))
        self.camera1_combo.bind("<<ComboboxSelected>>", self._camera_selection_changed)
        self.camera2_combo.bind("<<ComboboxSelected>>", self._camera_selection_changed)
        self.refresh_button = ttk.Button(
            frame, text="Refresh", command=self.refresh_devices
        )
        self.refresh_button.grid(row=0, column=2, rowspan=2, padx=(8, 0))
        ttk.Label(frame, text="Standalone camera 2 optical frame").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(7, 0)
        )
        ttk.Entry(frame, textvariable=self.cam2_frame_var).grid(
            row=2, column=1, sticky="ew", pady=(7, 0)
        )

    def _build_transform_section(self) -> None:
        self._tab_intro(
            self._tab_transform,
            "Connect both camera trees",
            "Recommended: run both ROS camera drivers and connect to their TF and clouds.",
        )
        ros = ttk.LabelFrame(
            self._tab_transform,
            text="Recommended  •  Live ROS 2 TF",
            padding=18,
            style="Section.TLabelframe",
        )
        ros.grid(row=1, column=0, sticky="new")
        ros.columnconfigure(1, weight=1)
        ros.columnconfigure(3, weight=1)
        ttk.Label(
            ros,
            text=(
                "Camera 1 supplies the known base pose. Camera 2 supplies its "
                "factory link-to-optical transform."
            ),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        ttk.Label(ros, text="Base frame").grid(row=1, column=0, sticky="w")
        ttk.Entry(ros, textvariable=self.base_frame_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 18)
        )
        self.ros_connect_button = ttk.Button(
            ros,
            text="Connect ROS TF",
            command=self.connect_ros_tf,
            style="Primary.TButton",
        )
        self.ros_connect_button.grid(row=1, column=2, sticky="w")
        self.ros_connection_label = ttk.Label(ros, text="Not connected")
        self.ros_connection_label.grid(row=1, column=3, sticky="w", padx=(10, 0))
        ttk.Label(ros, text="Cloud acquisition").grid(
            row=2, column=0, sticky="w", pady=(14, 0)
        )
        ttk.Combobox(
            ros,
            textvariable=self.cam1_acquisition_var,
            values=("Direct RealSense", "ROS PointCloud2"),
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", padx=(8, 18), pady=(14, 0))
        ttk.Label(ros, text="Camera 1 PointCloud2").grid(
            row=2, column=2, sticky="w", pady=(14, 0)
        )
        self.cam1_topic_combo = ttk.Combobox(
            ros,
            textvariable=self.cam1_pointcloud_topic_var,
            state="normal",
        )
        self.cam1_topic_combo.grid(
            row=2, column=3, sticky="ew", padx=(8, 0), pady=(14, 0)
        )
        ttk.Label(ros, text="Camera 1 cloud frame").grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Label(
            ros,
            textvariable=self.cam1_frame_var,
            relief="sunken",
            padding=(7, 3),
        ).grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(10, 0)
        )
        ttk.Label(ros, text="Camera 2 PointCloud2").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        self.cam2_topic_combo = ttk.Combobox(
            ros,
            textvariable=self.cam2_pointcloud_topic_var,
            state="normal",
        )
        self.cam2_topic_combo.grid(
            row=4, column=1, sticky="ew", padx=(8, 18), pady=(10, 0)
        )
        ttk.Label(ros, text="Camera 2 link frame").grid(
            row=4, column=2, sticky="w", pady=(10, 0)
        )
        self.cam2_link_combo = ttk.Combobox(
            ros,
            textvariable=self.cam2_link_frame_var,
            state="normal",
        )
        self.cam2_link_combo.grid(
            row=4, column=3, sticky="ew", padx=(8, 0), pady=(10, 0)
        )
        ttk.Label(ros, text="Camera 2 cloud frame").grid(
            row=5, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Label(
            ros,
            textvariable=self.cam2_frame_var,
            relief="sunken",
            padding=(7, 3),
        ).grid(
            row=5, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(10, 0)
        )
        ttk.Button(
            ros,
            text="Export temporary camera 2 identity TF…",
            command=self.export_dummy_camera2_tf,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(14, 0))

        self.manual_toggle_button = ttk.Button(
            self._tab_transform,
            text="▸  Manual transform or import from file",
            command=self._toggle_manual_transform,
        )
        self.manual_toggle_button.grid(
            row=2, column=0, sticky="w", pady=(16, 0)
        )
        frame = ttk.LabelFrame(
            self._tab_transform,
            text="Manual transform  (base ← camera 1)",
            padding=18,
            style="Section.TLabelframe",
        )
        self._manual_transform_frame = frame
        frame.grid(row=3, column=0, sticky="new", pady=(10, 0))
        frame.grid_remove()
        for column in range(8):
            frame.columnconfigure(column, weight=1 if column > 0 else 0)

        ttk.Label(frame, text="Translation (m)").grid(row=0, column=0, sticky="w")
        for index, label in enumerate(("X", "Y", "Z")):
            ttk.Label(frame, text=label).grid(row=0, column=index * 2 + 1)
            ttk.Entry(
                frame, width=10, textvariable=self.transform_vars[index]
            ).grid(row=0, column=index * 2 + 2, sticky="ew", padx=(2, 8))

        ttk.Label(frame, text="Quaternion XYZW").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        for index, label in enumerate(("QX", "QY", "QZ", "QW")):
            ttk.Label(frame, text=label).grid(
                row=1, column=index * 2 + 1, pady=(8, 0)
            )
            ttk.Entry(
                frame, width=10, textvariable=self.transform_vars[index + 3]
            ).grid(
                row=1,
                column=index * 2 + 2,
                sticky="ew",
                padx=(2, 8),
                pady=(8, 0),
            )

        names = ttk.Frame(frame)
        names.grid(row=2, column=0, columnspan=9, sticky="ew", pady=(9, 0))
        names.columnconfigure(1, weight=1)
        names.columnconfigure(3, weight=1)
        ttk.Label(names, text="Base frame").grid(row=0, column=0, sticky="w")
        ttk.Entry(names, textvariable=self.base_frame_var).grid(
            row=0, column=1, sticky="ew", padx=(5, 12)
        )
        ttk.Label(names, text="Camera 1 frame").grid(row=0, column=2, sticky="w")
        ttk.Entry(names, textvariable=self.cam1_frame_var).grid(
            row=0, column=3, sticky="ew", padx=(5, 12)
        )
        ttk.Button(names, text="Import TF…", command=self.import_transform).grid(
            row=0, column=4
        )
        ttk.Button(
            names,
            text="Use manual transform",
            command=self._use_manual_transform,
            style="Primary.TButton",
        ).grid(row=0, column=5, padx=(8, 0))

    def _toggle_manual_transform(self) -> None:
        self._manual_transform_expanded = not self._manual_transform_expanded
        if self._manual_transform_expanded:
            self._manual_transform_frame.grid()
            self.manual_toggle_button.configure(
                text="▾  Manual transform or import from file"
            )
        else:
            self._manual_transform_frame.grid_remove()
            self.manual_toggle_button.configure(
                text="▸  Manual transform or import from file"
            )

    def _use_manual_transform(self) -> None:
        try:
            self._known_transform()
        except ValueError as error:
            messagebox.showerror("Invalid transform", str(error), parent=self)
            return
        self.transform_source_var.set("Manual / file")
        self.status_var.set("Using the manually entered camera 1 transform.")

    def _build_capture_settings(self) -> None:
        self._tab_intro(
            self._tab_capture,
            "Capture and validate the scene",
            "Tune each depth range, capture once, then inspect both static clouds.",
        )
        frame = ttk.LabelFrame(
            self._tab_capture,
            text="Capture settings",
            padding=18,
            style="Section.TLabelframe",
        )
        frame.grid(row=1, column=0, sticky="new")
        options = (
            ("Warm-up frames", self.warmup_var),
            ("Accumulated frames", self.accumulate_var),
        )
        for index, (label, variable) in enumerate(options):
            ttk.Label(frame, text=label).grid(
                row=0, column=index * 2, sticky="w", padx=(0 if index == 0 else 16, 5)
            )
            ttk.Entry(frame, textvariable=variable, width=9).grid(
                row=0, column=index * 2 + 1
            )
        ttk.Label(frame, text="Cam 1 depth min/max (m)").grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(
            frame, textvariable=self.cam1_min_depth_var, width=8
        ).grid(row=1, column=1, pady=(12, 0))
        ttk.Entry(
            frame, textvariable=self.cam1_max_depth_var, width=8
        ).grid(row=1, column=2, pady=(12, 0))
        ttk.Label(frame, text="Cam 2 depth min/max (m)").grid(
            row=1, column=3, sticky="e", padx=(16, 5), pady=(12, 0)
        )
        ttk.Entry(
            frame, textvariable=self.cam2_min_depth_var, width=8
        ).grid(row=1, column=4, pady=(12, 0))
        ttk.Entry(
            frame, textvariable=self.cam2_max_depth_var, width=8
        ).grid(row=1, column=5, pady=(12, 0))

    def _build_registration_section(self) -> None:
        self._tab_intro(
            self._tab_registration,
            "Align the captured point clouds",
            "Start globally, or constrain the search with an approximate pose.",
        )
        frame = ttk.LabelFrame(
            self._tab_registration,
            text="Solver settings",
            padding=18,
            style="Section.TLabelframe",
        )
        frame.grid(row=1, column=0, sticky="new")
        options = (
            ("ICP voxel (m)", self.voxel_var),
            ("Global voxel (m)", self.global_voxel_var),
        )
        for index, (label, variable) in enumerate(options):
            ttk.Label(frame, text=label).grid(
                row=0, column=index * 2, sticky="w",
                padx=(0 if index == 0 else 18, 6),
            )
            ttk.Entry(frame, textvariable=variable, width=10).grid(
                row=0, column=index * 2 + 1
            )
        ttk.Label(frame, text="Refinement").grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Combobox(
            frame,
            textvariable=self.refinement_var,
            values=("point_to_plane", "colored"),
            state="readonly",
            width=19,
        ).grid(row=1, column=1, sticky="w", pady=(12, 0))
        ttk.Label(frame, text="Registration mode").grid(
            row=2, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Combobox(
            frame,
            textvariable=self.registration_mode_var,
            values=("FPFH global + ICP", "Approximate pose + ICP"),
            state="readonly",
            width=24,
        ).grid(row=2, column=1, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(frame, text="Approx. camera 2 XYZ (m)").grid(
            row=3, column=0, sticky="w", pady=(12, 0)
        )
        for index, label in enumerate(("X", "Y", "Z")):
            ttk.Label(frame, text=label).grid(
                row=3, column=index * 2 + 1, pady=(12, 0)
            )
            ttk.Entry(
                frame, textvariable=self.initial_pose_vars[index], width=8
            ).grid(row=3, column=index * 2 + 2, pady=(12, 0))
        ttk.Label(frame, text="Approx. roll/pitch/yaw (deg)").grid(
            row=4, column=0, sticky="w", pady=(12, 0)
        )
        for index, label in enumerate(("R", "P", "Y")):
            ttk.Label(frame, text=label).grid(
                row=4, column=index * 2 + 1, pady=(12, 0)
            )
            ttk.Entry(
                frame, textvariable=self.initial_pose_vars[index + 3], width=8
            ).grid(row=4, column=index * 2 + 2, pady=(12, 0))

    def _build_actions(self) -> None:
        capture_actions = ttk.Frame(self._tab_capture)
        capture_actions.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.go_button = ttk.Button(
            capture_actions,
            text="Capture streams",
            command=self.start_calibration,
            style="Primary.TButton",
        )
        self.go_button.grid(row=0, column=0)

        registration_actions = ttk.Frame(self._tab_registration)
        registration_actions.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.recalculate_button = ttk.Button(
            registration_actions,
            text="Run calibration",
            command=self.recalculate_captured,
            state="disabled",
            style="Primary.TButton",
        )
        self.recalculate_button.grid(row=0, column=0)

        result_actions = ttk.Frame(self._tab_result)
        result_actions.grid(row=2, column=0, sticky="e", pady=(14, 0))
        self.save_button = ttk.Button(
            result_actions,
            text="Export data…",
            command=self.export_result,
            state="disabled",
        )
        self.save_button.pack(side="right")
        self.save_launch_button = ttk.Button(
            result_actions,
            text="Export ROS 2 launch.py…",
            command=self.export_launch_result,
            state="disabled",
            style="Primary.TButton",
        )
        self.save_launch_button.pack(side="right", padx=(0, 8))

        footer = ttk.Frame(self, padding=(2, 12, 2, 0), style="App.TFrame")
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            textvariable=self.status_var,
            style="Status.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 0))

    def _build_previews(self) -> None:
        frame = ttk.LabelFrame(
            self._tab_capture,
            text="Captured views",
            padding=14,
            style="Section.TLabelframe",
        )
        frame.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        self._tab_capture.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self.preview1 = ttk.Label(
            frame, text="Camera 1 preview", anchor="center"
        )
        self.preview2 = ttk.Label(
            frame, text="Camera 2 preview", anchor="center"
        )
        self.preview1.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.preview2.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        controls = ttk.Frame(frame)
        controls.grid(row=1, column=0, columnspan=2, pady=(7, 0))
        self.view_overlay_button = ttk.Button(
            controls,
            text="Inspect registered overlay",
            command=lambda: self._open_cloud_viewer("overlay"),
            state="disabled",
        )
        self.view_overlay_button.grid(row=0, column=0, padx=3)
        self.view_3d_button = ttk.Button(
            controls,
            text="Inspect both clouds in 3D",
            command=self._open_interactive_3d_viewer,
            state="disabled",
        )
        self.view_3d_button.grid(row=0, column=1, padx=3)

    def _build_output(self) -> None:
        self._tab_intro(
            self._tab_result,
            "Calibration result",
            "Review quality metrics and export the base-to-camera-2 transform.",
        )
        frame = ttk.LabelFrame(
            self._tab_result,
            text="Camera 2 transform",
            padding=14,
            style="Section.TLabelframe",
        )
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.output = tk.Text(frame, height=12, wrap="none", state="disabled")
        self.output.grid(row=0, column=0, sticky="nsew")
        self.output.configure(
            background="#111827",
            foreground="#dbeafe",
            insertbackground="#dbeafe",
            selectbackground="#2563eb",
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=14,
            font=("TkFixedFont", 10),
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.output.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)

    def refresh_devices(self) -> None:
        self.status_var.set("Searching for RealSense cameras…")
        self.refresh_button.configure(state="disabled")

        def worker() -> None:
            try:
                self._events.put(("devices", list_devices()))
            except Exception as error:
                self._events.put(("error", error))

        threading.Thread(target=worker, daemon=True).start()

    def connect_ros_tf(self) -> None:
        """Attach the standalone GUI to a live ROS 2 TF tree."""
        if self._ros_tf is not None:
            self.transform_source_var.set("Live ROS TF")
            self.status_var.set("Already connected; refreshing TF frames…")
            self._ros_tf.wait_for_frames(  # type: ignore[attr-defined]
                lambda frames: self._events.put(("tf_frames", frames))
            )
            return
        self.ros_connect_button.configure(state="disabled")
        self.ros_connection_label.configure(text="Connecting…")
        self.status_var.set("Listening to /tf and /tf_static…")
        try:
            from .ros_tf import LiveRosTf

            self._ros_tf = LiveRosTf()
            self._ros_tf.wait_for_frames(
                lambda frames: self._events.put(("tf_frames", frames))
            )
        except Exception as error:
            self._ros_tf = None
            self.ros_connect_button.configure(state="normal")
            self.ros_connection_label.configure(text="Unavailable")
            messagebox.showerror("ROS TF connection failed", str(error), parent=self)

    def _on_close(self) -> None:
        if self._ros_tf is not None:
            try:
                self._ros_tf.close()  # type: ignore[attr-defined]
            except Exception:
                LOGGER.exception("Failed to close ROS TF listener")
        self.master.destroy()

    def _camera_selection_changed(self, _event: object | None = None) -> None:
        """Apply sensible range defaults for each selected camera model."""
        selections = (
            (
                self.camera1_var.get(),
                self.cam1_min_depth_var,
                self.cam1_max_depth_var,
            ),
            (
                self.camera2_var.get(),
                self.cam2_min_depth_var,
                self.cam2_max_depth_var,
            ),
        )
        for label, minimum_var, maximum_var in selections:
            device = self._devices.get(label)
            if device is None:
                continue
            if "D405" in device.name.upper():
                minimum_var.set("0.07")
                maximum_var.set("1.00")
            else:
                minimum_var.set("0.10")
                maximum_var.set("3.00")

    def import_transform(self) -> None:
        path = filedialog.askopenfilename(
            title="Import camera 1 transform",
            filetypes=[
                ("Supported transforms", "*.json *.npy *.txt *.csv *.launch.py"),
                ("All files", "*"),
            ],
        )
        if not path:
            return
        try:
            transform = load_transform(path)
            quaternion = quaternion_from_matrix(transform)
            values = [*transform[:3, 3], *quaternion]
            for variable, value in zip(self.transform_vars, values):
                variable.set(f"{value:.10g}")
            self.transform_source_var.set("Manual / file")
            self.status_var.set(f"Imported camera 1 TF from {Path(path).name}")
            if Path(path).name.endswith(".launch.py"):
                messagebox.showwarning(
                    "Verify transform frame",
                    "Captured geometry is now expressed in camera 1's native "
                    "depth optical frame. The imported launch transform is only "
                    "directly valid if its child frame is that depth optical "
                    "frame. A base-to-camera-link transform must first be "
                    "composed with the camera-link-to-depth-optical transform.",
                    parent=self,
                )
        except Exception as error:
            messagebox.showerror("Transform import failed", str(error), parent=self)

    def _known_transform(self) -> np.ndarray:
        try:
            values = [float(variable.get()) for variable in self.transform_vars]
        except ValueError as error:
            raise ValueError("All camera 1 TF fields must be numeric") from error
        return matrix_from_translation_quaternion(values[:3], values[3:])

    def export_dummy_camera2_tf(self) -> None:
        """Write an identity base-to-camera2-link transform for temporary bringup."""
        parent = self.base_frame_var.get().strip() or "base"
        child = self.cam2_link_frame_var.get().strip()
        if not child:
            messagebox.showerror(
                "Missing frame", "Camera 2 link frame is required.", parent=self
            )
            return
        path = filedialog.asksaveasfilename(
            title="Export temporary camera 2 identity transform",
            initialfile="camera_2_dummy_tf.launch.py",
            defaultextension=".launch.py",
            filetypes=[("ROS 2 Python launch file", "*.launch.py")],
        )
        if not path:
            return
        if not path.endswith(".launch.py"):
            path = f"{path.removesuffix('.py')}.launch.py"
        try:
            save_transform(
                path,
                np.eye(4),
                parent_frame=parent,
                child_frame=child,
                metadata={"temporary_identity_transform": True},
            )
            self.status_var.set(
                f"Exported temporary identity TF {Path(path).name}. "
                "Replace it with the calibration result."
            )
        except Exception as error:
            messagebox.showerror("Dummy TF export failed", str(error), parent=self)

    def start_calibration(self) -> None:
        try:
            use_ros_clouds = (
                self.cam1_acquisition_var.get() == "ROS PointCloud2"
            )
            if use_ros_clouds:
                if self._ros_tf is None:
                    raise ValueError(
                        "Connect to ROS TF before using ROS PointCloud2"
                    )
                cloud1_topic = self.cam1_pointcloud_topic_var.get().strip()
                cloud2_topic = self.cam2_pointcloud_topic_var.get().strip()
                if not cloud1_topic or not cloud2_topic:
                    raise ValueError("Both camera PointCloud2 topics are required")
                if cloud1_topic == cloud2_topic:
                    raise ValueError("Select two different PointCloud2 topics")
                device1 = device2 = None
            else:
                device2 = self._devices[self.camera2_var.get()]
                device1 = self._devices[self.camera1_var.get()]
                if device1.serial == device2.serial:
                    raise ValueError(
                        "Camera 1 and camera 2 must be different devices"
                    )
            warmup = int(self.warmup_var.get())
            accumulate = int(self.accumulate_var.get())
            min_depth_cam1 = float(self.cam1_min_depth_var.get())
            max_depth_cam1 = float(self.cam1_max_depth_var.get())
            min_depth_cam2 = float(self.cam2_min_depth_var.get())
            max_depth_cam2 = float(self.cam2_max_depth_var.get())
        except (KeyError, ValueError) as error:
            messagebox.showerror("Invalid configuration", str(error), parent=self)
            return

        self.go_button.configure(state="disabled")
        self.recalculate_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.save_launch_button.configure(state="disabled")
        self.view_overlay_button.configure(state="disabled")
        self.view_3d_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Capturing both streams… keep the scene static.")
        self._result = None
        self._T_base_cam2 = None
        self._pcd_cam1 = None
        self._pcd_cam2 = None
        self._captured_from_ros_pair = False

        def worker() -> None:
            try:
                if use_ros_clouds:
                    assert self._ros_tf is not None
                    (
                        pcd_cam1,
                        cloud1_frame,
                        pcd_cam2,
                        cloud2_frame,
                    ) = self._ros_tf.capture_pointcloud_pair(  # type: ignore[attr-defined]
                        cloud1_topic, cloud2_topic
                    )
                    cropped = []
                    for cloud, minimum, maximum in (
                        (pcd_cam1, min_depth_cam1, max_depth_cam1),
                        (pcd_cam2, min_depth_cam2, max_depth_cam2),
                    ):
                        points = np.asarray(cloud.points)
                        keep = np.flatnonzero(
                            (points[:, 2] >= minimum)
                            & (points[:, 2] <= maximum)
                        )
                        cropped.append(cloud.select_by_index(keep.tolist()))
                    pcd_cam1, pcd_cam2 = cropped
                    self._events.put(
                        ("ros_cloud_frames", (cloud1_frame, cloud2_frame))
                    )
                else:
                    assert device1 is not None and device2 is not None
                    pcd_cam1, pcd_cam2 = capture_pair(
                        device1.serial,
                        device2.serial,
                        warmup_frames=warmup,
                        accumulate_frames=accumulate,
                        min_depth_cam1_m=min_depth_cam1,
                        max_depth_cam1_m=max_depth_cam1,
                        min_depth_cam2_m=min_depth_cam2,
                        max_depth_cam2_m=max_depth_cam2,
                        preview_callback=lambda image1, image2: self._events.put(
                            ("preview", (image1, image2))
                        ),
                    )
                self._events.put(("clouds", (pcd_cam1, pcd_cam2)))
                if use_ros_clouds:
                    self._events.put(("ros_pair_capture", True))
                self._events.put(("capture_complete", None))
            except Exception as error:
                LOGGER.exception("Calibration failed")
                self._events.put(("calibration_error", error))

        threading.Thread(target=worker, daemon=True).start()

    def recalculate_captured(self) -> None:
        """Rerun registration on cached clouds without touching the cameras."""
        if self._pcd_cam1 is None or self._pcd_cam2 is None:
            messagebox.showinfo(
                "No capture", "Capture both cameras first.", parent=self
            )
            return
        try:
            config = RegistrationConfig(
                voxel_size=float(self.voxel_var.get()),
                global_voxel_size=float(self.global_voxel_var.get()),
                refinement=self.refinement_var.get(),  # type: ignore[arg-type]
            )
            use_global = self.registration_mode_var.get() == "FPFH global + ICP"
            guess_values = [
                float(variable.get()) for variable in self.initial_pose_vars
            ]
            T_base_cam2_guess = matrix_from_translation_euler(
                guess_values[:3], guess_values[3:]
            )
            live_ros = self.transform_source_var.get() == "Live ROS TF"
            if live_ros and self._ros_tf is None:
                raise ValueError("Connect to ROS TF before using live TF mode")
            if live_ros:
                base_frame = self.base_frame_var.get().strip()
                cam1_cloud = self.cam1_frame_var.get().strip()
                cam2_cloud = self.cam2_frame_var.get().strip()
                if not all((base_frame, cam1_cloud, cam2_cloud)):
                    raise ValueError(
                        "Base, resolved camera 1 cloud, and camera 2 output "
                        "frame names are required"
                    )
                cam2_link = self.cam2_link_frame_var.get().strip()
                if self._captured_from_ros_pair and not cam2_link:
                    raise ValueError("Camera 2 link frame is required")
                T_base_cam1_manual = None
            else:
                T_base_cam1_manual = self._known_transform()
                cam2_cloud = self.cam2_frame_var.get().strip()
                cam2_link = self.cam2_link_frame_var.get().strip()
                if self._captured_from_ros_pair and not cam2_link:
                    raise ValueError("Camera 2 link frame is required")
        except ValueError as error:
            messagebox.showerror("Invalid configuration", str(error), parent=self)
            return

        pcd_cam1 = self._pcd_cam1
        pcd_cam2 = self._pcd_cam2
        self.recalculate_button.configure(state="disabled")
        self.go_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.save_launch_button.configure(state="disabled")
        self.view_overlay_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Recalculating from the cached capture…")

        def worker() -> None:
            try:
                if live_ros:
                    assert self._ros_tf is not None
                    # tf2 resolves the complete base -> camera-link -> optical
                    # chain internally. Calibration only needs the frame in
                    # which the subscribed PointCloud2 is actually expressed.
                    T_base_cloud1 = self._ros_tf.lookup_matrix(  # type: ignore[attr-defined]
                        base_frame, cam1_cloud
                    )
                    if self._captured_from_ros_pair:
                        # This is an internal factory/static transform. The
                        # temporary base->camera2_link identity is deliberately
                        # not used as calibration data.
                        T_link2_cloud2 = self._ros_tf.lookup_matrix(  # type: ignore[attr-defined]
                            cam2_link, cam2_cloud
                        )
                        T_base_cloud2_guess = (
                            T_base_cam2_guess @ T_link2_cloud2
                        )
                    else:
                        T_link2_cloud2 = None
                        T_base_cloud2_guess = T_base_cam2_guess
                    initial = (
                        np.linalg.inv(T_base_cloud1) @ T_base_cloud2_guess
                    )
                else:
                    assert T_base_cam1_manual is not None
                    T_base_cloud1 = T_base_cam1_manual
                    if self._captured_from_ros_pair and self._ros_tf is not None:
                        T_link2_cloud2 = self._ros_tf.lookup_matrix(  # type: ignore[attr-defined]
                            cam2_link, cam2_cloud
                        )
                        T_base_cloud2_guess = (
                            T_base_cam2_guess @ T_link2_cloud2
                        )
                    else:
                        T_link2_cloud2 = None
                        T_base_cloud2_guess = T_base_cam2_guess
                    initial = (
                        np.linalg.inv(T_base_cloud1) @ T_base_cloud2_guess
                    )
                result = calibrate(
                    pcd_cam1,
                    pcd_cam2,
                    config=config,
                    return_result=True,
                    progress_callback=lambda message: self._events.put(
                        ("status", message)
                    ),
                    initial_transform=initial,
                    use_global_registration=use_global,
                )
                assert isinstance(result, RegistrationResult)
                T_base_cloud2 = (
                    T_base_cloud1 @ result.transformation
                )
                if T_link2_cloud2 is not None:
                    output_transform = (
                        T_base_cloud2 @ np.linalg.inv(T_link2_cloud2)
                    )
                    output_frame = cam2_link
                else:
                    output_transform = T_base_cloud2
                    output_frame = cam2_cloud
                self._events.put(
                    ("result", (result, output_transform, output_frame))
                )
            except Exception as error:
                LOGGER.exception("Recalculation failed")
                self._events.put(("calibration_error", error))

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _photo_from_rgb(image: np.ndarray, max_width: int = 410) -> tk.PhotoImage:
        """Create a Tk image from RGB data without requiring Pillow."""
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Preview must be an RGB image")
        step = max(1, int(np.ceil(image.shape[1] / max_width)))
        resized = np.ascontiguousarray(image[::step, ::step, :3], dtype=np.uint8)
        height, width = resized.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + resized.tobytes()
        return tk.PhotoImage(data=ppm, format="PPM")

    def _show_previews(self, image1: np.ndarray, image2: np.ndarray) -> None:
        photo1 = self._photo_from_rgb(image1)
        photo2 = self._photo_from_rgb(image2)
        self._preview_photos = (photo1, photo2)
        self.preview1.configure(image=photo1, text="")
        self.preview2.configure(image=photo2, text="")

    def _open_cloud_viewer(self, view: str) -> None:
        """Open software-rendered orthographic views without OpenGL/GLFW."""
        if self._pcd_cam1 is None or self._pcd_cam2 is None:
            return
        if view == "camera1":
            clouds = [
                (
                    np.asarray(self._pcd_cam1.points),
                    np.asarray(self._pcd_cam1.colors),
                )
            ]
            title = "Camera 1 captured cloud"
        elif view == "camera2":
            clouds = [
                (
                    np.asarray(self._pcd_cam2.points),
                    np.asarray(self._pcd_cam2.colors),
                )
            ]
            title = "Camera 2 captured cloud"
        else:
            if self._result is None:
                return
            target_points = np.asarray(self._pcd_cam1.points)
            source_points = np.asarray(self._pcd_cam2.points)
            rotation = self._result.transformation[:3, :3]
            translation = self._result.transformation[:3, 3]
            source_points = source_points @ rotation.T + translation
            clouds = [
                (
                    target_points,
                    np.tile([0.15, 0.85, 0.25], (len(target_points), 1)),
                ),
                (
                    source_points,
                    np.tile([0.95, 0.15, 0.75], (len(source_points), 1)),
                ),
            ]
            title = "Registered overlay: camera 1 green, camera 2 magenta"

        try:
            window = tk.Toplevel(self)
            window.title(title)
            views = (
                ("Optical perspective: X/Z / -Y/Z", None, 2, True),
                ("Top: X / Z", (0, 2), 1, False),
                ("Side: Z / -Y", (2, 1), 0, True),
            )
            photos: list[tk.PhotoImage] = []
            for column, (label, axes, depth_axis, flip_vertical) in enumerate(views):
                ttk.Label(window, text=label).grid(
                    row=0, column=column, pady=(8, 3)
                )
                if axes is None:
                    image = self._render_cloud_perspective(clouds)
                else:
                    image = self._render_cloud_projection(
                        clouds,
                        horizontal_axis=axes[0],
                        vertical_axis=axes[1],
                        depth_axis=depth_axis,
                        flip_vertical=flip_vertical,
                    )
                photo = self._photo_from_rgb(image, max_width=image.shape[1])
                photos.append(photo)
                ttk.Label(window, image=photo).grid(
                    row=1, column=column, padx=5, pady=(0, 5)
                )
            ttk.Label(
                window,
                text=(
                    "Perspective approximates the RGB view. Top and side use "
                    "equal metric scale; sparse holes are missing depth."
                ),
            ).grid(row=2, column=0, columnspan=3, pady=(2, 8))
            # Tk images are deleted if Python releases the final reference.
            window._cloud_photos = photos  # type: ignore[attr-defined]
        except Exception as error:
            messagebox.showerror("Cloud viewer failed", str(error), parent=self)

    def _open_interactive_3d_viewer(self) -> None:
        """Open two fast software orbit views with robust rotation centers."""
        if self._pcd_cam1 is None or self._pcd_cam2 is None:
            return

        window = tk.Toplevel(self)
        window.title("Interactive point-cloud inspection")
        window.geometry("1280x700")
        window.columnconfigure(0, weight=1)
        window.columnconfigure(1, weight=1)
        window.rowconfigure(1, weight=1)
        ttk.Label(
            window,
            text=(
                "Left-drag: orbit around cloud center • Middle/right-drag: pan "
                "• Wheel: zoom • Double-left-click: reset"
            ),
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(7, 2))
        clouds = (
            ("Camera 1 — native depth optical frame", self._pcd_cam1),
            ("Camera 2 — native depth optical frame", self._pcd_cam2),
        )
        viewers: list[_OrbitPointCloudView] = []
        for column, (title, cloud) in enumerate(clouds):
            points = np.asarray(cloud.points)
            colors = np.asarray(cloud.colors)
            if not len(points):
                continue
            # Build spatially representative display clouds. Calibration retains
            # the full captures and is unaffected by this viewer-only reduction.
            viewer_voxel = 0.008 if len(points) > 20_000 else 0.004
            display_cloud = cloud.voxel_down_sample(viewer_voxel)
            display_points = np.asarray(display_cloud.points)
            display_colors = np.asarray(display_cloud.colors)
            maximum_display_points = 12_000
            if len(display_points) > maximum_display_points:
                indices = np.linspace(
                    0,
                    len(display_points) - 1,
                    maximum_display_points,
                    dtype=np.int64,
                )
                display_points = display_points[indices]
                display_colors = (
                    display_colors[indices]
                    if len(display_colors) == len(display_cloud.points)
                    else np.empty((0, 3))
                )
            display_count = len(display_points)
            if len(display_colors) != display_count:
                display_colors = np.full((display_count, 3), 0.75)
            else:
                display_colors = np.clip(display_colors, 0.0, 1.0)
            viewer = _OrbitPointCloudView(
                window,
                display_points,
                display_colors,
                title,
            )
            viewer.grid(
                row=1,
                column=column,
                sticky="nsew",
                padx=(7 if column == 0 else 3, 3 if column == 0 else 7),
                pady=(2, 7),
            )
            viewers.append(viewer)
        window._pointcloud_viewers = viewers  # type: ignore[attr-defined]

    @staticmethod
    def _sample_cloud_arrays(
        clouds: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine bounded samples from one or more colored clouds."""
        points_parts: list[np.ndarray] = []
        color_parts: list[np.ndarray] = []
        for points, colors in clouds:
            if not len(points):
                continue
            stride = max(1, int(np.ceil(len(points) / 100_000)))
            sampled_points = np.asarray(points[::stride], dtype=np.float64)
            if len(colors) == len(points):
                sampled_colors = np.asarray(colors[::stride], dtype=np.float64)
            else:
                sampled_colors = np.full((len(sampled_points), 3), 0.8)
            valid = np.all(np.isfinite(sampled_points), axis=1)
            points_parts.append(sampled_points[valid])
            color_parts.append(sampled_colors[valid])
        if not points_parts:
            raise ValueError("The captured point cloud is empty")
        return (
            np.concatenate(points_parts),
            np.clip(np.concatenate(color_parts), 0.0, 1.0),
        )

    @staticmethod
    def _rasterize_points(
        horizontal: np.ndarray,
        vertical: np.ndarray,
        depth: np.ndarray,
        colors: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Rasterize projected coordinates while preserving equal axis scale."""
        low_x, high_x = np.percentile(horizontal, [1.0, 99.0])
        low_y, high_y = np.percentile(vertical, [1.0, 99.0])
        span_x, span_y = high_x - low_x, high_y - low_y
        if span_x < 1.0e-9 or span_y < 1.0e-9:
            raise ValueError("Cloud has insufficient extent for this view")
        padding = 10
        scale = min(
            (width - 2 * padding - 1) / span_x,
            (height - 2 * padding - 1) / span_y,
        )
        center_x = 0.5 * (low_x + high_x)
        center_y = 0.5 * (low_y + high_y)
        x = np.rint(width / 2 + (horizontal - center_x) * scale).astype(np.int64)
        y = np.rint(height / 2 - (vertical - center_y) * scale).astype(np.int64)
        visible = (
            (horizontal >= low_x)
            & (horizontal <= high_x)
            & (vertical >= low_y)
            & (vertical <= high_y)
            & (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
        x, y = x[visible], y[visible]
        colors = colors[visible]
        depth = depth[visible]
        order = np.argsort(depth)[::-1]
        x, y, colors = x[order], y[order], colors[order]
        image = np.full((height, width, 3), 22, dtype=np.uint8)
        rgb = np.rint(colors * 255).astype(np.uint8)
        image[y, x] = rgb
        image[np.minimum(y + 1, height - 1), x] = rgb
        image[y, np.minimum(x + 1, width - 1)] = rgb
        image[
            np.minimum(y + 1, height - 1),
            np.minimum(x + 1, width - 1),
        ] = rgb
        return image

    @classmethod
    def _render_cloud_perspective(
        cls,
        clouds: list[tuple[np.ndarray, np.ndarray]],
        *,
        width: int = 430,
        height: int = 340,
    ) -> np.ndarray:
        """Render the native optical view using normalized image coordinates."""
        points, colors = cls._sample_cloud_arrays(clouds)
        valid = points[:, 2] > 1.0e-6
        points, colors = points[valid], colors[valid]
        return cls._rasterize_points(
            points[:, 0] / points[:, 2],
            -points[:, 1] / points[:, 2],
            points[:, 2],
            colors,
            width=width,
            height=height,
        )

    @classmethod
    def _render_cloud_projection(
        cls,
        clouds: list[tuple[np.ndarray, np.ndarray]],
        *,
        horizontal_axis: int,
        vertical_axis: int,
        depth_axis: int,
        flip_vertical: bool,
        width: int = 430,
        height: int = 340,
    ) -> np.ndarray:
        """Rasterize colored points with NumPy for headless/Wayland systems."""
        points, colors = cls._sample_cloud_arrays(clouds)
        horizontal = points[:, horizontal_axis]
        vertical = points[:, vertical_axis]
        if flip_vertical:
            vertical = -vertical
        return cls._rasterize_points(
            horizontal,
            vertical,
            points[:, depth_axis],
            colors,
            width=width,
            height=height,
        )

    def export_result(self) -> None:
        if self._T_base_cam2 is None or self._result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export camera 2 transform",
            defaultextension=".json",
            filetypes=[
                ("JSON with metadata", "*.json"),
                ("NumPy matrix", "*.npy"),
                ("CSV matrix", "*.csv"),
                ("Text matrix", "*.txt"),
            ],
        )
        if not path:
            return
        try:
            save_transform(
                path,
                self._T_base_cam2,
                parent_frame=self.base_frame_var.get().strip() or "base",
                child_frame=self._result_child_frame,
                metadata={
                    "fitness": self._result.fitness,
                    "inlier_rmse_m": self._result.inlier_rmse,
                    "global_fitness": self._result.global_fitness,
                    "global_inlier_rmse_m": self._result.global_inlier_rmse,
                },
            )
            self.status_var.set(f"Exported {Path(path).name}")
        except Exception as error:
            messagebox.showerror("Export failed", str(error), parent=self)

    def export_launch_result(self) -> None:
        """Export the accepted transform as a ROS 2 static-TF launch file."""
        if self._T_base_cam2 is None or self._result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export camera 2 ROS 2 static transform",
            initialfile="camera_2_tf.launch.py",
            defaultextension=".launch.py",
            filetypes=[
                ("ROS 2 Python launch file", "*.launch.py"),
                ("Python file", "*.py"),
            ],
        )
        if not path:
            return
        if not path.endswith(".launch.py"):
            path = f"{path.removesuffix('.py')}.launch.py"
        try:
            save_transform(
                path,
                self._T_base_cam2,
                parent_frame=self.base_frame_var.get().strip() or "base",
                child_frame=self._result_child_frame,
                metadata={
                    "fitness": self._result.fitness,
                    "inlier_rmse_m": self._result.inlier_rmse,
                },
            )
            self.status_var.set(
                f"Exported {Path(path).name}. Stop the temporary identity "
                "publisher before launching this calibrated transform."
            )
        except Exception as error:
            messagebox.showerror(
                "Launch export failed", str(error), parent=self
            )

    def _display_result(
        self, result: RegistrationResult, T_base_cam2: np.ndarray
    ) -> None:
        quaternion = quaternion_from_matrix(T_base_cam2)
        text = (
            f"Convention: p_base = T_base_cam2 @ p_cam2\n"
            f"Camera 2 frame: {self._result_child_frame}\n\n"
            f"T_base_cam2:\n"
            f"{np.array2string(T_base_cam2, precision=9, suppress_small=True)}\n\n"
            f"Translation XYZ (m): "
            f"{np.array2string(T_base_cam2[:3, 3], precision=9)}\n"
            f"Quaternion XYZW: {np.array2string(quaternion, precision=9)}\n\n"
            f"Global fitness: {result.global_fitness:.4f}\n"
            f"Global RMSE: {result.global_inlier_rmse:.6f} m\n"
            f"Refined fitness: {result.fitness:.4f}\n"
            f"Refined RMSE: {result.inlier_rmse:.6f} m\n"
        )
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self._events.get_nowait()
                if event == "devices":
                    devices = payload
                    assert isinstance(devices, list)
                    self._devices = {device.label: device for device in devices}
                    labels = list(self._devices)
                    self.camera1_combo.configure(values=labels)
                    self.camera2_combo.configure(values=labels)
                    if labels:
                        self.camera1_var.set(labels[0])
                    if len(labels) > 1:
                        self.camera2_var.set(labels[1])
                    self._camera_selection_changed()
                    self.status_var.set(
                        f"Found {len(labels)} RealSense camera(s)."
                    )
                    self.refresh_button.configure(state="normal")
                elif event == "tf_frames":
                    if isinstance(payload, Exception):
                        self.ros_connect_button.configure(state="normal")
                        self.ros_connection_label.configure(text="TF read failed")
                        messagebox.showerror(
                            "ROS TF read failed", str(payload), parent=self
                        )
                    else:
                        frames = list(payload)  # type: ignore[arg-type]
                        self.cam2_link_combo.configure(values=frames)
                        current_link = self.cam2_link_frame_var.get()
                        if frames and current_link not in frames:
                            likely_links = [
                                frame
                                for frame in frames
                                if frame.endswith("link")
                                and (
                                    "camera_2" in frame
                                    or "camera2" in frame
                                    or "cam2" in frame
                                )
                            ]
                            if len(likely_links) == 1:
                                self.cam2_link_frame_var.set(likely_links[0])
                        topics = (
                            self._ros_tf.pointcloud_topics()  # type: ignore[attr-defined]
                            if self._ros_tf is not None
                            else []
                        )
                        self.cam1_topic_combo.configure(values=topics)
                        self.cam2_topic_combo.configure(values=topics)
                        current_topic1 = self.cam1_pointcloud_topic_var.get()
                        current_topic2 = self.cam2_pointcloud_topic_var.get()
                        if topics and current_topic1 not in topics:
                            preferred1 = next(
                                (
                                    topic for topic in topics
                                    if "camera_1" in topic or "camera1" in topic
                                ),
                                topics[0],
                            )
                            self.cam1_pointcloud_topic_var.set(preferred1)
                        if topics and current_topic2 not in topics:
                            preferred2 = next(
                                (
                                    topic for topic in topics
                                    if "camera_2" in topic or "camera2" in topic
                                ),
                                topics[-1],
                            )
                            self.cam2_pointcloud_topic_var.set(preferred2)
                        if "base" in frames:
                            self.base_frame_var.set("base")
                        self.transform_source_var.set("Live ROS TF")
                        self.cam1_acquisition_var.set("ROS PointCloud2")
                        self.ros_connect_button.configure(
                            state="normal", text="Refresh ROS TF"
                        )
                        self.ros_connection_label.configure(
                            text=f"Connected: {len(frames)} frames"
                        )
                        self.status_var.set(
                            "ROS TF connected. Select the two PointCloud2 topics "
                            "and verify the camera 2 link frame."
                        )
                elif event == "status":
                    self.status_var.set(str(payload))
                elif event == "preview":
                    image1, image2 = payload  # type: ignore[misc]
                    self._show_previews(image1, image2)
                elif event == "ros_cloud_frames":
                    cloud1_frame, cloud2_frame = payload  # type: ignore[misc]
                    self.cam1_frame_var.set(str(cloud1_frame))
                    self.cam2_frame_var.set(str(cloud2_frame))
                    self.status_var.set(
                        f"Received ROS clouds in '{cloud1_frame}' and "
                        f"'{cloud2_frame}'."
                    )
                elif event == "ros_pair_capture":
                    self._captured_from_ros_pair = bool(payload)
                elif event == "clouds":
                    self._pcd_cam1, self._pcd_cam2 = payload  # type: ignore[misc]
                    self.view_3d_button.configure(state="normal")
                    self.status_var.set(
                        "Captured clouds; inspect them before registration."
                    )
                elif event == "capture_complete":
                    self.progress.stop()
                    self.go_button.configure(state="normal")
                    self.recalculate_button.configure(state="normal")
                    self.status_var.set(
                        "Capture ready. Inspect clouds, then click Recalculate."
                    )
                    self.notebook.select(self._tab_capture)
                elif event == "result":
                    result, transform, output_frame = payload  # type: ignore[misc]
                    self._result = result
                    self._T_base_cam2 = transform
                    self._result_child_frame = output_frame
                    self._display_result(result, transform)
                    self.status_var.set("Calibration accepted.")
                    self.progress.stop()
                    self.go_button.configure(state="normal")
                    self.recalculate_button.configure(state="normal")
                    self.save_button.configure(state="normal")
                    self.save_launch_button.configure(state="normal")
                    self.view_overlay_button.configure(state="normal")
                    self.notebook.select(self._tab_result)
                elif event == "calibration_error":
                    self.progress.stop()
                    self.go_button.configure(state="normal")
                    if self._pcd_cam1 is not None:
                        self.recalculate_button.configure(state="normal")
                    self.status_var.set(
                        "Calibration failed; inspect both captured clouds."
                    )
                    messagebox.showerror(
                        "Calibration failed", str(payload), parent=self
                    )
                elif event == "viewer_error":
                    messagebox.showerror(
                        "3-D viewer failed", str(payload), parent=self
                    )
                elif event == "error":
                    self.refresh_button.configure(state="normal")
                    self.status_var.set("Could not enumerate cameras.")
                    messagebox.showerror(
                        "RealSense error", str(payload), parent=self
                    )
        except queue.Empty:
            pass
        self.after(100, self._drain_events)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if ttkb is not None:
        root = ttkb.Window(themename="darkly")
    else:
        root = tk.Tk()
        LOGGER.warning(
            "ttkbootstrap is not installed; using the basic Tk theme. "
            "Install the GUI extra for the modern theme: "
            "pip install 'multicam-icp-calib[gui]'"
        )
    CalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
