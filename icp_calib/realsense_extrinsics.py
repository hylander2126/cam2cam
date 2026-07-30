"""Command-line export of factory RealSense stream extrinsics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .realsense import factory_extrinsics, list_devices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read factory depth-to-stream extrinsics directly from a "
            "RealSense device using librealsense."
        )
    )
    parser.add_argument(
        "--serial",
        help="Device serial number; optional when exactly one camera is attached.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise print to stdout.",
    )
    args = parser.parse_args()

    serial = args.serial
    if not serial:
        devices = list_devices()
        if len(devices) != 1:
            labels = ", ".join(device.serial for device in devices) or "none"
            parser.error(
                "omit --serial only when one camera is attached; "
                f"detected serials: {labels}"
            )
        serial = devices[0].serial

    text = json.dumps(factory_extrinsics(serial), indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
