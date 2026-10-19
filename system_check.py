"""Branch-PC production smoke test, including optional live RTSP verification."""

import argparse
import ctypes
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from branch_config import (
    PROJECT_DIR,
    build_rtsp_url_from_config,
    load_branch_config,
    resolve_project_path,
    validate_config,
)
from runtime_storage import SQLiteStore, empty_state
from staff_face import StaffFaceRecognizer
from staff_uniform import StaffUniformDetector


class CheckRunner:
    def __init__(self):
        self.failed = []

    def check(self, name, callback):
        try:
            details = callback()
        except Exception as exc:
            self.failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            return
        suffix = f": {details}" if details else ""
        print(f"[PASS] {name}{suffix}")


def check_dependencies():
    import filterpy
    import openpyxl
    import scipy
    import torch
    import torchvision
    import ultralytics
    expected = {
        "OpenCV": (cv2.__version__, "5.0.0"),
        "NumPy": (np.__version__, "2.4.6"),
        "Ultralytics": (ultralytics.__version__, "8.4.177"),
        "Torch": (torch.__version__.split("+")[0], "2.14.1"),
        "Torchvision": (torchvision.__version__.split("+")[0], "0.29.1"),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError("version mismatch: " + ", ".join(mismatches))
    return (
        f"OpenCV {cv2.__version__}, NumPy {np.__version__}, "
        f"Ultralytics {ultralytics.__version__}, Torch {torch.__version__}"
    )


def check_windows_runtime():
    if sys.platform != "win32":
        return "not Windows; skipped"
    missing = []
    for library in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        try:
            ctypes.WinDLL(library)
        except OSError:
            missing.append(library)
    if missing:
        raise RuntimeError(
            "Microsoft Visual C++ 2015-2022 Redistributable (x64) is missing: "
            + ", ".join(missing)
        )
    return "Microsoft VC++ x64 runtime loaded"


def check_tkinter():
    import tkinter as tk

    interpreter = tk.Tcl()
    return f"Tcl/Tk {interpreter.eval('info patchlevel')}"


def check_pip_health():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError((result.stdout or result.stderr).strip())
    return (result.stdout or "No broken requirements").strip()


def check_yolo(config):
    from ultralytics import YOLO

    path = resolve_project_path(config["model"]["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    model = YOLO(str(path))
    results = model(
        np.zeros((320, 320, 3), dtype=np.uint8),
        classes=[0],
        imgsz=320,
        verbose=False,
    )
    if results is None:
        raise RuntimeError("YOLO returned no result object")
    return path.name


def check_face_models(config):
    face = config["face"]
    if not face.get("enabled", True):
        return "disabled by branch configuration"
    recognizer = StaffFaceRecognizer(
        resolve_project_path(face["detector_model"]),
        resolve_project_path(face["recognizer_model"]),
        resolve_project_path(face["profile_file"]),
        enabled=True,
    )
    if not recognizer.models_ready:
        raise RuntimeError(recognizer.load_error or "face models unavailable")
    return f"models loaded; enrolled employees={recognizer.enrolled_employee_count}"


def check_uniform_detector(config):
    staff = config["staff"]
    detector = StaffUniformDetector(
        resolve_project_path(staff.get("logo_reference", ""))
        if staff.get("logo_reference") else None,
        resolve_project_path(staff.get("enrollment_reference_dir", "")),
        profile_path=resolve_project_path(staff.get("profile_file", ""))
        if staff.get("profile_file") else None,
        enabled=staff.get("enabled", True),
        white_min_ratio=staff.get("white_min_ratio", 0.30),
        gray_min_ratio=staff.get("gray_min_ratio", 0.28),
        belt_min_ratio=staff.get("belt_min_ratio", 0.18),
    )
    if staff.get("enabled", True) and not detector.ready:
        raise RuntimeError(detector.load_error or "uniform detector unavailable")
    return (
        "clothing layout ready; "
        f"enrolled logo references={'yes' if detector.logo_ready else 'no (optional)'}; "
        f"branch profile={'yes' if detector.profile_ready else 'no (optional)'}"
    )


def check_storage_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        store = SQLiteStore(
            root / "check.sqlite3",
            "system_check",
            "127.0.0.1",
            root / "reports",
            report_refresh_delay_s=0.25,
        )
        state = empty_state()
        state.update(total_in=1, peak_inside=1)
        store.add_event(1, "IN", state)
        report = store.export_daily_excel(state["business_date"])
        store.close(state, export_today=False)
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("Excel report was not created")
    return "SQLite commit + Excel export"


def check_camera(config):
    url = build_rtsp_url_from_config(config)
    camera = config["camera"]
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(camera.get("open_timeout_ms", 5000)))
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(camera.get("read_timeout_ms", 2000)))
    opened = cap.open(url, cv2.CAP_FFMPEG) or cap.open(url)
    if not opened:
        cap.release()
        raise RuntimeError("RTSP stream could not be opened")
    frames = 0
    shape = None
    deadline = time.monotonic() + 12
    try:
        while frames < 10 and time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                frames += 1
                shape = frame.shape
    finally:
        cap.release()
    if frames < 10:
        raise RuntimeError(f"received only {frames}/10 valid frames")
    return f"10 frames at {shape[1]}x{shape[0]}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true", help="also test the live RTSP stream")
    args = parser.parse_args(argv)
    runner = CheckRunner()
    config = load_branch_config()

    print("People Counter - Production System Check")
    print("=" * 50)
    runner.check(
        "Python version",
        lambda: sys.version.split()[0]
        if sys.version_info[:2] == (3, 11) and struct.calcsize("P") * 8 == 64
        else (_ for _ in ()).throw(RuntimeError("64-bit Python 3.11 is required")),
    )
    runner.check("Microsoft runtime", check_windows_runtime)
    runner.check("Configuration GUI runtime", check_tkinter)
    runner.check("Required dependencies", check_dependencies)
    runner.check("Python dependency health", check_pip_health)
    runner.check("Active branch configuration", lambda: validate_config(config))
    runner.check("Free disk space", lambda: f"{shutil.disk_usage(PROJECT_DIR).free / (1024 ** 3):.1f} GB free" if shutil.disk_usage(PROJECT_DIR).free >= 2 * 1024 ** 3 else (_ for _ in ()).throw(RuntimeError("less than 2 GB free")))
    runner.check("YOLO model and inference", lambda: check_yolo(config))
    runner.check("Face models", lambda: check_face_models(config))
    runner.check("Uniform detector", lambda: check_uniform_detector(config))
    runner.check("SQLite and Excel", check_storage_round_trip)
    if args.camera:
        runner.check("Live camera RTSP", lambda: check_camera(config))

    print("=" * 50)
    if runner.failed:
        print(f"RESULT: FAILED ({len(runner.failed)} check(s))")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
