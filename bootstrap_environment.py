"""Create and verify the reproducible local Python environment."""

from pathlib import Path
import struct
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"


def run(command):
    subprocess.run(command, cwd=str(PROJECT_DIR), check=True)


def supported_python():
    return sys.version_info[:2] == (3, 11) and struct.calcsize("P") * 8 == 64


def environment_is_healthy():
    if not VENV_PYTHON.is_file():
        return False
    probe = (
        "import cv2,numpy,scipy,filterpy,openpyxl,ultralytics,torch,torchvision;"
        "assert cv2.__version__=='5.0.0';"
        "assert torch.__version__.split('+')[0]=='2.14.1';"
        "print('Environment verified')"
    )
    return subprocess.run(
        [str(VENV_PYTHON), "-c", probe],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def tkinter_is_healthy(python):
    probe = "import tkinter as tk; interpreter=tk.Tcl(); interpreter.eval('info patchlevel')"
    return subprocess.run(
        [str(python), "-c", probe],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def main():
    if not supported_python():
        print(
            "ERROR: Install 64-bit Python 3.11 and enable 'Add Python to PATH'.\n"
            f"Detected: Python {sys.version.split()[0]}, {struct.calcsize('P') * 8}-bit."
        )
        return 2

    if not tkinter_is_healthy(sys.executable):
        print(
            "ERROR: Python Tcl/Tk is missing or damaged, so the configuration "
            "screen cannot open. Run the Python 3.11 x64 installer, choose "
            "Modify/Repair, enable 'tcl/tk and IDLE', then restart Windows."
        )
        return 5

    if environment_is_healthy():
        print("Local environment is ready.")
        return 0

    try:
        if not VENV_PYTHON.is_file():
            print("Creating the local Python environment...")
            run([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("Installing the tested project requirements. This can take several minutes...")
        run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements.txt")])
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: Python package installation failed (exit code {exc.returncode}).")
        return 3

    if not environment_is_healthy():
        print(
            "ERROR: Packages installed but could not be imported. Install the "
            "Microsoft Visual C++ 2015-2022 Redistributable (x64), restart "
            "Windows, then run this file again."
        )
        return 4
    print("Local environment installed and verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
