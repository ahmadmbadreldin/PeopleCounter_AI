"""Keep the People Counter alive and restart it after a crash or stale heartbeat."""

import ctypes
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from branch_config import load_branch_config, resolve_project_path, safe_branch_id
from runtime_logging import configure_rotating_logger


PROJECT_DIR = Path(__file__).resolve().parent


def _setting(config, section, key, default):
    return config.get(section, {}).get(key, default)


def _single_instance(branch_id):
    """Return a live Windows mutex handle, or None if another copy exists."""
    if os.name != "nt":
        return True
    name = f"Local\\ColorsBeautyPeopleCounter_{safe_branch_id(branch_id)}"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool
    handle = kernel32.CreateMutexW(None, False, name)
    already_exists = ctypes.get_last_error() == 183
    if not handle or already_exists:
        if handle:
            kernel32.CloseHandle(handle)
        return None
    return handle


def _close_mutex(handle):
    if os.name != "nt" or handle in (None, True):
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool
    kernel32.CloseHandle(handle)


def heartbeat_is_stale(path, stale_seconds):
    path = Path(path)
    if not path.exists():
        return True
    try:
        return time.time() - path.stat().st_mtime > float(stale_seconds)
    except OSError:
        return True


def _stop_child(process, logger):
    if process is None or process.poll() is not None:
        return
    logger.warning("Stopping unresponsive People Counter process pid=%s", process.pid)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.error("Process did not terminate; killing pid=%s", process.pid)
        process.kill()
        process.wait(timeout=5)


def main():
    config = load_branch_config()
    branch_id = config.get("branch_id", "default_branch")
    runtime = config.get("runtime", {})
    heartbeat = resolve_project_path(
        runtime.get("heartbeat_file", f"runtime/{branch_id}_heartbeat.json")
    )
    stale_seconds = float(runtime.get("watchdog_stale_s", 45))
    startup_grace = float(runtime.get("watchdog_startup_grace_s", 150))
    restart_delay = float(runtime.get("restart_delay_s", 5))
    logger = configure_rotating_logger(
        "watchdog",
        resolve_project_path("logs/watchdog.log"),
        runtime.get("log_max_bytes", 5_242_880),
        runtime.get("log_backup_count", 5),
    )

    mutex = _single_instance(branch_id)
    if mutex is None:
        logger.info("Another watchdog instance is already running; exiting")
        return 0

    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    if os.name == "nt" and pythonw.exists():
        python = pythonw
    app = PROJECT_DIR / "people_counter_hikvision.py"
    recent_restarts = deque()
    child = None

    logger.info("Watchdog started for branch %s", branch_id)
    try:
        while True:
            now = time.monotonic()
            while recent_restarts and now - recent_restarts[0] > 600:
                recent_restarts.popleft()
            if len(recent_restarts) >= 5:
                logger.error("Restart storm detected; cooling down for 5 minutes")
                time.sleep(300)
                recent_restarts.clear()

            try:
                heartbeat.unlink(missing_ok=True)
            except OSError:
                pass
            child = subprocess.Popen(
                [str(python), str(app)],
                cwd=str(PROJECT_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            recent_restarts.append(time.monotonic())
            started = time.monotonic()
            logger.info("People Counter launched pid=%s", child.pid)

            restart_reason = None
            while child.poll() is None:
                age = time.monotonic() - started
                if age > startup_grace and heartbeat_is_stale(heartbeat, stale_seconds):
                    restart_reason = f"heartbeat stale for more than {stale_seconds}s"
                    logger.error("People Counter is unresponsive: %s", restart_reason)
                    _stop_child(child, logger)
                    break
                time.sleep(5)

            return_code = child.poll()
            if return_code == 0 and restart_reason is None:
                logger.info("People Counter closed normally; watchdog exiting")
                return 0
            logger.warning(
                "People Counter stopped (code=%s, reason=%s); restarting in %.1fs",
                return_code,
                restart_reason or "process crashed",
                restart_delay,
            )
            time.sleep(restart_delay)
    except KeyboardInterrupt:
        logger.info("Watchdog stopped manually")
        _stop_child(child, logger)
        return 0
    finally:
        _close_mutex(mutex)


if __name__ == "__main__":
    raise SystemExit(main())
