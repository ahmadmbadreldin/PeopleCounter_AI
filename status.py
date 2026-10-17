"""Human-readable branch heartbeat status."""

import json
import time
from pathlib import Path

from branch_config import load_branch_config, resolve_project_path


def main():
    config = load_branch_config()
    branch_id = config.get("branch_id", "default_branch")
    stale_seconds = float(config.get("runtime", {}).get("watchdog_stale_s", 45))
    heartbeat = resolve_project_path(
        config.get("runtime", {}).get(
            "heartbeat_file", f"runtime/{branch_id}_heartbeat.json"
        )
    )
    print(f"Branch: {branch_id}")
    if not Path(heartbeat).exists():
        print("Program: NOT STARTED (no heartbeat file)")
        return 1
    try:
        payload = json.loads(Path(heartbeat).read_text(encoding="utf-8"))
        age = max(0.0, time.time() - Path(heartbeat).stat().st_mtime)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Heartbeat: UNREADABLE ({exc})")
        return 1

    healthy = payload.get("status") == "running" and age <= stale_seconds
    print(f"Program: {'RUNNING' if healthy else 'NOT HEALTHY'}")
    print(f"Last heartbeat: {payload.get('timestamp', 'unknown')} ({age:.1f}s ago)")
    print(f"Camera: {'CONNECTED' if payload.get('camera_connected') else 'DISCONNECTED'}")
    print(f"Last camera frame age: {payload.get('camera_last_frame_age_s', 'unknown')}s")
    print(f"IN: {payload.get('entered', 0)}")
    print(f"OUT: {payload.get('exited', 0)}")
    print(f"INSIDE: {payload.get('inside', 0)}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
