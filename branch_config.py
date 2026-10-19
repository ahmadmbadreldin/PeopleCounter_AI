"""Load, validate, and save per-branch People Counter configuration."""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from urllib.parse import quote


PROJECT_DIR = Path(__file__).resolve().parent
ACTIVE_CONFIG_PATH = PROJECT_DIR / "branch_config.json"
BRANCHES_DIR = PROJECT_DIR / "branches"


DEFAULT_CONFIG = {
    "branch_id": "default_branch",
    "camera": {
        "ip": "192.168.1.64",
        "username": "admin",
        "password": "",
        "rtsp_port": 554,
        "stream_channel": "101",
        "rtsp_url": "",
        "open_timeout_ms": 5000,
        "read_timeout_ms": 2000,
        "reconnect_delay_s": 1.0,
    },
    "counting": {
        "calibrated": False,
        "geometry_mode": "polygon_line",
        "base_width": 1280,
        "base_height": 720,
        "zone_x1": 150,
        "zone_y1": 170,
        "zone_x2": 450,
        "zone_y2": 260,
        "zone_polygon": [[150, 170], [450, 170], [450, 260], [150, 260]],
        "line_p1": [150, 215],
        "line_p2": [450, 215],
        "inside_sign": 1,
        "inside_point": [300, 245],
        "line_position_ratio": 0.50,
        "in_direction": "top_to_bottom",
        "line_hysteresis": 15,
        "crossing_cooldown_s": 0.75,
    },
    "model": {
        "path": "yolov8n.pt",
        "confidence": 0.40,
        "nms_iou": 0.40,
        "image_size": 640,
        "device": "",
        "process_every_n_frames": 1,
    },
    "tracking": {
        "max_age": 30,
        "min_hits": 2,
        "iou_threshold": 0.20,
        "min_object_width": 30,
        "min_object_height": 30,
    },
    "staff": {
        "enabled": True,
        "logo_reference": "",
        "enrollment_reference_dir": "",
        "profile_file": "",
        "min_person_width": 70,
        "min_person_height": 130,
        "white_min_ratio": 0.25,
        "gray_min_ratio": 0.22,
        "belt_min_ratio": 0.18,
        "frame_threshold": 0.54,
        "track_threshold": 0.58,
        "strong_frame_threshold": 0.78,
        "min_positive_frames": 2,
        "min_negative_frames": 6,
        "min_decision_frames": 6,
        "process_every_n_frames": 2,
        "decision_timeout_s": 3.50,
        "belt_bonus": 0.05,
        "logo_bonus": 0.10,
        "fast_customer_factor": 0.55,
        "fast_customer_frames": 4,
    },
    "uniform_dataset": {
        "directory": "branches/default_branch/uniform_dataset",
        "collection_duration_s": 60,
        "max_samples_per_session": 180,
        "sample_interval_s": 0.35,
        "min_person_width": 90,
        "min_person_height": 180,
        "min_blur_score": 35.0,
        "head_crop_ratio": 0.18,
    },
    "face": {
        "enabled": True,
        "detector_model": "models/face_detection_yunet_2026may.onnx",
        "recognizer_model": "models/face_recognition_sface_2021dec.onnx",
        "profile_file": "",
        "match_threshold": 0.36,
        "support_threshold": 0.29,
        "min_face_size": 38,
        "min_positive_frames": 2,
        "process_every_n_frames": 2,
    },
    "storage": {
        "database": "data/default_branch.sqlite3",
        "reports_dir": "reports/default_branch",
        "event_retention_days": 90,
        "report_retention_days": 365,
        "report_refresh_delay_s": 2.0,
        "state_snapshot_interval_s": 10,
    },
    "runtime": {
        "heartbeat_file": "runtime/default_branch_heartbeat.json",
        "heartbeat_interval_s": 5,
        "watchdog_stale_s": 45,
        "watchdog_startup_grace_s": 150,
        "restart_delay_s": 5,
        "log_file": "logs/default_branch_runtime.log",
        "log_max_bytes": 5242880,
        "log_backup_count": 5,
    },
    "display": {
        "width": 1100,
        "height": 700,
        "window_name": "Store People Counter - Hikvision",
    },
    "sound": {
        "file": "beep-01a.wav",
        "cooldown_s": 0.20,
    },
}


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def safe_branch_id(value):
    value = re.sub(r"[^\w-]+", "_", str(value).strip(), flags=re.UNICODE)
    value = value.strip("_")
    if not value:
        raise ValueError("Branch ID is required and must contain letters or numbers.")
    return value[:64]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def point_in_polygon(point, polygon):
    """Boundary-inclusive ray casting used for configuration validation."""
    x, y = map(float, point)
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = map(float, polygon[index])
        x2, y2 = map(float, polygon[(index + 1) % count])
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-6 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside
    return inside


def polygon_area(polygon):
    total = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(total) / 2.0


def polygon_is_simple(polygon):
    """Return False when non-adjacent polygon edges cross each other."""
    def orientation(a, b, c):
        value = ((b[1] - a[1]) * (c[0] - b[0])) - (
            (b[0] - a[0]) * (c[1] - b[1])
        )
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else 2

    def on_segment(a, b, c):
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    def intersects(a, b, c, d):
        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        if o1 != o2 and o3 != o4:
            return True
        return (
            (o1 == 0 and on_segment(a, c, b))
            or (o2 == 0 and on_segment(a, d, b))
            or (o3 == 0 and on_segment(c, a, d))
            or (o4 == 0 and on_segment(c, b, d))
        )

    count = len(polygon)
    for first in range(count):
        a, b = polygon[first], polygon[(first + 1) % count]
        for second in range(first + 1, count):
            if second in (first, (first + 1) % count):
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = polygon[second], polygon[(second + 1) % count]
            if intersects(a, b, c, d):
                return False
    return True


def load_branch_config(path=ACTIVE_CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load branch config '{path}': {exc}") from exc
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    loaded_counting = loaded.get("counting", {})
    if "zone_polygon" not in loaded_counting:
        counting = merged["counting"]
        x1, y1 = int(counting["zone_x1"]), int(counting["zone_y1"])
        x2, y2 = int(counting["zone_x2"]), int(counting["zone_y2"])
        ratio = float(counting.get("line_position_ratio", 0.50))
        line_y = int(round(y1 + (y2 - y1) * ratio))
        counting["geometry_mode"] = "polygon_line"
        counting["zone_polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        counting["line_p1"] = [x1, line_y]
        counting["line_p2"] = [x2, line_y]
        counting["inside_sign"] = (
            1 if counting.get("in_direction") == "top_to_bottom" else -1
        )
        inside_y = int(round(line_y + (y2 - y1) * 0.25 * counting["inside_sign"]))
        inside_y = min(max(inside_y, y1), y2)
        counting["inside_point"] = [int(round((x1 + x2) / 2)), inside_y]
    branch_id = safe_branch_id(merged.get("branch_id", "default_branch"))
    # Upgrade branch files created before SQLite/heartbeat support without
    # making different branches share the default_branch runtime files.
    if "database" not in loaded.get("storage", {}):
        merged["storage"]["database"] = f"data/{branch_id}.sqlite3"
    if "reports_dir" not in loaded.get("storage", {}):
        merged["storage"]["reports_dir"] = f"reports/{branch_id}"
    if "heartbeat_file" not in loaded.get("runtime", {}):
        merged["runtime"]["heartbeat_file"] = f"runtime/{branch_id}_heartbeat.json"
    if "log_file" not in loaded.get("runtime", {}):
        merged["runtime"]["log_file"] = f"logs/{branch_id}_runtime.log"
    if "directory" not in loaded.get("uniform_dataset", {}):
        merged["uniform_dataset"]["directory"] = (
            f"branches/{branch_id}/uniform_dataset"
        )
    if "profile_file" not in loaded.get("staff", {}):
        merged["staff"]["profile_file"] = (
            f"branches/{branch_id}/uniform_profile.json"
        )
    if "min_negative_frames" not in loaded.get("staff", {}):
        merged["staff"]["min_negative_frames"] = int(
            loaded.get("staff", {}).get(
                "min_decision_frames", DEFAULT_CONFIG["staff"]["min_negative_frames"]
            )
        )
    return merged


def validate_config(config):
    branch_id = safe_branch_id(config.get("branch_id", ""))
    camera = config.get("camera", {})
    if not camera.get("rtsp_url"):
        if not str(camera.get("ip", "")).strip():
            raise ValueError("Camera IP is required.")
        if not str(camera.get("username", "")).strip():
            raise ValueError("Camera username is required.")
        if not str(camera.get("password", "")):
            raise ValueError("Camera password is required.")
    port = int(camera.get("rtsp_port", 554))
    if not 1 <= port <= 65535:
        raise ValueError("RTSP port must be between 1 and 65535.")

    counting = config.get("counting", {})
    width = int(counting.get("base_width", 0))
    height = int(counting.get("base_height", 0))
    x1, x2 = int(counting.get("zone_x1", 0)), int(counting.get("zone_x2", 0))
    y1, y2 = int(counting.get("zone_y1", 0)), int(counting.get("zone_y2", 0))
    if width <= 0 or height <= 0:
        raise ValueError("A camera preview is required before saving.")
    if not bool(counting.get("calibrated", False)):
        raise ValueError("Test the camera and calibrate the entrance shape before saving.")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Draw a valid entrance shape on the camera preview.")
    polygon = counting.get("zone_polygon", [])
    if not isinstance(polygon, list) or not 3 <= len(polygon) <= 20:
        raise ValueError("Draw an entrance shape with 3 to 20 points.")
    for point in polygon:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not 0 <= float(point[0]) < width
            or not 0 <= float(point[1]) < height
        ):
            raise ValueError("Entrance shape contains a point outside the camera image.")
    if not polygon_is_simple(polygon):
        raise ValueError("Entrance shape edges must not cross each other.")
    if polygon_area(polygon) < 500:
        raise ValueError("The entrance shape is too small.")
    line_p1 = counting.get("line_p1", [])
    line_p2 = counting.get("line_p2", [])
    if not (
        isinstance(line_p1, (list, tuple)) and len(line_p1) == 2
        and isinstance(line_p2, (list, tuple)) and len(line_p2) == 2
    ):
        raise ValueError("Draw the counting line inside the entrance shape.")
    line_length_sq = (
        (float(line_p2[0]) - float(line_p1[0])) ** 2
        + (float(line_p2[1]) - float(line_p1[1])) ** 2
    )
    if line_length_sq < 25 ** 2:
        raise ValueError("The counting line is too short.")
    for point in (line_p1, line_p2):
        if not (0 <= float(point[0]) < width and 0 <= float(point[1]) < height):
            raise ValueError("Counting line contains a point outside the camera image.")
        if not point_in_polygon(point, polygon):
            raise ValueError("Both ends of the counting line must be inside the entrance shape.")
    for step in range(1, 20):
        fraction = step / 20.0
        sample = (
            float(line_p1[0]) + (float(line_p2[0]) - float(line_p1[0])) * fraction,
            float(line_p1[1]) + (float(line_p2[1]) - float(line_p1[1])) * fraction,
        )
        if not point_in_polygon(sample, polygon):
            raise ValueError("The complete counting line must stay inside the entrance shape.")
    if int(counting.get("inside_sign", 0)) not in (-1, 1):
        raise ValueError("Choose the inside-store side of the counting line.")
    inside_point = counting.get("inside_point", [])
    if not isinstance(inside_point, (list, tuple)) or len(inside_point) != 2:
        raise ValueError("Click inside the store to choose the IN side.")
    if not point_in_polygon(inside_point, polygon):
        raise ValueError("The IN-side point must be inside the entrance shape.")
    line_position = float(counting.get("line_position_ratio", 0.50))
    if not 0.10 <= line_position <= 0.90:
        raise ValueError("Counting line position must be between 10% and 90%.")
    if counting.get("in_direction") not in ("top_to_bottom", "bottom_to_top"):
        raise ValueError("Select a valid entrance direction.")

    staff = config.get("staff", {})
    for key in ("white_min_ratio", "gray_min_ratio", "belt_min_ratio"):
        value = float(staff.get(key, DEFAULT_CONFIG["staff"][key]))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Staff setting '{key}' must be between 0 and 1.")
    thresholds = [
        float(staff.get("frame_threshold", 0.54)),
        float(staff.get("track_threshold", 0.58)),
        float(staff.get("strong_frame_threshold", 0.78)),
    ]
    if not all(0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Staff confidence thresholds must be between 0 and 1.")
    if not thresholds[0] <= thresholds[1] <= thresholds[2]:
        raise ValueError("Staff thresholds must satisfy frame <= track <= strong.")
    for key in (
        "min_positive_frames", "min_negative_frames", "process_every_n_frames",
        "fast_customer_frames",
    ):
        if int(staff.get(key, DEFAULT_CONFIG["staff"][key])) < 1:
            raise ValueError(f"Staff setting '{key}' must be at least 1.")
    if int(staff.get("fast_customer_frames", 4)) < 2:
        raise ValueError("Fast customer confirmation requires at least 2 frames.")
    if not 0.5 <= float(staff.get("decision_timeout_s", 3.5)) <= 30.0:
        raise ValueError("Staff decision timeout must be between 0.5 and 30 seconds.")
    if not 0.0 <= float(staff.get("belt_bonus", 0.05)) <= 0.05:
        raise ValueError("Belt bonus must be between 0 and 0.05.")
    if not 0.0 <= float(staff.get("logo_bonus", 0.10)) <= 0.10:
        raise ValueError("Logo bonus must be between 0 and 0.10.")
    if not 0.10 <= float(staff.get("fast_customer_factor", 0.55)) <= 0.90:
        raise ValueError("Fast customer factor must be between 0.10 and 0.90.")
    return branch_id


def save_branch_config(config, active_path=ACTIVE_CONFIG_PATH, branches_dir=BRANCHES_DIR):
    config = _deep_merge(DEFAULT_CONFIG, config)
    branch_id = validate_config(config)
    config["branch_id"] = branch_id
    config["storage"]["database"] = f"data/{branch_id}.sqlite3"
    config["storage"]["reports_dir"] = f"reports/{branch_id}"
    config["runtime"]["heartbeat_file"] = f"runtime/{branch_id}_heartbeat.json"
    config["runtime"]["log_file"] = f"logs/{branch_id}_runtime.log"
    config["face"]["profile_file"] = f"branches/{branch_id}/staff_faces.npz"
    config["staff"]["enrollment_reference_dir"] = (
        f"branches/{branch_id}/uniform_refs"
    )
    config["staff"]["profile_file"] = f"branches/{branch_id}/uniform_profile.json"
    config["uniform_dataset"]["directory"] = (
        f"branches/{branch_id}/uniform_dataset"
    )

    active_path = Path(active_path)
    branches_dir = Path(branches_dir)
    branches_dir.mkdir(parents=True, exist_ok=True)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(config["storage"]["reports_dir"]).mkdir(
        parents=True, exist_ok=True
    )
    resolve_project_path(config["uniform_dataset"]["directory"]).mkdir(
        parents=True, exist_ok=True
    )
    branch_path = branches_dir / f"{branch_id}.json"

    payload = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    temporary = active_path.with_suffix(active_path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, active_path)
    branch_path.write_text(payload, encoding="utf-8")
    return active_path, branch_path


def build_rtsp_url_from_config(config):
    camera = config["camera"]
    explicit = str(camera.get("rtsp_url", "")).strip()
    if explicit:
        return explicit
    user = quote(str(camera["username"]), safe="")
    password = quote(str(camera["password"]), safe="")
    return (
        f"rtsp://{user}:{password}@{camera['ip']}:{int(camera['rtsp_port'])}/"
        f"Streaming/channels/{camera['stream_channel']}"
    )


def list_saved_branches():
    if not BRANCHES_DIR.exists():
        return []
    return sorted(BRANCHES_DIR.glob("*.json"), key=lambda path: path.stem.lower())
