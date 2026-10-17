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
        "base_width": 1280,
        "base_height": 720,
        "zone_x1": 150,
        "zone_y1": 170,
        "zone_x2": 450,
        "zone_y2": 260,
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
        "min_person_width": 70,
        "min_person_height": 130,
        "white_min_ratio": 0.30,
        "frame_threshold": 0.56,
        "track_threshold": 0.60,
        "min_positive_frames": 3,
        "decision_timeout_s": 2.50,
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
    "logging": {
        "file": "logs/default_branch_store_log.xlsx",
        "queue_size": 100,
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


def load_branch_config(path=ACTIVE_CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load branch config '{path}': {exc}") from exc
    return _deep_merge(DEFAULT_CONFIG, loaded)


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
        raise ValueError("Test the camera and draw the entrance rectangle before saving.")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Draw a valid entrance rectangle on the camera preview.")
    if counting.get("in_direction") not in ("top_to_bottom", "bottom_to_top"):
        raise ValueError("Select a valid entrance direction.")
    return branch_id


def save_branch_config(config, active_path=ACTIVE_CONFIG_PATH, branches_dir=BRANCHES_DIR):
    config = _deep_merge(DEFAULT_CONFIG, config)
    branch_id = validate_config(config)
    config["branch_id"] = branch_id
    config["logging"]["file"] = f"logs/{branch_id}_store_log.xlsx"
    config["face"]["profile_file"] = f"branches/{branch_id}/staff_faces.npz"
    config["staff"]["enrollment_reference_dir"] = (
        f"branches/{branch_id}/uniform_refs"
    )

    active_path = Path(active_path)
    branches_dir = Path(branches_dir)
    branches_dir.mkdir(parents=True, exist_ok=True)
    active_path.parent.mkdir(parents=True, exist_ok=True)
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
