#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPE / Store People Counter - Production-oriented implementation

Fixed: Hybrid Centroid+IoU tracking to prevent ID swapping, 
and Target-Y point tracking for accurate line crossing detection.
"""

import os
import time
import socket
import threading
from dataclasses import dataclass
from collections import deque
from datetime import date, datetime
from pathlib import Path
from queue import Queue, Empty
from urllib.parse import quote

import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from staff_face import StaffFaceRecognizer
from staff_uniform import StaffUniformDetector
from branch_config import (
    ACTIVE_CONFIG_PATH,
    load_branch_config,
    resolve_project_path,
    validate_config,
)
from runtime_logging import configure_rotating_logger
from runtime_lock import BranchInstanceLock
from runtime_storage import HeartbeatWriter, SQLiteStore


# Configure FFmpeg RTSP socket timeout options globally before video capture opens
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"


# ============================================================
# 1. CONFIGURATION
# ============================================================

_BRANCH_CONFIG = load_branch_config()


def _setting(section, key, default=None):
    return _BRANCH_CONFIG.get(section, {}).get(key, default)


def _optional_project_path(value):
    value = str(value or "").strip()
    return str(resolve_project_path(value)) if value else ""


@dataclass
class Config:
    # Values come from branch_config.json, written by setup_branch.py.
    branch_id: str = str(_BRANCH_CONFIG.get("branch_id", "default_branch"))

    # ---------- Hikvision ----------
    camera_ip: str = str(_setting("camera", "ip", "192.168.1.64"))
    camera_username: str = str(_setting("camera", "username", "admin"))
    camera_password: str = str(_setting("camera", "password", ""))
    camera_rtsp_url: str = str(_setting("camera", "rtsp_url", ""))
    rtsp_port: int = int(_setting("camera", "rtsp_port", 554))
    stream_channel: str = str(_setting("camera", "stream_channel", "101"))
    camera_open_timeout_ms: int = int(_setting("camera", "open_timeout_ms", 5000))
    camera_read_timeout_ms: int = int(_setting("camera", "read_timeout_ms", 2000))
    reconnect_delay_s: float = float(_setting("camera", "reconnect_delay_s", 1.0))

    # ---------- Model ----------
    model_path: str = str(resolve_project_path(_setting("model", "path", "yolov8n.pt")))
    confidence: float = float(_setting("model", "confidence", 0.40))
    nms_iou: float = float(_setting("model", "nms_iou", 0.40))
    image_size: int = int(_setting("model", "image_size", 640))
    device: str = str(_setting("model", "device", ""))
    process_every_n_frames: int = max(1, int(_setting("model", "process_every_n_frames", 1)))

    # ---------- Tracking ----------
    max_age: int = int(_setting("tracking", "max_age", 30))
    min_hits: int = int(_setting("tracking", "min_hits", 2))
    iou_threshold: float = float(_setting("tracking", "iou_threshold", 0.20))
    min_object_width: int = int(_setting("tracking", "min_object_width", 30))
    min_object_height: int = int(_setting("tracking", "min_object_height", 30))

    # ---------- Staff uniform detection ----------
    staff_detection_enabled: bool = bool(_setting("staff", "enabled", True))
    staff_logo_reference: str = _optional_project_path(
        _setting("staff", "logo_reference", "")
    )
    staff_enrollment_reference_dir: str = str(resolve_project_path(
        _setting("staff", "enrollment_reference_dir", "")
        or f"branches/{_BRANCH_CONFIG.get('branch_id', 'default_branch')}/uniform_refs"
    ))
    staff_profile_file: str = str(resolve_project_path(
        _setting("staff", "profile_file", "")
        or f"branches/{_BRANCH_CONFIG.get('branch_id', 'default_branch')}/uniform_profile.json"
    ))
    staff_min_person_width: int = int(_setting("staff", "min_person_width", 70))
    staff_min_person_height: int = int(_setting("staff", "min_person_height", 130))
    staff_white_min_ratio: float = float(_setting("staff", "white_min_ratio", 0.25))
    staff_gray_min_ratio: float = float(_setting("staff", "gray_min_ratio", 0.22))
    staff_belt_min_ratio: float = float(_setting("staff", "belt_min_ratio", 0.18))
    staff_frame_threshold: float = float(_setting("staff", "frame_threshold", 0.54))
    staff_track_threshold: float = float(_setting("staff", "track_threshold", 0.58))
    staff_strong_frame_threshold: float = float(_setting("staff", "strong_frame_threshold", 0.78))
    staff_min_positive_frames: int = int(_setting("staff", "min_positive_frames", 2))
    staff_min_negative_frames: int = int(_setting(
        "staff", "min_negative_frames", _setting("staff", "min_decision_frames", 6)
    ))
    staff_process_every_n_frames: int = max(
        1, int(_setting("staff", "process_every_n_frames", 2))
    )
    staff_decision_timeout_s: float = float(_setting("staff", "decision_timeout_s", 3.50))
    staff_belt_bonus: float = float(_setting("staff", "belt_bonus", 0.05))
    staff_logo_bonus: float = float(_setting("staff", "logo_bonus", 0.10))
    staff_fast_customer_factor: float = float(_setting("staff", "fast_customer_factor", 0.55))
    staff_fast_customer_frames: int = int(_setting("staff", "fast_customer_frames", 4))

    # ---------- Per-branch staff face recognition ----------
    face_detection_enabled: bool = bool(_setting("face", "enabled", True))
    face_detector_model: str = str(resolve_project_path(
        _setting("face", "detector_model", "models/face_detection_yunet_2026may.onnx")
    ))
    face_recognizer_model: str = str(resolve_project_path(
        _setting("face", "recognizer_model", "models/face_recognition_sface_2021dec.onnx")
    ))
    face_profile_file: str = str(resolve_project_path(
        _setting("face", "profile_file", "")
        or f"branches/{_BRANCH_CONFIG.get('branch_id', 'default_branch')}/staff_faces.npz"
    ))
    face_match_threshold: float = float(_setting("face", "match_threshold", 0.36))
    face_support_threshold: float = float(_setting("face", "support_threshold", 0.29))
    face_min_size: int = int(_setting("face", "min_face_size", 38))
    face_min_positive_frames: int = int(_setting("face", "min_positive_frames", 2))
    face_process_every_n_frames: int = max(
        1, int(_setting("face", "process_every_n_frames", 2))
    )

    # ---------- Counting geometry ----------
    base_width: int = int(_setting("counting", "base_width", 1280))
    base_height: int = int(_setting("counting", "base_height", 720))
    zone_x1: int = int(_setting("counting", "zone_x1", 150))
    zone_y1: int = int(_setting("counting", "zone_y1", 170))
    zone_x2: int = int(_setting("counting", "zone_x2", 450))
    zone_y2: int = int(_setting("counting", "zone_y2", 260))
    zone_polygon: tuple = tuple(
        tuple(map(float, point))
        for point in _setting(
            "counting", "zone_polygon",
            [[150, 170], [450, 170], [450, 260], [150, 260]],
        )
    )
    line_p1: tuple = tuple(map(float, _setting("counting", "line_p1", [150, 215])))
    line_p2: tuple = tuple(map(float, _setting("counting", "line_p2", [450, 215])))
    inside_sign: int = int(_setting("counting", "inside_sign", 1))
    line_position_ratio: float = float(_setting("counting", "line_position_ratio", 0.50))
    in_direction: str = str(_setting("counting", "in_direction", "top_to_bottom"))
    line_hysteresis: int = int(_setting("counting", "line_hysteresis", 15))
    crossing_cooldown_s: float = float(_setting("counting", "crossing_cooldown_s", 0.75))
    inside_timeout_s: float = 600.0  # legacy, unused

    # ---------- Durable storage and unattended runtime ----------
    database_file: str = str(resolve_project_path(
        _setting("storage", "database", "data/default_branch.sqlite3")
    ))
    reports_dir: str = str(resolve_project_path(
        _setting("storage", "reports_dir", "reports/default_branch")
    ))
    event_retention_days: int = int(_setting("storage", "event_retention_days", 90))
    report_retention_days: int = int(_setting("storage", "report_retention_days", 365))
    report_refresh_delay_s: float = float(
        _setting("storage", "report_refresh_delay_s", 2.0)
    )
    state_snapshot_interval_s: float = float(
        _setting("storage", "state_snapshot_interval_s", 10)
    )
    heartbeat_file: str = str(resolve_project_path(
        _setting("runtime", "heartbeat_file", "runtime/default_branch_heartbeat.json")
    ))
    heartbeat_interval_s: float = float(_setting("runtime", "heartbeat_interval_s", 5))
    runtime_log_file: str = str(resolve_project_path(
        _setting("runtime", "log_file", "logs/default_branch_runtime.log")
    ))
    log_max_bytes: int = int(_setting("runtime", "log_max_bytes", 5_242_880))
    log_backup_count: int = int(_setting("runtime", "log_backup_count", 5))

    # ---------- UI ----------
    display_width: int = int(_setting("display", "width", 1100))
    display_height: int = int(_setting("display", "height", 700))
    window_name: str = str(_setting("display", "window_name", "Store People Counter - Hikvision"))

    # ---------- Sound ----------
    sound_file: str = str(resolve_project_path(_setting("sound", "file", "beep-01a.wav")))
    sound_cooldown_s: float = float(_setting("sound", "cooldown_s", 0.20))


CONFIG = Config()
LOGGER = configure_rotating_logger(
    "people_counter",
    CONFIG.runtime_log_file,
    CONFIG.log_max_bytes,
    CONFIG.log_backup_count,
)


def build_rtsp_url(cfg: Config) -> str:
    explicit_url = cfg.camera_rtsp_url.strip()
    if explicit_url:
        return explicit_url

    if not cfg.camera_password:
        raise RuntimeError("HIKVISION_PASSWORD is not set.")

    user = quote(cfg.camera_username, safe="")
    password = quote(cfg.camera_password, safe="")
    return (
        f"rtsp://{user}:{password}@{cfg.camera_ip}:{cfg.rtsp_port}/"
        f"Streaming/channels/{cfg.stream_channel}"
    )


# ============================================================
# 2. CAMERA / HOST INFORMATION
# ============================================================

def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((CONFIG.camera_ip, 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


LOCAL_IP = get_local_ip()


def display_window_is_open(window_name: str) -> bool:
    """Allow mouse-only remote sessions to close the OpenCV window with X."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def counting_signature(config):
    counting = config.get("counting", {})
    keys = (
        "base_width", "base_height", "zone_x1", "zone_y1", "zone_x2",
        "zone_y2", "zone_polygon", "line_p1", "line_p2", "inside_sign",
        "line_position_ratio", "in_direction", "line_hysteresis",
        "crossing_cooldown_s",
    )
    return tuple(counting.get(key) for key in keys)


CONFIG_RESTART_EXIT_CODE = 75


# ============================================================
# 4. SOUND SYSTEM
# ============================================================

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    winsound = None
    SOUND_AVAILABLE = False


def play_sound(sound_file: str) -> None:
    if not SOUND_AVAILABLE:
        return
    def _play_worker():
        try:
            path = Path(sound_file)
            if path.exists():
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.Beep(1000, 150)
        except Exception:
            pass
    threading.Thread(target=_play_worker, daemon=True).start()


# ============================================================
# 5. TRACKER
# ============================================================

class Track:
    def __init__(self, detection, track_id: int):
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Tune Kalman filter to trust measurements more, prediction less
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = np.asarray(detection[:4], dtype=float).reshape(4, 1)
        self.kf.x[4:] = 0.0

        self.hits = 1
        self.no_losses = 0
        self.time_since_update = 0
        self.recovered_after_gap = False
        self.last_observed_bbox = tuple(map(float, detection[:4]))

        self.detection_width = float(detection[2])
        self.detection_height = float(detection[3])
        self.is_small_object = False

        self.count_side = None
        self.candidate_side = None
        self.candidate_frames = 0
        self.last_event_time = 0.0
        self.last_event = None
        self.staff_scores = deque(maxlen=20)
        self.staff_negative_frames = 0
        self.staff_score = 0.0
        self.staff_status = "UNKNOWN"
        self.uniform_status = "UNKNOWN"
        self.uniform_clear_customer = False
        self.uniform_clear_customer_frames = 0
        self.staff_last_evidence = None
        self.face_scores = deque(maxlen=24)
        self.face_score = 0.0
        self.face_employee = ""
        self.face_observations = 0
        self.face_last_evidence = None
        self.update_small_object_status()

    def update_small_object_status(self, min_width=30, min_height=30):
        self.is_small_object = (
            self.detection_width < min_width
            or self.detection_height < min_height
        )

    def update(self, detection, min_width=30, min_height=30):
        self.recovered_after_gap = self.time_since_update > 1
        self.last_observed_bbox = tuple(map(float, detection[:4]))
        arr = np.asarray(detection[:4], dtype=float).reshape(4, 1)
        self.kf.update(arr)
        self.hits += 1
        self.no_losses = 0
        self.time_since_update = 0

        self.detection_width = float(detection[2])
        self.detection_height = float(detection[3])
        self.update_small_object_status(min_width, min_height)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def get_state(self):
        state = self.kf.x[:4].flatten()
        state[2] = max(1.0, state[2])
        state[3] = max(1.0, state[3])
        return state

    def get_center(self):
        state = self.get_state()
        return float(state[0]), float(state[1])

    def get_bbox(self):
        return tuple(map(float, self.get_state()))

    def get_size(self):
        return self.detection_width, self.detection_height

    def get_count_point(self):
        """Bottom-center of the latest real YOLO box (approximate foot point)."""
        cx, cy, _width, height = self.last_observed_bbox
        return float(cx), float(cy + height / 2.0)

    def update_count_side(self, line_y: float, hysteresis: float, min_stable_frames=2):
        # Never advance the crossing state using predicted-only positions.
        # Use the bottom-center of the latest real detection rather than a
        # Kalman extrapolation.  This makes the event correspond to the
        # person's feet crossing the configured doorway line.
        if self.time_since_update != 0:
            return self.count_side, self.count_side
        if self.recovered_after_gap:
            # Re-acquired after missing frames: re-initialize the side, do not
            # turn a long unseen interval into a fabricated crossing.
            self.count_side = None
            self.candidate_side = None
            self.candidate_frames = 0
            self.recovered_after_gap = False
        _target_x, target_y = self.get_count_point()
        if target_y < line_y - hysteresis:
            new_side = "ABOVE"
        elif target_y > line_y + hysteresis:
            new_side = "BELOW"
        else:
            self.candidate_side = None
            self.candidate_frames = 0
            return self.count_side, self.count_side
        if new_side != self.candidate_side:
            self.candidate_side = new_side
            self.candidate_frames = 1
        else:
            self.candidate_frames += 1
        previous = self.count_side
        if self.candidate_frames >= min_stable_frames:
            self.count_side = new_side
        return previous, self.count_side

    def update_polygon_side(
        self, line_p1, line_p2, inside_sign, hysteresis, min_stable_frames=2
    ):
        """Update stable OUTSIDE/INSIDE state relative to an arbitrary line."""
        if self.time_since_update != 0:
            return self.count_side, self.count_side
        if self.recovered_after_gap:
            self.count_side = None
            self.candidate_side = None
            self.candidate_frames = 0
            self.recovered_after_gap = False
        foot_x, foot_y = self.get_count_point()
        x1, y1 = line_p1
        x2, y2 = line_p2
        dx, dy = x2 - x1, y2 - y1
        length = max(1.0, float(np.hypot(dx, dy)))
        signed_distance = ((dx * (foot_y - y1)) - (dy * (foot_x - x1))) / length
        directed_distance = signed_distance * int(inside_sign)
        if directed_distance > hysteresis:
            new_side = "INSIDE"
        elif directed_distance < -hysteresis:
            new_side = "OUTSIDE"
        else:
            self.candidate_side = None
            self.candidate_frames = 0
            return self.count_side, self.count_side
        if new_side != self.candidate_side:
            self.candidate_side = new_side
            self.candidate_frames = 1
        else:
            self.candidate_frames += 1
        previous = self.count_side
        if self.candidate_frames >= min_stable_frames:
            self.count_side = new_side
        return previous, self.count_side

    def check_polygon_crossing(
        self, line_p1, line_p2, inside_sign, hysteresis, now, cooldown
    ):
        if self.time_since_update != 0:
            return None
        previous, current = self.update_polygon_side(
            line_p1, line_p2, inside_sign, hysteresis
        )
        if previous not in ("OUTSIDE", "INSIDE") or current == previous:
            return None
        if now - self.last_event_time < cooldown:
            return None
        event = "IN" if previous == "OUTSIDE" and current == "INSIDE" else "OUT"
        self.last_event = event
        self.last_event_time = now
        return event

    def check_crossing(self, line_y: float, hysteresis: float, now: float,
                       cooldown: float, in_direction="top_to_bottom"):
        if self.time_since_update != 0:
            return None
        previous, current = self.update_count_side(line_y, hysteresis)
        if previous not in ("ABOVE", "BELOW") or current == previous:
            return None
        if now - self.last_event_time < cooldown:
            return None
        moved_down = previous == "ABOVE" and current == "BELOW"
        if in_direction == "bottom_to_top":
            event = "OUT" if moved_down else "IN"
        else:
            event = "IN" if moved_down else "OUT"
        self.last_event = event
        self.last_event_time = now
        return event


class IoUTracker:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.25,
                 min_width=30, min_height=30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.min_width = min_width
        self.min_height = min_height
        self.next_id = 0
        self.tracks = []

    @staticmethod
    def iou(bbox1, bbox2):
        x1, y1, w1, h1 = map(float, bbox1)
        x2, y2, w2, h2 = map(float, bbox2)

        a1x, a1y = x1 - w1 / 2, y1 - h1 / 2
        a2x, a2y = x2 - w2 / 2, y2 - h2 / 2
        b1x, b1y = x1 + w1 / 2, y1 + h1 / 2
        b2x, b2y = x2 + w2 / 2, y2 + h2 / 2

        ix1, iy1 = max(a1x, a2x), max(a1y, a2y)
        ix2, iy2 = min(b1x, b2x), min(b1y, b2y)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0.0

    def associate(self, detections):
        if not self.tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(self.tracks))), []

        cost = np.ones((len(self.tracks), len(detections)), dtype=float)
        for i, track in enumerate(self.tracks):
            pred_bbox = track.get_bbox()
            px, py, pw, ph = pred_bbox
            
            for j, det in enumerate(detections):
                dx, dy, dw, dh = det
                
                # FIX: Hybrid Tracking. If IoU drops because the bounding box size 
                # fluctuated, we now fall back to Centroid Distance to maintain the ID.
                iou_score = self.iou(pred_bbox, det)
                
                dist = np.hypot(px - dx, py - dy)
                diag = np.hypot(pw, ph)
                dist_score = max(0.0, 1.0 - (dist / max(diag, 1.0)))

                # Combine scores, prioritizing whichever is higher
                match_score = max(iou_score, dist_score * 0.75)
                cost[i, j] = 1.0 - match_score

        rows, cols = linear_sum_assignment(cost)
        matches = []
        matched_tracks = set()
        matched_detections = set()

        for r, c in zip(rows, cols):
            if cost[r, c] <= 1.0 - self.iou_threshold:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_detections.add(c)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_detections = self.associate(detections)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx], self.min_width, self.min_height
            )

        alive = []
        for i, track in enumerate(self.tracks):
            if i in unmatched_tracks:
                track.no_losses += 1
                if track.hits < self.min_hits or track.no_losses > self.max_age:
                    continue
            alive.append(track)

        for det_idx in unmatched_detections:
            alive.append(Track(detections[det_idx], self.next_id))
            self.next_id += 1

        self.tracks = alive

    def get_tracks(self):
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]

    def reset(self):
        self.tracks.clear()
        self.next_id = 0


# ============================================================
# 6. CAMERA THREAD
# ============================================================

class CameraThread(threading.Thread):
    def __init__(self, camera_url: str, cfg: Config):
        super().__init__(daemon=True, name="HikvisionCamera")
        self.camera_url = camera_url
        self.cfg = cfg
        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.connected = False
        self._connection_state_known = False
        self._disconnected_since_monotonic = None
        self.last_frame_monotonic = None
        self.connection_events = Queue(maxsize=100)

    def _set_connected(self, connected, reason=""):
        connected = bool(connected)
        if self._connection_state_known and connected == self.connected:
            return
        now_mono = time.monotonic()
        downtime_seconds = None
        if connected:
            if self._disconnected_since_monotonic is not None:
                downtime_seconds = round(
                    max(0.0, now_mono - self._disconnected_since_monotonic), 2
                )
            self._disconnected_since_monotonic = None
        else:
            self._disconnected_since_monotonic = now_mono
        self.connected = connected
        self._connection_state_known = True
        event = {
            "connected": connected,
            "reason": str(reason),
            "downtime_seconds": downtime_seconds,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        try:
            self.connection_events.put_nowait(event)
        except Exception:
            pass

    def drain_connection_events(self):
        events = []
        while True:
            try:
                events.append(self.connection_events.get_nowait())
            except Empty:
                return events

    def get_frame(self, last_sequence=-1):
        with self.lock:
            if self.frame is None or self.sequence == last_sequence:
                return None, last_sequence
            return self.frame.copy(), self.sequence

    def _open(self):
        try:
            cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(self.camera_url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def run(self):
        cap = None
        while self.running.is_set():
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None: cap.release()
                    cap = self._open()
                    if not cap.isOpened():
                        self._set_connected(False, "RTSP open failed")
                        time.sleep(self.cfg.reconnect_delay_s)
                        continue
                    self._set_connected(True, "RTSP stream opened")

                ret, frame = cap.read()
                if not ret or frame is None:
                    self._set_connected(False, "Camera frame read failed")
                    cap.release()
                    cap = None
                    time.sleep(self.cfg.reconnect_delay_s)
                    continue

                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                    self.last_frame_monotonic = time.monotonic()

            except Exception as exc:
                self._set_connected(False, f"Camera exception: {exc}")
                if cap is not None: cap.release()
                cap = None
                time.sleep(self.cfg.reconnect_delay_s)

        self._set_connected(False, "Camera thread stopped")
        if cap is not None:
            cap.release()

    def stop(self):
        self.running.clear()


# ============================================================
# 7. STORE COUNTER
# ============================================================

class StoreCounter:
    def __init__(self, cfg: Config, storage: SQLiteStore):
        self.cfg = cfg
        self.storage = storage
        self.business_date = date.today().isoformat()

        print("Loading YOLO model...")
        self.model = YOLO(cfg.model_path)
        print("YOLO model loaded")

        self.tracker = IoUTracker(
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_threshold,
            min_width=cfg.min_object_width,
            min_height=cfg.min_object_height,
        )
        self.staff_detector = StaffUniformDetector(
            reference_path=cfg.staff_logo_reference,
            enrollment_reference_dir=cfg.staff_enrollment_reference_dir,
            profile_path=cfg.staff_profile_file,
            enabled=cfg.staff_detection_enabled,
            min_person_width=cfg.staff_min_person_width,
            min_person_height=cfg.staff_min_person_height,
            white_min_ratio=cfg.staff_white_min_ratio,
            gray_min_ratio=cfg.staff_gray_min_ratio,
            belt_min_ratio=cfg.staff_belt_min_ratio,
            frame_threshold=cfg.staff_frame_threshold,
            track_threshold=cfg.staff_track_threshold,
            strong_frame_threshold=cfg.staff_strong_frame_threshold,
            min_positive_frames=cfg.staff_min_positive_frames,
            min_negative_frames=cfg.staff_min_negative_frames,
            belt_bonus=cfg.staff_belt_bonus,
            logo_bonus=cfg.staff_logo_bonus,
            fast_customer_factor=cfg.staff_fast_customer_factor,
            fast_customer_frames=cfg.staff_fast_customer_frames,
        )
        if self.staff_detector.ready:
            print(
                "Staff uniform detection enabled (white upper + gray trousers; "
                f"logo references={'yes' if self.staff_detector.logo_ready else 'no'}, "
                f"branch profile={'yes' if self.staff_detector.profile_ready else 'no'})"
            )
        elif cfg.staff_detection_enabled:
            print(
                "WARNING: Staff uniform detection disabled: "
                f"{self.staff_detector.load_error or 'reference has too few features'}"
            )

        self.face_recognizer = StaffFaceRecognizer(
            detector_model=cfg.face_detector_model,
            recognizer_model=cfg.face_recognizer_model,
            profile_file=cfg.face_profile_file,
            enabled=cfg.face_detection_enabled,
            match_threshold=cfg.face_match_threshold,
            support_threshold=cfg.face_support_threshold,
            min_face_size=cfg.face_min_size,
            min_positive_frames=cfg.face_min_positive_frames,
        )
        if self.face_recognizer.ready:
            print(
                "Staff face recognition enabled: "
                f"{self.face_recognizer.enrolled_employee_count} employee(s)"
            )
        elif cfg.face_detection_enabled:
            print(
                "Staff face recognition waiting for enrollment: "
                f"{self.face_recognizer.load_error or cfg.face_profile_file}"
            )

        restored = self.storage.prepare_business_day(self.business_date)
        self.total_in = restored["total_in"]
        self.total_out = restored["total_out"]
        self.inside_ids = set()
        self.inside_timestamps = {}  # legacy; occupancy no longer expires
        self.peak_inside = restored["peak_inside"]
        self.staff_ignored = restored["staff_ignored"]
        self.unknown_crossings = restored["unknown_crossings"]
        self.pending_crossings = []

        self.zone = (cfg.zone_x1, cfg.zone_y1, cfg.zone_x2, cfg.zone_y2)
        self.zone_polygon = tuple(cfg.zone_polygon)
        self.line_p1 = tuple(cfg.line_p1)
        self.line_p2 = tuple(cfg.line_p2)
        self.line_hysteresis = cfg.line_hysteresis

        self.track_colors = {}

        self.frame_counter = 0
        self.fps = 0.0
        self._fps_timer = time.monotonic()
        self._fps_frames = 0
        self.last_sound_time = 0.0
        self._last_state_save = time.monotonic()
        self._model_error_count = 0
        self._model_error_active = False

        LOGGER.info(
            "Counter restored for %s: IN=%s OUT=%s STAFF=%s UNKNOWN=%s",
            self.business_date,
            self.total_in,
            self.total_out,
            self.staff_ignored,
            self.unknown_crossings,
        )

    def scaled_geometry(self, width, height):
        sx = width / self.cfg.base_width
        sy = height / self.cfg.base_height
        polygon = tuple(
            (int(round(x * sx)), int(round(y * sy)))
            for x, y in self.cfg.zone_polygon
        )
        line_p1 = (
            int(round(self.cfg.line_p1[0] * sx)),
            int(round(self.cfg.line_p1[1] * sy)),
        )
        line_p2 = (
            int(round(self.cfg.line_p2[0] * sx)),
            int(round(self.cfg.line_p2[1] * sy)),
        )
        hysteresis = self.cfg.line_hysteresis * ((sx + sy) / 2.0)
        return polygon, line_p1, line_p2, hysteresis

    def apply_counting_config(self, counting):
        """Apply a newly saved doorway without resetting persisted totals."""
        for key in (
            "base_width", "base_height", "zone_x1", "zone_y1", "zone_x2",
            "zone_y2", "line_hysteresis",
        ):
            setattr(self.cfg, key, int(counting[key]))
        self.cfg.in_direction = str(counting["in_direction"])
        self.cfg.line_position_ratio = float(counting.get("line_position_ratio", 0.50))
        self.cfg.zone_polygon = tuple(
            tuple(map(float, point)) for point in counting["zone_polygon"]
        )
        self.cfg.line_p1 = tuple(map(float, counting["line_p1"]))
        self.cfg.line_p2 = tuple(map(float, counting["line_p2"]))
        self.cfg.inside_sign = int(counting["inside_sign"])
        self.cfg.crossing_cooldown_s = float(counting["crossing_cooldown_s"])
        self.zone = (
            self.cfg.zone_x1, self.cfg.zone_y1,
            self.cfg.zone_x2, self.cfg.zone_y2,
        )
        self.zone_polygon = tuple(self.cfg.zone_polygon)
        self.line_p1 = tuple(self.cfg.line_p1)
        self.line_p2 = tuple(self.cfg.line_p2)
        self.line_hysteresis = self.cfg.line_hysteresis
        # Old track-side history belongs to the old line and could fabricate a
        # crossing immediately after calibration changes.
        self.pending_crossings.clear()
        self.inside_ids.clear()
        self.inside_timestamps.clear()
        self.track_colors.clear()
        self.tracker.reset()

    def get_color(self, track_id):
        if track_id not in self.track_colors:
            rng = np.random.default_rng(track_id)
            self.track_colors[track_id] = tuple(int(v) for v in rng.integers(0, 255, size=3))
        return self.track_colors[track_id]

    def is_valid_object(self, track):
        w, h = track.get_size()
        return w >= self.cfg.min_object_width and h >= self.cfg.min_object_height

    def detect(self, frame):
        kwargs = {
            "classes": [0],
            "conf": self.cfg.confidence,
            "iou": self.cfg.nms_iou,
            "imgsz": self.cfg.image_size,
            "verbose": False,
        }
        if self.cfg.device:
            kwargs["device"] = self.cfg.device

        try:
            results = self.model(frame, **kwargs)
        except Exception as exc:
            self._model_error_count += 1
            if self._model_error_count == 1 or self._model_error_count % 10 == 0:
                LOGGER.exception(
                    "YOLO inference failed (%s consecutive failures)",
                    self._model_error_count,
                )
                self.storage.log_health(
                    "YOLO_ERROR",
                    f"consecutive={self._model_error_count}; {type(exc).__name__}: {exc}",
                    self.business_date,
                )
            self._model_error_active = True
            if self._model_error_count >= 30:
                raise RuntimeError("YOLO failed 30 consecutive times") from exc
            return []

        if self._model_error_active:
            self.storage.log_health(
                "YOLO_RECOVERED",
                f"Recovered after {self._model_error_count} consecutive error(s)",
                self.business_date,
            )
            LOGGER.info("YOLO recovered after %s error(s)", self._model_error_count)
        self._model_error_count = 0
        self._model_error_active = False
        detections = []

        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append([cx, cy, w, h])

        return detections

    def process_frame(self, frame):
        self.maybe_roll_business_day()
        height, width = frame.shape[:2]
        polygon, line_p1, line_p2, hysteresis = self.scaled_geometry(width, height)
        self.zone_polygon = polygon
        self.line_p1 = line_p1
        self.line_p2 = line_p2

        self.frame_counter += 1
        self._fps_frames += 1
        now_mono = time.monotonic()
        if now_mono - self._fps_timer >= 1.0:
            self.fps = self._fps_frames / (now_mono - self._fps_timer)
            self._fps_frames = 0
            self._fps_timer = now_mono

        detections = self.detect(frame)
        self.tracker.update(detections)
        tracks = self.tracker.get_tracks()

        now = time.monotonic()
        active_track_ids = {t.track_id for t in tracks}

        for tid in list(self.track_colors.keys()):
            if tid not in active_track_ids:
                del self.track_colors[tid]
        # Tracker IDs cannot survive a process restart and are not the source
        # of truth for occupancy, so expired IDs must not accumulate forever.
        self.inside_ids.intersection_update(active_track_ids)
        for tid in list(self.inside_timestamps):
            if tid not in active_track_ids:
                del self.inside_timestamps[tid]

        # Build a track-level staff decision from multiple real observations.
        # This is intentionally done before crossing events are resolved.
        for track in tracks:
            if not self.is_valid_object(track):
                continue
            if (
                track.staff_status != "STAFF"
                and track.time_since_update == 0
                and self.frame_counter % self.cfg.staff_process_every_n_frames == 0
            ):
                evidence = self.staff_detector.analyze(frame, track.last_observed_bbox)
                track.uniform_status = self.staff_detector.update_track(track, evidence)

            if (
                track.staff_status != "STAFF"
                and
                track.time_since_update == 0
                and
                self.face_recognizer.ready
                and self.frame_counter % self.cfg.face_process_every_n_frames == 0
            ):
                face_evidence = self.face_recognizer.analyze(
                    frame, track.last_observed_bbox
                )
                self.face_recognizer.update_track(track, face_evidence)
            self._update_hybrid_staff_status(track)

        self._resolve_pending_crossings(now)

        for track in tracks:
            if not self.is_valid_object(track):
                continue
            # Only points inside the calibrated entrance polygon can count.
            foot_x, foot_y = track.get_count_point()
            contour = np.asarray(polygon, dtype=np.float32)
            if cv2.pointPolygonTest(contour, (foot_x, foot_y), False) < 0:
                continue
            event = track.check_polygon_crossing(
                line_p1,
                line_p2,
                self.cfg.inside_sign,
                hysteresis,
                now,
                self.cfg.crossing_cooldown_s,
            )

            if event is not None:
                self._handle_crossing(track, event, now)

            if track.track_id in self.inside_ids:
                self.inside_timestamps[track.track_id] = now

        # Occupancy is an event ledger, not the number of active tracker IDs.
        # If a tracker ID expires inside the shop, the person remains inside.
        current_inside = max(0, self.total_in - self.total_out)
        self.peak_inside = max(self.peak_inside, current_inside)

        if now_mono - self._last_state_save >= self.cfg.state_snapshot_interval_s:
            self.storage.save_state(self.state_snapshot())
            self._last_state_save = now_mono

        return self.draw(
            frame, tracks, polygon, line_p1, line_p2, hysteresis, current_inside
        )

    def _update_hybrid_staff_status(self, track):
        """Mark staff when either face OR uniform is independently confirmed."""
        if not self.face_recognizer.ready:
            track.staff_status = track.uniform_status
            return track.staff_status

        matching_scores = [
            score for score, employee_id in track.face_scores
            if employee_id == track.face_employee
        ]
        face_positive_frames = sum(
            score >= self.cfg.face_match_threshold for score in matching_scores
        )
        face_confirmed = (
            face_positive_frames >= self.cfg.face_min_positive_frames
            or any(
                score >= self.cfg.face_match_threshold + 0.08
                for score in matching_scores
            )
        )
        # Strict OR: either recognizer may confirm staff on its own.  Weak
        # evidence from both recognizers is not combined into a confirmation.
        if face_confirmed or track.uniform_status == "STAFF":
            track.staff_status = "STAFF"
        elif (
            track.uniform_status == "CUSTOMER"
            and getattr(track, "uniform_clear_customer", False)
        ):
            track.staff_status = "CUSTOMER"
        elif (
            track.face_observations >= 3
            and track.face_score < self.cfg.face_support_threshold
            and track.uniform_status == "CUSTOMER"
        ):
            track.staff_status = "CUSTOMER"
        else:
            track.staff_status = "UNKNOWN"
        return track.staff_status

    def _handle_crossing(self, track, event, now):
        if track.staff_status == "STAFF":
            self._commit_crossing(track, event, "STAFF")
        elif track.staff_status == "CUSTOMER" or not self.staff_detector.ready:
            self._commit_crossing(track, event, "CUSTOMER")
        else:
            # Keep the event while later frames provide a better torso/logo view.
            duplicate = any(
                pending["track"].track_id == track.track_id
                and pending["event"] == event
                for pending in self.pending_crossings
            )
            if not duplicate:
                self.pending_crossings.append({
                    "track": track,
                    "event": event,
                    "created": now,
                })

    def _resolve_pending_crossings(self, now):
        remaining = []
        for pending in self.pending_crossings:
            track = pending["track"]
            if track.staff_status == "STAFF":
                self._commit_crossing(track, pending["event"], "STAFF")
            elif track.staff_status == "CUSTOMER":
                self._commit_crossing(track, pending["event"], "CUSTOMER")
            elif now - pending["created"] >= self.cfg.staff_decision_timeout_s:
                partial_check = getattr(
                    self.staff_detector, "has_partial_staff_evidence", lambda _track: False
                )
                if partial_check(track):
                    self._commit_crossing(track, pending["event"], "STAFF")
                else:
                    self._commit_crossing(track, pending["event"], "UNKNOWN")
            else:
                remaining.append(pending)
        self.pending_crossings = remaining

    def _commit_crossing(self, track, event, person_type):
        if person_type == "STAFF":
            self.staff_ignored += 1
            self._log_event(track.track_id, f"STAFF_{event}_IGNORED", play_alert=False)
            return

        if person_type == "UNKNOWN":
            self.unknown_crossings += 1
        if event == "IN":
            self.total_in += 1
            self.inside_ids.add(track.track_id)
            self.inside_timestamps[track.track_id] = time.monotonic()
        elif event == "OUT":
            self.total_out += 1
            self.inside_ids.discard(track.track_id)
            self.inside_timestamps.pop(track.track_id, None)
        log_event = event if person_type == "CUSTOMER" else f"{event}_UNKNOWN_COUNTED"
        self._log_event(track.track_id, log_event)

    def _log_event(self, track_id, event, play_alert=True):
        self.storage.add_event(track_id, event, self.state_snapshot())

        now = time.monotonic()
        if play_alert and now - self.last_sound_time >= self.cfg.sound_cooldown_s:
            play_sound(self.cfg.sound_file)
            self.last_sound_time = now

        print(
            f"Track {track_id} {event} | "
            f"IN={self.total_in} OUT={self.total_out} "
            f"INSIDE={max(0, self.total_in - self.total_out)}"
        )

    def draw(self, frame, tracks, polygon, line_p1, line_p2, hysteresis, current_inside):
        overlay = frame.copy()
        contour = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [contour], (0, 255, 255))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [contour], True, (0, 255, 255), 3)

        dx, dy = line_p2[0] - line_p1[0], line_p2[1] - line_p1[1]
        length = max(1.0, float(np.hypot(dx, dy)))
        nx, ny = -dy / length, dx / length
        band = np.asarray([
            (int(line_p1[0] + nx * hysteresis), int(line_p1[1] + ny * hysteresis)),
            (int(line_p2[0] + nx * hysteresis), int(line_p2[1] + ny * hysteresis)),
            (int(line_p2[0] - nx * hysteresis), int(line_p2[1] - ny * hysteresis)),
            (int(line_p1[0] - nx * hysteresis), int(line_p1[1] - ny * hysteresis)),
        ], dtype=np.int32)
        cv2.polylines(frame, [band], True, (255, 180, 0), 2)
        cv2.line(frame, line_p1, line_p2, (0, 0, 0), 8)
        cv2.line(frame, line_p1, line_p2, (0, 255, 0), 4)
        midpoint = (
            int(round((line_p1[0] + line_p2[0]) / 2.0)),
            int(round((line_p1[1] + line_p2[1]) / 2.0)),
        )
        arrow_end = (
            int(round(midpoint[0] + nx * self.cfg.inside_sign * 55)),
            int(round(midpoint[1] + ny * self.cfg.inside_sign * 55)),
        )
        cv2.arrowedLine(frame, midpoint, arrow_end, (0, 255, 255), 4, tipLength=0.25)
        cv2.putText(
            frame, "IN", (arrow_end[0] + 6, arrow_end[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA,
        )
        line_label_x = min(line_p1[0], line_p2[0]) + 8
        line_label_y = max(24, min(line_p1[1], line_p2[1]) - 12)
        cv2.putText(frame, "COUNTING LINE", (line_label_x, line_label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(frame, "COUNTING LINE", (line_label_x, line_label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for track in tracks:
            x, y, w, h = track.get_bbox()
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            small = not self.is_valid_object(track)
            color = (128, 128, 128) if small else self.get_color(track.track_id)
            if track.track_id in self.inside_ids:
                color = (0, 255, 0)
            if track.staff_status == "STAFF":
                color = (255, 0, 255)

            thickness = 1 if small else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # The red dot is the exact bottom-center point used for counting.
            foot_x, foot_y = track.get_count_point()
            cv2.circle(frame, (int(foot_x), int(foot_y)), 7, (0, 0, 255), -1)

            side = track.count_side or "?"
            label = f"ID:{track.track_id} {side}"
            if self.staff_detector.ready:
                label += f" {track.staff_status} U:{track.staff_score:.2f}"
                evidence = track.staff_last_evidence
                if evidence is not None and evidence.usable:
                    label += (
                        f" W:{evidence.white_ratio:.2f}"
                        f" G:{evidence.gray_ratio:.2f}"
                        f" B:{evidence.belt_ratio:.2f}"
                        f" P:{evidence.profile_similarity:.2f}"
                    )
            if self.face_recognizer.ready and track.face_employee:
                label += f" F:{track.face_employee}:{track.face_score:.2f}"
            if track.last_event:
                label += f" {track.last_event}"

            cv2.putText(frame, label, (x1, max(15, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        panel_width = min(400, max(320, frame.shape[1] - 20))
        panel_height = 185
        panel_x = max(10, frame.shape[1] - panel_width - 10)
        panel_y = 10
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
        cv2.putText(frame, "STORE COUNTER", (panel_x + 16, panel_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (panel_x + panel_width - 105, panel_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.line(frame, (panel_x + 12, panel_y + 48),
                 (panel_x + panel_width - 12, panel_y + 48), (100, 100, 100), 1)

        metric_font = cv2.FONT_HERSHEY_SIMPLEX
        metric_scale = 0.82
        metric_thickness = 2
        cv2.putText(frame, f"IN: {self.total_in}", (panel_x + 16, panel_y + 82),
                    metric_font, metric_scale, (0, 255, 0), metric_thickness)
        cv2.putText(frame, f"OUT: {self.total_out}", (panel_x + 16, panel_y + 122),
                    metric_font, metric_scale, (0, 0, 255), metric_thickness)
        cv2.putText(frame, f"STAFF IGNORED: {self.staff_ignored}",
                    (panel_x + 16, panel_y + 162), metric_font,
                    metric_scale, (255, 0, 255), metric_thickness)

        return frame

    def reset(self):
        self.total_in = 0
        self.total_out = 0
        self.peak_inside = 0
        self.staff_ignored = 0
        self.unknown_crossings = 0
        self.pending_crossings.clear()
        self.inside_ids.clear()
        self.inside_timestamps.clear()
        self.track_colors.clear()
        self.tracker.reset()
        self.storage.save_state(self.state_snapshot())

    def state_snapshot(self):
        return {
            "business_date": self.business_date,
            "total_in": int(self.total_in),
            "total_out": int(self.total_out),
            "staff_ignored": int(self.staff_ignored),
            "unknown_crossings": int(self.unknown_crossings),
            "peak_inside": int(self.peak_inside),
        }

    def maybe_roll_business_day(self):
        today = date.today().isoformat()
        if today == self.business_date:
            return False
        previous = self.business_date
        self.storage.save_state(self.state_snapshot())
        self.storage.export_daily_excel(previous)
        self.business_date = today
        self.total_in = 0
        self.total_out = 0
        self.peak_inside = 0
        self.staff_ignored = 0
        self.unknown_crossings = 0
        self.pending_crossings.clear()
        self.inside_ids.clear()
        self.inside_timestamps.clear()
        self.track_colors.clear()
        self.tracker.reset()
        self.storage.save_state(self.state_snapshot())
        self.storage.log_health("BUSINESS_DAY_STARTED", f"Previous day: {previous}", today)
        LOGGER.info("Business day rolled over from %s to %s", previous, today)
        return True

    def export_report(self):
        report = self.storage.export_daily_excel(self.business_date)
        LOGGER.info("Daily hourly report exported: %s", report)
        return report

    def close(self):
        self.storage.close(self.state_snapshot(), export_today=True)


# ============================================================
# 8. MAIN
# ============================================================

def main():
    heartbeat = HeartbeatWriter(CONFIG.heartbeat_file)
    heartbeat.write(
        status="starting", pid=os.getpid(), branch_id=CONFIG.branch_id,
        camera_connected=False,
    )
    try:
        validate_config(_BRANCH_CONFIG)
    except ValueError as exc:
        print(f"Configuration is incomplete: {exc}")
        print("Run setup_branch.py or double-click Start People Counter.cmd.")
        LOGGER.error("Configuration is incomplete: %s", exc)
        heartbeat.write(status="configuration_error", pid=os.getpid(), error=str(exc))
        return 2
    try:
        camera_url = build_rtsp_url(CONFIG)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        LOGGER.error("Unable to build camera URL: %s", exc)
        heartbeat.write(status="configuration_error", pid=os.getpid(), error=str(exc))
        return 2

    instance_lock = BranchInstanceLock(CONFIG.branch_id)
    if not instance_lock.acquired:
        LOGGER.warning("Another counter instance is already running for %s", CONFIG.branch_id)
        heartbeat.write(
            status="already_running", pid=os.getpid(), branch_id=CONFIG.branch_id
        )
        return 4

    storage = SQLiteStore(
        CONFIG.database_file, CONFIG.branch_id, CONFIG.camera_ip,
        CONFIG.reports_dir, CONFIG.event_retention_days,
        CONFIG.report_retention_days, CONFIG.report_refresh_delay_s,
    )
    storage.run_maintenance()
    storage.log_health("PROGRAM_STARTING", f"pid={os.getpid()}")
    counter = None
    camera = None
    exit_code = 0

    try:
        counter = StoreCounter(CONFIG, storage)
        camera = CameraThread(camera_url, CONFIG)
        camera.start()

        cv2.namedWindow(CONFIG.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONFIG.window_name, CONFIG.display_width, CONFIG.display_height)

        process_counter = 0
        last_processed_frame = None
        last_sequence = -1
        last_heartbeat = 0.0
        active_counting_signature = counting_signature(_BRANCH_CONFIG)
        storage.log_health("PROGRAM_RUNNING", "Model loaded and main loop started")
        LOGGER.info("People Counter running for branch %s", CONFIG.branch_id)

        while True:
            if not display_window_is_open(CONFIG.window_name):
                LOGGER.info("Display window was closed")
                break
            now_mono = time.monotonic()
            for connection_event in camera.drain_connection_events():
                event_name = (
                    "CAMERA_CONNECTED" if connection_event["connected"]
                    else "CAMERA_DISCONNECTED"
                )
                details = connection_event["reason"]
                if connection_event.get("downtime_seconds") is not None:
                    details += (
                        f"; outage_duration_seconds="
                        f"{connection_event['downtime_seconds']}"
                    )
                storage.log_health(event_name, details)
                LOGGER.info("%s: %s", event_name, details)

            if now_mono - last_heartbeat >= CONFIG.heartbeat_interval_s:
                counter.maybe_roll_business_day()
                try:
                    # Read the small JSON file on every heartbeat.  Comparing only
                    # its mtime proved unreliable when a remote PC replaced the
                    # file quickly or synced an older timestamp.
                    refreshed = load_branch_config(ACTIVE_CONFIG_PATH)
                    validate_config(refreshed)
                    if refreshed.get("branch_id") != CONFIG.branch_id:
                        storage.log_health(
                            "ACTIVE_BRANCH_CHANGED",
                            f"from={CONFIG.branch_id}; to={refreshed.get('branch_id')}",
                        )
                        LOGGER.info(
                            "Active branch changed from %s to %s; restarting",
                            CONFIG.branch_id,
                            refreshed.get("branch_id"),
                        )
                        exit_code = CONFIG_RESTART_EXIT_CODE
                        break
                    new_signature = counting_signature(refreshed)
                    if new_signature != active_counting_signature:
                        counter.apply_counting_config(refreshed["counting"])
                        active_counting_signature = new_signature
                        storage.log_health(
                            "COUNTING_ZONE_RELOADED",
                            f"zone={counter.zone}; direction={counter.cfg.in_direction}",
                        )
                        LOGGER.info(
                            "Reloaded counting zone %s direction=%s",
                            counter.zone,
                            counter.cfg.in_direction,
                        )
                except (OSError, ValueError, RuntimeError) as exc:
                    LOGGER.error("Unable to reload saved counting config: %s", exc)
                last_frame_age = None
                if camera.last_frame_monotonic is not None:
                    last_frame_age = round(
                        max(0.0, now_mono - camera.last_frame_monotonic), 2
                    )
                heartbeat.write(
                    status="running",
                    pid=os.getpid(),
                    branch_id=CONFIG.branch_id,
                    business_date=counter.business_date,
                    camera_connected=camera.connected,
                    camera_last_frame_age_s=last_frame_age,
                    fps=round(counter.fps, 2),
                    entered=counter.total_in,
                    exited=counter.total_out,
                    inside=max(0, counter.total_in - counter.total_out),
                    staff_ignored=counter.staff_ignored,
                    unknown_counted=counter.unknown_crossings,
                )
                last_heartbeat = now_mono

            frame, seq = camera.get_frame(last_sequence)
            if frame is None:
                cv2.waitKey(1)
                time.sleep(0.005)
                continue
            last_sequence = seq

            process_counter += 1
            if process_counter % CONFIG.process_every_n_frames == 0:
                last_processed_frame = counter.process_frame(frame)
                cv2.imshow(CONFIG.window_name, last_processed_frame)
            else:
                display_frame = last_processed_frame if last_processed_frame is not None else frame
                cv2.imshow(CONFIG.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                counter.reset()
            elif key == ord("e"):
                report = counter.export_report()
                print(f"Daily Excel report: {report}")

    except KeyboardInterrupt:
        LOGGER.info("Stopped from keyboard")
    except Exception as exc:
        exit_code = 1
        LOGGER.exception("Fatal main-loop error")
        try:
            storage.log_health("PROGRAM_CRASH", f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        if camera is not None:
            camera.stop()
            camera.join(timeout=3.0)
        if counter is not None:
            try:
                storage.log_health("PROGRAM_STOPPED", f"exit_code={exit_code}")
                counter.close()
            except Exception:
                LOGGER.exception("Failed while closing the counter")
        else:
            try:
                storage.close(export_today=False)
            except Exception:
                LOGGER.exception("Failed while closing storage after startup error")
        cv2.destroyAllWindows()
        heartbeat.write(
            status="stopped" if exit_code == 0 else "crashed",
            pid=os.getpid(), branch_id=CONFIG.branch_id, exit_code=exit_code,
        )
        instance_lock.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
