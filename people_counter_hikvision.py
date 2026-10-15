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
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from urllib.parse import quote

import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from staff_uniform import StaffUniformDetector
from branch_config import load_branch_config, resolve_project_path, validate_config


# Configure FFmpeg RTSP socket timeout options globally before video capture opens
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"


# ============================================================
# 1. CONFIGURATION
# ============================================================

_BRANCH_CONFIG = load_branch_config()


def _setting(section, key, default=None):
    return _BRANCH_CONFIG.get(section, {}).get(key, default)


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
    staff_logo_reference: str = str(resolve_project_path(
        _setting("staff", "logo_reference", "WhatsApp Image 2026-10-15 at 10.16.27 AM.jpeg")
    ))
    staff_min_person_width: int = int(_setting("staff", "min_person_width", 70))
    staff_min_person_height: int = int(_setting("staff", "min_person_height", 130))
    staff_white_min_ratio: float = float(_setting("staff", "white_min_ratio", 0.30))
    staff_frame_threshold: float = float(_setting("staff", "frame_threshold", 0.62))
    staff_track_threshold: float = float(_setting("staff", "track_threshold", 0.68))
    staff_min_positive_frames: int = int(_setting("staff", "min_positive_frames", 3))
    staff_decision_timeout_s: float = float(_setting("staff", "decision_timeout_s", 1.25))

    # ---------- Counting geometry ----------
    base_width: int = int(_setting("counting", "base_width", 1280))
    base_height: int = int(_setting("counting", "base_height", 720))
    zone_x1: int = int(_setting("counting", "zone_x1", 150))
    zone_y1: int = int(_setting("counting", "zone_y1", 170))
    zone_x2: int = int(_setting("counting", "zone_x2", 450))
    zone_y2: int = int(_setting("counting", "zone_y2", 260))
    in_direction: str = str(_setting("counting", "in_direction", "top_to_bottom"))
    line_hysteresis: int = int(_setting("counting", "line_hysteresis", 15))
    crossing_cooldown_s: float = float(_setting("counting", "crossing_cooldown_s", 0.75))
    inside_timeout_s: float = 600.0  # legacy, unused

    # ---------- Logging ----------
    log_file: str = str(resolve_project_path(
        _setting("logging", "file", "logs/default_branch_store_log.xlsx")
    ))
    log_queue_size: int = int(_setting("logging", "queue_size", 100))

    # ---------- UI ----------
    display_width: int = int(_setting("display", "width", 1100))
    display_height: int = int(_setting("display", "height", 700))
    window_name: str = str(_setting("display", "window_name", "Store People Counter - Hikvision"))

    # ---------- Sound ----------
    sound_file: str = str(resolve_project_path(_setting("sound", "file", "beep-01a.wav")))
    sound_cooldown_s: float = float(_setting("sound", "cooldown_s", 0.20))


CONFIG = Config()


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


# ============================================================
# 3. ASYNCHRONOUS EXCEL LOGGER
# ============================================================

class ExcelLogger:
    def __init__(self, filename: str, camera_ip: str, queue_size: int = 100):
        self.filename = Path(filename)
        self.camera_ip = camera_ip
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker = None
        self.enabled = False
        self._lock = threading.Lock()

        try:
            import openpyxl  # noqa: F401
            self.enabled = True
        except ImportError:
            print("⚠️ openpyxl not installed; Excel logging disabled.")
            return

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_workbook()
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="ExcelLogger",
            daemon=True,
        )
        self.worker.start()
        print(f"📊 Excel logging enabled: {self.filename}")

    def _ensure_workbook(self) -> None:
        from openpyxl import Workbook
        if self.filename.exists():
            return
        wb = Workbook()
        ws = wb.active
        ws.title = "Events"
        ws.append([
            "Camera_IP", "Date", "Time", "Track_ID", "Event",
            "Entered_Count", "Exited_Count", "Inside_Count",
        ])
        with self._lock:
            wb.save(self.filename)
        wb.close()

    def add_event(self, track_id: int, event: str,
                  entered: int, exited: int, inside: int) -> None:
        if not self.enabled:
            return
        row = (
            self.camera_ip,
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            int(track_id),
            event,
            int(entered),
            int(exited),
            int(inside),
        )
        try:
            self.queue.put_nowait(row)
        except Exception:
            pass

    def _append_rows(self, rows) -> None:
        from openpyxl import load_workbook
        with self._lock:
            wb = load_workbook(self.filename)
            ws = wb["Events"]
            for row in rows:
                ws.append(row)
            wb.save(self.filename)
            wb.close()

    def _worker_loop(self) -> None:
        pending = []
        last_flush = time.monotonic()
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                row = self.queue.get(timeout=0.25)
                pending.append(row)
                self.queue.task_done()
            except Empty:
                pass

            now = time.monotonic()
            if pending and (len(pending) >= 10 or now - last_flush >= 2.0):
                try:
                    self._append_rows(pending)
                    pending.clear()
                    last_flush = now
                except Exception as exc:
                    if len(pending) > 500:
                        pending = pending[-250:]
                    time.sleep(1.0)
        if pending:
            try:
                self._append_rows(pending)
            except Exception:
                pass

    def close(self) -> None:
        if not self.enabled:
            return
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=5.0)

    def show_report(self) -> None:
        if not self.enabled or not self.filename.exists():
            return
        try:
            import pandas as pd
            with self._lock:
                df = pd.read_excel(self.filename, engine="openpyxl")
            print("\n" + "=" * 70)
            print("📊 EXCEL REPORT")
            print(f"Total records: {len(df)}")
            if not df.empty:
                print("\nLast 5 records:")
                print(df.tail(5).to_string(index=False))
            print("=" * 70)
        except Exception:
            pass


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
        self.staff_score = 0.0
        self.staff_status = "UNKNOWN"
        self.staff_last_evidence = None
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

    def update_count_side(self, line_y: float, hysteresis: float, min_stable_frames=2):
        # Never advance the crossing state using predicted-only positions.
        # Use last observed detection center rather than a Kalman extrapolation.
        if self.time_since_update != 0:
            return self.count_side, self.count_side
        if self.recovered_after_gap:
            # Re-acquired after missing frames: re-initialize the side, do not
            # turn a long unseen interval into a fabricated crossing.
            self.count_side = None
            self.candidate_side = None
            self.candidate_frames = 0
            self.recovered_after_gap = False
        target_y = self.last_observed_bbox[1]
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
                        self.connected = False
                        time.sleep(self.cfg.reconnect_delay_s)
                        continue
                    self.connected = True

                ret, frame = cap.read()
                if not ret or frame is None:
                    self.connected = False
                    cap.release()
                    cap = None
                    time.sleep(self.cfg.reconnect_delay_s)
                    continue

                with self.lock:
                    self.frame = frame
                    self.sequence += 1

            except Exception:
                self.connected = False
                if cap is not None: cap.release()
                cap = None
                time.sleep(self.cfg.reconnect_delay_s)

        self.connected = False
        if cap is not None:
            cap.release()

    def stop(self):
        self.running.clear()


# ============================================================
# 7. STORE COUNTER
# ============================================================

class StoreCounter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        print("🔄 Loading YOLO model...")
        self.model = YOLO(cfg.model_path)
        print("✅ YOLO model loaded")

        self.tracker = IoUTracker(
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_threshold,
            min_width=cfg.min_object_width,
            min_height=cfg.min_object_height,
        )
        self.staff_detector = StaffUniformDetector(
            reference_path=cfg.staff_logo_reference,
            enabled=cfg.staff_detection_enabled,
            min_person_width=cfg.staff_min_person_width,
            min_person_height=cfg.staff_min_person_height,
            white_min_ratio=cfg.staff_white_min_ratio,
            frame_threshold=cfg.staff_frame_threshold,
            track_threshold=cfg.staff_track_threshold,
            min_positive_frames=cfg.staff_min_positive_frames,
        )
        if self.staff_detector.ready:
            print(f"Staff uniform detection enabled: {cfg.staff_logo_reference}")
        elif cfg.staff_detection_enabled:
            print(
                "WARNING: Staff uniform detection disabled: "
                f"{self.staff_detector.load_error or 'reference has too few features'}"
            )

        self.total_in = 0
        self.total_out = 0
        self.inside_ids = set()
        self.inside_timestamps = {}  # legacy; occupancy no longer expires
        self.peak_inside = 0
        self.staff_ignored = 0
        self.unknown_crossings = 0
        self.pending_crossings = []

        self.zone = (cfg.zone_x1, cfg.zone_y1, cfg.zone_x2, cfg.zone_y2)
        self.count_line_y = (cfg.zone_y1 + cfg.zone_y2) / 2.0
        self.line_hysteresis = cfg.line_hysteresis

        self.track_colors = {}
        self.ignored_small_track_ids = set()

        self.frame_counter = 0
        self.fps = 0.0
        self._fps_timer = time.monotonic()
        self._fps_frames = 0
        self.last_sound_time = 0.0

        self.excel_logger = ExcelLogger(cfg.log_file, cfg.camera_ip, cfg.log_queue_size)

    def scaled_geometry(self, width, height):
        sx = width / self.cfg.base_width
        sy = height / self.cfg.base_height
        zone = (
            int(self.cfg.zone_x1 * sx),
            int(self.cfg.zone_y1 * sy),
            int(self.cfg.zone_x2 * sx),
            int(self.cfg.zone_y2 * sy),
        )
        line_y = ((self.cfg.zone_y1 + self.cfg.zone_y2) / 2.0) * sy
        hysteresis = self.cfg.line_hysteresis * sy
        return zone, line_y, hysteresis

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

        results = self.model(frame, **kwargs)
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
        height, width = frame.shape[:2]
        zone, line_y, hysteresis = self.scaled_geometry(width, height)
        self.zone = zone
        self.count_line_y = line_y

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
            if tid not in active_track_ids and tid not in self.inside_ids:
                del self.track_colors[tid]

        # Build a track-level staff decision from multiple real observations.
        # This is intentionally done before crossing events are resolved.
        for track in tracks:
            if not self.is_valid_object(track):
                self.ignored_small_track_ids.add(track.track_id)
                continue
            evidence = self.staff_detector.analyze(frame, track.last_observed_bbox)
            self.staff_detector.update_track(track, evidence)

        self._resolve_pending_crossings(now)

        for track in tracks:
            if not self.is_valid_object(track):
                continue
            # Count only observed tracks whose detected center intersects the
            # configured horizontal segment, not people outside the entrance.
            cx = track.last_observed_bbox[0]
            if not (zone[0] <= cx <= zone[2]):
                continue
            event = track.check_crossing(
                line_y,
                hysteresis,
                now,
                self.cfg.crossing_cooldown_s,
                self.cfg.in_direction,
            )

            if event is not None:
                self._handle_crossing(track, event, now)

            if track.track_id in self.inside_ids:
                self.inside_timestamps[track.track_id] = now

        # Occupancy is an event ledger, not the number of active tracker IDs.
        # If a tracker ID expires inside the shop, the person remains inside.
        current_inside = max(0, self.total_in - self.total_out)
        self.peak_inside = max(self.peak_inside, current_inside)

        return self.draw(frame, tracks, zone, line_y, hysteresis, current_inside)

    def _handle_crossing(self, track, event, now):
        if track.staff_status == "STAFF":
            self._commit_crossing(track, event, "STAFF")
        elif track.staff_status == "CUSTOMER" or not self.staff_detector.ready:
            self._commit_crossing(track, event, "CUSTOMER")
        else:
            # Keep the event while later frames provide a better torso/logo view.
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
        log_event = event if person_type == "CUSTOMER" else f"{event}_UNKNOWN"
        self._log_event(track.track_id, log_event)

    def _log_event(self, track_id, event, play_alert=True):
        self.excel_logger.add_event(
            track_id=track_id,
            event=event,
            entered=self.total_in,
            exited=self.total_out,
            inside=max(0, self.total_in - self.total_out),
        )

        now = time.monotonic()
        if play_alert and now - self.last_sound_time >= self.cfg.sound_cooldown_s:
            play_sound(self.cfg.sound_file)
            self.last_sound_time = now

        print(
            f"🚶 Track {track_id} {event} | "
            f"IN={self.total_in} OUT={self.total_out} "
            f"INSIDE={max(0, self.total_in - self.total_out)}"
        )

    def draw(self, frame, tracks, zone, line_y, hysteresis, current_inside):
        overlay = frame.copy()
        cv2.rectangle(overlay, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 255), 2)

        band_top = int(line_y - hysteresis)
        band_bottom = int(line_y + hysteresis)
        cv2.rectangle(frame, (zone[0], band_top), (zone[2], band_bottom), (255, 180, 0), 1)
        cv2.line(frame, (zone[0], int(line_y)), (zone[2], int(line_y)), (0, 255, 0), 2)

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
            
            # VISUAL FIX: Draw a red dot exactly where the counter is measuring
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            side = track.count_side or "?"
            label = f"ID:{track.track_id} {side}"
            if self.staff_detector.ready:
                label += f" {track.staff_status}:{track.staff_score:.2f}"
            if track.last_event:
                label += f" {track.last_event}"

            cv2.putText(frame, label, (x1, max(15, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        panel_x = max(0, frame.shape[1] - 255)
        panel_y = 10
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 245, panel_y + 170), (0, 0, 0), -1)
        cv2.putText(frame, "STORE COUNTER", (panel_x + 10, panel_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"IN: {self.total_in}", (panel_x + 10, panel_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.total_out}", (panel_x + 10, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(frame, f"INSIDE: {current_inside}", (panel_x + 10, panel_y + 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (panel_x + 125, panel_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"STAFF IGNORED: {self.staff_ignored}",
                    (panel_x + 10, panel_y + 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 0, 255), 1)
        cv2.putText(frame, f"UNKNOWN: {self.unknown_crossings}",
                    (panel_x + 10, panel_y + 143), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 255), 1)

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

    def close(self):
        self.excel_logger.close()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    try:
        validate_config(_BRANCH_CONFIG)
    except ValueError as exc:
        print(f"Configuration is incomplete: {exc}")
        print("Run setup_branch.py or double-click Start People Counter.cmd.")
        return
    try:
        camera_url = build_rtsp_url(CONFIG)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    counter = StoreCounter(CONFIG)
    camera = CameraThread(camera_url, CONFIG)
    camera.start()

    cv2.namedWindow(CONFIG.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG.window_name, CONFIG.display_width, CONFIG.display_height)

    process_counter = 0
    last_processed_frame = None
    last_sequence = -1

    try:
        while True:
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
                counter.excel_logger.show_report()

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        camera.join(timeout=3.0)
        counter.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
