"""Collect consented, head-cropped uniform classifier samples from a branch camera."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from branch_config import (
    build_rtsp_url_from_config,
    load_branch_config,
    resolve_project_path,
    safe_branch_id,
)
from staff_uniform import StaffUniformDetector


VALID_LABELS = {"staff", "customer"}


def body_crop_without_head(frame, xyxy, head_crop_ratio=0.18):
    """Return a clipped neck-to-feet crop from an XYXY person detection."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = max(0, min(frame_w, int(round(x1))))
    x2 = max(0, min(frame_w, int(round(x2))))
    y1 = max(0, min(frame_h, int(round(y1))))
    y2 = max(0, min(frame_h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    body_top = y1 + int(round((y2 - y1) * float(head_crop_ratio)))
    body_top = max(y1, min(y2, body_top))
    if x2 - x1 < 45 or y2 - body_top < 80:
        return None
    return frame[body_top:y2, x1:x2].copy()


def blur_score(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_signature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def sufficiently_different(signature, recent_signatures, threshold=2.5):
    if not recent_signatures:
        return True
    return all(float(np.mean(np.abs(signature - old))) >= threshold for old in recent_signatures)


def choose_largest_person(result, min_width, min_height):
    """Choose one controlled-session subject, avoiding accidental bystanders."""
    if result is None or result.boxes is None:
        return None
    best = None
    best_area = 0.0
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id != 0:
            continue
        coordinates = box.xyxy[0].detach().cpu().numpy().astype(float)
        width = coordinates[2] - coordinates[0]
        height = coordinates[3] - coordinates[1]
        if width < min_width or height < min_height:
            continue
        area = width * height
        if area > best_area:
            best = coordinates
            best_area = area
    return best


def _write_jpeg(path, image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True


def _write_session_metadata(directory, payload):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{payload['session_id']}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def update_uniform_profile(path, label, feature_vectors, session_id, max_samples=1200):
    """Append a reviewed enrollment session to the branch's active profile."""
    path = Path(path)
    payload = {"format_version": 1, "samples": {"staff": [], "customer": []}, "sessions": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    samples = payload.setdefault("samples", {}).setdefault(label, [])
    accepted = []
    for vector in feature_vectors:
        if len(vector) != 3 or not all(np.isfinite(vector)):
            continue
        clipped = [float(np.clip(value, 0.0, 1.0)) for value in vector]
        if label == "staff" and not (
            clipped[0] >= 0.12 and clipped[1] >= 0.10
        ):
            continue
        accepted.append(clipped)
    samples.extend(accepted)
    payload["samples"][label] = samples[-int(max_samples):]
    payload.setdefault("sessions", []).append({
        "session_id": session_id,
        "label": label,
        "accepted_features": len(accepted),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    payload["sessions"] = payload["sessions"][-100:]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return len(accepted)


def _open_camera(config):
    camera = config["camera"]
    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(camera.get("open_timeout_ms", 5000)))
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(camera.get("read_timeout_ms", 2000)))
    url = build_rtsp_url_from_config(config)
    if not capture.open(url, cv2.CAP_FFMPEG):
        capture.open(url)
    return capture


def collect(label, activate_profile=False):
    label = str(label).strip().lower()
    if label not in VALID_LABELS:
        raise ValueError("Dataset label must be STAFF or CUSTOMER.")

    config = load_branch_config()
    branch_id = safe_branch_id(config.get("branch_id", "default_branch"))
    settings = config.get("uniform_dataset", {})
    root = resolve_project_path(
        settings.get("directory", f"branches/{branch_id}/uniform_dataset")
    )
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)

    duration = max(10.0, float(settings.get("collection_duration_s", 60)))
    max_samples = max(10, int(settings.get("max_samples_per_session", 180)))
    interval = max(0.15, float(settings.get("sample_interval_s", 0.35)))
    min_width = max(45, int(settings.get("min_person_width", 90)))
    min_height = max(90, int(settings.get("min_person_height", 180)))
    min_blur = max(0.0, float(settings.get("min_blur_score", 35.0)))
    head_crop_ratio = float(np.clip(settings.get("head_crop_ratio", 0.18), 0.12, 0.30))

    model_cfg = config.get("model", {})
    model_path = resolve_project_path(model_cfg.get("path", "yolov8n.pt"))
    model = YOLO(str(model_path))
    feature_detector = StaffUniformDetector(
        reference_path=None,
        enabled=True,
        min_person_width=min_width,
        min_person_height=min_height,
    )
    capture = _open_camera(config)
    if not capture.isOpened():
        raise RuntimeError("Could not open the branch camera for uniform data collection.")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    window = f"Uniform Dataset - {label.upper()} - automatic collection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1100, 700)
    started = time.monotonic()
    collect_after = started + 3.0
    last_sample = 0.0
    saved = 0
    rejected_blur = 0
    rejected_duplicate = 0
    recent_signatures = []
    feature_vectors = []
    logo_references_saved = 0
    cancelled = False

    try:
        while time.monotonic() - collect_after < duration and saved < max_samples:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            now = time.monotonic()
            display = frame.copy()
            selected = None

            if now >= collect_after and now - last_sample >= interval:
                results = model.predict(
                    frame,
                    conf=float(model_cfg.get("confidence", 0.40)),
                    iou=float(model_cfg.get("nms_iou", 0.40)),
                    imgsz=int(model_cfg.get("image_size", 640)),
                    classes=[0],
                    verbose=False,
                )
                selected = choose_largest_person(
                    results[0] if results else None, min_width, min_height
                )
                last_sample = now
                if selected is not None:
                    crop = body_crop_without_head(frame, selected, head_crop_ratio)
                    if crop is not None:
                        sharpness = blur_score(crop)
                        if sharpness < min_blur:
                            rejected_blur += 1
                        else:
                            signature = sample_signature(crop)
                            if sufficiently_different(signature, recent_signatures):
                                timestamp_ms = int(time.time() * 1000)
                                filename = (
                                    f"{branch_id}_{label}_{session_id}_"
                                    f"{saved + 1:04d}_{timestamp_ms}.jpg"
                                )
                                if _write_jpeg(label_dir / filename, crop):
                                    saved += 1
                                    center_bbox = [
                                        float((selected[0] + selected[2]) / 2.0),
                                        float((selected[1] + selected[3]) / 2.0),
                                        float(selected[2] - selected[0]),
                                        float(selected[3] - selected[1]),
                                    ]
                                    uniform_evidence = feature_detector.analyze(
                                        frame, center_bbox
                                    )
                                    if uniform_evidence.usable:
                                        feature_vectors.append([
                                            uniform_evidence.white_ratio,
                                            uniform_evidence.gray_ratio,
                                            uniform_evidence.belt_ratio,
                                        ])
                                    active_elapsed = max(0.0, now - collect_after)
                                    if (
                                        activate_profile
                                        and label == "staff"
                                        and active_elapsed <= 15.0
                                        and logo_references_saved < 12
                                    ):
                                        torso_bounds = feature_detector.torso_bounds(
                                            frame.shape, center_bbox
                                        )
                                        tx1, ty1, tx2, ty2 = torso_bounds
                                        torso = frame[ty1:ty2, tx1:tx2]
                                        reference_dir = resolve_project_path(
                                            config.get("staff", {}).get(
                                                "enrollment_reference_dir", ""
                                            )
                                            or f"branches/{branch_id}/uniform_refs"
                                        )
                                        reference_dir.mkdir(parents=True, exist_ok=True)
                                        reference_path = reference_dir / (
                                            f"uniform_{session_id}_{logo_references_saved + 1:02d}.jpg"
                                        )
                                        if torso.size and _write_jpeg(reference_path, torso):
                                            logo_references_saved += 1
                                    recent_signatures.append(signature)
                                    recent_signatures = recent_signatures[-8:]
                            else:
                                rejected_duplicate += 1

            if selected is not None:
                x1, y1, x2, y2 = map(int, selected)
                privacy_y = y1 + int((y2 - y1) * head_crop_ratio)
                cv2.rectangle(display, (x1, privacy_y), (x2, y2), (0, 255, 0), 3)
                cv2.line(display, (x1, privacy_y), (x2, privacy_y), (0, 255, 255), 2)

            remaining = max(0, int(round(duration - max(0.0, now - collect_after))))
            countdown = max(0, int(round(collect_after - now)))
            cv2.rectangle(display, (10, 10), (min(display.shape[1] - 10, 950), 125), (0, 0, 0), -1)
            cv2.putText(
                display, f"LABEL: {label.upper()}   SAVED: {saved}/{max_samples}",
                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
            )
            active_elapsed = max(0.0, now - collect_after)
            if countdown:
                instruction = f"Starting in {countdown}s - stand alone in view"
            elif activate_profile and label == "staff":
                if active_elapsed <= 15:
                    pose_prompt = "FACE CAMERA - show front logo"
                elif active_elapsed <= 28:
                    pose_prompt = "TURN LEFT"
                elif active_elapsed <= 41:
                    pose_prompt = "TURN RIGHT"
                elif active_elapsed <= 55:
                    pose_prompt = "TURN BACK"
                else:
                    pose_prompt = "WALK NATURALLY"
                instruction = f"{pose_prompt} - {remaining}s remaining"
            else:
                instruction = f"Move front / back / left / right - {remaining}s remaining"
            cv2.putText(display, instruction, (25, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (255, 255, 255), 2)
            cv2.putText(display, "Automatic finish. Q/ESC or close window to cancel.",
                        (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cancelled = True
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    cancelled = True
                    break
            except cv2.error:
                cancelled = True
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    payload = {
        "session_id": session_id,
        "branch_id": branch_id,
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "saved_samples": saved,
        "rejected_blur": rejected_blur,
        "rejected_duplicate": rejected_duplicate,
        "head_crop_ratio": head_crop_ratio,
        "cancelled": cancelled,
        "logo_references_saved": logo_references_saved,
    }
    metadata_path = _write_session_metadata(root / "_sessions", payload)
    profile_samples = 0
    if activate_profile and saved:
        profile_path = resolve_project_path(
            config.get("staff", {}).get("profile_file", "")
            or f"branches/{branch_id}/uniform_profile.json"
        )
        profile_samples = update_uniform_profile(
            profile_path, label, feature_vectors, session_id
        )
    if saved == 0:
        return 2, f"No usable {label.upper()} samples were saved. Check camera view and lighting."
    state = "cancelled after partial save" if cancelled else "completed"
    return 0, (
        f"{label.upper()} collection {state}: {saved} samples saved in "
        f"{label_dir}. Active profile samples added: {profile_samples}. "
        f"Front-logo references added: {logo_references_saved}. "
        f"Session details: {metadata_path.name}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect controlled uniform dataset samples.")
    parser.add_argument("--label", required=True, choices=sorted(VALID_LABELS))
    parser.add_argument(
        "--activate-profile",
        action="store_true",
        help="also append accepted feature vectors to the active branch profile",
    )
    args = parser.parse_args(argv)
    try:
        code, message = collect(args.label, activate_profile=args.activate_profile)
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"Uniform data collection failed: {exc}")
        return 2
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
