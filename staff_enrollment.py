"""Interactive staff enrollment using the installed branch camera."""

import argparse
from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np

from branch_config import build_rtsp_url_from_config, load_branch_config, resolve_project_path
from staff_face import StaffFaceRecognizer, safe_employee_id


TARGET_SAMPLES = 24
MIN_SAVE_SAMPLES = 10


def chest_crop(frame, face_box):
    x, y, width, height = face_box
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, int(x - 0.75 * width))
    x2 = min(frame_w, int(x + 1.75 * width))
    y1 = max(0, int(y + 0.95 * height))
    y2 = min(frame_h, int(y + 4.2 * height))
    if x2 - x1 < 45 or y2 - y1 < 55:
        return None
    return frame[y1:y2, x1:x2].copy()


def save_uniform_references(directory, employee_id, samples):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob(f"{employee_id}_*.jpg"):
        old.unlink(missing_ok=True)
    saved = 0
    for index, image in enumerate(samples[:9], start=1):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            continue
        (directory / f"{employee_id}_{index:02d}.jpg").write_bytes(encoded.tobytes())
        saved += 1
    return saved


def open_camera(url, config):
    camera = config["camera"]
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(camera.get("open_timeout_ms", 5000)))
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(camera.get("read_timeout_ms", 2000)))
    if not cap.open(url, cv2.CAP_FFMPEG):
        cap.open(url)
    return cap


def required_pose_text(counts):
    needs = []
    if counts["FRONT"] < 8:
        needs.append("LOOK FRONT")
    if counts["LEFT"] < 3:
        needs.append("TURN LEFT")
    if counts["RIGHT"] < 3:
        needs.append("TURN RIGHT")
    return " | ".join(needs) if needs else "GOOD COVERAGE - KEEP MOVING NATURALLY"


def enroll(employee_id):
    employee_id = safe_employee_id(employee_id)
    config = load_branch_config()
    face_cfg = config.get("face", {})
    recognizer = StaffFaceRecognizer(
        detector_model=resolve_project_path(face_cfg.get("detector_model", "")),
        recognizer_model=resolve_project_path(face_cfg.get("recognizer_model", "")),
        profile_file=resolve_project_path(face_cfg.get("profile_file", "")),
        enabled=True,
        match_threshold=face_cfg.get("match_threshold", 0.36),
        support_threshold=face_cfg.get("support_threshold", 0.29),
        min_face_size=face_cfg.get("min_face_size", 38),
        min_positive_frames=face_cfg.get("min_positive_frames", 2),
    )
    if not recognizer.models_ready:
        raise RuntimeError(recognizer.load_error or "Face models are unavailable.")

    cap = open_camera(build_rtsp_url_from_config(config), config)
    if not cap.isOpened():
        raise RuntimeError("Could not open the branch camera for staff enrollment.")

    window = "Staff Enrollment - SPACE finish / Q cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1100, 700)
    embeddings = []
    uniform_samples = []
    pose_counts = Counter()
    last_capture = 0.0
    started = time.monotonic()
    cancelled = False

    try:
        while time.monotonic() - started < 120:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            embedding, face_box, confidence, pose = recognizer.extract_embedding(frame)
            now = time.monotonic()
            accepted = False
            if embedding is not None and now - last_capture >= 0.28:
                diverse = not embeddings or max(float(old @ embedding) for old in embeddings) < 0.9985
                if diverse:
                    embeddings.append(embedding)
                    pose_counts[pose] += 1
                    last_capture = now
                    accepted = True
                    crop = chest_crop(frame, face_box)
                    if crop is not None and len(uniform_samples) < 12:
                        uniform_samples.append(crop)

            if face_box:
                x, y, width, height = face_box
                color = (0, 255, 0) if accepted else (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
                cv2.putText(frame, f"{pose}  quality:{confidence:.2f}",
                            (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

            cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 820), 150), (0, 0, 0), -1)
            cv2.putText(frame, f"EMPLOYEE: {employee_id}", (25, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"SAMPLES: {len(embeddings)}/{TARGET_SAMPLES}   "
                        f"FRONT:{pose_counts['FRONT']} LEFT:{pose_counts['LEFT']} "
                        f"RIGHT:{pose_counts['RIGHT']}", (25, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, required_pose_text(pose_counts), (25, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: save when ready   Q: cancel", (25, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            enough_poses = sum(pose_counts[p] > 0 for p in ("FRONT", "LEFT", "RIGHT")) >= 2
            if len(embeddings) >= TARGET_SAMPLES and enough_poses:
                break
            if key in (ord("q"), 27):
                cancelled = True
                break
            if key == 32 and len(embeddings) >= MIN_SAVE_SAMPLES:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if cancelled:
        return 1, "Enrollment cancelled."
    if len(embeddings) < MIN_SAVE_SAMPLES:
        return 2, f"Only {len(embeddings)} usable samples; at least {MIN_SAVE_SAMPLES} are required."

    saved_faces = recognizer.save_employee(employee_id, embeddings)
    uniform_dir = resolve_project_path(
        config.get("staff", {}).get("enrollment_reference_dir", "")
    )
    saved_uniforms = save_uniform_references(uniform_dir, employee_id, uniform_samples)
    return 0, (
        f"Saved {saved_faces} face samples and {saved_uniforms} uniform references "
        f"for {employee_id}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--employee-id", required=True)
    args = parser.parse_args()
    try:
        code, message = enroll(args.employee_id)
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"Enrollment failed: {exc}")
        return 2
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

