"""Local, per-branch staff face enrollment and recognition.

Only normalized SFace embeddings and employee codes are persisted.  Raw face
images are neither required nor stored by this module.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceEvidence:
    score: float = 0.0
    employee_id: str = ""
    usable: bool = False
    face_box: tuple = ()
    pose: str = "UNKNOWN"


def safe_employee_id(value):
    value = re.sub(r"[^\w-]+", "_", str(value).strip(), flags=re.UNICODE).strip("_")
    if not value:
        raise ValueError("Employee name/code is required.")
    return value[:64]


class StaffFaceRecognizer:
    """OpenCV YuNet + SFace recognizer backed by a branch-local profile."""

    def __init__(
        self,
        detector_model,
        recognizer_model,
        profile_file,
        enabled=True,
        match_threshold=0.36,
        support_threshold=0.29,
        min_face_size=38,
        min_positive_frames=2,
    ):
        self.enabled = bool(enabled)
        self.detector_model = Path(detector_model)
        self.recognizer_model = Path(recognizer_model)
        self.profile_file = Path(profile_file)
        self.match_threshold = float(match_threshold)
        self.support_threshold = float(support_threshold)
        self.min_face_size = int(min_face_size)
        self.min_positive_frames = int(min_positive_frames)
        self.detector = None
        self.recognizer = None
        self.employee_ids = np.empty((0,), dtype=str)
        self.embeddings = np.empty((0, 128), dtype=np.float32)
        self.load_error = None

        if self.enabled:
            self._load_models()
            self.load_profiles()

    @property
    def models_ready(self):
        return self.detector is not None and self.recognizer is not None

    @property
    def ready(self):
        return self.models_ready and len(self.embeddings) > 0

    @property
    def enrolled_employee_count(self):
        return len(set(self.employee_ids.tolist()))

    def _load_models(self):
        missing = [
            str(path)
            for path in (self.detector_model, self.recognizer_model)
            if not path.exists()
        ]
        if missing:
            self.load_error = "Missing face model(s): " + ", ".join(missing)
            return
        try:
            detector_path = self._opencv_model_path(self.detector_model)
            recognizer_path = self._opencv_model_path(self.recognizer_model)
            self.detector = cv2.FaceDetectorYN_create(
                detector_path, "", (320, 320), 0.78, 0.3, 100
            )
            self.recognizer = cv2.FaceRecognizerSF_create(
                recognizer_path, ""
            )
            # Force lazy ONNX parsing now, so setup reports model/path errors
            # immediately instead of silently collecting zero samples.
            self.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            warm_feature = self.recognizer.feature(
                np.zeros((112, 112, 3), dtype=np.uint8)
            )
            if warm_feature is None or warm_feature.size == 0:
                raise RuntimeError("SFace warm-up returned no embedding.")
        except cv2.error as exc:
            self.detector = None
            self.recognizer = None
            self.load_error = f"Unable to load face models: {exc}"
        except RuntimeError as exc:
            self.detector = None
            self.recognizer = None
            self.load_error = str(exc)

    @staticmethod
    def _opencv_model_path(path):
        """Prefer an ASCII relative path for OpenCV DNN on Windows.

        Python can open Unicode paths, but some OpenCV ONNX loaders cannot.
        The project models live below the process working directory, so the
        relative form avoids passing the Arabic project path into C++.
        """
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(Path.cwd().resolve())
        except ValueError:
            return str(resolved)
        candidate = str(relative)
        return candidate if candidate.isascii() else str(resolved)

    def load_profiles(self):
        self.employee_ids = np.empty((0,), dtype=str)
        self.embeddings = np.empty((0, 128), dtype=np.float32)
        if not self.profile_file.exists():
            return
        try:
            with np.load(self.profile_file, allow_pickle=False) as payload:
                ids = payload["employee_ids"].astype(str)
                embeddings = payload["embeddings"].astype(np.float32)
            if embeddings.ndim != 2 or len(ids) != len(embeddings):
                raise ValueError("profile dimensions do not match")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.employee_ids = ids
            self.embeddings = embeddings / np.maximum(norms, 1e-8)
        except (OSError, ValueError, KeyError) as exc:
            self.load_error = f"Unable to read face profile: {exc}"

    @staticmethod
    def _clip_person_roi(frame_shape, bbox):
        height, width = frame_shape[:2]
        cx, cy, box_w, box_h = map(float, bbox)
        x1 = max(0, int(cx - box_w * 0.55))
        x2 = min(width, int(cx + box_w * 0.55))
        y1 = max(0, int(cy - box_h * 0.55))
        y2 = min(height, int(cy + box_h * 0.12))
        return x1, y1, x2, y2

    @staticmethod
    def _pose_from_landmarks(face):
        # YuNet: bbox, right eye, left eye, nose, mouth corners, score.
        eye_left_x, eye_right_x = float(face[4]), float(face[6])
        nose_x = float(face[8])
        lo, hi = sorted((eye_left_x, eye_right_x))
        span = max(1.0, hi - lo)
        ratio = (nose_x - lo) / span
        if ratio < 0.38:
            return "LEFT"
        if ratio > 0.62:
            return "RIGHT"
        return "FRONT"

    def extract_embedding(self, frame, bbox=None):
        if not self.models_ready or frame is None or frame.size == 0:
            return None, (), 0.0, "UNKNOWN"
        if bbox is None:
            height, width = frame.shape[:2]
            bounds = (0, 0, width, height)
        else:
            bounds = self._clip_person_roi(frame.shape, bbox)
        x1, y1, x2, y2 = bounds
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < self.min_face_size or roi.shape[1] < self.min_face_size:
            return None, (), 0.0, "UNKNOWN"

        try:
            self.detector.setInputSize((roi.shape[1], roi.shape[0]))
            _, faces = self.detector.detect(roi)
        except cv2.error:
            return None, (), 0.0, "UNKNOWN"
        if faces is None or len(faces) == 0:
            return None, (), 0.0, "UNKNOWN"

        usable = [
            face for face in faces
            if min(float(face[2]), float(face[3])) >= self.min_face_size
        ]
        if not usable:
            return None, (), 0.0, "UNKNOWN"
        face = max(usable, key=lambda item: float(item[2] * item[3] * item[-1]))
        try:
            aligned = self.recognizer.alignCrop(roi, face)
            feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        except cv2.error:
            return None, (), 0.0, "UNKNOWN"
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm <= 1e-8:
            return None, (), 0.0, "UNKNOWN"
        feature /= norm
        face_box = (
            int(x1 + face[0]),
            int(y1 + face[1]),
            int(face[2]),
            int(face[3]),
        )
        return feature, face_box, float(face[-1]), self._pose_from_landmarks(face)

    def analyze(self, frame, bbox):
        embedding, face_box, _, pose = self.extract_embedding(frame, bbox)
        if embedding is None:
            return FaceEvidence()
        if not self.ready:
            return FaceEvidence(usable=True, face_box=face_box, pose=pose)
        scores = self.embeddings @ embedding
        index = int(np.argmax(scores))
        return FaceEvidence(
            score=float(scores[index]),
            employee_id=str(self.employee_ids[index]),
            usable=True,
            face_box=face_box,
            pose=pose,
        )

    def update_track(self, track, evidence):
        if not evidence.usable:
            return
        track.face_observations += 1
        track.face_last_evidence = evidence
        if evidence.employee_id:
            track.face_scores.append((float(evidence.score), evidence.employee_id))
        if not track.face_scores:
            return
        best_by_employee = {}
        for score, employee_id in track.face_scores:
            best_by_employee.setdefault(employee_id, []).append(score)
        employee_id, scores = max(
            best_by_employee.items(), key=lambda item: max(item[1])
        )
        strongest = sorted(scores, reverse=True)[: max(1, self.min_positive_frames)]
        track.face_score = float(np.mean(strongest))
        track.face_employee = employee_id

    def save_employee(self, employee_id, embeddings):
        employee_id = safe_employee_id(employee_id)
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) < 1:
            raise ValueError("No usable face samples were collected.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-8)

        keep_other = self.employee_ids != employee_id
        old_ids = self.employee_ids[keep_other]
        old_vectors = self.embeddings[keep_other]
        employee_vectors = [
            vector.copy() for vector in self.embeddings[self.employee_ids == employee_id]
        ]
        for vector in vectors:
            if not employee_vectors or max(
                float(existing @ vector) for existing in employee_vectors
            ) < 0.9995:
                employee_vectors.append(vector.copy())
        # Multiple sessions improve lighting/angle coverage without allowing
        # one employee profile to grow forever.
        employee_vectors = employee_vectors[-80:]
        employee_array = np.asarray(employee_vectors, dtype=np.float32)
        new_ids = np.asarray([employee_id] * len(employee_array), dtype=str)
        all_ids = np.concatenate((old_ids, new_ids))
        all_vectors = (
            np.vstack((old_vectors, employee_array))
            if len(old_vectors) else employee_array
        )

        self.profile_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.profile_file.with_suffix(self.profile_file.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                employee_ids=all_ids,
                embeddings=all_vectors.astype(np.float32),
                format_version=np.asarray([1], dtype=np.int16),
            )
        os.replace(temporary, self.profile_file)
        self.load_profiles()
        return len(employee_array)

