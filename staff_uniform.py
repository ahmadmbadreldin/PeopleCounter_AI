"""Conservative, training-free staff-uniform detection.

The detector intentionally requires both a mostly-white torso and geometric
feature matches to the supplied company logo.  A single frame never decides a
track: callers aggregate evidence over several observed frames.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StaffEvidence:
    score: float = 0.0
    white_ratio: float = 0.0
    magenta_ratio: float = 0.0
    feature_matches: int = 0
    feature_inliers: int = 0
    usable: bool = False


class StaffUniformDetector:
    """Match a logo-bearing white uniform inside a person's torso crop.

    This is a demo fallback for installations that have no real camera
    training data.  It is deliberately biased toward false negatives: weak or
    unclear evidence remains CUSTOMER/UNKNOWN and is never excluded.
    """

    def __init__(
        self,
        reference_path,
        enabled=True,
        min_person_width=70,
        min_person_height=130,
        white_min_ratio=0.30,
        white_max_saturation=85,
        white_min_value=135,
        magenta_min_ratio=0.0005,
        min_feature_matches=6,
        min_feature_inliers=4,
        frame_threshold=0.62,
        track_threshold=0.68,
        min_positive_frames=3,
        history_size=20,
    ):
        self.enabled = bool(enabled)
        self.reference_path = Path(reference_path)
        self.min_person_width = int(min_person_width)
        self.min_person_height = int(min_person_height)
        self.white_min_ratio = float(white_min_ratio)
        self.white_max_saturation = int(white_max_saturation)
        self.white_min_value = int(white_min_value)
        self.magenta_min_ratio = float(magenta_min_ratio)
        self.min_feature_matches = int(min_feature_matches)
        self.min_feature_inliers = int(min_feature_inliers)
        self.frame_threshold = float(frame_threshold)
        self.track_threshold = float(track_threshold)
        self.min_positive_frames = int(min_positive_frames)
        self.history_size = int(history_size)

        self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=7)
        self.reference_gray = None
        self.reference_keypoints = []
        self.reference_descriptors = None
        self.load_error = None

        if self.enabled:
            self._load_reference()

    @property
    def ready(self):
        return (
            self.enabled
            and self.reference_descriptors is not None
            and len(self.reference_keypoints) >= self.min_feature_matches
        )

    @staticmethod
    def _crop_nonwhite(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        content = (hsv[:, :, 1] > 35) | (hsv[:, :, 2] < 225)
        ys, xs = np.where(content)
        if not len(xs):
            return image
        pad = max(4, int(0.03 * max(image.shape[:2])))
        x1, x2 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad + 1)
        y1, y2 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad + 1)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _read_image(path):
        """Read an image reliably when its Windows path contains Unicode."""
        try:
            encoded = np.fromfile(str(path), dtype=np.uint8)
        except OSError:
            return None
        if encoded.size == 0:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _load_reference(self):
        image = self._read_image(self.reference_path)
        if image is None:
            self.load_error = f"Unable to read staff logo reference: {self.reference_path}"
            self.enabled = False
            return
        image = self._crop_nonwhite(image)
        longest = max(image.shape[:2])
        if longest > 900:
            scale = 900.0 / longest
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.reference_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(
            self.reference_gray, None
        )
        if self.reference_descriptors is None:
            self.load_error = f"No usable features found in staff logo: {self.reference_path}"
            self.enabled = False

    @staticmethod
    def torso_bounds(frame_shape, bbox):
        """Return a clipped upper-body ROI for a center-format person bbox."""
        height, width = frame_shape[:2]
        cx, cy, box_w, box_h = map(float, bbox)
        person_x1 = cx - box_w / 2.0
        person_y1 = cy - box_h / 2.0
        x1 = int(round(person_x1 + 0.08 * box_w))
        x2 = int(round(person_x1 + 0.92 * box_w))
        y1 = int(round(person_y1 + 0.10 * box_h))
        y2 = int(round(person_y1 + 0.68 * box_h))
        return (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )

    def _feature_evidence(self, torso_gray):
        keypoints, descriptors = self.orb.detectAndCompute(torso_gray, None)
        if descriptors is None or len(keypoints) < 4 or not self.ready:
            return 0, 0, 0.0

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
        good = []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 4:
            return len(good), 0, 0.0

        source = np.float32(
            [self.reference_keypoints[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        target = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        except cv2.error:
            mask = None
        inliers = int(mask.sum()) if mask is not None else 0
        match_strength = min(1.0, len(good) / max(1, self.min_feature_matches))
        inlier_strength = min(1.0, inliers / max(1, self.min_feature_inliers))
        return len(good), inliers, match_strength * inlier_strength

    def analyze(self, frame, bbox):
        if not self.ready:
            return StaffEvidence()
        _, _, box_w, box_h = map(float, bbox)
        if box_w < self.min_person_width or box_h < self.min_person_height:
            return StaffEvidence()

        x1, y1, x2, y2 = self.torso_bounds(frame.shape, bbox)
        if x2 - x1 < 30 or y2 - y1 < 30:
            return StaffEvidence()
        torso = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        white = (
            (hsv[:, :, 1] <= self.white_max_saturation)
            & (hsv[:, :, 2] >= self.white_min_value)
        )
        # OpenCV hue for the supplied pink/magenta accent is near 160.
        magenta = (
            (hsv[:, :, 0] >= 140)
            & (hsv[:, :, 0] <= 179)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 70)
        )
        white_ratio = float(np.mean(white))
        magenta_ratio = float(np.mean(magenta))

        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        matches, inliers, feature_score = self._feature_evidence(gray)
        if white_ratio < self.white_min_ratio or matches < self.min_feature_matches:
            score = 0.0
        else:
            white_score = float(np.clip(
                (white_ratio - self.white_min_ratio) / max(0.01, 0.75 - self.white_min_ratio),
                0.0,
                1.0,
            ))
            magenta_score = float(np.clip(
                magenta_ratio / max(self.magenta_min_ratio, 1e-6), 0.0, 1.0
            ))
            score = 0.30 * white_score + 0.50 * feature_score + 0.20 * magenta_score
            if inliers < self.min_feature_inliers:
                score *= 0.5

        return StaffEvidence(
            score=float(np.clip(score, 0.0, 1.0)),
            white_ratio=white_ratio,
            magenta_ratio=magenta_ratio,
            feature_matches=matches,
            feature_inliers=inliers,
            usable=True,
        )

    def update_track(self, track, evidence):
        if not evidence.usable:
            return track.staff_status
        track.staff_scores.append(float(evidence.score))
        track.staff_last_evidence = evidence
        positives = [score for score in track.staff_scores if score >= self.frame_threshold]
        if positives:
            strongest = sorted(positives, reverse=True)[: self.min_positive_frames]
            track.staff_score = float(np.mean(strongest))
        else:
            track.staff_score = 0.0

        if (
            len(positives) >= self.min_positive_frames
            and track.staff_score >= self.track_threshold
        ):
            track.staff_status = "STAFF"
        elif len(track.staff_scores) >= self.min_positive_frames:
            track.staff_status = "CUSTOMER"
        else:
            track.staff_status = "UNKNOWN"
        return track.staff_status
