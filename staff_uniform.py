"""Track-level uniform detection with optional enrolled logo references."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StaffEvidence:
    score: float = 0.0
    white_ratio: float = 0.0
    gray_ratio: float = 0.0
    belt_ratio: float = 0.0
    white_score: float = 0.0
    gray_score: float = 0.0
    upper_visible: bool = False
    lower_visible: bool = False
    magenta_ratio: float = 0.0
    feature_matches: int = 0
    feature_inliers: int = 0
    profile_similarity: float = 0.0
    usable: bool = False


class StaffUniformDetector:
    """Recognize the store uniform from clothing layout and an optional logo."""

    def __init__(
        self,
        reference_path,
        enrollment_reference_dir=None,
        profile_path=None,
        enabled=True,
        min_person_width=70,
        min_person_height=130,
        white_min_ratio=0.30,
        gray_min_ratio=0.28,
        belt_min_ratio=0.18,
        white_max_saturation=85,
        white_min_value=170,
        magenta_min_ratio=0.0005,
        min_feature_matches=6,
        min_feature_inliers=4,
        frame_threshold=0.62,
        track_threshold=0.68,
        strong_frame_threshold=0.78,
        min_positive_frames=3,
        min_negative_frames=6,
        min_decision_frames=None,
        belt_bonus=0.05,
        logo_bonus=0.10,
        fast_customer_factor=0.55,
        fast_customer_frames=4,
        history_size=20,
    ):
        self.enabled = bool(enabled)
        self.reference_path = Path(reference_path) if reference_path else None
        self.enrollment_reference_dir = (
            Path(enrollment_reference_dir) if enrollment_reference_dir else None
        )
        self.profile_path = Path(profile_path) if profile_path else None
        self.min_person_width = int(min_person_width)
        self.min_person_height = int(min_person_height)
        self.white_min_ratio = float(white_min_ratio)
        self.gray_min_ratio = float(gray_min_ratio)
        self.belt_min_ratio = float(belt_min_ratio)
        self.white_max_saturation = int(white_max_saturation)
        self.white_min_value = int(white_min_value)
        self.magenta_min_ratio = float(magenta_min_ratio)
        self.min_feature_matches = int(min_feature_matches)
        self.min_feature_inliers = int(min_feature_inliers)
        self.frame_threshold = float(frame_threshold)
        self.track_threshold = float(track_threshold)
        self.strong_frame_threshold = float(strong_frame_threshold)
        self.min_positive_frames = int(min_positive_frames)
        if min_decision_frames is not None:
            min_negative_frames = min_decision_frames
        self.min_negative_frames = max(int(min_negative_frames), self.min_positive_frames)
        # Backward-compatible name used by older integrations/tests.
        self.min_decision_frames = self.min_negative_frames
        self.belt_bonus = float(np.clip(belt_bonus, 0.0, 0.05))
        self.logo_bonus = float(np.clip(logo_bonus, 0.0, 0.10))
        self.fast_customer_factor = float(np.clip(fast_customer_factor, 0.10, 0.90))
        self.fast_customer_frames = max(2, int(fast_customer_frames))
        self.history_size = int(history_size)

        self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=7)
        self.reference_gray = None
        self.reference_keypoints = []
        self.reference_descriptors = None
        self.reference_features = []
        self.profile_staff = np.empty((0, 3), dtype=np.float32)
        self.profile_customer = np.empty((0, 3), dtype=np.float32)
        self.load_error = None

        if self.enabled:
            self._load_reference()
            self._load_profile()

    @property
    def ready(self):
        # Clothing-layout detection works immediately; enrolled references
        # only add the stronger front-logo cue.
        return self.enabled

    @property
    def logo_ready(self):
        return self.enabled and bool(self.reference_features)

    @property
    def profile_ready(self):
        return self.enabled and len(self.profile_staff) >= 10

    def effective_min_ratios(self):
        """Return conservative per-branch thresholds learned from staff samples."""
        if not self.profile_ready:
            return self.white_min_ratio, self.gray_min_ratio
        staff = self.profile_staff[:, :2]
        low = np.percentile(staff, 10, axis=0) * 0.90
        white = float(np.clip(low[0], 0.12, self.white_min_ratio))
        gray = float(np.clip(low[1], 0.10, self.gray_min_ratio))
        return white, gray

    def _load_profile(self):
        if not self.profile_path or not self.profile_path.is_file():
            return
        try:
            payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
            samples = payload.get("samples", {})
            staff = np.asarray(samples.get("staff", []), dtype=np.float32)
            customer = np.asarray(samples.get("customer", []), dtype=np.float32)
            if staff.size:
                staff = staff.reshape(-1, 3)
                staff = staff[np.all(np.isfinite(staff), axis=1)]
                self.profile_staff = np.clip(staff, 0.0, 1.0)
            if customer.size:
                customer = customer.reshape(-1, 3)
                customer = customer[np.all(np.isfinite(customer), axis=1)]
                self.profile_customer = np.clip(customer, 0.0, 1.0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.load_error = f"Unable to read uniform profile: {exc}"

    @staticmethod
    def _robust_distance(vector, samples):
        center = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - center), axis=0) * 1.4826
        scale = np.maximum(mad, np.asarray([0.06, 0.06, 0.08], dtype=np.float32))
        return float(np.mean(np.abs((vector - center) / scale)))

    def _profile_evidence(self, white_ratio, gray_ratio, belt_ratio):
        if not self.profile_ready:
            return 0.0, False
        # Preserve the known clothing structure even if a bad enrollment file
        # is supplied. This prevents a background-only profile from matching.
        if white_ratio < 0.12 or gray_ratio < 0.10:
            return 0.0, False
        # Profiles remain stored as three values for compatibility, but belt is
        # intentionally excluded from the decision distance.
        vector = np.asarray([white_ratio, gray_ratio], dtype=np.float32)
        staff_samples = self.profile_staff[:, :2]
        fixed_scale = np.asarray([0.12, 0.12], dtype=np.float32)
        nearest_staff = float(np.min(np.mean(
            np.abs((staff_samples - vector) / fixed_scale), axis=1
        )))
        center = np.median(staff_samples, axis=0)
        mad = np.median(np.abs(staff_samples - center), axis=0) * 1.4826
        scale = np.maximum(mad, np.asarray([0.06, 0.06], dtype=np.float32))
        robust_distance = float(np.mean(np.abs((vector - center) / scale)))
        staff_distance = min(robust_distance, nearest_staff)
        similarity = float(np.exp(-0.5 * staff_distance))
        confirmed = staff_distance <= 2.0
        if len(self.profile_customer) >= 10:
            customer_samples = self.profile_customer[:, :2]
            nearest_customer = float(np.min(np.mean(
                np.abs((customer_samples - vector) / fixed_scale), axis=1
            )))
            confirmed = confirmed and nearest_staff + 0.30 < nearest_customer
        return similarity, confirmed

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
        paths = []
        if self.reference_path and self.reference_path.is_file():
            paths.append(self.reference_path)
        if self.enrollment_reference_dir and self.enrollment_reference_dir.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                paths.extend(sorted(self.enrollment_reference_dir.glob(pattern)))

        # Bound ORB work when many employees/sessions have been enrolled.
        def modified_time(path):
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        paths = sorted(set(paths), key=modified_time)[-48:]

        for path in paths:
            image = self._read_image(path)
            if image is None:
                continue
            # Keep the original supplied logo tightly cropped.  Enrollment
            # references are already chest-only crops from the branch camera.
            if self.reference_path and path == self.reference_path:
                image = self._crop_nonwhite(image)
            longest = max(image.shape[:2])
            if longest > 900:
                scale = 900.0 / longest
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < self.min_feature_matches:
                continue
            self.reference_features.append((keypoints, descriptors))
            if self.reference_descriptors is None:
                self.reference_gray = gray
                self.reference_keypoints = keypoints
                self.reference_descriptors = descriptors

        if not self.reference_features:
            self.load_error = (
                "No enrolled logo references found; using clothing colours and layout."
            )

    @staticmethod
    def torso_bounds(frame_shape, bbox):
        """Return a clipped upper-body ROI for a center-format person bbox."""
        height, width = frame_shape[:2]
        cx, cy, box_w, box_h = map(float, bbox)
        person_x1 = cx - box_w / 2.0
        person_y1 = cy - box_h / 2.0
        x1 = int(round(person_x1 + 0.18 * box_w))
        x2 = int(round(person_x1 + 0.82 * box_w))
        # Focus on the formal shirt chest: below the face and above the belt.
        # The real logo exists only on the front chest, so back views are left
        # for face/track evidence rather than guessed from shirt colour alone.
        y1 = int(round(person_y1 + 0.20 * box_h))
        y2 = int(round(person_y1 + 0.56 * box_h))
        return (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )

    def _feature_evidence(self, torso_gray):
        keypoints, descriptors = self.orb.detectAndCompute(torso_gray, None)
        if descriptors is None or len(keypoints) < 4 or not self.logo_ready:
            return 0, 0, 0.0

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        best = (0, 0, 0.0)
        for reference_keypoints, reference_descriptors in self.reference_features:
            pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
            good = []
            for pair in pairs:
                if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
                    good.append(pair[0])
            if len(good) < 4:
                if len(good) > best[0]:
                    best = (len(good), 0, 0.0)
                continue

            source = np.float32(
                [reference_keypoints[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            target = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            try:
                _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            except cv2.error:
                mask = None
            inliers = int(mask.sum()) if mask is not None else 0
            match_strength = min(1.0, len(good) / max(1, self.min_feature_matches))
            inlier_strength = min(1.0, inliers / max(1, self.min_feature_inliers))
            candidate = (len(good), inliers, match_strength * inlier_strength)
            if candidate[2] > best[2]:
                best = candidate
        return best

    def analyze(self, frame, bbox):
        if not self.ready:
            return StaffEvidence()
        _, _, box_w, box_h = map(float, bbox)
        if box_w < self.min_person_width or box_h < self.min_person_height:
            return StaffEvidence()

        height, width = frame.shape[:2]
        cx, cy, box_w, box_h = map(float, bbox)
        person_x1, person_y1 = cx - box_w / 2.0, cy - box_h / 2.0

        def relative_crop(left, top, right, bottom, min_width=24, min_height=24):
            intended = (
                person_x1 + left * box_w, person_y1 + top * box_h,
                person_x1 + right * box_w, person_y1 + bottom * box_h,
            )
            ix1, iy1, ix2, iy2 = intended
            intended_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
            rx1, ry1 = max(0, int(round(ix1))), max(0, int(round(iy1)))
            rx2, ry2 = min(width, int(round(ix2))), min(height, int(round(iy2)))
            visible_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
            visible = (
                rx2 - rx1 >= min_width and ry2 - ry1 >= min_height
                and visible_area / intended_area >= 0.85
            )
            if not visible:
                return None, False
            return frame[ry1:ry2, rx1:rx2], True

        torso, upper_visible = relative_crop(0.18, 0.20, 0.82, 0.56)
        trousers, lower_visible = relative_crop(0.25, 0.60, 0.75, 0.92)
        waist, _ = relative_crop(0.18, 0.50, 0.82, 0.64, 20, 10)
        if not upper_visible or not lower_visible:
            return StaffEvidence()
        if torso is None or trousers is None or torso.size == 0 or trousers.size == 0:
            return StaffEvidence()
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)

        neutral_lab = (
            (np.abs(lab[:, :, 1].astype(np.int16) - 128) <= 18)
            & (np.abs(lab[:, :, 2].astype(np.int16) - 128) <= 18)
        )
        white = (
            (hsv[:, :, 1] <= self.white_max_saturation)
            & (hsv[:, :, 2] >= self.white_min_value)
            & neutral_lab
        )
        # Retained for diagnostics/backward-compatible logs only. The real
        # embroidered chest logo is black, so magenta is not decision evidence.
        magenta = (
            (hsv[:, :, 0] >= 140)
            & (hsv[:, :, 0] <= 179)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 70)
        )
        white_ratio = float(np.mean(white))
        magenta_ratio = float(np.mean(magenta))

        gray_ratio = 0.0
        belt_ratio = 0.0
        trousers_hsv = cv2.cvtColor(trousers, cv2.COLOR_BGR2HSV)
        trousers_lab = cv2.cvtColor(trousers, cv2.COLOR_BGR2LAB)
        trousers_neutral = (
            (np.abs(trousers_lab[:, :, 1].astype(np.int16) - 128) <= 20)
            & (np.abs(trousers_lab[:, :, 2].astype(np.int16) - 128) <= 20)
        )
        gray = (
            (trousers_hsv[:, :, 1] <= 80)
            & trousers_neutral
            & (trousers_hsv[:, :, 2] >= 55)
            & (trousers_hsv[:, :, 2] <= 215)
        )
        gray_ratio = float(np.mean(gray))
        if waist is not None and waist.size:
            waist_hsv = cv2.cvtColor(waist, cv2.COLOR_BGR2HSV)
            black = waist_hsv[:, :, 2] <= 85
            # A belt is a horizontal dark run. The best row ratio is more
            # useful than the mean because the waist ROI also includes shirt.
            belt_ratio = float(np.max(np.mean(black, axis=1)))

        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        matches, inliers, feature_score = self._feature_evidence(gray)
        effective_white_min, effective_gray_min = self.effective_min_ratios()
        white_score = float(np.clip(
            white_ratio / max(0.01, effective_white_min * 1.8), 0.0, 1.0
        ))
        gray_score = float(np.clip(
            gray_ratio / max(0.01, effective_gray_min * 1.8), 0.0, 1.0
        ))
        score = 0.58 * white_score + 0.42 * gray_score
        profile_similarity, profile_confirmed = self._profile_evidence(
            white_ratio, gray_ratio, belt_ratio
        )
        clothing_confirmed = (
            white_ratio >= effective_white_min
            and gray_ratio >= effective_gray_min
        )
        if clothing_confirmed:
            score = max(score, self.track_threshold + 0.01)

        if clothing_confirmed and belt_ratio >= self.belt_min_ratio:
            score += self.belt_bonus * min(1.0, belt_ratio / max(0.01, self.belt_min_ratio))

        if profile_confirmed:
            score = max(
                score,
                self.frame_threshold + 0.02,
                self.track_threshold + 0.02,
                profile_similarity,
            )

        if clothing_confirmed and matches >= self.min_feature_matches:
            verified_logo = feature_score if inliers >= self.min_feature_inliers else feature_score * 0.5
            score += self.logo_bonus * verified_logo

        # A belt, logo, or profile never confirms a uniform without both core
        # colour regions. Profile evidence only helps thresholds/diagnostics.
        if not clothing_confirmed:
            score = min(score, self.frame_threshold - 0.01)

        return StaffEvidence(
            score=float(np.clip(score, 0.0, 1.0)),
            white_ratio=white_ratio,
            gray_ratio=gray_ratio,
            belt_ratio=belt_ratio,
            white_score=white_score,
            gray_score=gray_score,
            upper_visible=upper_visible,
            lower_visible=lower_visible,
            magenta_ratio=magenta_ratio,
            feature_matches=matches,
            feature_inliers=inliers,
            profile_similarity=profile_similarity,
            usable=True,
        )

    def update_track(self, track, evidence):
        if getattr(track, "staff_status", "UNKNOWN") == "STAFF":
            return "STAFF"
        if not evidence.usable:
            return track.staff_status
        track.staff_scores.append(float(evidence.score))
        track.staff_last_evidence = evidence
        positives = [score for score in track.staff_scores if score >= self.frame_threshold]
        white_min, gray_min = self.effective_min_ratios()
        clear_nonuniform = (
            evidence.white_ratio < white_min * self.fast_customer_factor
            and evidence.gray_ratio < gray_min * self.fast_customer_factor
        )
        clear_frames = int(getattr(track, "uniform_clear_customer_frames", 0))
        clear_frames = min(self.history_size, clear_frames + 1) if clear_nonuniform else 0
        track.uniform_clear_customer_frames = clear_frames
        track.uniform_clear_customer = clear_frames >= self.fast_customer_frames
        plausible_partial = (
            evidence.white_ratio >= white_min * 0.75
            and evidence.gray_ratio >= gray_min * 0.75
        )
        negative_frames = int(getattr(track, "staff_negative_frames", 0))
        if evidence.score >= self.frame_threshold:
            negative_frames = 0
        elif not plausible_partial:
            negative_frames = min(self.history_size, negative_frames + 1)
        track.staff_negative_frames = negative_frames
        if positives:
            strongest = sorted(positives, reverse=True)[: self.min_positive_frames]
            track.staff_score = float(np.mean(strongest))
        else:
            track.staff_score = 0.0

        strong = evidence.score >= self.strong_frame_threshold
        if strong or (
            len(positives) >= self.min_positive_frames
            and track.staff_score >= self.track_threshold
        ):
            track.uniform_clear_customer = False
            track.staff_status = "STAFF"
        elif track.uniform_clear_customer:
            # Both required colours are clearly absent in a fully visible ROI.
            # A short consecutive-frame confirmation avoids one-frame noise.
            track.staff_status = "CUSTOMER"
        elif negative_frames >= self.min_negative_frames and not positives:
            track.staff_status = "CUSTOMER"
        else:
            track.staff_status = "UNKNOWN"
        return track.staff_status

    def has_partial_staff_evidence(self, track):
        """Conservative timeout bias: do not count plausible staff as customers."""
        scores = list(getattr(track, "staff_scores", ()))
        if any(score >= self.frame_threshold for score in scores):
            return True
        evidence = getattr(track, "staff_last_evidence", None)
        if evidence is None or not evidence.usable:
            return False
        white_min, gray_min = self.effective_min_ratios()
        return (
            evidence.white_ratio >= white_min * 0.75
            and evidence.gray_ratio >= gray_min * 0.75
        )
