from collections import Counter, deque
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

from staff_enrollment import chest_crop, required_pose_text, save_uniform_references
from staff_face import FaceEvidence, StaffFaceRecognizer, safe_employee_id
from staff_uniform import StaffEvidence, StaffUniformDetector
from uniform_data_collection import (
    blur_score,
    body_crop_without_head,
    sample_signature,
    sufficiently_different,
    update_uniform_profile,
)


class FaceComponentTests(unittest.TestCase):
    def test_employee_id_validation(self):
        self.assertEqual(safe_employee_id(" Ahmed Ali #1 "), "Ahmed_Ali_1")
        with self.assertRaises(ValueError):
            safe_employee_id("@@@")

    def test_pose_classification_boundaries(self):
        face = np.zeros(15, dtype=np.float32)
        face[4], face[6] = 10, 30
        face[8] = 20
        self.assertEqual(StaffFaceRecognizer._pose_from_landmarks(face), "FRONT")
        face[8] = 14
        self.assertEqual(StaffFaceRecognizer._pose_from_landmarks(face), "LEFT")
        face[8] = 27
        self.assertEqual(StaffFaceRecognizer._pose_from_landmarks(face), "RIGHT")

    def test_person_roi_is_clipped_to_frame(self):
        bounds = StaffFaceRecognizer._clip_person_roi((100, 120, 3), [5, 8, 80, 90])
        self.assertTrue(all(value >= 0 for value in bounds))
        self.assertLessEqual(bounds[2], 120)
        self.assertLessEqual(bounds[3], 100)

    def test_corrupt_profile_is_rejected_without_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "faces.npz"
            path.write_bytes(b"not-an-npz")
            recognizer = StaffFaceRecognizer("", "", path, enabled=False)
            recognizer.load_profiles()
            self.assertEqual(len(recognizer.embeddings), 0)
            self.assertIn("Unable to read face profile", recognizer.load_error)

    def test_invalid_embedding_collection_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            recognizer = StaffFaceRecognizer("", "", Path(folder) / "f.npz", enabled=False)
            with self.assertRaises(ValueError):
                recognizer.save_employee("employee", [])

    def test_analyze_selects_highest_similarity_employee(self):
        recognizer = StaffFaceRecognizer("", "", "unused.npz", enabled=False)
        recognizer.detector = object()
        recognizer.recognizer = object()
        recognizer.employee_ids = np.asarray(["A", "B"])
        recognizer.embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        recognizer.extract_embedding = lambda frame, bbox: (
            np.asarray([0.1, 0.995], dtype=np.float32), (1, 2, 3, 4), 0.9, "FRONT"
        )
        evidence = recognizer.analyze(np.zeros((10, 10, 3), dtype=np.uint8), None)
        self.assertEqual(evidence.employee_id, "B")
        self.assertGreater(evidence.score, 0.9)

    def test_track_face_score_aggregates_best_employee(self):
        recognizer = StaffFaceRecognizer("", "", "unused.npz", enabled=False)
        recognizer.min_positive_frames = 2
        track = types.SimpleNamespace(
            face_observations=0,
            face_last_evidence=None,
            face_scores=deque(maxlen=10),
            face_score=0.0,
            face_employee="",
        )
        recognizer.update_track(track, FaceEvidence(0.3, "A", True))
        recognizer.update_track(track, FaceEvidence(0.8, "B", True))
        recognizer.update_track(track, FaceEvidence(0.7, "B", True))
        self.assertEqual(track.face_employee, "B")
        self.assertAlmostEqual(track.face_score, 0.75)


class EnrollmentHelperTests(unittest.TestCase):
    def test_chest_crop_valid_and_too_small(self):
        frame = np.zeros((500, 400, 3), dtype=np.uint8)
        self.assertIsNotNone(chest_crop(frame, (100, 50, 60, 60)))
        self.assertIsNone(chest_crop(frame, (1, 1, 5, 5)))

    def test_pose_prompt_reports_missing_angles(self):
        text = required_pose_text(Counter(FRONT=8, LEFT=0, RIGHT=3))
        self.assertEqual(text, "TURN LEFT")
        self.assertIn("GOOD COVERAGE", required_pose_text(Counter(FRONT=8, LEFT=3, RIGHT=3)))

    def test_uniform_reference_save_accumulates_and_is_capped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "emp_99.jpg"
            old.write_bytes(b"old")
            samples = [np.full((60, 80, 3), index, dtype=np.uint8) for index in range(12)]
            saved = save_uniform_references(root, "emp", samples)
            self.assertEqual(saved, 13)
            self.assertTrue(old.exists())
            self.assertEqual(len(list(root.glob("emp_*.jpg"))), 13)


class UniformDatasetHelperTests(unittest.TestCase):
    def test_body_crop_excludes_head_and_clips_detection(self):
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:50] = 255
        crop = body_crop_without_head(frame, (-10, 0, 110, 200), head_crop_ratio=0.25)
        self.assertEqual(crop.shape, (150, 100, 3))
        self.assertEqual(int(crop.max()), 0)
        self.assertIsNone(body_crop_without_head(frame, (1, 1, 10, 20)))

    def test_blur_and_duplicate_filters(self):
        flat = np.full((180, 90, 3), 127, dtype=np.uint8)
        checker = np.zeros_like(flat)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        self.assertGreater(blur_score(checker), blur_score(flat))
        first = sample_signature(flat)
        changed = sample_signature(checker)
        self.assertFalse(sufficiently_different(first, [first]))
        self.assertTrue(sufficiently_different(changed, [first]))

    def test_repeated_sessions_build_active_branch_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / "uniform_profile.json"
            vectors = [[0.62, 0.68, 0.72] for _ in range(12)]
            self.assertEqual(
                update_uniform_profile(profile, "staff", vectors, "session_1"),
                12,
            )
            detector = StaffUniformDetector(None, profile_path=profile, enabled=True)
            self.assertTrue(detector.profile_ready)
            similarity, confirmed = detector._profile_evidence(0.61, 0.67, 0.70)
            self.assertTrue(confirmed)
            self.assertGreater(similarity, 0.8)

    def test_legacy_three_value_profile_calibrates_white_and_gray_only(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / "uniform_profile.json"
            profile.write_text(
                '{"samples":{"staff":' + str([[0.20, 0.18, 0.0]] * 12).replace("'", '"') + '}}',
                encoding="utf-8",
            )
            detector = StaffUniformDetector(
                None, profile_path=profile, enabled=True,
                white_min_ratio=0.25, gray_min_ratio=0.22,
            )
            white_min, gray_min = detector.effective_min_ratios()
            self.assertLess(white_min, detector.white_min_ratio)
            self.assertLess(gray_min, detector.gray_min_ratio)
            similarity, confirmed = detector._profile_evidence(0.20, 0.18, 0.0)
            self.assertTrue(confirmed)
            self.assertGreater(similarity, 0.8)


class UniformEdgeCaseTests(unittest.TestCase):
    def test_disabled_detector_is_not_ready(self):
        detector = StaffUniformDetector("missing.jpg", enabled=False)
        self.assertFalse(detector.ready)

    def test_missing_logo_reference_uses_clothing_fallback(self):
        detector = StaffUniformDetector("missing.jpg", enabled=True)
        self.assertTrue(detector.ready)
        self.assertFalse(detector.logo_ready)
        self.assertIsNotNone(detector.load_error)

    def test_torso_bounds_stay_inside_frame(self):
        bounds = StaffUniformDetector.torso_bounds((100, 120, 3), [5, 5, 100, 150])
        self.assertTrue(0 <= bounds[0] <= bounds[2] <= 120)
        self.assertTrue(0 <= bounds[1] <= bounds[3] <= 100)

    def test_unusable_evidence_does_not_change_track(self):
        detector = StaffUniformDetector("", enabled=False)
        track = types.SimpleNamespace(staff_status="UNKNOWN")
        self.assertEqual(detector.update_track(track, StaffEvidence()), "UNKNOWN")

    def test_unusable_evidence_is_not_added_to_history(self):
        detector = StaffUniformDetector(None, enabled=True)
        track = types.SimpleNamespace(
            staff_status="UNKNOWN", staff_scores=deque(maxlen=5),
            staff_last_evidence=None, staff_score=0.0,
        )
        detector.update_track(track, StaffEvidence(score=0.0, usable=False))
        self.assertEqual(list(track.staff_scores), [])
        self.assertIsNone(track.staff_last_evidence)


if __name__ == "__main__":
    unittest.main()
