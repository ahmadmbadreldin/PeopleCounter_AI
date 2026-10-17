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

    def test_uniform_reference_save_is_atomic_in_scope_and_capped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "emp_99.jpg"
            old.write_bytes(b"old")
            samples = [np.full((60, 80, 3), index, dtype=np.uint8) for index in range(12)]
            saved = save_uniform_references(root, "emp", samples)
            self.assertEqual(saved, 9)
            self.assertFalse(old.exists())
            self.assertEqual(len(list(root.glob("emp_*.jpg"))), 9)


class UniformEdgeCaseTests(unittest.TestCase):
    def test_disabled_or_missing_reference_is_not_ready(self):
        detector = StaffUniformDetector("missing.jpg", enabled=False)
        self.assertFalse(detector.ready)
        detector = StaffUniformDetector("missing.jpg", enabled=True)
        self.assertFalse(detector.ready)
        self.assertIsNotNone(detector.load_error)

    def test_torso_bounds_stay_inside_frame(self):
        bounds = StaffUniformDetector.torso_bounds((100, 120, 3), [5, 5, 100, 150])
        self.assertTrue(0 <= bounds[0] <= bounds[2] <= 120)
        self.assertTrue(0 <= bounds[1] <= bounds[3] <= 100)

    def test_unusable_evidence_does_not_change_track(self):
        detector = StaffUniformDetector("", enabled=False)
        track = types.SimpleNamespace(staff_status="UNKNOWN")
        self.assertEqual(detector.update_track(track, StaffEvidence()), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
