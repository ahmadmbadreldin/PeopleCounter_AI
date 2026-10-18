"""Offline regression tests for crossing state, no camera or model required."""
import importlib.util
from copy import deepcopy
import sys
import tempfile
import types
import unittest
import numpy as np
import cv2

# Lightweight test stand-ins for optional runtime dependencies, not used in production.
class FakeKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.x = np.zeros((dim_x, 1))
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
    def predict(self):
        self.x = self.F @ self.x
    def update(self, z):
        self.x[:4] = z

try:
    import filterpy.kalman
except ImportError:
    filterpy = types.ModuleType('filterpy')
    kalman = types.ModuleType('filterpy.kalman')
    kalman.KalmanFilter = FakeKalmanFilter
    filterpy.kalman = kalman
    sys.modules['filterpy'] = filterpy
    sys.modules['filterpy.kalman'] = kalman
# The unit tests exercise tracking/state logic and never load a YOLO model.
# Always use a stand-in so tests remain offline, fast, and side-effect free.
ultra = types.ModuleType('ultralytics')
ultra.YOLO = lambda *args, **kwargs: None
sys.modules['ultralytics'] = ultra

from pathlib import Path
spec = importlib.util.spec_from_file_location('counter', Path(__file__).with_name('people_counter_hikvision.py'))
counter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = counter
spec.loader.exec_module(counter)

from staff_uniform import StaffEvidence, StaffUniformDetector
from staff_face import StaffFaceRecognizer
from branch_config import (
    DEFAULT_CONFIG,
    build_rtsp_url_from_config,
    load_branch_config,
    save_branch_config,
    validate_config,
)

class TrackingTests(unittest.TestCase):
    def new_track(self, foot_y=50):
        return counter.Track([200, foot_y - 50, 60, 100], 4)

    def observed(self, track, foot_y):
        track.predict()
        track.update([200, foot_y - 50, 60, 100])
        self.tick = getattr(self, "tick", 1000) + 1
        return track.check_crossing(100, 10, self.tick, 0.75)

    def test_red_count_point_is_bottom_center_of_detection(self):
        track = counter.Track([220, 140, 80, 180], 9)
        self.assertEqual(track.get_count_point(), (220.0, 230.0))

    def test_normal_entry_and_exit(self):
        t = self.new_track()
        self.assertIsNone(t.check_crossing(100, 10, 1000, 0.75))
        self.observed(t, 55)
        self.assertIsNone(self.observed(t, 125))
        self.assertEqual(self.observed(t, 130), 'IN')
        self.assertIsNone(self.observed(t, 60))
        self.assertEqual(self.observed(t, 55), 'OUT')

    def test_reversed_entrance_direction(self):
        t = self.new_track()
        t.check_crossing(100, 10, 1000, 0.75, 'bottom_to_top')
        t.predict()
        t.update([200, 5, 60, 100])
        t.check_crossing(100, 10, 1001, 0.75, 'bottom_to_top')
        t.predict()
        t.update([200, 75, 60, 100])
        self.assertIsNone(t.check_crossing(100, 10, 1002, 0.75, 'bottom_to_top'))
        t.predict()
        t.update([200, 80, 60, 100])
        self.assertEqual(
            t.check_crossing(100, 10, 1003, 0.75, 'bottom_to_top'),
            'OUT',
        )

    def test_predicted_position_never_counts(self):
        t = self.new_track()
        t.check_crossing(100, 10, 1, .75)
        self.observed(t, 55)
        t.predict()
        t.kf.x[1] = 140
        self.assertIsNone(t.check_crossing(100, 10, 2, .75))
        self.assertEqual(t.count_side, 'ABOVE')

    def test_reappearance_does_not_count_unobserved_crossing(self):
        t = self.new_track()
        t.check_crossing(100, 10, 1, .75)
        self.observed(t, 55)
        t.predict()
        t.predict()
        t.update([200, 80, 60, 100])
        self.assertIsNone(t.check_crossing(100, 10, 10, .75))
        self.assertIsNone(self.observed(t, 135))
        self.assertEqual(t.count_side, 'BELOW')

    def test_expired_tracks_not_returned_as_observations(self):
        tracker = counter.IoUTracker(min_hits=1)
        tracker.update([[200, 50, 60, 100]])
        self.assertEqual(len(tracker.get_tracks()), 1)
        tracker.update([])
        self.assertEqual(len(tracker.get_tracks()), 0)

    def test_hysteresis_no_jitter_count(self):
        t = self.new_track()
        t.check_crossing(100, 10, 1, .75)
        self.observed(t, 50)
        for y in (95, 103, 106, 99, 91, 98):
            self.assertIsNone(self.observed(t, y))
        self.assertEqual(t.count_side, 'ABOVE')


class StaffUniformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_directory = tempfile.TemporaryDirectory()
        reference = Path(cls.reference_directory.name) / 'logo.png'
        logo = np.full((180, 180, 3), 255, dtype=np.uint8)
        cv2.circle(logo, (90, 90), 58, (0, 0, 0), 8)
        cv2.line(logo, (91, 28), (145, 58), (0, 0, 0), 7)
        cv2.line(logo, (91, 152), (146, 122), (0, 0, 0), 7)
        cv2.putText(logo, 'C', (48, 125), cv2.FONT_HERSHEY_SIMPLEX,
                    2.6, (0, 0, 0), 6)
        cv2.imwrite(str(reference), logo)
        cls.detector = StaffUniformDetector(
            reference,
            min_person_width=50,
            min_person_height=100,
            min_feature_matches=5,
            min_feature_inliers=4,
            min_positive_frames=3,
        )

    @classmethod
    def tearDownClass(cls):
        cls.reference_directory.cleanup()

    def test_blank_white_shirt_is_not_staff(self):
        frame = np.full((500, 300, 3), 180, dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (250, 450), (245, 245, 245), -1)
        evidence = self.detector.analyze(frame, [150, 250, 200, 400])
        self.assertTrue(evidence.usable)
        self.assertEqual(evidence.score, 0.0)

    def test_reference_logo_on_white_shirt_is_strong_evidence(self):
        reference = self.detector._read_image(self.detector.reference_path)
        logo = self.detector._crop_nonwhite(reference)
        logo = cv2.resize(logo, (140, 140), interpolation=cv2.INTER_AREA)
        frame = np.full((500, 300, 3), 180, dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (250, 450), (245, 245, 245), -1)
        frame[120:260, 80:220] = logo
        evidence = self.detector.analyze(frame, [150, 250, 200, 400])
        self.assertGreaterEqual(evidence.feature_inliers, 4)
        self.assertGreaterEqual(evidence.score, self.detector.frame_threshold)

    def test_repeated_strong_evidence_marks_track_as_staff(self):
        track = counter.Track([150, 250, 200, 400], 99)
        evidence = StaffEvidence(score=0.90, usable=True)
        self.detector.update_track(track, evidence)
        self.detector.update_track(track, evidence)
        self.assertEqual(track.staff_status, 'UNKNOWN')
        self.detector.update_track(track, evidence)
        self.assertEqual(track.staff_status, 'STAFF')

    def test_weak_evidence_defaults_to_customer(self):
        track = counter.Track([150, 250, 200, 400], 100)
        evidence = StaffEvidence(score=0.10, usable=True)
        for _ in range(3):
            self.detector.update_track(track, evidence)
        self.assertEqual(track.staff_status, 'CUSTOMER')


class CrossingDecisionTests(unittest.TestCase):
    def new_counter(self):
        store = counter.StoreCounter.__new__(counter.StoreCounter)
        store.cfg = types.SimpleNamespace(staff_decision_timeout_s=1.25)
        store.staff_detector = types.SimpleNamespace(ready=True)
        store.pending_crossings = []
        store.staff_ignored = 0
        store.unknown_crossings = 0
        store.total_in = 0
        store.total_out = 0
        store.inside_ids = set()
        store.inside_timestamps = {}
        store.logged = []
        store._log_event = lambda track_id, event, play_alert=True: store.logged.append(event)
        return store

    def test_unknown_crossing_waits_then_counts_as_customer(self):
        store = self.new_counter()
        track = counter.Track([150, 50, 100, 200], 7)
        store._handle_crossing(track, 'IN', 10.0)
        self.assertEqual(store.total_in, 0)
        store._resolve_pending_crossings(11.0)
        self.assertEqual(store.total_in, 0)
        store._resolve_pending_crossings(11.3)
        self.assertEqual(store.total_in, 1)
        self.assertEqual(store.unknown_crossings, 1)
        self.assertIn('IN_UNKNOWN_COUNTED', store.logged)

    def test_confirmed_staff_crossing_is_ignored(self):
        store = self.new_counter()
        track = counter.Track([150, 50, 100, 200], 8)
        track.staff_status = 'STAFF'
        store._handle_crossing(track, 'IN', 10.0)
        self.assertEqual(store.total_in, 0)
        self.assertEqual(store.staff_ignored, 1)
        self.assertIn('STAFF_IN_IGNORED', store.logged)


class HybridStaffDecisionTests(unittest.TestCase):
    def new_counter(self):
        store = counter.StoreCounter.__new__(counter.StoreCounter)
        store.cfg = types.SimpleNamespace(
            face_match_threshold=0.36,
            face_support_threshold=0.29,
            face_min_positive_frames=2,
            staff_white_min_ratio=0.30,
        )
        store.face_recognizer = types.SimpleNamespace(ready=True)
        return store

    def test_repeated_face_match_marks_staff(self):
        store = self.new_counter()
        track = counter.Track([150, 250, 200, 400], 10)
        track.uniform_status = 'CUSTOMER'
        track.face_employee = 'employee_01'
        track.face_scores.extend([(0.40, 'employee_01'), (0.41, 'employee_01')])
        track.face_score = 0.405
        track.face_observations = 2
        self.assertEqual(store._update_hybrid_staff_status(track), 'STAFF')

    def test_supporting_face_plus_white_uniform_marks_staff(self):
        store = self.new_counter()
        track = counter.Track([150, 250, 200, 400], 11)
        track.uniform_status = 'CUSTOMER'
        track.face_employee = 'employee_01'
        track.face_scores.extend([(0.31, 'employee_01'), (0.32, 'employee_01')])
        track.face_score = 0.315
        track.face_observations = 2
        track.staff_last_evidence = StaffEvidence(
            score=0.1, white_ratio=0.60, usable=True
        )
        self.assertEqual(store._update_hybrid_staff_status(track), 'STAFF')

    def test_repeated_negative_face_and_uniform_marks_customer(self):
        store = self.new_counter()
        track = counter.Track([150, 250, 200, 400], 12)
        track.uniform_status = 'CUSTOMER'
        track.face_employee = 'employee_01'
        track.face_scores.extend([(0.10, 'employee_01'), (0.12, 'employee_01')])
        track.face_score = 0.11
        track.face_observations = 3
        self.assertEqual(store._update_hybrid_staff_status(track), 'CUSTOMER')


class BranchConfigTests(unittest.TestCase):
    def valid_config(self):
        config = deepcopy(DEFAULT_CONFIG)
        config['branch_id'] = 'branch_01'
        config['camera']['password'] = 'p@ss word'
        config['counting']['calibrated'] = True
        return config

    def test_valid_config_and_encoded_rtsp_url(self):
        config = self.valid_config()
        self.assertEqual(validate_config(config), 'branch_01')
        url = build_rtsp_url_from_config(config)
        self.assertIn('p%40ss%20word', url)
        self.assertTrue(url.endswith('/Streaming/channels/101'))

    def test_uncalibrated_config_is_rejected(self):
        config = self.valid_config()
        config['counting']['calibrated'] = False
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_branch_is_saved_as_active_and_named_copy(self):
        config = self.valid_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, branch = save_branch_config(
                config,
                active_path=root / 'branch_config.json',
                branches_dir=root / 'branches',
            )
            self.assertTrue(active.exists())
            self.assertEqual(branch.name, 'branch_01.json')
            self.assertEqual(load_branch_config(branch)['branch_id'], 'branch_01')


class FaceProfileTests(unittest.TestCase):
    def test_employee_embeddings_are_saved_per_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'staff_faces.npz'
            recognizer = StaffFaceRecognizer('', '', profile, enabled=False)
            vectors = np.eye(3, 128, dtype=np.float32)
            recognizer.save_employee('employee_01', vectors)
            self.assertTrue(profile.exists())
            self.assertEqual(recognizer.enrolled_employee_count, 1)
            self.assertEqual(recognizer.embeddings.shape, (3, 128))

            replacement = np.ones((2, 128), dtype=np.float32)
            recognizer.save_employee('employee_01', replacement)
            self.assertEqual(recognizer.embeddings.shape, (2, 128))

if __name__ == '__main__':
    unittest.main()
