import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from test_tracking import counter
from runtime_lock import BranchInstanceLock


class FakeStorage:
    def __init__(self):
        self.health = []
        self.states = []
        self.events = []
        self.reports = []

    def log_health(self, event, details="", business_date=None):
        self.health.append((event, details, business_date))

    def save_state(self, state):
        self.states.append(dict(state))

    def add_event(self, track_id, event, state):
        self.events.append((track_id, event, dict(state)))

    def export_monthly_excel(self, month):
        self.reports.append(month)
        return month


class TrackerMathTests(unittest.TestCase):
    def test_iou_identity_disjoint_and_partial(self):
        box = [50, 50, 20, 20]
        self.assertAlmostEqual(counter.IoUTracker.iou(box, box), 1.0)
        self.assertEqual(counter.IoUTracker.iou(box, [200, 200, 20, 20]), 0.0)
        self.assertGreater(counter.IoUTracker.iou(box, [55, 50, 20, 20]), 0.0)

    def test_tracker_preserves_id_for_nearby_detection(self):
        tracker = counter.IoUTracker(min_hits=1, iou_threshold=0.1)
        tracker.update([[100, 100, 60, 120]])
        first_id = tracker.get_tracks()[0].track_id
        tracker.update([[104, 103, 62, 118]])
        self.assertEqual(tracker.get_tracks()[0].track_id, first_id)

    def test_tracker_reset_clears_tracks_and_id_sequence(self):
        tracker = counter.IoUTracker(min_hits=1)
        tracker.update([[100, 100, 60, 120]])
        tracker.reset()
        self.assertEqual(tracker.tracks, [])
        self.assertEqual(tracker.next_id, 0)

    def test_small_object_status_changes_after_update(self):
        track = counter.Track([10, 10, 10, 10], 1)
        track.update_small_object_status(30, 30)
        self.assertTrue(track.is_small_object)
        track.update([10, 10, 100, 200], 30, 30)
        self.assertFalse(track.is_small_object)


class CameraStateTests(unittest.TestCase):
    def make_camera(self):
        cfg = types.SimpleNamespace(reconnect_delay_s=0.01)
        return counter.CameraThread("rtsp://unused", cfg)

    def test_connection_transitions_are_emitted_once(self):
        camera = self.make_camera()
        camera._set_connected(False, "open failed")
        camera._set_connected(False, "duplicate")
        time.sleep(0.01)
        camera._set_connected(True, "opened")
        events = camera.drain_connection_events()
        self.assertEqual(len(events), 2)
        self.assertFalse(events[0]["connected"])
        self.assertTrue(events[1]["connected"])
        self.assertIsNotNone(events[1]["downtime_seconds"])

    def test_get_frame_returns_only_new_sequence_and_a_copy(self):
        camera = self.make_camera()
        camera.frame = np.ones((4, 4, 3), dtype=np.uint8)
        camera.sequence = 3
        frame, sequence = camera.get_frame(2)
        self.assertEqual(sequence, 3)
        frame[:] = 0
        self.assertTrue(np.all(camera.frame == 1))
        self.assertEqual(camera.get_frame(3), (None, 3))

    def test_display_window_close_is_detected_without_keyboard(self):
        with patch.object(counter.cv2, "getWindowProperty", return_value=0):
            self.assertFalse(counter.display_window_is_open("counter"))
        with patch.object(counter.cv2, "getWindowProperty", return_value=1):
            self.assertTrue(counter.display_window_is_open("counter"))
        with patch.object(counter.cv2, "getWindowProperty", side_effect=counter.cv2.error("closed")):
            self.assertFalse(counter.display_window_is_open("counter"))


class YoloFailureTests(unittest.TestCase):
    def make_store(self, model):
        store = counter.StoreCounter.__new__(counter.StoreCounter)
        store.cfg = types.SimpleNamespace(
            confidence=0.4,
            nms_iou=0.4,
            image_size=640,
            device="",
        )
        store.model = model
        store.storage = FakeStorage()
        store.business_date = "2026-10-17"
        store._model_error_count = 0
        store._model_error_active = False
        return store

    def test_transient_yolo_error_returns_no_detections_and_logs_health(self):
        def failing(*args, **kwargs):
            raise RuntimeError("GPU temporarily unavailable")

        store = self.make_store(failing)
        self.assertEqual(store.detect(np.zeros((20, 20, 3), dtype=np.uint8)), [])
        self.assertEqual(store._model_error_count, 1)
        self.assertEqual(store.storage.health[0][0], "YOLO_ERROR")

    def test_yolo_recovery_is_logged_and_counter_resets(self):
        calls = iter((RuntimeError("temporary"), []))

        def model(*args, **kwargs):
            result = next(calls)
            if isinstance(result, Exception):
                raise result
            return result

        store = self.make_store(model)
        store.detect(np.zeros((20, 20, 3), dtype=np.uint8))
        store.detect(np.zeros((20, 20, 3), dtype=np.uint8))
        self.assertEqual(store._model_error_count, 0)
        self.assertIn("YOLO_RECOVERED", [item[0] for item in store.storage.health])

    def test_30_consecutive_yolo_errors_escalate_to_watchdog(self):
        store = self.make_store(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        for _ in range(29):
            self.assertEqual(store.detect(frame), [])
        with self.assertRaisesRegex(RuntimeError, "30 consecutive"):
            store.detect(frame)


class CounterStateTests(unittest.TestCase):
    def make_store(self):
        store = counter.StoreCounter.__new__(counter.StoreCounter)
        store.cfg = types.SimpleNamespace(sound_cooldown_s=100, sound_file="none")
        store.storage = FakeStorage()
        store.business_date = "2026-10-17"
        store.total_in = 4
        store.total_out = 1
        store.staff_ignored = 2
        store.unknown_crossings = 1
        store.peak_inside = 3
        store.last_sound_time = time.monotonic()
        store.pending_crossings = []
        store.inside_ids = set()
        store.inside_timestamps = {}
        store.track_colors = {}
        store.tracker = counter.IoUTracker()
        return store

    def test_log_event_commits_full_counter_snapshot(self):
        store = self.make_store()
        store._log_event(8, "IN", play_alert=False)
        _, event, state = store.storage.events[0]
        self.assertEqual(event, "IN")
        self.assertEqual(state["total_in"], 4)
        self.assertEqual(state["total_out"], 1)

    def test_reset_clears_and_persists_all_counters(self):
        store = self.make_store()
        store.reset()
        self.assertEqual(store.total_in, 0)
        self.assertEqual(store.total_out, 0)
        self.assertEqual(store.storage.states[-1]["peak_inside"], 0)

    def test_business_day_rollover_exports_and_resets(self):
        store = self.make_store()

        class Tomorrow:
            @staticmethod
            def today():
                return types.SimpleNamespace(isoformat=lambda: "2026-10-18")

        with patch.object(counter, "date", Tomorrow):
            self.assertTrue(store.maybe_roll_business_day())
        self.assertEqual(store.storage.reports, ["2026-10"])
        self.assertEqual(store.business_date, "2026-10-18")
        self.assertEqual(store.total_in, 0)
        self.assertIn("BUSINESS_DAY_STARTED", [item[0] for item in store.storage.health])

    def test_new_doorway_is_applied_without_resetting_totals(self):
        store = self.make_store()
        store.cfg.base_width = 1280
        store.cfg.base_height = 720
        store.cfg.zone_x1 = 10
        store.cfg.zone_y1 = 20
        store.cfg.zone_x2 = 300
        store.cfg.zone_y2 = 200
        store.cfg.line_position_ratio = 0.5
        store.cfg.line_hysteresis = 15
        store.cfg.in_direction = "top_to_bottom"
        store.cfg.crossing_cooldown_s = 0.75
        store.tracker.update([[100, 100, 60, 120]])
        store.apply_counting_config({
            "base_width": 640, "base_height": 360,
            "zone_x1": 200, "zone_y1": 30,
            "zone_x2": 500, "zone_y2": 300,
            "line_position_ratio": 0.75,
            "line_hysteresis": 8,
            "in_direction": "bottom_to_top",
            "crossing_cooldown_s": 1.0,
        })
        self.assertEqual(store.zone, (200, 30, 500, 300))
        self.assertEqual(store.cfg.in_direction, "bottom_to_top")
        self.assertEqual(store.count_line_y, 232.5)
        self.assertEqual(store.total_in, 4)
        self.assertEqual(store.tracker.tracks, [])


class SingleInstanceTests(unittest.TestCase):
    def test_only_one_counter_lock_per_branch(self):
        if counter.os.name != "nt":
            self.skipTest("Windows named mutex test")
        branch = f"unit_test_{time.time_ns()}"
        first = BranchInstanceLock(branch)
        second = BranchInstanceLock(branch)
        try:
            self.assertTrue(first.acquired)
            self.assertFalse(second.acquired)
        finally:
            second.close()
            first.close()


if __name__ == "__main__":
    unittest.main()
