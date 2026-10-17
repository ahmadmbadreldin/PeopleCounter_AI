import json
import tempfile
import time
import unittest
from pathlib import Path

from runtime_storage import HeartbeatWriter, SQLiteStore, empty_state
from watchdog import heartbeat_is_stale


class RuntimeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.database = root / "data" / "branch.sqlite3"
        self.reports = root / "reports"

    def tearDown(self):
        self.temp.cleanup()

    def make_store(self):
        return SQLiteStore(
            self.database,
            "test_branch",
            "192.168.1.52",
            self.reports,
        )

    def test_state_and_events_survive_restart(self):
        store = self.make_store()
        state = empty_state("2026-10-17")
        state.update(total_in=7, total_out=2, peak_inside=5)
        store.add_event(14, "IN", state)
        store.close(state, export_today=False)

        reopened = self.make_store()
        restored = reopened.prepare_business_day("2026-10-17")
        self.assertEqual(restored["total_in"], 7)
        self.assertEqual(restored["total_out"], 2)
        reopened.close(restored, export_today=False)

    def test_excel_is_generated_as_daily_report(self):
        store = self.make_store()
        state = empty_state("2026-10-17")
        state.update(total_in=1, peak_inside=1)
        store.add_event(1, "IN", state)
        store.log_health("CAMERA_CONNECTED", "test", "2026-10-17")
        report = store.export_daily_excel("2026-10-17")
        self.assertTrue(report.exists())
        self.assertEqual(report.name, "2026-10-17.xlsx")
        store.close(state, export_today=False)

    def test_daily_excel_refreshes_automatically_after_event(self):
        store = SQLiteStore(
            self.database,
            "test_branch",
            "192.168.1.52",
            self.reports,
            report_refresh_delay_s=0.25,
        )
        state = empty_state()
        state.update(total_in=1, peak_inside=1)
        store.add_event(2, "IN", state)
        report = self.reports / f"{state['business_date']}.xlsx"
        deadline = time.monotonic() + 4.0
        while not report.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(report.exists())
        store.close(state, export_today=False)

    def test_new_business_day_resets_counters_and_exports_previous_day(self):
        store = self.make_store()
        old = empty_state("2026-10-16")
        old.update(total_in=3, total_out=1, peak_inside=2)
        store.add_event(9, "OUT", old)
        current = store.prepare_business_day("2026-10-17")
        self.assertEqual(current["total_in"], 0)
        self.assertTrue((self.reports / "2026-10-16.xlsx").exists())
        store.close(current, export_today=False)


class HeartbeatTests(unittest.TestCase):
    def test_heartbeat_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "heartbeat.json"
            writer = HeartbeatWriter(path)
            writer.write(status="running", camera_connected=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "running")
            self.assertTrue(payload["camera_connected"])
            self.assertFalse(heartbeat_is_stale(path, 30))


if __name__ == "__main__":
    unittest.main()
