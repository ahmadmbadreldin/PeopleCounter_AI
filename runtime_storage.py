"""SQLite state/event storage, daily Excel reports, and atomic heartbeat files."""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path


STATE_KEYS = (
    "total_in",
    "total_out",
    "staff_ignored",
    "unknown_crossings",
    "peak_inside",
)
LOGGER = logging.getLogger("people_counter.storage")


def empty_state(business_date=None):
    return {
        "business_date": business_date or date.today().isoformat(),
        "total_in": 0,
        "total_out": 0,
        "staff_ignored": 0,
        "unknown_crossings": 0,
        "peak_inside": 0,
    }


class SQLiteStore:
    """Small, durable operational store. Excel is generated only as a report."""

    def __init__(
        self,
        database,
        branch_id,
        camera_ip,
        reports_dir,
        event_retention_days=90,
        report_retention_days=365,
        report_refresh_delay_s=2.0,
    ):
        self.database = Path(database)
        self.reports_dir = Path(reports_dir)
        self.branch_id = str(branch_id)
        self.camera_ip = str(camera_ip)
        self.event_retention_days = max(7, int(event_retention_days))
        self.report_retention_days = max(7, int(report_retention_days))
        self.report_refresh_delay_s = max(0.25, float(report_refresh_delay_s))
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self.connection = sqlite3.connect(
            str(self.database), timeout=15.0, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA busy_timeout=15000")
            self._create_schema()
        self._report_requested = threading.Event()
        self._report_stop = threading.Event()
        self._export_lock = threading.Lock()
        self._report_thread = threading.Thread(
            target=self._report_loop,
            name="DailyExcelReport",
            daemon=True,
        )
        self._report_thread.start()

    def _create_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS counter_state (
                branch_id TEXT PRIMARY KEY,
                business_date TEXT NOT NULL,
                total_in INTEGER NOT NULL DEFAULT 0,
                total_out INTEGER NOT NULL DEFAULT 0,
                staff_ignored INTEGER NOT NULL DEFAULT 0,
                unknown_crossings INTEGER NOT NULL DEFAULT 0,
                peak_inside INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                business_date TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                camera_ip TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event TEXT NOT NULL,
                entered INTEGER NOT NULL,
                exited INTEGER NOT NULL,
                inside INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_branch_date
                ON events(branch_id, business_date, id);
            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                business_date TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_health_branch_date
                ON health_events(branch_id, business_date, id);
            """
        )
        self.connection.commit()

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _state_from_row(self, row):
        state = empty_state(row["business_date"])
        for key in STATE_KEYS:
            state[key] = int(row[key])
        return state

    def prepare_business_day(self, business_date=None):
        today = business_date or date.today().isoformat()
        previous_date = None
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM counter_state WHERE branch_id = ?",
                (self.branch_id,),
            ).fetchone()
            if row is None:
                state = empty_state(today)
                self._write_state_locked(state)
                self.connection.commit()
            elif row["business_date"] == today:
                state = self._state_from_row(row)
            else:
                previous_date = row["business_date"]
                state = empty_state(today)
                self._write_state_locked(state)
                self.connection.commit()
        if previous_date:
            self.export_daily_excel(previous_date)
        self.export_missing_past_reports(today)
        return state

    def _write_state_locked(self, state):
        self.connection.execute(
            """
            INSERT INTO counter_state (
                branch_id, business_date, total_in, total_out,
                staff_ignored, unknown_crossings, peak_inside, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(branch_id) DO UPDATE SET
                business_date=excluded.business_date,
                total_in=excluded.total_in,
                total_out=excluded.total_out,
                staff_ignored=excluded.staff_ignored,
                unknown_crossings=excluded.unknown_crossings,
                peak_inside=excluded.peak_inside,
                updated_at=excluded.updated_at
            """,
            (
                self.branch_id,
                state["business_date"],
                int(state["total_in"]),
                int(state["total_out"]),
                int(state["staff_ignored"]),
                int(state["unknown_crossings"]),
                int(state["peak_inside"]),
                self._now(),
            ),
        )
    def save_state(self, state):
        with self._lock:
            self._write_state_locked(state)
            self.connection.commit()

    def add_event(self, track_id, event, state):
        now = self._now()
        inside = max(0, int(state["total_in"]) - int(state["total_out"]))
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    """
                    INSERT INTO events (
                        timestamp, business_date, branch_id, camera_ip,
                        track_id, event, entered, exited, inside
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        state["business_date"],
                        self.branch_id,
                        self.camera_ip,
                        int(track_id),
                        str(event),
                        int(state["total_in"]),
                        int(state["total_out"]),
                        inside,
                    ),
                )
                self._write_state_locked(state)
            except Exception:
                self.connection.rollback()
                raise
            else:
                self.connection.commit()
        self.request_report()

    def log_health(self, event, details="", business_date=None):
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO health_events (
                    timestamp, business_date, branch_id, event, details
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self._now(),
                    business_date or date.today().isoformat(),
                    self.branch_id,
                    str(event),
                    str(details)[:2000],
                ),
            )
            self.connection.commit()
        self.request_report()

    def request_report(self):
        """Refresh today's workbook soon without blocking video processing."""
        if not self._closed:
            self._report_requested.set()

    def _report_loop(self):
        while not self._report_stop.is_set():
            if not self._report_requested.wait(timeout=1.0):
                continue
            if self._report_stop.is_set():
                break
            # Group events that happen within a short burst into one workbook save.
            time.sleep(self.report_refresh_delay_s)
            if self._report_stop.is_set():
                break
            self._report_requested.clear()
            try:
                self.export_daily_excel(date.today().isoformat())
            except Exception:
                LOGGER.exception("Unable to refresh the daily Excel report")
                # Excel may temporarily lock the workbook while a user has it
                # open. Keep SQLite authoritative and retry in the background.
                self._report_requested.set()
                self._report_stop.wait(timeout=30.0)

    def export_missing_past_reports(self, today=None):
        today = today or date.today().isoformat()
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT business_date FROM events
                WHERE branch_id = ? AND business_date < ?
                UNION
                SELECT business_date FROM health_events
                WHERE branch_id = ? AND business_date < ?
                """,
                (self.branch_id, today, self.branch_id, today),
            ).fetchall()
        for row in rows:
            report = self.reports_dir / f"{row['business_date']}.xlsx"
            if not report.exists():
                self.export_daily_excel(row["business_date"])

    def export_daily_excel(self, business_date=None):
        with self._export_lock:
            return self._export_daily_excel_unlocked(business_date)

    def _export_daily_excel_unlocked(self, business_date=None):
        from openpyxl import Workbook
        from openpyxl.styles import Font

        report_date = business_date or date.today().isoformat()
        with self._lock:
            events = self.connection.execute(
                """SELECT timestamp, camera_ip, track_id, event, entered, exited, inside
                   FROM events WHERE branch_id = ? AND business_date = ? ORDER BY id""",
                (self.branch_id, report_date),
            ).fetchall()
            health = self.connection.execute(
                """SELECT timestamp, event, details FROM health_events
                   WHERE branch_id = ? AND business_date = ? ORDER BY id""",
                (self.branch_id, report_date),
            ).fetchall()

        final_in = int(events[-1]["entered"]) if events else 0
        final_out = int(events[-1]["exited"]) if events else 0
        max_inside = max((int(row["inside"]) for row in events), default=0)
        staff_ignored = sum("STAFF_" in row["event"] for row in events)
        unknown_counted = sum("UNKNOWN_COUNTED" in row["event"] for row in events)

        wb = Workbook()
        summary = wb.active
        summary.title = "Summary"
        summary.append(["Branch", self.branch_id])
        summary.append(["Date", report_date])
        summary.append(["Entered", final_in])
        summary.append(["Exited", final_out])
        summary.append(["Inside at last event", max(0, final_in - final_out)])
        summary.append(["Peak inside", max_inside])
        summary.append(["Staff ignored", staff_ignored])
        summary.append(["Unknown counted as customer", unknown_counted])
        for cell in summary[1]:
            cell.font = Font(bold=True)

        ws = wb.create_sheet("Events")
        ws.append([
            "Timestamp", "Camera IP", "Track ID", "Event",
            "Entered", "Exited", "Inside",
        ])
        for row in events:
            ws.append(list(row))
        for cell in ws[1]:
            cell.font = Font(bold=True)

        hs = wb.create_sheet("Health")
        hs.append(["Timestamp", "Event", "Details"])
        for row in health:
            hs.append(list(row))
        for cell in hs[1]:
            cell.font = Font(bold=True)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        target = self.reports_dir / f"{report_date}.xlsx"
        temporary = target.with_suffix(".tmp.xlsx")
        wb.save(temporary)
        wb.close()
        os.replace(temporary, target)
        return target

    def run_maintenance(self):
        cutoff = (date.today() - timedelta(days=self.event_retention_days)).isoformat()
        with self._lock:
            self.connection.execute(
                "DELETE FROM events WHERE branch_id = ? AND business_date < ?",
                (self.branch_id, cutoff),
            )
            self.connection.execute(
                "DELETE FROM health_events WHERE branch_id = ? AND business_date < ?",
                (self.branch_id, cutoff),
            )
            self.connection.commit()
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        report_cutoff = date.today() - timedelta(days=self.report_retention_days)
        for report in self.reports_dir.glob("????-??-??.xlsx"):
            try:
                if date.fromisoformat(report.stem) < report_cutoff:
                    report.unlink()
            except (OSError, ValueError):
                continue

    def close(self, state=None, export_today=True):
        if self._closed:
            return
        self._report_stop.set()
        self._report_requested.set()
        self._report_thread.join(timeout=self.report_refresh_delay_s + 10.0)
        if state is not None:
            self.save_state(state)
        if export_today:
            self.export_daily_excel(
                state["business_date"] if state else date.today().isoformat()
            )
        with self._lock:
            self.connection.close()
            self._closed = True


class HeartbeatWriter:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **payload):
        data = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            **payload,
        }
        temporary = self.filename.with_suffix(self.filename.suffix + ".tmp")
        with self._lock:
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.filename)
        return data
