"""SQLite state/event storage, monthly Excel reports, and atomic heartbeat files."""

import json
import calendar
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
            name="MonthlyExcelReport",
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
            self.export_monthly_excel(previous_date[:7])
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
        """Refresh this month's workbook soon without blocking video processing."""
        if not self._closed:
            self._report_requested.set()

    def _report_loop(self):
        last_periodic_refresh = time.monotonic()
        while not self._report_stop.is_set():
            requested = self._report_requested.wait(timeout=1.0)
            periodic = time.monotonic() - last_periodic_refresh >= 60.0
            if not requested and not periodic:
                continue
            if self._report_stop.is_set():
                break
            # Group events that happen within a short burst into one workbook save.
            if requested:
                time.sleep(self.report_refresh_delay_s)
            if self._report_stop.is_set():
                break
            self._report_requested.clear()
            try:
                self.export_monthly_excel(date.today().strftime("%Y-%m"))
                last_periodic_refresh = time.monotonic()
            except Exception:
                LOGGER.exception("Unable to refresh the monthly Excel report")
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
        months = sorted({row["business_date"][:7] for row in rows})
        for month in months:
            report = self.reports_dir / f"{month}.xlsx"
            if not report.exists():
                self.export_monthly_excel(month)

    def export_monthly_excel(self, report_month=None):
        with self._export_lock:
            return self._export_monthly_excel_unlocked(report_month)

    def export_daily_excel(self, business_date=None):
        """Compatibility wrapper: refresh the month containing this date."""
        value = business_date or date.today().isoformat()
        return self.export_monthly_excel(str(value)[:7])

    @staticmethod
    def _duration_seconds(rows, start_event, stop_events, report_day, now):
        """Sum event-state intervals for one day, including an open interval."""
        start = None
        total = 0.0
        for row in rows:
            event = row["event"]
            timestamp = datetime.fromisoformat(row["timestamp"])
            if event == start_event:
                if start is None:
                    start = timestamp
            elif event in stop_events and start is not None:
                total += max(0.0, (timestamp - start).total_seconds())
                start = None
        if start is not None:
            if report_day == now.date().isoformat():
                end = now
            else:
                end = start.replace(hour=23, minute=59, second=59)
            total += max(0.0, (end - start).total_seconds())
        return int(round(total))

    def _export_monthly_excel_unlocked(self, report_month=None):
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        month = str(report_month or date.today().strftime("%Y-%m"))[:7]
        try:
            year, month_number = (int(part) for part in month.split("-"))
            first_day = date(year, month_number, 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("Report month must use YYYY-MM format.") from exc
        last_day = date(year, month_number, calendar.monthrange(year, month_number)[1])
        now = datetime.now().astimezone()
        visible_last_day = min(last_day, now.date()) if first_day <= now.date() else last_day
        first_text = first_day.isoformat()
        last_text = last_day.isoformat()
        with self._lock:
            events = self.connection.execute(
                """SELECT timestamp, business_date, camera_ip, track_id, event, entered, exited, inside
                   FROM events WHERE branch_id = ? AND business_date BETWEEN ? AND ?
                   ORDER BY business_date, id""",
                (self.branch_id, first_text, last_text),
            ).fetchall()
            health = self.connection.execute(
                """SELECT timestamp, business_date, event, details FROM health_events
                   WHERE branch_id = ? AND business_date BETWEEN ? AND ?
                   ORDER BY business_date, id""",
                (self.branch_id, first_text, last_text),
            ).fetchall()

        events_by_day = {}
        for row in events:
            day = row["business_date"]
            events_by_day.setdefault(day, []).append(row)
        health_by_day = {}
        for row in health:
            health_by_day.setdefault(row["business_date"], []).append(row)

        daily_rows = []
        cursor = first_day
        while cursor <= visible_last_day:
            day = cursor.isoformat()
            day_events = events_by_day.get(day, [])
            day_health = health_by_day.get(day, [])
            final_in = int(day_events[-1]["entered"]) if day_events else 0
            final_out = int(day_events[-1]["exited"]) if day_events else 0
            peak_inside = max((int(row["inside"]) for row in day_events), default=0)
            staff_ignored = sum("STAFF_" in row["event"] for row in day_events)
            unknown_counted = sum("UNKNOWN_COUNTED" in row["event"] for row in day_events)
            camera_downtime = self._duration_seconds(
                day_health,
                "CAMERA_DISCONNECTED",
                {"CAMERA_CONNECTED", "PROGRAM_STOPPED"},
                day,
                now,
            )
            program_uptime = self._duration_seconds(
                day_health,
                "PROGRAM_RUNNING",
                {"PROGRAM_STOPPED"},
                day,
                now,
            )
            daily_rows.append({
                "date": day,
                "in": final_in,
                "out": final_out,
                "closing_inside": max(0, final_in - final_out),
                "peak_inside": peak_inside,
                "staff_ignored": staff_ignored,
                "unknown_counted": unknown_counted,
                "camera_downtime": camera_downtime,
                "program_uptime": program_uptime,
            })
            cursor += timedelta(days=1)

        wb = Workbook()
        summary = wb.active
        summary.title = "Monthly Summary"
        summary.merge_cells("A1:H1")
        summary["A1"] = "MONTHLY PEOPLE COUNTER REPORT"
        summary["A1"].font = Font(bold=True, size=16, color="FFFFFF")
        summary["A1"].fill = PatternFill("solid", fgColor="1F4E78")
        summary["A1"].alignment = Alignment(horizontal="center")
        summary["A2"] = "Branch"
        summary["B2"] = self.branch_id
        summary["D2"] = "Month"
        summary["E2"] = month

        total_in = sum(row["in"] for row in daily_rows)
        total_out = sum(row["out"] for row in daily_rows)
        highest_peak = max((row["peak_inside"] for row in daily_rows), default=0)
        total_staff = sum(row["staff_ignored"] for row in daily_rows)
        total_unknown = sum(row["unknown_counted"] for row in daily_rows)
        total_downtime = sum(row["camera_downtime"] for row in daily_rows)
        total_uptime = sum(row["program_uptime"] for row in daily_rows)
        totals = (
            ("Total IN", total_in, "Total OUT", total_out, "Highest Peak", highest_peak),
            ("Staff Ignored", total_staff, "Unknown Counted", total_unknown,
             "Camera Downtime", total_downtime / 86400.0),
            ("Program Uptime", total_uptime / 86400.0, "", "", "", ""),
        )
        for row_index, values in enumerate(totals, start=4):
            for column, value in enumerate(values, start=1):
                summary.cell(row=row_index, column=column, value=value)
        summary["F5"].number_format = "[h]:mm:ss"
        summary["B6"].number_format = "[h]:mm:ss"

        header_row = 9
        headers = (
            "Date", "IN", "OUT", "Closing Inside", "Peak Inside",
            "Staff Ignored", "Unknown Counted", "Camera Downtime",
        )
        for column, value in enumerate(headers, start=1):
            cell = summary.cell(row=header_row, column=column, value=value)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="4472C4")
            cell.alignment = Alignment(horizontal="center")
        for row_index, item in enumerate(daily_rows, start=header_row + 1):
            values = (
                item["date"], item["in"], item["out"], item["closing_inside"],
                item["peak_inside"], item["staff_ignored"],
                item["unknown_counted"], item["camera_downtime"] / 86400.0,
            )
            for column, value in enumerate(values, start=1):
                summary.cell(row=row_index, column=column, value=value)
            summary.cell(row=row_index, column=8).number_format = "[h]:mm:ss"
        summary.freeze_panes = f"A{header_row + 1}"
        summary.auto_filter.ref = f"A{header_row}:H{header_row + len(daily_rows)}"
        widths = (14, 11, 11, 18, 15, 17, 19, 20)
        for index, width in enumerate(widths, start=1):
            summary.column_dimensions[get_column_letter(index)].width = width

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        target = self.reports_dir / f"{month}.xlsx"
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
        for report in self.reports_dir.glob("????-??.xlsx"):
            try:
                report_month = date.fromisoformat(f"{report.stem}-01")
                if report_month < report_cutoff.replace(day=1):
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
            self.export_monthly_excel(
                (state["business_date"] if state else date.today().isoformat())[:7]
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
