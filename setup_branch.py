"""One-screen branch setup, RTSP test, and visual entrance calibration."""

import base64
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from queue import Empty, Queue

import cv2

from branch_config import (
    ACTIVE_CONFIG_PATH,
    PROJECT_DIR,
    build_rtsp_url_from_config,
    list_saved_branches,
    load_branch_config,
    save_branch_config,
    validate_config,
)


class BranchSetupApp:
    PREVIEW_WIDTH = 960
    PREVIEW_HEIGHT = 540

    def __init__(self, root):
        self.root = root
        self.root.title("People Counter - Branch Setup")
        self.root.minsize(1050, 790)
        self.config = load_branch_config()
        self.frame = None
        self.photo = None
        self.image_scale = 1.0
        self.image_offset = (0, 0)
        self.selection_start = None
        self.zone_original = None
        self.zone_confirmed = False
        self.zone_canvas_id = None
        self.line_canvas_id = None
        self.direction_canvas_id = None
        self.direction_text_id = None
        self.start_requested = False
        self.busy = False
        self.capture_queue = Queue()

        self.branch_var = tk.StringVar()
        self.ip_var = tk.StringVar()
        self.user_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.port_var = tk.StringVar()
        self.stream_var = tk.StringVar()
        self.direction_var = tk.StringVar()
        self.saved_branch_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Enter camera details, then click Test Connection.")

        self._build_ui()
        self._populate_form(self.config)
        self._refresh_saved_branches()
        self.root.after(100, self._poll_capture_queue)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        saved = ttk.LabelFrame(outer, text="Saved branches / الفروع المحفوظة", padding=8)
        saved.pack(fill="x")
        self.saved_combo = ttk.Combobox(
            saved, textvariable=self.saved_branch_var, state="readonly", width=35
        )
        self.saved_combo.pack(side="left", padx=(0, 8))
        ttk.Button(saved, text="Load Branch", command=self._load_saved_branch).pack(side="left")
        ttk.Label(
            saved,
            text="Create a new Branch ID or load an existing branch.",
        ).pack(side="left", padx=15)

        form = ttk.LabelFrame(outer, text="Camera settings / بيانات الكاميرا", padding=10)
        form.pack(fill="x", pady=(10, 8))
        fields = [
            ("Branch ID", self.branch_var, 24, False),
            ("Camera IP", self.ip_var, 20, False),
            ("Username", self.user_var, 16, False),
            ("Password", self.password_var, 18, True),
            ("RTSP Port", self.port_var, 8, False),
        ]
        for column, (label, variable, width, secret) in enumerate(fields):
            group = ttk.Frame(form)
            group.grid(row=0, column=column, padx=5, sticky="w")
            ttk.Label(group, text=label).pack(anchor="w")
            ttk.Entry(group, textvariable=variable, width=width, show="*" if secret else "").pack()

        stream_group = ttk.Frame(form)
        stream_group.grid(row=0, column=len(fields), padx=5, sticky="w")
        ttk.Label(stream_group, text="Stream").pack(anchor="w")
        ttk.Combobox(
            stream_group,
            textvariable=self.stream_var,
            values=("101", "102"),
            width=7,
        ).pack()

        self.test_button = ttk.Button(form, text="Test Connection", command=self._test_connection)
        self.test_button.grid(row=0, column=len(fields) + 1, padx=(12, 4), pady=(15, 0))

        calibration = ttk.LabelFrame(
            outer,
            text="Entrance calibration / ارسم مستطيل حول فتحة الباب",
            padding=8,
        )
        calibration.pack(fill="both", expand=True)
        ttk.Label(
            calibration,
            text=(
                "After the preview appears, drag from one corner of the entrance to the opposite "
                "corner. The green center line is the counting line."
            ),
        ).pack(anchor="w", pady=(0, 5))
        self.canvas = tk.Canvas(
            calibration,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            background="#202020",
            highlightthickness=1,
            highlightbackground="#555555",
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._start_selection)
        self.canvas.bind("<B1-Motion>", self._move_selection)
        self.canvas.bind("<ButtonRelease-1>", self._finish_selection)

        direction = ttk.LabelFrame(outer, text="Entrance direction / اتجاه الدخول", padding=8)
        direction.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(
            direction,
            text="Outside is above the line; entering moves DOWN  ↓",
            variable=self.direction_var,
            value="top_to_bottom",
            command=self._draw_direction_arrow,
        ).pack(side="left", padx=(0, 30))
        ttk.Radiobutton(
            direction,
            text="Outside is below the line; entering moves UP  ↑",
            variable=self.direction_var,
            value="bottom_to_top",
            command=self._draw_direction_arrow,
        ).pack(side="left")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Label(footer, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="Start Active Config", command=self._start_saved).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(footer, text="Save & Start", command=self._save_and_start).pack(side="right")

        ttk.Label(
            outer,
            foreground="#8a5a00",
            text=(
                "Demo note: the camera password is stored locally in branch_config.json. "
                "Do not share or commit that file."
            ),
        ).pack(anchor="w", pady=(7, 0))

    def _refresh_saved_branches(self):
        names = [path.stem for path in list_saved_branches()]
        self.saved_combo["values"] = names

    def _populate_form(self, config):
        camera = config["camera"]
        counting = config["counting"]
        self.branch_var.set(config.get("branch_id", ""))
        self.ip_var.set(camera.get("ip", ""))
        self.user_var.set(camera.get("username", "admin"))
        self.password_var.set(camera.get("password", ""))
        self.port_var.set(str(camera.get("rtsp_port", 554)))
        self.stream_var.set(str(camera.get("stream_channel", "101")))
        self.direction_var.set(counting.get("in_direction", "top_to_bottom"))
        self.zone_confirmed = bool(counting.get("calibrated", False))
        self.zone_original = (
            int(counting.get("zone_x1", 0)),
            int(counting.get("zone_y1", 0)),
            int(counting.get("zone_x2", 0)),
            int(counting.get("zone_y2", 0)),
        )

    def _load_saved_branch(self):
        name = self.saved_branch_var.get().strip()
        if not name:
            messagebox.showinfo("Select branch", "Select a saved branch first.")
            return
        path = next((item for item in list_saved_branches() if item.stem == name), None)
        if path is None:
            messagebox.showerror("Missing branch", "The selected branch file no longer exists.")
            return
        try:
            self.config = load_branch_config(path)
        except RuntimeError as exc:
            messagebox.showerror("Configuration error", str(exc))
            return
        self.frame = None
        self.canvas.delete("all")
        self.zone_canvas_id = None
        self.line_canvas_id = None
        self.direction_canvas_id = None
        self.direction_text_id = None
        self._populate_form(self.config)
        self.status_var.set("Branch loaded. Click Test Connection to refresh the preview.")

    def _form_config(self):
        config = deepcopy(self.config)
        config["branch_id"] = self.branch_var.get().strip()
        config["camera"].update({
            "ip": self.ip_var.get().strip(),
            "username": self.user_var.get().strip(),
            "password": self.password_var.get(),
            "rtsp_port": int(self.port_var.get().strip()),
            "stream_channel": self.stream_var.get().strip(),
            "rtsp_url": "",
        })
        config["counting"]["in_direction"] = self.direction_var.get()
        config["counting"]["calibrated"] = bool(self.zone_confirmed)
        if self.frame is not None and self.zone_original is not None:
            height, width = self.frame.shape[:2]
            x1, y1, x2, y2 = self.zone_original
            config["counting"].update({
                "base_width": int(width),
                "base_height": int(height),
                "zone_x1": int(x1),
                "zone_y1": int(y1),
                "zone_x2": int(x2),
                "zone_y2": int(y2),
            })
        return config

    def _test_connection(self):
        if self.busy:
            return
        try:
            config = self._form_config()
            if not config["camera"]["ip"] or not config["camera"]["password"]:
                raise ValueError("Enter Camera IP and Password first.")
            int(config["camera"]["rtsp_port"])
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Camera settings", str(exc))
            return
        self.busy = True
        self.test_button.configure(state="disabled")
        self.status_var.set("Connecting to RTSP and reading a frame...")
        threading.Thread(
            target=self._capture_worker,
            args=(config,),
            daemon=True,
        ).start()

    def _capture_worker(self, config):
        url = build_rtsp_url_from_config(config)
        camera = config["camera"]
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(camera.get("open_timeout_ms", 5000)))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(camera.get("read_timeout_ms", 2000)))
        opened = cap.open(url, cv2.CAP_FFMPEG)
        if not opened:
            opened = cap.open(url)
        frame = None
        error = None
        if opened:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                ok, candidate = cap.read()
                if ok and candidate is not None and candidate.size:
                    frame = candidate
                    break
            if frame is None:
                error = "Connected, but no video frame was received. Check stream 101/102 and codec."
        else:
            error = "Could not open RTSP. Check IP, password, port, network, and stream channel."
        cap.release()
        self.capture_queue.put((frame, error))

    def _poll_capture_queue(self):
        try:
            frame, error = self.capture_queue.get_nowait()
        except Empty:
            pass
        else:
            self._capture_finished(frame, error)
        if self.root.winfo_exists():
            self.root.after(100, self._poll_capture_queue)

    def _capture_finished(self, frame, error):
        self.busy = False
        self.test_button.configure(state="normal")
        if error:
            self.status_var.set(error)
            messagebox.showerror("Camera test failed", error)
            return
        self.frame = frame
        self._show_frame()
        height, width = frame.shape[:2]
        self.status_var.set(
            f"Camera connected: {width}x{height}. Drag a rectangle around the entrance."
        )

    def _show_frame(self):
        if self.frame is None:
            return
        self.root.update_idletasks()
        canvas_w = max(self.PREVIEW_WIDTH, self.canvas.winfo_width())
        canvas_h = max(self.PREVIEW_HEIGHT, self.canvas.winfo_height())
        height, width = self.frame.shape[:2]
        self.image_scale = min(canvas_w / width, canvas_h / height)
        shown_w = max(1, int(width * self.image_scale))
        shown_h = max(1, int(height * self.image_scale))
        self.image_offset = ((canvas_w - shown_w) // 2, (canvas_h - shown_h) // 2)
        shown = cv2.resize(self.frame, (shown_w, shown_h), interpolation=cv2.INTER_AREA)
        shown = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", shown)
        if not ok:
            messagebox.showerror("Preview error", "Could not prepare the camera preview.")
            return
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.photo = tk.PhotoImage(data=data, format="png")
        self.canvas.delete("all")
        self.zone_canvas_id = None
        self.line_canvas_id = None
        self.direction_canvas_id = None
        self.direction_text_id = None
        ox, oy = self.image_offset
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")
        self._draw_saved_zone()

    def _canvas_to_image(self, x, y):
        if self.frame is None:
            return None
        ox, oy = self.image_offset
        height, width = self.frame.shape[:2]
        image_x = int(round((x - ox) / self.image_scale))
        image_y = int(round((y - oy) / self.image_scale))
        if not (0 <= image_x < width and 0 <= image_y < height):
            return None
        return image_x, image_y

    def _image_to_canvas(self, x, y):
        ox, oy = self.image_offset
        return ox + x * self.image_scale, oy + y * self.image_scale

    def _start_selection(self, event):
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.selection_start = point
        if self.zone_canvas_id is not None:
            self.canvas.delete(self.zone_canvas_id)
        if self.line_canvas_id is not None:
            self.canvas.delete(self.line_canvas_id)
        if self.direction_canvas_id is not None:
            self.canvas.delete(self.direction_canvas_id)
        if self.direction_text_id is not None:
            self.canvas.delete(self.direction_text_id)
        self.zone_canvas_id = None
        self.line_canvas_id = None
        self.direction_canvas_id = None
        self.direction_text_id = None

    def _move_selection(self, event):
        point = self._canvas_to_image(event.x, event.y)
        if self.selection_start is None or point is None:
            return
        self._draw_selection(self.selection_start, point)

    def _finish_selection(self, event):
        point = self._canvas_to_image(event.x, event.y)
        if self.selection_start is None or point is None:
            self.selection_start = None
            return
        x1, x2 = sorted((self.selection_start[0], point[0]))
        y1, y2 = sorted((self.selection_start[1], point[1]))
        self.selection_start = None
        if x2 - x1 < 25 or y2 - y1 < 25:
            self.status_var.set("The entrance rectangle is too small. Draw it again.")
            return
        self.zone_original = (x1, y1, x2, y2)
        self.zone_confirmed = True
        self._draw_selection((x1, y1), (x2, y2))
        self.status_var.set(
            f"Entrance selected: ({x1}, {y1}) to ({x2}, {y2}). Choose the IN direction."
        )

    def _draw_selection(self, first, second):
        x1, y1 = self._image_to_canvas(*first)
        x2, y2 = self._image_to_canvas(*second)
        if self.zone_canvas_id is None:
            self.zone_canvas_id = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="#00ffff", width=3
            )
            self.line_canvas_id = self.canvas.create_line(
                x1, (y1 + y2) / 2, x2, (y1 + y2) / 2, fill="#00ff00", width=3
            )
        else:
            self.canvas.coords(self.zone_canvas_id, x1, y1, x2, y2)
            self.canvas.coords(self.line_canvas_id, x1, (y1 + y2) / 2, x2, (y1 + y2) / 2)
        self._draw_direction_arrow()

    def _draw_direction_arrow(self):
        if self.zone_canvas_id is None:
            return
        x1, y1, x2, y2 = self.canvas.coords(self.zone_canvas_id)
        center_x = (x1 + x2) / 2.0
        margin = max(10.0, (y2 - y1) * 0.18)
        if self.direction_var.get() == "bottom_to_top":
            start_y, end_y = y2 - margin, y1 + margin
        else:
            start_y, end_y = y1 + margin, y2 - margin
        if self.direction_canvas_id is None:
            self.direction_canvas_id = self.canvas.create_line(
                center_x,
                start_y,
                center_x,
                end_y,
                fill="#ffff00",
                width=4,
                arrow=tk.LAST,
                arrowshape=(14, 18, 7),
            )
            self.direction_text_id = self.canvas.create_text(
                center_x + 25,
                (start_y + end_y) / 2.0,
                text="IN",
                fill="#ffff00",
                font=("Segoe UI", 12, "bold"),
            )
        else:
            self.canvas.coords(
                self.direction_canvas_id, center_x, start_y, center_x, end_y
            )
            self.canvas.coords(
                self.direction_text_id, center_x + 25, (start_y + end_y) / 2.0
            )

    def _draw_saved_zone(self):
        if self.zone_original is None or self.frame is None:
            return
        x1, y1, x2, y2 = self.zone_original
        height, width = self.frame.shape[:2]
        old_w = max(1, int(self.config["counting"].get("base_width", width)))
        old_h = max(1, int(self.config["counting"].get("base_height", height)))
        if old_w != width or old_h != height:
            x1, x2 = int(x1 * width / old_w), int(x2 * width / old_w)
            y1, y2 = int(y1 * height / old_h), int(y2 * height / old_h)
            self.zone_original = (x1, y1, x2, y2)
        self._draw_selection((x1, y1), (x2, y2))

    def _save_and_start(self):
        try:
            config = self._form_config()
            validate_config(config)
            active_path, branch_path = save_branch_config(config)
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Cannot save", str(exc))
            return
        self.config = config
        self._refresh_saved_branches()
        self.status_var.set(f"Saved {branch_path.name}. Starting counter...")
        self.start_requested = True
        self.root.after(250, self.root.destroy)

    def _start_saved(self):
        try:
            config = load_branch_config(ACTIVE_CONFIG_PATH)
            validate_config(config)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Saved configuration", str(exc))
            return
        self.start_requested = True
        self.root.destroy()


def main():
    root = tk.Tk()
    app = BranchSetupApp(root)
    root.mainloop()
    if app.start_requested:
        return subprocess.call(
            [sys.executable, str(PROJECT_DIR / "people_counter_hikvision.py")],
            cwd=str(PROJECT_DIR),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
