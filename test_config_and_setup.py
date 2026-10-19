import json
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

import branch_config
from branch_config import (
    DEFAULT_CONFIG,
    build_rtsp_url_from_config,
    list_saved_branches,
    load_branch_config,
    safe_branch_id,
    save_branch_config,
    validate_config,
)
from setup_branch import BranchSetupApp


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class ConfigValidationTests(unittest.TestCase):
    def valid(self):
        config = deepcopy(DEFAULT_CONFIG)
        config["branch_id"] = "Cairo_01"
        config["camera"].update(ip="192.168.1.52", username="admin", password="secret")
        config["counting"]["calibrated"] = True
        return config

    def test_branch_and_employee_style_identifiers_are_sanitized(self):
        self.assertEqual(safe_branch_id(" Nasr City / 01 "), "Nasr_City_01")
        self.assertEqual(len(safe_branch_id("x" * 100)), 64)
        with self.assertRaises(ValueError):
            safe_branch_id(" !@#$ ")

    def test_explicit_rtsp_url_does_not_require_camera_password(self):
        config = self.valid()
        config["camera"].update(password="", rtsp_url="rtsp://example/live")
        self.assertEqual(validate_config(config), "Cairo_01")
        self.assertEqual(build_rtsp_url_from_config(config), "rtsp://example/live")

    def test_required_camera_fields_and_port_are_validated(self):
        for field in ("ip", "username", "password"):
            config = self.valid()
            config["camera"][field] = ""
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(config)
        for port in (0, 65536):
            config = self.valid()
            config["camera"]["rtsp_port"] = port
            with self.subTest(port=port), self.assertRaises(ValueError):
                validate_config(config)

    def test_invalid_geometry_and_direction_are_rejected(self):
        invalid_updates = (
            {"base_width": 0},
            {"zone_x1": 450, "zone_x2": 150},
            {"zone_y2": 9999},
            {"line_position_ratio": 0.05},
            {"line_position_ratio": 0.95},
            {"in_direction": "sideways"},
        )
        for update in invalid_updates:
            config = self.valid()
            config["counting"].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_config(config)

    def test_polygon_and_angled_line_geometry_are_validated(self):
        config = self.valid()
        config["counting"].update(
            zone_x1=100,
            zone_y1=100,
            zone_x2=700,
            zone_y2=600,
            zone_polygon=[[100, 100], [700, 170], [620, 600], [160, 540]],
            line_p1=[140, 300],
            line_p2=[650, 350],
            inside_point=[400, 470],
            inside_sign=1,
        )
        self.assertEqual(validate_config(config), "Cairo_01")

        crossing = deepcopy(config)
        crossing["counting"]["zone_polygon"] = [
            [100, 100], [700, 600], [700, 100], [100, 600]
        ]
        with self.assertRaisesRegex(ValueError, "edges must not cross"):
            validate_config(crossing)

        outside_line = deepcopy(config)
        outside_line["counting"].update(
            zone_polygon=[
                [100, 100], [700, 100], [700, 600],
                [450, 600], [450, 250], [350, 250], [350, 600], [100, 600]
            ],
            line_p1=[200, 500],
            line_p2=[550, 500],
        )
        with self.assertRaisesRegex(ValueError, "complete counting line"):
            validate_config(outside_line)

    def test_corrupt_json_reports_a_clear_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "branch.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unable to load branch config"):
                load_branch_config(path)

    def test_legacy_config_is_upgraded_to_branch_specific_runtime_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.json"
            path.write_text(json.dumps({"branch_id": "Legacy_02"}), encoding="utf-8")
            config = load_branch_config(path)
            self.assertEqual(config["storage"]["database"], "data/Legacy_02.sqlite3")
            self.assertIn("Legacy_02", config["runtime"]["heartbeat_file"])
            self.assertEqual(
                config["counting"]["zone_polygon"],
                [[150, 170], [450, 170], [450, 260], [150, 260]],
            )
            self.assertEqual(config["counting"]["line_p1"], [150, 215])
            self.assertEqual(config["counting"]["inside_sign"], 1)

    def test_legacy_uniform_decision_frames_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.json"
            path.write_text(
                json.dumps({
                    "branch_id": "Legacy_Uniform",
                    "staff": {"min_decision_frames": 9},
                }),
                encoding="utf-8",
            )
            config = load_branch_config(path)
            self.assertEqual(config["staff"]["min_negative_frames"], 9)

    def test_save_creates_named_config_and_report_directory(self):
        config = self.valid()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(branch_config, "PROJECT_DIR", root):
                active, named = save_branch_config(
                    config,
                    active_path=root / "branch_config.json",
                    branches_dir=root / "branches",
                )
            self.assertTrue(active.exists())
            self.assertTrue(named.exists())
            self.assertTrue((root / "reports" / "Cairo_01").is_dir())
            self.assertTrue(
                (root / "branches" / "Cairo_01" / "uniform_dataset").is_dir()
            )

    def test_saved_branches_are_sorted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("z_branch.json", "A_branch.json"):
                (root / name).write_text("{}", encoding="utf-8")
            with patch.object(branch_config, "BRANCHES_DIR", root):
                self.assertEqual(
                    [path.name for path in list_saved_branches()],
                    ["A_branch.json", "z_branch.json"],
                )


class SetupGeometryTests(unittest.TestCase):
    def make_app(self):
        app = BranchSetupApp.__new__(BranchSetupApp)
        app.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        app.image_scale = 0.5
        app.image_offset = (10, 20)
        return app

    def test_canvas_image_coordinate_round_trip(self):
        app = self.make_app()
        image_point = app._canvas_to_image(110, 70)
        self.assertEqual(image_point, (200, 100))
        self.assertEqual(app._image_to_canvas(*image_point), (110.0, 70.0))
        self.assertIsNone(app._canvas_to_image(0, 0))
        self.assertEqual(app._canvas_to_image(0, 0, clamp=True), (0, 0))

    def test_drag_is_committed_before_mouse_release(self):
        app = self.make_app()
        app.zone_original = (10, 10, 20, 20)
        app.zone_confirmed = False
        self.assertTrue(app._commit_selection((700, 500), (100, 100)))
        self.assertEqual(app.zone_original, (100, 100, 700, 500))
        self.assertTrue(app.zone_confirmed)

    def test_too_small_drag_cannot_reuse_old_zone(self):
        app = self.make_app()
        app.zone_original = None
        app.zone_confirmed = False
        self.assertFalse(app._commit_selection((100, 100), (110, 110)))
        self.assertIsNone(app.zone_original)
        self.assertFalse(app.zone_confirmed)

    def test_form_config_copies_camera_and_selected_zone(self):
        app = self.make_app()
        app.config = deepcopy(DEFAULT_CONFIG)
        app.branch_var = Value("Branch_9")
        app.ip_var = Value("10.0.0.12")
        app.user_var = Value("viewer")
        app.password_var = Value("pass")
        app.port_var = Value("8554")
        app.stream_var = Value("102")
        app.direction_var = Value("bottom_to_top")
        app.line_position_var = Value(72.0)
        app.zone_confirmed = True
        app.zone_original = (100, 200, 600, 500)
        app.polygon_original = []
        app.line_original = None
        app.inside_point_original = None
        app.inside_sign = 0
        result = app._form_config()
        self.assertEqual(result["branch_id"], "Branch_9")
        self.assertEqual(result["camera"]["rtsp_port"], 8554)
        self.assertEqual(result["camera"]["stream_channel"], "102")
        self.assertEqual(result["counting"]["base_width"], 1280)
        self.assertEqual(result["counting"]["zone_x2"], 600)
        self.assertEqual(result["counting"]["zone_polygon"][1], [600, 200])
        self.assertEqual(result["counting"]["line_p2"], [600, 416])
        self.assertEqual(result["counting"]["inside_sign"], -1)
        self.assertEqual(result["counting"]["in_direction"], "bottom_to_top")
        self.assertEqual(result["counting"]["line_position_ratio"], 0.72)


if __name__ == "__main__":
    unittest.main()
