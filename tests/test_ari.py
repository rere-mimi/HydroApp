"""Multi-ARI flow scenarios overlay water level and velocity."""

import json
import multiprocessing
import pickle
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app import app
from hydroscreen import (
    HydroScreenError,
    normalize_flow_scenarios,
    parse_user_slope,
    run_screening,
)


def _finish_with(fake):
    """Stand in for the screening process and keep the arguments the API passed."""

    def start(kwargs, result_path):
        start.kwargs = kwargs
        Path(result_path).write_bytes(pickle.dumps({"ok": fake}))
        proc = multiprocessing.Process(target=time.sleep, args=(0,), daemon=True)
        proc.start()
        proc.join(5)
        return proc

    return start


def _write_channel_dem(path, west=174.770, north=-41.280, res=0.0001, height=40, width=80):
    grid = np.zeros((height, width), dtype=np.float32)
    for col in range(width):
        grid[:, col] = 30.0 - 0.1 * col
    grid[height // 2, :] -= 4.0
    transform = from_origin(west, north, res, res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(grid, 1)
    lat = north - (height / 2) * res
    lon = west + (width / 2) * res
    centerline = [
        [west + 5 * res, lat],
        [west + (width - 5) * res, lat],
    ]
    return lat, lon, centerline


class NormalizeFlowScenariosTests(unittest.TestCase):
    def test_default_is_100_year(self):
        scenarios = normalize_flow_scenarios(10)
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["years"], 100)
        self.assertEqual(scenarios[0]["key"], "100y")
        self.assertEqual(scenarios[0]["flow_m3_s"], 10.0)
        self.assertEqual(scenarios[0]["color"], "#7c3aed")

    def test_keeps_selected_order_and_colours(self):
        scenarios = normalize_flow_scenarios(
            10,
            [
                {"years": 10, "flow_m3_s": 4},
                {"years": 1000, "flow_m3_s": 20},
                {"years": 100, "flow_m3_s": 9},
            ],
        )
        self.assertEqual([item["years"] for item in scenarios], [10, 1000, 100])
        self.assertEqual(scenarios[0]["label"], "10-year ARI")
        self.assertEqual(scenarios[1]["label"], "1,000-year ARI")
        self.assertEqual(scenarios[0]["color"], "#0284c7")
        self.assertEqual(scenarios[1]["color"], "#be123c")

    def test_rejects_unknown_and_empty_selection(self):
        with self.assertRaises(HydroScreenError):
            normalize_flow_scenarios(10, [{"years": 20, "flow_m3_s": 1}])
        with self.assertRaises(HydroScreenError):
            normalize_flow_scenarios(10, [])
        with self.assertRaises(HydroScreenError):
            normalize_flow_scenarios(10, [{"years": 10, "flow_m3_s": -1}])


class MultiAriScreeningTests(unittest.TestCase):
    def test_run_screening_solves_each_selected_ari(self):
        with tempfile.TemporaryDirectory() as tmp:
            dem_path = Path(tmp) / "dem.tif"
            lat, lon, centerline = _write_channel_dem(dem_path)
            result = run_screening(
                lat=lat,
                lon=lon,
                dem=str(dem_path),
                outdir=str(Path(tmp) / "out"),
                interval=80.0,
                along_m=80.0,
                length=40.0,
                sample_spacing=10.0,
                centerline_coords=centerline,
                flow_m3_s=4.0,
                flow_scenarios=[
                    {"years": 10, "flow_m3_s": 2.0},
                    {"years": 100, "flow_m3_s": 8.0},
                ],
                mannings_n=0.035,
            )
        keys = [item["key"] for item in result["layout"]["flow_scenarios"]]
        self.assertEqual(keys, ["10y", "100y"])
        self.assertAlmostEqual(result["layout"]["flow_m3_s"], 2.0)
        self.assertGreater(result["layout"]["slope"], 0)
        self.assertEqual(result["layout"]["slope_source"], "centreline")
        self.assertAlmostEqual(result["layout"]["slope"], result["layout"]["calculated_slope"])
        self.assertIn("slope", result["centerline_profile"])
        self.assertIn("fit_slope", result["centerline_profile"])
        self.assertIn("fit_intercept", result["centerline_profile"])
        first = result["transects"][0]
        self.assertIn("10y", first["aris"])
        self.assertIn("100y", first["aris"])
        low = first["aris"]["10y"]["water_level_m"]
        high = first["aris"]["100y"]["water_level_m"]
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreaterEqual(high, low)
        self.assertGreater(first["aris"]["100y"]["velocity_m_s"], 0)
        summary = result["summary"][0]
        self.assertIn("10y", summary["aris"])
        self.assertAlmostEqual(summary["target_discharge_m3_s"], 2.0)


class AriApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_invalid_ari_is_rejected_before_screening(self):
        with patch("app._start_screening_process") as mock_run:
            response = self.client.post(
                "/api/run",
                data={
                    "lat": -41.2865,
                    "lon": 174.7762,
                    "interval": 50,
                    "length": 200,
                    "sample_spacing": 1,
                    "mannings_n": 0.035,
                    "flow": 10,
                    "aris": json.dumps([{"years": 20, "flow_m3_s": 3}]),
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("return period", response.get_json()["error"].lower())
        mock_run.assert_not_called()

    def test_run_passes_selected_aris(self):
        fake = {
            "centerline_source": "drawn",
            "used_synthetic_centerline": False,
            "layout": {
                "flow_m3_s": 4.0,
                "flow_scenarios": [
                    {"years": 10, "key": "10y", "label": "10-year ARI", "flow_m3_s": 4.0, "color": "#0284c7"},
                    {"years": 100, "key": "100y", "label": "100-year ARI", "flow_m3_s": 10.0, "color": "#7c3aed"},
                ],
                "slope": 0.002,
                "mannings_n": 0.035,
                "n_transects": 0,
                "sample_spacing_m": 1,
            },
            "centerline": [],
            "centerline_profile": {"distance_m": [], "elevation_m": [], "slope": 0.002},
            "transects": [],
            "summary": [],
            "summary_xlsx": "summary.xlsx",
        }
        starter = _finish_with(fake)
        with patch("app._start_screening_process", side_effect=starter):
            response = self.client.post(
                "/api/run",
                data={
                    "lat": -41.2865,
                    "lon": 174.7762,
                    "interval": 50,
                    "length": 200,
                    "sample_spacing": 1,
                    "mannings_n": 0.035,
                    "flow": 10,
                    "aris": json.dumps([
                        {"years": 10, "flow_m3_s": 4},
                        {"years": 100, "flow_m3_s": 10},
                    ]),
                },
            )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True)[:400])
        kwargs = starter.kwargs
        self.assertEqual([item["years"] for item in kwargs["flow_scenarios"]], [10, 100])
        payload = response.get_json()
        self.assertEqual(len(payload["layout"]["flow_scenarios"]), 2)
        self.assertEqual(payload["centerline_profile"]["slope"], 0.002)
        self.assertIsNone(kwargs["slope"])

    def test_specified_slope_is_passed_through(self):
        fake = {
            "centerline_source": "drawn",
            "used_synthetic_centerline": False,
            "layout": {"slope": 0.01, "slope_source": "user", "n_transects": 0, "sample_spacing_m": 1},
            "centerline": [],
            "centerline_profile": {"distance_m": [], "elevation_m": [], "slope": 0.01},
            "transects": [],
            "summary": [],
            "summary_xlsx": "summary.xlsx",
        }
        starter = _finish_with(fake)
        with patch("app._start_screening_process", side_effect=starter):
            response = self.client.post(
                "/api/run",
                data={
                    "lat": -41.2865,
                    "lon": 174.7762,
                    "interval": 50,
                    "length": 200,
                    "sample_spacing": 1,
                    "mannings_n": 0.035,
                    "flow": 10,
                    "slope_mode": "specified",
                    "slope": "0.01",
                },
            )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True)[:400])
        self.assertAlmostEqual(starter.kwargs["slope"], 0.01)

    def test_specified_slope_rejects_missing_zero_and_too_steep(self):
        base = {
            "lat": -41.2865,
            "lon": 174.7762,
            "interval": 50,
            "length": 200,
            "sample_spacing": 1,
            "mannings_n": 0.035,
            "flow": 10,
            "slope_mode": "specified",
        }
        with patch("app._start_screening_process") as mock_run:
            missing = self.client.post("/api/run", data=base)
            zero = self.client.post("/api/run", data={**base, "slope": "0"})
            steep = self.client.post("/api/run", data={**base, "slope": "2"})
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(zero.status_code, 400)
        self.assertEqual(steep.status_code, 400)
        self.assertIn("slope", missing.get_json()["error"].lower())
        self.assertIn("percent", zero.get_json()["error"].lower())
        self.assertIn("percent", steep.get_json()["error"].lower())
        mock_run.assert_not_called()


class UserSlopeTests(unittest.TestCase):
    def test_parse_user_slope_accepts_rise_over_run(self):
        self.assertIsNone(parse_user_slope(None))
        self.assertIsNone(parse_user_slope(""))
        self.assertAlmostEqual(parse_user_slope("0.002"), 0.002)
        self.assertAlmostEqual(parse_user_slope(1), 1.0)

    def test_parse_user_slope_rejects_bad_values(self):
        for value in (0, -0.01, 1.01, "steep", float("nan")):
            with self.assertRaises(HydroScreenError):
                parse_user_slope(value)

    def test_specified_slope_replaces_the_centreline_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            dem_path = Path(tmp) / "dem.tif"
            lat, lon, centerline = _write_channel_dem(dem_path)
            default = run_screening(
                lat=lat,
                lon=lon,
                dem=str(dem_path),
                outdir=str(Path(tmp) / "default"),
                interval=80.0,
                along_m=80.0,
                length=40.0,
                sample_spacing=10.0,
                centerline_coords=centerline,
                flow_m3_s=4.0,
                mannings_n=0.035,
            )
            specified = run_screening(
                lat=lat,
                lon=lon,
                dem=str(dem_path),
                outdir=str(Path(tmp) / "specified"),
                interval=80.0,
                along_m=80.0,
                length=40.0,
                sample_spacing=10.0,
                centerline_coords=centerline,
                flow_m3_s=4.0,
                mannings_n=0.035,
                slope=0.01,
            )
        self.assertEqual(default["layout"]["slope_source"], "centreline")
        self.assertAlmostEqual(default["layout"]["slope"], default["layout"]["calculated_slope"])
        self.assertAlmostEqual(specified["layout"]["slope"], 0.01)
        self.assertEqual(specified["layout"]["slope_source"], "user")
        self.assertAlmostEqual(specified["summary"][0]["slope"], 0.01)
        self.assertAlmostEqual(specified["centerline_profile"]["slope"], 0.01)
        self.assertGreater(specified["layout"]["calculated_slope"], 0)
        self.assertNotAlmostEqual(specified["layout"]["calculated_slope"], 0.01, places=4)
        self.assertAlmostEqual(
            specified["centerline_profile"]["fit_slope"],
            default["centerline_profile"]["fit_slope"],
        )

    def test_specified_slope_runs_when_the_centreline_fit_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dem_path = Path(tmp) / "dem.tif"
            lat, lon, centerline = _write_channel_dem(dem_path)
            with patch(
                "hydroscreen.centerline_slope_and_profile",
                return_value=(None, np.array([]), np.array([]), None, None),
            ):
                result = run_screening(
                    lat=lat,
                    lon=lon,
                    dem=str(dem_path),
                    outdir=str(Path(tmp) / "out"),
                    interval=80.0,
                    along_m=80.0,
                    length=40.0,
                    sample_spacing=10.0,
                    centerline_coords=centerline,
                    flow_m3_s=4.0,
                    mannings_n=0.035,
                    slope=0.01,
                )
                with self.assertRaises(HydroScreenError):
                    run_screening(
                        lat=lat,
                        lon=lon,
                        dem=str(dem_path),
                        outdir=str(Path(tmp) / "out2"),
                        interval=80.0,
                        along_m=80.0,
                        length=40.0,
                        sample_spacing=10.0,
                        centerline_coords=centerline,
                        flow_m3_s=4.0,
                        mannings_n=0.035,
                    )
        self.assertAlmostEqual(result["layout"]["slope"], 0.01)
        self.assertEqual(result["layout"]["slope_source"], "user")
        self.assertIsNone(result["layout"]["calculated_slope"])
        self.assertIsNone(result["centerline_profile"]["fit_slope"])


class AriSummaryMarkupTests(unittest.TestCase):
    def test_index_summary_uses_transect_blocks(self):
        client = app.test_client()
        html = client.get("/").get_data(as_text=True)
        self.assertIn('id="results-blocks"', html)
        self.assertNotIn('id="results-table"', html)
        self.assertNotIn("<th>Samples</th>", html)
        self.assertNotIn("<th>Width (m)</th>", html)
        self.assertNotIn("<th>Area (m²)</th>", html)
        self.assertNotIn("<th>Hyd. radius (m)</th>", html)

    def test_client_groups_return_periods_inside_each_transect(self):
        root = Path(__file__).resolve().parents[1]
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        css = (root / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn("function renderTransectBlock", js)
        self.assertIn("transect-block", js)
        self.assertIn("transect-ari-table", js)
        self.assertIn("Water level (max depth) (m)", js)
        self.assertIn("Flow (m³/s)", js)
        self.assertIn("resultsArisExpanded", js)
        self.assertIn("Orange numbered dots = transect stations", js)
        self.assertIn(".transect-block", css)
        self.assertIn(".transect-ari-table", css)
        self.assertIn(".transect-block-scroll", css)


if __name__ == "__main__":
    unittest.main()
