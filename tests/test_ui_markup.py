"""The map UI must expose working Run and Cross-section controls."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app, output_folder_name, unique_output_folder, _safe_output_name
from hydroscreen import plan_linz_clip

SAMPLE_DEM = Path(__file__).resolve().parent / "fixtures" / "sample_dem.tif"


def _fake_extract(lat, lon, out_tif, progress=None, **kwargs):
    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SAMPLE_DEM, out_tif)
    return str(out_tif), plan_linz_clip(lat, lon)


def _fake_iter_extract(lat, lon, out_tif, **kwargs):
    path, plan = _fake_extract(lat, lon, out_tif)
    yield {"percent": 8, "message": "Downloading", "plan": plan}
    yield {"percent": 100, "message": "DEM ready", "path": path, "plan": plan}


class UiMarkupTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_includes_run_and_cross_section_controls(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('id="run-btn"', html)
        self.assertIn('id="mode-xs"', html)
        self.assertIn('id="run-status"', html)
        self.assertIn("novalidate", html)
        self.assertIn('id="busy"', html)
        self.assertIn('id="busy-bar"', html)
        self.assertIn('id="busy-pct"', html)
        self.assertIn('id="dem-opacity"', html)
        self.assertIn('name="upstream"', html)
        self.assertIn('name="downstream"', html)
        self.assertIn('name="lateral"', html)
        self.assertIn('id="clip-dem"', html)
        self.assertIn("Clip DEM", html)
        self.assertIn("LiDAR downloads only when you click Clip DEM", html)
        self.assertIn("5,000 m", html)
        self.assertIn("area of interest", html)
        self.assertIn("busy-spinner", html)
        self.assertIn("xs-measure", html)
        self.assertIn('id="xs-measure"', html)
        self.assertIn('id="xs-title"', html)
        self.assertIn("id=\"dem-tiles\"", html)
        self.assertLess(html.index('id="clip-dem"'), html.index('id="dem-tiles"'))
        self.assertIn("LINZ layer 121859", html)
        self.assertIn('id="open-hydrology"', html)
        self.assertIn('id="hydro-modal"', html)
        self.assertIn("NIWA flood flows", html)
        self.assertIn("Use matching flows", html)
        self.assertIn("not a continuous DEM for all of New Zealand", html)
        self.assertIn("outputs/dems", html)
        self.assertIn('name="slope_mode"', html)
        self.assertIn("Calculated from the centreline", html)
        self.assertIn("Specified slope", html)
        self.assertIn('name="slope"', html)
        self.assertIn(">m/m<", html)
        self.assertIn("0.002 is 1 in 500", html)
        self.assertIn("class=\"app-frame\"", html)
        self.assertIn("class=\"workspace\"", html)
        self.assertIn("class=\"map-stage\"", html)
        self.assertIn("area of interest", html)
        self.assertIn("Return period (ARI)", html)
        self.assertIn("1,000-year", html)
        self.assertIn('id="ari-list"', html)
        self.assertIn('data-ari="10"', html)
        self.assertIn('data-ari="25"', html)
        self.assertIn('data-ari="50"', html)
        self.assertIn('data-ari="100"', html)
        self.assertIn('data-ari="1000"', html)
        self.assertNotIn("Bundled Wellington", html)
        self.assertNotIn("Upload a GeoTIFF", html)
        self.assertIn('id="results-blocks"', html)
        self.assertIn('id="results-ari-toggle"', html)
        self.assertIn("Collapse return periods", html)
        self.assertNotIn(">Samples<", html)
        self.assertNotIn("Hyd. radius", html)
        self.assertNotIn('id="results-table"', html)
        self.assertIn("Download results workbook", html)
        css = Path(__file__).resolve().parents[1] / "static" / "style.css"
        styles = css.read_text(encoding="utf-8")
        self.assertIn("max-height: calc(100dvh - 6.2rem)", styles)
        self.assertIn("overscroll-behavior: contain", styles)
        js = Path(__file__).resolve().parents[1] / "static" / "app.js"
        client = js.read_text(encoding="utf-8")
        self.assertNotIn("scheduleDemPreview", client)
        self.assertIn("function clipSelectedDem()", client)
        self.assertIn("function syncInfoScrollLimit()", client)
        self.assertIn('clipDemBtn.addEventListener("click", clipSelectedDem)', client)
        self.assertEqual(client.count("refreshDemOverlay();"), 1)
        self.assertIn("DEM download", client)
        self.assertIn("Downloaded DEM", client)
        self.assertIn("/api/dem-tiles", client)
        self.assertIn("Screening DEM saved to", client)
        self.assertIn('navigator.sendBeacon("/api/stop"', client)
        self.assertIn("pagehide", client)

    def test_dem_preview_stream_reports_percent(self):
        with patch("hydroscreen.iter_extract_linz_dem_for_bridge", side_effect=_fake_iter_extract):
            response = self.client.post(
                "/api/dem-preview?stream=1",
                data={
                    "lat": -41.2865,
                    "lon": 174.7762,
                    "stream": "1",
                },
            )
            body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200, body[:500])
        events = [
            json.loads(line)
            for line in body.splitlines()
            if line.strip()
        ]
        self.assertTrue(events)
        percents = [item["percent"] for item in events if "percent" in item]
        self.assertTrue(percents)
        self.assertEqual(percents[0], 0)
        self.assertEqual(percents[-1], 100)
        done = events[-1]
        self.assertTrue(done.get("png"))
        self.assertTrue(done.get("bounds"))
        self.assertIn("BQ31", done.get("tiles") or [])
        self.assertTrue(any("BQ31.tiff" in (item.get("message") or "") for item in events))

    def test_dem_tiles_names_the_sheet_without_downloading(self):
        response = self.client.post(
            "/api/dem-tiles",
            data={"lat": -41.2865, "lon": 174.7762},
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIn("BQ31", payload["tiles"])
        self.assertIn("BQ31.tiff", payload["files"])

    def test_dem_tiles_outside_new_zealand_is_empty(self):
        response = self.client.post("/api/dem-tiles", data={"lat": -20, "lon": 160})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["tiles"], [])
        self.assertEqual(payload["files"], [])

    def test_cross_section_api_still_accepts_two_points(self):
        with patch("hydroscreen.extract_linz_dem_for_bridge", side_effect=_fake_extract):
            response = self.client.post(
                "/api/cross-section",
                data={
                    "lat1": -41.2865,
                    "lon1": 174.770,
                    "lat2": -41.2865,
                    "lon2": 174.782,
                    "sample_spacing": 5,
                },
            )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertGreater(response.get_json()["n_samples"], 5)


class OutputFolderTests(unittest.TestCase):
    def test_bridge_name_is_the_output_folder(self):
        self.assertEqual(output_folder_name("Colombo Street Bridge"), "Colombo Street Bridge")
        self.assertEqual(output_folder_name("  Avon/Bridge  "), "Avon Bridge")
        self.assertTrue(_safe_output_name("Colombo Street Bridge"))
        self.assertFalse(_safe_output_name(".."))
        self.assertFalse(_safe_output_name("secret/file"))

    def test_unnamed_bridge_gets_a_generated_folder(self):
        self.assertRegex(output_folder_name("   "), r"^\d{8}-\d{6}-[0-9a-f]{6}$")
        self.assertRegex(output_folder_name("CON"), r"^\d{8}-\d{6}-[0-9a-f]{6}$")

    def test_an_existing_bridge_folder_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "Avon Bridge").mkdir()
            self.assertEqual(unique_output_folder("Avon Bridge", root=tmp), "Avon Bridge (2)")


if __name__ == "__main__":
    unittest.main()
