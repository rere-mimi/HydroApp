"""NIWA Henderson & Collins REC1 lookup for the hydrology window."""

import unittest
from unittest.mock import patch

from app import app
from hydroscreen import NIWA_FLOOD_QUERY, HydroScreenError, lookup_niwa_flood, nearest_niwa_reach


def _feature(name, flow, path):
    return {
        "attributes": {
            "Rivername": name,
            "NZREACH": 10010739,
            "Areakm2": 12.5,
            "Strm_Order": 3,
            "H_C18_10y": flow,
            "H_C18_50y": flow + 2,
            "H_C18_100y": flow + 4,
            "H_C18_1000": flow + 8,
            "HCse_100y": 1.5,
        },
        "geometry": {"paths": [path]},
    }


class NearestReachTests(unittest.TestCase):
    def test_picks_the_closer_reach_and_marks_screening_years(self):
        pin_lon, pin_lat = 174.0, -41.0
        near = _feature("Maitai", 10, [[174.0002, -41.0], [174.0005, -41.0]])
        far = _feature("Other", 40, [[174.02, -41.0], [174.03, -41.0]])
        reach = nearest_niwa_reach(pin_lat, pin_lon, [far, near])
        self.assertEqual(reach["river"], "Maitai")
        self.assertEqual(reach["nzreach"], 10010739)
        self.assertLess(reach["distance_m"], 80)
        flows = {item["years"]: item for item in reach["flows"]}
        self.assertAlmostEqual(flows[100]["flow_m3_s"], 14)
        self.assertTrue(flows[100]["screening"])
        self.assertFalse(flows[20]["screening"])
        self.assertFalse(flows[None]["screening"])

    def test_blank_river_name_is_empty(self):
        feature = _feature("  ", 1, [[174.0, -41.0], [174.001, -41.0]])
        reach = nearest_niwa_reach(-41.0, 174.0, [feature])
        self.assertIsNone(reach["river"])

    def test_nztm_geometry_is_returned_as_lonlat(self):
        feature = _feature(
            "Avon",
            8,
            [[1748000.0, 5428000.0], [1748100.0, 5428000.0]],
        )
        reach = nearest_niwa_reach(-41.29, 174.78, [feature])
        self.assertLess(abs(reach["line"][0][0]), 180)
        self.assertLess(abs(reach["line"][0][1]), 90)
        self.assertLess(reach["distance_m"], 5000)


class LookupRequestTests(unittest.TestCase):
    def test_uses_the_first_search_radius_that_returns_a_reach(self):
        feature = _feature("Maitai", 6, [[173.2842, -41.271], [173.285, -41.271]])

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"features": [feature]}

        with patch("hydroscreen.requests.get", return_value=Response()) as get:
            reach = lookup_niwa_flood(-41.271, 173.284)
        self.assertEqual(get.call_args.args[0], NIWA_FLOOD_QUERY)
        self.assertEqual(get.call_args.kwargs["params"]["distance"], "200")
        self.assertIn("H_C18_100y", get.call_args.kwargs["params"]["outFields"])
        self.assertEqual(reach["river"], "Maitai")
        self.assertEqual(reach["missing_screening_years"], [25])
        self.assertIn("2016", reach["note"])

    def test_reports_when_the_service_cannot_be_reached(self):
        with patch("hydroscreen.requests.get", side_effect=OSError("offline")):
            with self.assertRaises(HydroScreenError):
                lookup_niwa_flood(-41.271, 173.284)


class HydrologyApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_requires_a_pin(self):
        response = self.client.post("/api/hydrology", data={})
        self.assertEqual(response.status_code, 400)

    def test_returns_the_reach(self):
        with patch("app.lookup_niwa_flood", return_value={"river": "Maitai", "flows": []}) as lookup:
            response = self.client.post("/api/hydrology", data={"lat": -41.271, "lon": 173.284})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["river"], "Maitai")
        self.assertAlmostEqual(lookup.call_args.args[0], -41.271)


if __name__ == "__main__":
    unittest.main()
