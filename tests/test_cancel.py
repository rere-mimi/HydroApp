"""Tests for stopping a screening run so parameters can be changed."""

import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app import app, _supervise_process
from hydroscreen import HydroScreenCancelled, _run_interruptibly, run_screening, solve_water_level


class CancelScreeningTests(unittest.TestCase):
    def test_pre_set_event_stops_before_work(self):
        cancelled = threading.Event()
        cancelled.set()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(HydroScreenCancelled):
                run_screening(
                    lat=-41.2865,
                    lon=174.7762,
                    dem=str(Path(tmp) / "missing.tif"),
                    outdir=str(Path(tmp) / "out"),
                    cancel_event=cancelled,
                )

    def test_interruptible_call_stops_during_wait(self):
        started = threading.Event()
        cancelled = threading.Event()

        def slow():
            started.set()
            time.sleep(8)
            return "done"

        def stopper():
            started.wait(timeout=2)
            cancelled.set()

        threading.Thread(target=stopper, daemon=True).start()
        t0 = time.time()
        with self.assertRaises(HydroScreenCancelled):
            _run_interruptibly(slow, cancelled)
        self.assertLess(time.time() - t0, 2.0)

    def test_solve_water_level_stops_when_cancelled(self):
        cancelled = threading.Event()
        cancelled.set()
        dists = np.array([0.0, 2.0, 12.0, 14.0])
        elevs = np.array([2.0, 0.0, 0.0, 2.0])
        with self.assertRaises(HydroScreenCancelled):
            solve_water_level(dists, elevs, 8.0, 0.035, 0.002, cancel_event=cancelled)

    def test_run_completes_when_not_cancelled(self):
        west, north = 174.770, -41.280
        res = 0.0001
        height, width = 20, 40
        grid = np.zeros((height, width), dtype=np.float32)
        for col in range(width):
            grid[:, col] = 20.0 - 0.05 * col
        transform = from_origin(west, north, res, res)
        lon = west + (width / 2) * res
        lat = north - (height / 2) * res
        with tempfile.TemporaryDirectory() as tmp:
            dem_path = Path(tmp) / "dem.tif"
            with rasterio.open(
                dem_path,
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
            result = run_screening(
                lat=lat,
                lon=lon,
                dem=str(dem_path),
                outdir=str(Path(tmp) / "out"),
                interval=40.0,
                along_m=40.0,
                length=20.0,
                sample_spacing=10.0,
                centerline_coords=[[west + 2 * res, lat], [west + (width - 2) * res, lat]],
                flow_m3_s=1.0,
                cancel_event=threading.Event(),
            )
            self.assertGreaterEqual(len(result["summary"]), 1)


class StopEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_stop_missing_job_id(self):
        response = self.client.post("/api/stop", json={})
        self.assertEqual(response.status_code, 400)

    def test_stop_unknown_job_is_ok(self):
        response = self.client.post("/api/stop", json={"job_id": "abcd1234-job"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json().get("cancelled"))

    def test_stop_kills_a_stuck_screening_process(self):
        started = threading.Event()

        def fake_start(kwargs, result_path):
            proc = multiprocessing.Process(target=time.sleep, args=(30,), daemon=True)
            proc.start()
            started.set()
            return proc

        holder = {}

        def post_run():
            holder["client"] = app.test_client()
            holder["response"] = holder["client"].post(
                "/api/run",
                data={"lat": "-41.2865", "lon": "174.7762", "job_id": "stop-job-01"},
            )

        with patch("app._start_screening_process", side_effect=fake_start):
            worker = threading.Thread(target=post_run, daemon=True)
            worker.start()
            self.assertTrue(started.wait(5), "screening process did not start")
            t0 = time.time()
            stopped = app.test_client().post("/api/stop", json={"job_id": "stop-job-01"})
            worker.join(5)
        self.assertFalse(worker.is_alive(), "screening request did not finish after stop")
        self.assertLess(time.time() - t0, 4.0)
        self.assertEqual(stopped.status_code, 200)
        self.assertTrue(stopped.get_json().get("cancelled"))
        self.assertIsNotNone(holder.get("response"))
        self.assertEqual(holder["response"].status_code, 409)
        self.assertTrue(holder["response"].get_json().get("cancelled"))

    def test_supervise_kills_a_sleeping_process(self):
        proc = multiprocessing.Process(target=time.sleep, args=(30,), daemon=True)
        proc.start()
        cancelled = threading.Event()

        def stopper():
            time.sleep(0.2)
            cancelled.set()

        threading.Thread(target=stopper, daemon=True).start()
        t0 = time.time()
        with self.assertRaises(HydroScreenCancelled):
            _supervise_process(proc, cancelled)
        self.assertLess(time.time() - t0, 3.0)
        self.assertFalse(proc.is_alive())
        proc.close()


if __name__ == "__main__":
    unittest.main()
