# test_benchmark.py

import os
import json
import tempfile

import numpy as np
import pytest

import benchmark

def test_load_ground_truth(monkeypatch, tmp_path):
    # ─── Monkey-patch VideoCapture to report 5 frames ─────────────────────────
    class DummyCap:
        def __init__(self, path):
            pass
        def get(self, prop_id):
            return 5
        def release(self):
            pass

    monkeypatch.setattr(benchmark.cv2, "VideoCapture", DummyCap)

    # ─── Create fake ground-truth JSON files for frames 1 and 3 ──────────────
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()

    for fid, (cx, cy) in [(1, (10, 11)), (3, (30, 33))]:
        data = [{"class_id": 0, "center_x": cx, "center_y": cy, "bbox": [0,0,0,0]}]
        fname = f"frame_{fid:04d}_ground_truth.json"
        (gt_dir / fname).write_text(json.dumps(data))

    # ─── Temporarily point VIDEOS["Test"] at our tmp folder ───────────────────
    benchmark.VIDEOS["Test"] = ("dummy.mp4", str(gt_dir))

    arr = benchmark.load_ground_truth("Test")
    # Should have length 5
    assert len(arr) == 5

    # Frame 1 → index 0
    assert arr[0] == (10, 11)
    # Frame 2 → index 1
    assert arr[1] is None
    # Frame 3 → index 2
    assert arr[2] == (30, 33)
    # Others untouched
    assert arr[3] is None and arr[4] is None

def test_bbox_to_center_conversion():
    # replicate the integer‐midpoint logic from run_tracker
    def bbox_to_center(x1, y1, x2, y2):
        return ((x1 + x2)//2, (y1 + y2)//2)

    # even dims
    assert bbox_to_center(10, 20, 30, 40) == (20, 30)
    # odd dims
    assert bbox_to_center(0, 0, 3, 5) == (1, 2)

def test_kalman_filter_on_linear_motion():
    # synthetic measurements along y = 2x
    measurements = [(i, float(i), float(2*i)) for i in range(10)]
    kf = benchmark.initialize_kf()

    preds = []
    for _, x, y in measurements:
        m = np.array([[np.float32(x)], [np.float32(y)]])
        kf.correct(m)
        p = kf.predict()
        preds.append((p[0,0], p[1,0]))

    # final prediction should be near (10,20)
    final = preds[-1]
    assert pytest.approx(final[0], rel=1e-2) == 10.0
    assert pytest.approx(final[1], rel=1e-2) == 20.0

def test_particle_filter_reproducibility_and_motion():
    # synthetic measurements along y = x
    measurements = [(i, float(i), float(i)) for i in range(10)]

    def run_pf():
        np.random.seed(0)
        pf = benchmark.PF(100, np.array([0.0, 0.0]), (100, 100))
        ests = []
        for _, x, y in measurements:
            pf.predict()
            pf.update(np.array([x, y]))
            pf.resample()
            ests.append(tuple(pf.estimate()))
        return ests

    first = run_pf()
    second = run_pf()
    # reproducibility under fixed seed
    assert first == second

    # sanity: first estimate near (0,0)
    assert np.hypot(*first[0]) < 5.0

    # subsequent estimates should track roughly along y = x
    diffs = [first[i][1] - first[i][0] for i in range(1, len(first))]
    assert all(abs(d) < 20 for d in diffs)
