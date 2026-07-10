"""FallDetector 단위 테스트."""

import numpy as np
import pytest
from arda.detection.fall_detector import FallDetector
from arda.processing.pointcloud import PointCloud


def _make_pc(z: float, doppler: float = 0.0, n: int = 5) -> PointCloud:
    pts = [{"x": 0.0, "y": 2.0, "z": z, "doppler": doppler, "snr": 20.0, "noise": 5.0} for _ in range(n)]
    return PointCloud(pts)


def test_no_fall_when_stationary():
    detector = FallDetector(history_window=10)
    for _ in range(10):
        fell = detector.update(_make_pc(z=1.5, doppler=0.0))
    assert not fell


def test_fall_detected_on_height_drop():
    detector = FallDetector(history_window=10)
    # 처음 몇 프레임은 높은 위치
    for _ in range(5):
        detector.update(_make_pc(z=1.5, doppler=0.0))
    # 급격한 하강
    fell = False
    for _ in range(5):
        fell = detector.update(_make_pc(z=0.1, doppler=-1.2))
    assert fell


def test_reset_clears_state():
    detector = FallDetector(history_window=10)
    for _ in range(5):
        detector.update(_make_pc(z=1.5))
    detector.reset()
    fell = detector.update(_make_pc(z=1.5))
    assert not fell
