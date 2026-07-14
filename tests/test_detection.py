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


def test_predicted_centroid_none_before_first_observation():
    detector = FallDetector()
    assert detector.predicted_centroid() is None
    detector.update(_make_pc(z=1.0))
    assert detector.predicted_centroid() is not None


def test_kalman_smooths_single_spurious_jump():
    # 몇 프레임 안정적으로 추적하다가 노이즈로 인한 스퓨리어스 관측이 한
    # 프레임 섞여도, 칼만 필터가 예측과 블렌딩하므로 무게중심이 그 관측치로
    # 통째로 점프하지 않고 일부만 움직여야 한다 (자기보정).
    detector = FallDetector(history_window=10)
    for _ in range(5):
        detector.update(_make_pc(z=1.0, doppler=0.0))
    stable_z = float(detector.last_centroid[2])

    detector.update(_make_pc(z=3.0, doppler=0.0))
    jumped_z = float(detector.last_centroid[2])

    assert abs(jumped_z - stable_z) < abs(3.0 - stable_z)
