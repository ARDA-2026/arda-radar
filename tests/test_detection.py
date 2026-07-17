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


def test_no_fall_while_bouncing_back_up_after_drop():
    # 피크에서 충분히 하강한 뒤 바닥 근처에서 다시 위로 향하기 시작하면
    # (바운스, 혹은 노이즈로 무게중심이 끌려 올라가는 경우) — 아직 피크보다
    # 낮은 값이라도 "낙하 중"으로 보면 안 된다 (궤적이 위로 향하는 중이므로).
    detector = FallDetector(history_window=10)
    for z in [0.6, 0.6, 0.6, 0.5, 0.35, 0.2, 0.1, 0.05]:
        detector.update(_make_pc(z=z, doppler=-0.5))

    # 여기까지는 하강 궤적이 확정되어 낙하로 판정된 상태
    assert detector.update(_make_pc(z=0.1, doppler=0.3))  # 살짝 반등 — 아직 완만함

    # 뚜렷하게 위로 향하는 프레임이 이어지면 더 이상 낙하로 보지 않아야 한다
    fell = detector.update(_make_pc(z=0.3, doppler=0.5))
    assert not fell


def test_fall_detected_on_low_fast_drop():
    # data/failDetecting.png 재현: 피크 Z가 0.38m로 낮고(원래 임계값 0.40m
    # 미만), 소실 전까지 유효 프레임이 3개뿐인 낮고 빠른 실제 낙하.
    # 궤적 자체는 단조 하강이라 감지되어야 한다.
    detector = FallDetector(history_window=10)
    fell = False
    for z in [0.38, 0.25, -0.22, -0.65]:
        fell = detector.update(_make_pc(z=z, doppler=-0.5))
    assert fell


def test_choose_target_coasts_through_brief_gap_instead_of_jumping():
    # data/failTracking.png 재현: 실제 낙하물의 클러스터가 한두 프레임
    # 형성되지 않고, 그 자리에 무관한 정지 클러스터(예: 손/가구)만 남으면
    # 즉시 그걸로 "재포착"하면 안 된다 — 몇 프레임은 코스팅하며 놓쳐야 한다.
    detector = FallDetector(history_window=10)
    for z in [0.55, 0.44, 0.22, 0.08]:
        target = detector.choose_target([_make_pc(z=z, doppler=-0.3)], airborne_z=0.40)
        detector.update(target)

    stationary_elsewhere = _make_pc(z=0.45, doppler=0.0)  # 실제 물체와 무관한 정지 클러스터

    for _ in range(2):  # 유예 프레임(REACQUIRE_GRACE_FRAMES) 이내
        target = detector.choose_target([stationary_elsewhere], airborne_z=0.40)
        assert len(target) == 0  # 즉시 재포착하지 않고 이번 프레임은 놓친다
        detector.update(target)

    # 유예 프레임을 넘기면 그제서야 전역 재탐색을 허용한다
    target = detector.choose_target([stationary_elsewhere], airborne_z=0.40)
    assert len(target) > 0


def test_freefall_detected_regardless_of_starting_height():
    # 경로 3: 피크가 PEAK_Z_THRESHOLD(0.37m)를 한참 밑도는 낮은 위치에서
    # 처음 포착돼도, 최근 궤적이 중력 가속과 일치하는 자유낙하 패턴이면
    # 낙하로 판정해야 한다 (narrow-ROI 테스트처럼 시작 위치가 낮거나
    # 임의인 경우 대비).
    g, dt = 9.8, 0.1
    h0, v0 = 0.15, 0.5  # 피크(0.15m) 자체가 이미 임계값 미만
    zs = [h0 - v0 * t - 0.5 * g * t * t for t in (i * dt for i in range(5))]
    assert max(zs) < 0.37  # 이 시나리오가 실제로 기존 피크 기준을 못 넘는지 확인

    detector = FallDetector(history_window=10)
    fell = False
    for z in zs:
        fell = detector.update(_make_pc(z=z, doppler=-0.5))
    assert fell


def test_no_fall_on_constant_velocity_descent():
    # 등속(가속 없는) 하강은 자유낙하가 아니므로 경로 3이 반응하면 안 된다.
    detector = FallDetector(history_window=10)
    fell = False
    for z in [0.20, 0.15, 0.10, 0.05, 0.00, -0.05]:
        fell = detector.update(_make_pc(z=z, doppler=-0.2))
    assert not fell
