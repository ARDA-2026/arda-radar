"""FallDetector / Track 단위 테스트 (다중 추적)."""

import numpy as np
import pytest
from arda.detection.fall_detector import FallDetector, Track
from arda.processing.pointcloud import PointCloud


def _make_pc(z: float, doppler: float = 0.0, x: float = 0.0, n: int = 5) -> PointCloud:
    pts = [{"x": x, "y": 2.0, "z": z, "doppler": doppler, "snr": 20.0, "noise": 5.0} for _ in range(n)]
    return PointCloud(pts)


def test_no_fall_when_stationary():
    detector = FallDetector(history_window=10)
    fell = False
    for _ in range(10):
        fell = detector.update([_make_pc(z=1.5, doppler=0.0)])
    assert not fell


def test_fall_detected_on_height_drop():
    detector = FallDetector(history_window=10)
    # 처음 몇 프레임은 높은 위치 (피크 형성)
    for _ in range(3):
        detector.update([_make_pc(z=1.5, doppler=0.0)])
    # 물리적으로 말이 되는(중력 가속) 연속 하강 — 매칭 시 가속도 타당성
    # 체크(MAX_MATCH_ACCEL)를 통과하려면 한 물체가 100ms 만에 순간이동하듯
    # 큰 폭으로 뛰면 안 되고, 실제 자유낙하 물리를 따라야 같은 트랙이
    # 계속 매칭된다.
    g, dt, h0 = 9.8, 0.1, 1.5
    fell = False
    for i in range(1, 5):
        t = i * dt
        z = h0 - 0.5 * g * t * t
        fell = detector.update([_make_pc(z=z, doppler=-1.2)])
    assert fell
    assert len(detector.tracks) == 1


def test_reset_clears_state():
    detector = FallDetector(history_window=10)
    for _ in range(5):
        detector.update([_make_pc(z=1.5)])
    detector.reset()
    fell = detector.update([_make_pc(z=1.5)])
    assert not fell
    assert len(detector.tracks) == 1  # 리셋 후 새로 하나만 생성됨


def test_predicted_centroid_none_before_first_observation():
    track = Track(track_id=1)
    assert track.predicted_centroid() is None
    track.update(np.array([0.0, 2.0, 1.0]))
    assert track.predicted_centroid() is not None


def test_kalman_smooths_single_spurious_jump():
    # 몇 프레임 안정적으로 추적하다가 노이즈로 인한 스퓨리어스 관측이 한
    # 프레임 섞여도, 칼만 필터가 예측과 블렌딩하므로 무게중심이 그 관측치로
    # 통째로 점프하지 않고 일부만 움직여야 한다 (자기보정).
    track = Track(track_id=1, history_window=10)
    for _ in range(5):
        track.update(np.array([0.0, 2.0, 1.0]))
    stable_z = float(track.last_centroid[2])

    track.update(np.array([0.0, 2.0, 3.0]))
    jumped_z = float(track.last_centroid[2])

    assert abs(jumped_z - stable_z) < abs(3.0 - stable_z)


def test_fall_stays_confirmed_after_bounce_back_up():
    # 한 번 낙하로 확정된 트랙은 이후 궤적이 반등하더라도(바운스, 혹은
    # 바닥 노이즈로 무게중심이 끌려 올라가는 경우) 계속 낙하로 보고해야
    # 한다 — 원래는 매 프레임 다시 판정해서 반등하면 즉시 미확정으로
    # 되돌렸는데, 착지 후 바닥 근처 센서 노이즈로 반등/정체 조건에
    # 반복해서 걸렸다 풀렸다 하면서 같은 낙하 사건이 한 녹화 안에서 여러
    # 번 "새로 감지됨"으로 재발화하는 문제가 있었다(narrow_roi 재생에서
    # 확인, data/reference/wrongChoice.png 관련 분석). 한 번 확정된 사실은
    # 그 트랙이 살아있는 동안 번복되지 않는다(래치).
    detector = FallDetector(history_window=10)
    fell = False
    for z in [0.6, 0.6, 0.6, 0.5, 0.35, 0.2, 0.1, 0.05]:
        fell = detector.update([_make_pc(z=z, doppler=-0.5)])
    assert fell  # 이미 낙하로 확정된 상태

    # 뚜렷하게 위로 향하는 프레임이 이어져도 여전히 낙하로 보고해야 한다
    assert detector.update([_make_pc(z=0.3, doppler=0.5)])
    assert detector.update([_make_pc(z=0.5, doppler=0.5)])


def test_fall_detected_on_low_fast_drop():
    # data/failDetecting.png 재현: 피크 Z가 0.38m로 낮고(원래 임계값 0.40m
    # 미만), 소실 전까지 유효 프레임이 3개뿐인 낮고 빠른 실제 낙하.
    # 궤적 자체는 단조 하강이라 감지되어야 한다.
    detector = FallDetector(history_window=10)
    fell = False
    for z in [0.38, 0.25, -0.22, -0.65]:
        fell = detector.update([_make_pc(z=z, doppler=-0.5)])
    assert fell


def test_falling_track_and_nearby_static_cluster_stay_separate():
    # data/reference/failTracking.png 재현 의도 — 다중 추적 버전: 낙하 중인
    # 트랙과 무관한(멀리 떨어진) 정지 클러스터가 동시에 있어도 서로 다른
    # 트랙으로 유지되어야 한다. 매칭이 잘못돼 낙하 트랙이 정지 클러스터를
    # 흡수하거나, 반대로 낙하 트랙이 몇 프레임 미매칭됐다고 곧바로 삭제되면
    # 안 된다.
    detector = FallDetector(history_window=10)
    for z in [0.55, 0.44, 0.22, 0.08]:
        detector.update([_make_pc(z=z, x=0.0, doppler=-0.3)])
    assert len(detector.tracks) == 1
    falling_id = detector.tracks[0].id

    # 실제 물체와 무관한, 멀리 떨어진(x=1.5) 정지 클러스터
    stationary_elsewhere = _make_pc(z=0.45, x=1.5, doppler=0.0)
    for _ in range(3):
        detector.update([stationary_elsewhere])

    ids = {t.id for t in detector.tracks}
    assert falling_id in ids          # 낙하 트랙이 흡수되지 않고 유지됨(코스팅 중)
    assert len(detector.tracks) == 2  # 정지 클러스터는 별도 트랙으로 생성됨


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
        fell = detector.update([_make_pc(z=z, doppler=-0.5)])
    assert fell


def test_no_fall_on_constant_velocity_descent():
    # 등속(가속 없는) 하강은 자유낙하가 아니므로 경로 3이 반응하면 안 된다.
    detector = FallDetector(history_window=10)
    fell = False
    for z in [0.20, 0.15, 0.10, 0.05, 0.00, -0.05]:
        fell = detector.update([_make_pc(z=z, doppler=-0.2)])
    assert not fell


def test_independent_tracks_only_freefalling_one_triggers():
    # 사용자 시나리오: 한 물체는 자유낙하하고, 다른 하나는 (비닐봉지처럼)
    # 가속 없이 불규칙하게 내려간다 — 두 후보를 모두 별도 트랙으로 추적하되,
    # 실제로 자유낙하 특징을 보이는 트랙만 낙하로 확정되어야 한다.
    detector = FallDetector(history_window=10)

    g, dt = 9.8, 0.1
    h0, v0 = 0.15, 0.5
    freefall_zs = [h0 - v0 * t - 0.5 * g * t * t for t in (i * dt for i in range(6))]
    constant_velocity_zs = [0.20, 0.15, 0.10, 0.05, 0.00, -0.05]  # 가속 없는(자유낙하 아닌) 하강

    fell = False
    for ff_z, cv_z in zip(freefall_zs, constant_velocity_zs):
        clusters = [_make_pc(z=ff_z, x=0.0, doppler=-0.5),
                    _make_pc(z=cv_z, x=1.5, doppler=-0.2)]
        fell = detector.update(clusters) or fell

    assert fell
    assert len(detector.tracks) == 2
    fallen     = [t for t in detector.tracks if t.fell]
    not_fallen = [t for t in detector.tracks if not t.fell]
    assert len(fallen) == 1
    assert len(not_fallen) == 1
