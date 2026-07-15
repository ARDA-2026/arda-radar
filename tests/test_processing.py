"""PointCloud 및 클러스터링 단위 테스트."""

import numpy as np
import pytest
from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points, select_target


def _pts(coords: list[tuple], snr: float = 20.0) -> list[dict]:
    return [{"x": x, "y": y, "z": z, "doppler": 0.0, "snr": snr, "noise": 5.0} for x, y, z in coords]


def _cluster(coords: list[tuple], doppler: float = 0.0) -> PointCloud:
    pc = PointCloud(_pts(coords))
    return PointCloud._from_arrays(pc.xyz, np.full(len(pc), doppler, dtype=np.float32), pc.snr)


def test_pointcloud_len():
    pc = PointCloud(_pts([(0, 1, 0), (1, 2, 0)]))
    assert len(pc) == 2


def test_filter_snr_removes_low_snr():
    pts = _pts([(0, 1, 0)], snr=5.0) + _pts([(1, 2, 0)], snr=20.0)
    pc = PointCloud(pts).filter_snr(min_snr=10.0)
    assert len(pc) == 1


def test_filter_roi():
    pts = _pts([(0, 2, 0), (10, 2, 0)])  # 두 번째는 ROI 밖
    pc = PointCloud(pts).filter_roi(x_range=(-5, 5))
    assert len(pc) == 1


def test_centroid_empty():
    pc = PointCloud([])
    assert pc.centroid() is None


def test_clustering_two_groups():
    group_a = [(0.0 + i * 0.1, 0.0, 0.0) for i in range(5)]
    group_b = [(5.0 + i * 0.1, 0.0, 0.0) for i in range(5)]
    pc = PointCloud(_pts(group_a + group_b))
    clusters = cluster_points(pc, eps=0.3, min_samples=3)
    assert len(clusters) == 2


def test_select_target_without_last_centroid_prefers_airborne_falling():
    # 게이팅 없으면 기존 우선순위대로 공중+하향 클러스터를 고른다
    noise  = _cluster([(3.0, 2.0, 0.1)], doppler=0.0)          # 정적, 낮은 Z
    falling = _cluster([(0.0, 1.0, 0.6)], doppler=-0.5)        # 공중 + 하향
    target = select_target([noise, falling])
    assert target.centroid()[2] == pytest.approx(0.6)


def test_select_target_keeps_tracking_nearby_object_over_more_extreme_distant_noise():
    # 계속 추적 중인(근처+조건 만족) 물체는, 더 멀리서 조건을 '더 세게'
    # 만족하는 노이즈(더 음수인 도플러)에게 타겟을 뺏기지 않아야 한다
    last_centroid = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    tracked = _cluster([(0.02, 1.0, 0.52)], doppler=-0.2)       # 계속 낙하 중, 근처
    noise   = _cluster([(3.0, 2.0, 0.6)], doppler=-0.8)         # 멀리 있고 도플러 더 음수

    target = select_target([noise, tracked], last_centroid=last_centroid, max_jump=0.5)

    assert target.centroid()[2] == pytest.approx(0.52)


def test_select_target_falls_back_when_nothing_nearby():
    # 근처에 후보가 전혀 없으면 게이팅을 풀고 전체 클러스터에서 재탐색한다
    last_centroid = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    falling = _cluster([(0.0, 1.0, 0.6)], doppler=-0.5)

    target = select_target([falling], last_centroid=last_centroid, max_jump=0.5)

    assert target.centroid()[2] == pytest.approx(0.6)


def test_select_target_reacquires_when_locked_cluster_no_longer_qualifies():
    # 직전 위치 근처에 있는 것이 더 이상 물리적으로 낙하물처럼 보이지 않으면
    # (정적 노이즈에 락인된 상태) 게이팅을 풀고 실제 낙하 중인 물체를 재탐색한다
    last_centroid = np.array([0.0, 1.0, 0.1], dtype=np.float32)
    stale_noise = _cluster([(0.0, 1.0, 0.1)], doppler=0.0)      # 근처지만 정적 — 조건 미달
    falling     = _cluster([(1.5, 1.5, 0.6)], doppler=-0.5)     # 멀리 있지만 실제 낙하 중

    target = select_target([stale_noise, falling], last_centroid=last_centroid, max_jump=0.5)

    assert target.centroid()[2] == pytest.approx(0.6)


def test_select_target_prefers_closer_candidate_over_nearby_but_more_extreme_noise():
    # 노이즈가 추적 중인 물체 '근처'(게이팅 범위 안)에 뜨더라도, 조건을 더
    # 세게(도플러 더 음수) 만족한다는 이유로 실제 추적 물체를 밀어내면 안 된다.
    # 둘 다 근처/둘 다 조건 만족일 때는 직전 위치에 더 가까운 쪽이 이겨야 한다.
    last_centroid = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    tracked = _cluster([(0.05, 1.0, 0.52)], doppler=-0.2)        # 직전 위치에 아주 가까움
    noise   = _cluster([(0.3, 1.0, 0.6)], doppler=-0.9)          # 게이팅 범위 안이지만 더 멀고,
                                                                   # 도플러는 더 음수(더 '세게' 만족)

    target = select_target([noise, tracked], last_centroid=last_centroid, max_jump=0.5)

    assert target.centroid()[2] == pytest.approx(0.52)


def test_select_target_rejects_upward_noise_even_when_euclidean_closer():
    # 하강 중인 물체 바로 위에 노이즈가 뜨면, 유클리드 거리상 더 가깝더라도
    # (위로 올라가는 후보이므로) 무시하고 계속 아래로 찍히는 실제 물체를 따라간다
    last_centroid = np.array([0.0, 1.0, 0.50], dtype=np.float32)
    noise   = _cluster([(0.0, 1.0, 0.58)], doppler=-0.2)   # 바로 위, 거리 0.08 — 더 가까움
    tracked = _cluster([(0.0, 1.0, 0.30)], doppler=-0.3)   # 계속 하강 중, 거리 0.20 — 더 멀지만 아래

    target = select_target([noise, tracked], last_centroid=last_centroid, max_jump=0.5)

    assert target.centroid()[2] == pytest.approx(0.30)


def test_select_target_global_reacquire_prefers_descending_over_elevated():
    # data/wrongTracking.png 재현: 근접 게이팅 범위 안에는 아무 후보도 없어
    # 전역 재탐색으로 넘어갈 때, 실제로 하강 중인(비상승) 클러스터가 있다면
    # 단지 "공중(Z>=airborne_z)"이라는 이유만으로 더 위에 있는(엉뚱한/정지된)
    # 클러스터를 우선하면 안 된다.
    last_centroid = np.array([0.0, 1.0, 0.40], dtype=np.float32)
    elevated   = _cluster([(0.0, 1.0, 0.56)], doppler=0.0)   # 상승 — 공중 조건(tier2)만 만족
    descending = _cluster([(0.0, 1.0, 0.10)], doppler=-0.5)  # 비상승 — 하향 조건(tier3) 만족

    # max_jump을 작게 줘서 둘 다 근접 게이팅에는 못 들고 전역 재탐색으로 넘어가게 한다
    target = select_target([elevated, descending], last_centroid=last_centroid,
                           max_jump=0.05, strict=False)

    assert target.centroid()[2] == pytest.approx(0.10)
