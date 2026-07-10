"""DBSCAN 기반 포인트 클러스터링 및 타겟 선택."""

import numpy as np
from sklearn.cluster import DBSCAN

from .pointcloud import PointCloud


def cluster_points(pc: PointCloud, eps: float = 0.5, min_samples: int = 3) -> list[PointCloud]:
    """DBSCAN으로 포인트를 클러스터링하고 각 클러스터를 PointCloud로 반환한다."""
    if len(pc) < min_samples:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pc.xyz)
    clusters = []
    for label in set(labels):
        if label == -1:  # 노이즈 제외
            continue
        mask = labels == label
        clusters.append(PointCloud._from_arrays(pc.xyz[mask], pc.doppler[mask], pc.snr[mask]))
    return clusters


def select_target(clusters: list[PointCloud],
                  airborne_z: float = 0.40,
                  fall_doppler: float = -0.1) -> PointCloud:
    """낙하 물체 추적을 위한 클러스터 선택.

    정적 물체(도플러 ≈ 0)보다 이동 물체를 우선하기 위해 네 단계로 선택한다.

    1순위: 공중(Z >= airborne_z) + 하향 이동(doppler < fall_doppler) — 낙하 중
    2순위: 공중에 있는 클러스터 (이동 방향 무관)
    3순위: 하향 이동 중인 클러스터 (Z 무관) — 낙하 물체가 정적 물체 아래로 내려간 경우
    4순위: 가장 큰 클러스터 (fallback)
    """
    if not clusters:
        return PointCloud([])

    def cz(c: PointCloud) -> float:
        cen = c.centroid()
        return float(cen[2]) if cen is not None else -999.0

    def mdop(c: PointCloud) -> float:
        return float(np.mean(c.doppler)) if len(c) > 0 else 0.0

    # 1순위: 공중 + 하향 이동
    cand = [c for c in clusters if cz(c) >= airborne_z and mdop(c) < fall_doppler]
    if cand:
        return min(cand, key=mdop)  # 도플러가 가장 음수인 것 (가장 빠르게 하향)

    # 2순위: 공중
    cand = [c for c in clusters if cz(c) >= airborne_z]
    if cand:
        return max(cand, key=cz)

    # 3순위: 하향 이동 (낙하 물체가 정적 물체 Z 아래로 내려간 경우)
    cand = [c for c in clusters if mdop(c) < fall_doppler]
    if cand:
        return min(cand, key=mdop)

    # 4순위: fallback — 가장 큰 클러스터
    return max(clusters, key=len)
