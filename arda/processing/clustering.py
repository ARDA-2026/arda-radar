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
                  fall_doppler: float = -0.1,
                  last_centroid: np.ndarray | None = None,
                  max_jump: float = 0.5,
                  max_rise: float = 0.05) -> PointCloud:
    """낙하 물체 추적을 위한 클러스터 선택.

    정적 물체(도플러 ≈ 0)보다 이동 물체를 우선하기 위해 네 단계로 선택한다.

    last_centroid는 보통 FallDetector.predicted_centroid()(칼만 필터의
    이번 프레임 예측 위치)를 넘긴다 — 직전 관측치를 그대로 쓰면 가속하며
    낙하하는 물체는 매 프레임 그 지점에서 점점 멀어지는 반면, 근처의
    정지 노이즈는 계속 가까이 남아 있어 게이팅이 역설적으로 노이즈에
    유리해진다. 예측 위치를 쓰면 게이팅 기준점 자체가 궤적을 따라
    이동하므로 이 문제가 없다.

    최초 포착/재포착 시 (1~3순위, last_centroid 없거나 근처에 유효 후보 없을 때):
      1순위: 공중(Z >= airborne_z) + 하향 이동(doppler < fall_doppler) — 낙하 중
      2순위: 공중에 있는 클러스터 (이동 방향 무관)
      3순위: 하향 이동 중인 클러스터 (Z 무관) — 낙하 물체가 정적 물체 아래로 내려간 경우
      4순위: 가장 큰 클러스터 (fallback, 게이팅 미적용)

    last_centroid가 주어지면, 그로부터 max_jump(m) 이내 + Z가 last_centroid
    대비 max_rise(m) 넘게 올라가지 않았으면서(=하강/수평 유지, 노이즈 수준의
    미세한 흔들림만 허용) 위 1~3순위 조건을 하나라도 만족하는 클러스터들
    중, last_centroid에 가장 가까운 것을 그대로 추적 대상으로 선택한다
    (같은 물체의 연속 = 계속 아래로 찍히는 궤적으로 간주).

    두 가지를 동시에 막기 위한 설계다:
    - 조건을 절대값 기준으로 비교하면(예: 도플러가 더 음수인 쪽 우선),
      근처에 새로 나타난 노이즈가 조건을 더 세게 만족할 때 실제 추적
      물체보다 우선시될 수 있다 → "가장 가까운 것"을 우선한다.
    - 단순히 "가장 가까운 것"만 보면, 하강 중인 물체 바로 위에 노이즈가
      뜰 때 유클리드 거리상 더 가깝다는 이유로 무게중심이 위로 튈 수
      있다 → Z가 올라가는 후보는 애초에 후보군에서 제외한다.

    근처(및 비상승)에 조건을 만족하는 후보가 하나도 없으면(추적이
    노이즈에 락인되었거나 실제 물체가 화면에 없는 경우) 게이팅을 풀고
    전체 클러스터에서 1~3순위를 다시 탐색한다 — 그래야 실제 낙하 물체가
    직전 위치와 무관하게(최초 포착, 재포착 등) 나타났을 때도 놓치지 않는다.
    """
    if not clusters:
        return PointCloud([])

    def cz(c: PointCloud) -> float:
        cen = c.centroid()
        return float(cen[2]) if cen is not None else -999.0

    def mdop(c: PointCloud) -> float:
        return float(np.mean(c.doppler)) if len(c) > 0 else 0.0

    def dist_to_last(c: PointCloud) -> float:
        cen = c.centroid()
        if cen is None or last_centroid is None:
            return float("inf")
        return float(np.linalg.norm(cen - last_centroid))

    def is_not_rising(c: PointCloud) -> bool:
        """last_centroid 대비 Z가 max_rise 넘게 올라가지 않았는지 (하강 궤적 유지)."""
        if last_centroid is None:
            return True
        return cz(c) <= float(last_centroid[2]) + max_rise

    def qualifies(c: PointCloud) -> bool:
        """1~3순위 중 하나라도 만족하는지 (공중 또는 하향 이동)."""
        return cz(c) >= airborne_z or mdop(c) < fall_doppler

    def pick(pool: list[PointCloud]) -> PointCloud | None:
        """주어진 후보군에서 1~3순위 로직만 적용 (없으면 None)."""
        cand = [c for c in pool if cz(c) >= airborne_z and mdop(c) < fall_doppler]
        if cand:
            return min(cand, key=mdop)  # 도플러가 가장 음수인 것 (가장 빠르게 하향)

        cand = [c for c in pool if cz(c) >= airborne_z]
        if cand:
            return max(cand, key=cz)

        cand = [c for c in pool if mdop(c) < fall_doppler]
        if cand:
            return min(cand, key=mdop)

        return None

    if last_centroid is not None:
        nearby_valid = [c for c in clusters
                        if dist_to_last(c) <= max_jump
                        and is_not_rising(c)
                        and qualifies(c)]
        if nearby_valid:
            return min(nearby_valid, key=dist_to_last)  # 직전 위치에 가장 가까운 것 = 같은 물체
        # 근처(비상승)에 물리적으로 유의미한 후보가 없음 — 게이팅 해제 후 전역 재탐색

    result = pick(clusters)
    if result is not None:
        return result

    # 4순위: fallback — 어디서도 못 찾았으면 가장 큰 클러스터
    return max(clusters, key=len)
