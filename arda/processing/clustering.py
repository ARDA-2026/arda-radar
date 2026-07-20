"""DBSCAN 기반 포인트 클러스터링."""

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
