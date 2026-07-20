"""PointCloud 및 클러스터링 단위 테스트."""

from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points


def _pts(coords: list[tuple], snr: float = 20.0) -> list[dict]:
    return [{"x": x, "y": y, "z": z, "doppler": 0.0, "snr": snr, "noise": 5.0} for x, y, z in coords]


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
