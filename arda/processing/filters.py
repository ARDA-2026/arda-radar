"""신호 필터링 유틸리티 (속도, 높이 등)."""

import numpy as np
from .pointcloud import PointCloud


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def filter_stationary(pc: PointCloud, min_abs_doppler: float = 0.1) -> PointCloud:
    """도플러 속도가 거의 없는 정적 포인트를 제거한다."""
    mask = np.abs(pc.doppler) >= min_abs_doppler
    return PointCloud._from_arrays(pc.xyz[mask], pc.doppler[mask], pc.snr[mask])
