"""포인트 클라우드 전처리 유틸리티."""

import numpy as np


class PointCloud:
    """레이더 프레임에서 추출한 포인트 클라우드를 래핑한다."""

    def __init__(self, points: list[dict]):
        if points:
            arr = np.array([[p["x"], p["y"], p["z"], p["doppler"], p["snr"]] for p in points], dtype=np.float32)
        else:
            arr = np.empty((0, 5), dtype=np.float32)
        self.xyz = arr[:, :3]
        self.doppler = arr[:, 3]
        self.snr = arr[:, 4]

    def __len__(self) -> int:
        return len(self.xyz)

    def filter_snr(self, min_snr: float = 10.0) -> "PointCloud":
        mask = self.snr >= min_snr
        return self._from_arrays(self.xyz[mask], self.doppler[mask], self.snr[mask])

    def filter_roi(self, x_range=(-1.5, 1.5), y_range=(0.3, 2.5), z_range=(-0.2, 2.2)) -> "PointCloud":
        """관심 영역(ROI) 밖의 포인트를 제거한다."""
        mask = (
            (self.xyz[:, 0] >= x_range[0]) & (self.xyz[:, 0] <= x_range[1])
            & (self.xyz[:, 1] >= y_range[0]) & (self.xyz[:, 1] <= y_range[1])
            & (self.xyz[:, 2] >= z_range[0]) & (self.xyz[:, 2] <= z_range[1])
        )
        return self._from_arrays(self.xyz[mask], self.doppler[mask], self.snr[mask])

    def centroid(self) -> np.ndarray | None:
        if len(self) == 0:
            return None
        return self.xyz.mean(axis=0)

    @classmethod
    def _from_arrays(cls, xyz, doppler, snr) -> "PointCloud":
        obj = cls.__new__(cls)
        obj.xyz = xyz
        obj.doppler = doppler
        obj.snr = snr
        return obj
