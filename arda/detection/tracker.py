"""간단한 칼만 필터 기반 단일 객체 트래커."""

import numpy as np


class KalmanTracker:
    """3D 위치를 추정하는 단순 칼만 필터 (등속 모델)."""

    def __init__(self, dt: float = 0.1):
        # 상태벡터: [x, y, z, vx, vy, vz]
        n = 6
        self.x = np.zeros((n, 1))
        self.P = np.eye(n) * 500.0
        self.F = np.eye(n)
        for i in range(3):
            self.F[i, i + 3] = dt
        self.H = np.zeros((3, n))
        for i in range(3):
            self.H[i, i] = 1.0
        self.R = np.eye(3) * 0.1
        self.Q = np.eye(n) * 0.01

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].flatten()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = measurement.reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return self.x[:3].flatten()
