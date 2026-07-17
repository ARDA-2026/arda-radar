"""간단한 칼만 필터 기반 단일 객체 트래커."""

import numpy as np

GRAVITY = 9.8  # m/s² — 게이팅 예측에 참고하는 중력가속도 (tracker.py 자체 상태에는 반영 안 함)


class KalmanTracker:
    """3D 위치를 추정하는 단순 칼만 필터 (등속 모델).

    유지·보정되는 상태(self.x/self.P)는 X/Y/Z 모두 순수 등속도 모델이다.
    Z축에 중력가속도를 항상 반영하는 것도 시도해봤지만, 정지된(혹은 손에
    들려 있는) 물체를 계속 추적할 때 매 프레임 중력만큼 아래로 당겨지고
    보정 게인이 1보다 작아 완전히 상쇄되지 않아 — 실제로는 가만히 있는데
    필터링된 높이가 서서히 내려가는 편향이 누적됐다(정지 낙하 오탐 발생).

    그래서 중력은 이 영속 상태에는 넣지 않고, 게이팅에만 쓰는 1프레임
    미리보기(FallDetector.predicted_centroid())에서 그때그때 더해 쓴다 —
    상태에 누적되지 않으니 정지 물체 편향 없이, 실제 낙하 중에만 게이팅
    기준점이 더 정확해진다.
    """

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
