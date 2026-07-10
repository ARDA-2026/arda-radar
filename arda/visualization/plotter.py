"""Matplotlib 기반 실시간 3D 포인트 클라우드 시각화."""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


class RealtimePlotter:
    """비블로킹 모드로 프레임마다 포인트 클라우드를 갱신한다."""

    def __init__(self, xlim=(-5, 5), ylim=(0, 8), zlim=(-2, 3)):
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self._scatter = None
        plt.ion()
        plt.show()

    def update(self, xyz: np.ndarray, fall_detected: bool = False) -> None:
        if self._scatter:
            self._scatter.remove()
        color = "red" if fall_detected else "cyan"
        self._scatter = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=10)
        title = "FALL DETECTED!" if fall_detected else "Monitoring..."
        self.ax.set_title(title, color="red" if fall_detected else "black")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        plt.close(self.fig)
