"""레이더 포인트 클라우드 실시간 모니터링 — 낙하 감지 없이 원시 데이터 시각화."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.processing.pointcloud import PointCloud
from arda.processing.filters import filter_stationary
from arda.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"

# ROI (단거리 실험 기준)
X_LIM = (-1.5, 1.5)
Y_LIM = (0.3, 2.5)
Z_LIM = (-0.2, 2.2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="레이더 포인트 클라우드 모니터")
    p.add_argument("--cli-port", default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--min-snr", type=float, default=8.0)
    p.add_argument("--all-points", action="store_true", help="정지 포인트도 표시 (도플러 필터 비활성화)")
    return p.parse_args()


class Monitor:
    def __init__(self, all_points: bool = False):
        self.all_points = all_points
        self._setup_plot()

    def _setup_plot(self):
        self.fig = plt.figure(figsize=(12, 5))
        self.fig.suptitle("ARDA — 레이더 모니터", fontsize=13)
        gs = gridspec.GridSpec(1, 2, figure=self.fig, wspace=0.35)

        # 왼쪽: 상단 뷰 (x-y, 위에서 아래로)
        self.ax_top = self.fig.add_subplot(gs[0])
        self.ax_top.set_title("Top View  (X – Y)")
        self.ax_top.set_xlabel("X (m)  ← 좌/우 →")
        self.ax_top.set_ylabel("Y (m)  거리")
        self.ax_top.set_xlim(*X_LIM)
        self.ax_top.set_ylim(*Y_LIM)
        self.ax_top.set_aspect("equal")
        self.ax_top.grid(True, alpha=0.3)
        # 센서 위치 표시
        self.ax_top.plot(0, 0, marker="^", color="black", markersize=10, label="센서")
        self.ax_top.legend(loc="upper right", fontsize=8)

        # 오른쪽: 측면 뷰 (y-z, 옆에서 봄)
        self.ax_side = self.fig.add_subplot(gs[1])
        self.ax_side.set_title("Side View  (Y – Z)")
        self.ax_side.set_xlabel("Y (m)  거리")
        self.ax_side.set_ylabel("Z (m)  높이")
        self.ax_side.set_xlim(*Y_LIM)
        self.ax_side.set_ylim(*Z_LIM)
        self.ax_side.set_aspect("equal")
        self.ax_side.grid(True, alpha=0.3)
        # 바닥선
        self.ax_side.axhline(0, color="brown", linewidth=1, linestyle="--", alpha=0.5, label="바닥")
        self.ax_side.legend(loc="upper right", fontsize=8)

        self._sc_top = self.ax_top.scatter([], [], s=30, c=[], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        self._sc_side = self.ax_side.scatter([], [], s=30, c=[], cmap="coolwarm", vmin=-1.0, vmax=1.0)

        # 도플러 컬러바
        cb = self.fig.colorbar(self._sc_top, ax=self.ax_side, fraction=0.046, pad=0.04)
        cb.set_label("Doppler (m/s)", fontsize=8)

        self._txt_info = self.fig.text(0.5, 0.01, "", ha="center", fontsize=9, color="gray")

        plt.ion()
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.show()

    def update(self, frame: dict) -> None:
        pc = PointCloud(frame["points"])
        pc = pc.filter_snr(self._min_snr).filter_roi(
            x_range=X_LIM, y_range=Y_LIM, z_range=Z_LIM
        )
        if not self.all_points:
            pc = filter_stationary(pc, min_abs_doppler=0.05)

        n = len(pc)
        if n > 0:
            colors = np.clip(pc.doppler, -1.0, 1.0)

            self._sc_top.set_offsets(np.c_[pc.xyz[:, 0], pc.xyz[:, 1]])
            self._sc_top.set_array(colors)

            self._sc_side.set_offsets(np.c_[pc.xyz[:, 1], pc.xyz[:, 2]])
            self._sc_side.set_array(colors)
        else:
            self._sc_top.set_offsets(np.empty((0, 2)))
            self._sc_side.set_offsets(np.empty((0, 2)))

        self._txt_info.set_text(
            f"Frame #{frame['frame_number']}   포인트: {n}개   "
            f"{'전체 포인트' if self.all_points else '움직이는 포인트만'}"
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_min_snr(self, v: float):
        self._min_snr = v

    def close(self):
        plt.close(self.fig)


def main() -> None:
    args = parse_args()
    monitor = Monitor(all_points=args.all_points)
    monitor.set_min_snr(args.min_snr)

    sensor = IWR6843Sensor(args.cli_port, args.data_port)
    sensor.configure(args.config)

    logger.info("모니터링 시작 (Ctrl+C로 종료)")
    try:
        with sensor:
            while True:
                frame = sensor.read_frame()
                if frame is None:
                    continue
                monitor.update(frame)
    except KeyboardInterrupt:
        logger.info("종료")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
