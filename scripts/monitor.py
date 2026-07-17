"""레이더 포인트 클라우드 실시간 모니터링 — 낙하 감지 없이 원시 데이터 시각화.

--true-distance로 실측 거리(m)를 주면, 그 거리에 놓은 보정용 물체를 레이더가
어떻게 보고하는지(좌표·거리) 실시간으로 비교한다 — 레이더가 실제보다 얼마나
어긋나게 보고하는지(오차) 확인하는 용도. 특히 근접 거리(roi.y 하한 부근)에서
레이더 자체 근접장 잡음으로 거리/도플러가 신뢰할 수 없어지는 구간을 찾을 때
쓴다.

실행:
    uv run scripts/monitor.py
    uv run scripts/monitor.py --true-distance 0.5   # 센서에서 0.5m에 물체를 두고 비교
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.processing.pointcloud import PointCloud
from arda.processing.filters import filter_stationary
from arda.processing.clustering import cluster_points
from arda.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"

# ROI (단거리 실험 기준) — 보정 목적이므로 settings.yaml의 운영용 ROI보다
# 넓게, 근접 구간(0.02m~)부터 폭넓게 살펴볼 수 있게 잡는다.
X_LIM = (-1.5, 1.5)
Y_LIM = (0.02, 2.5)
Z_LIM = (-0.2, 2.2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="레이더 포인트 클라우드 모니터")
    p.add_argument("--cli-port", default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--min-snr", type=float, default=8.0)
    p.add_argument("--all-points", action="store_true", help="정지 포인트도 표시 (도플러 필터 비활성화)")
    p.add_argument("--y-min", type=float, default=Y_LIM[0], help="ROI 전방 거리 하한 (m)")
    p.add_argument("--y-max", type=float, default=Y_LIM[1], help="ROI 전방 거리 상한 (m)")
    p.add_argument("--true-distance", type=float, default=None,
                   help="보정용 물체까지의 실측 직선 거리 (m) — 주면 레이더 보고값과 오차를 실시간 비교")
    p.add_argument("--cluster-eps", type=float, default=0.15)
    p.add_argument("--cluster-min-samples", type=int, default=2)
    return p.parse_args()


class Monitor:
    def __init__(self, all_points: bool = False, y_range: tuple[float, float] = Y_LIM,
                 true_distance: float | None = None,
                 cluster_eps: float = 0.15, cluster_min_samples: int = 2):
        self.all_points = all_points
        self.y_range = y_range
        self.true_distance = true_distance
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
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
        self.ax_top.set_ylim(*self.y_range)
        self.ax_top.set_aspect("equal")
        self.ax_top.grid(True, alpha=0.3)
        # 센서 위치 표시
        self.ax_top.plot(0, 0, marker="^", color="black", markersize=10, label="센서")

        # 오른쪽: 측면 뷰 (y-z, 옆에서 봄)
        self.ax_side = self.fig.add_subplot(gs[1])
        self.ax_side.set_title("Side View  (Y – Z)")
        self.ax_side.set_xlabel("Y (m)  거리")
        self.ax_side.set_ylabel("Z (m)  높이")
        self.ax_side.set_xlim(*self.y_range)
        self.ax_side.set_ylim(*Z_LIM)
        self.ax_side.set_aspect("equal")
        self.ax_side.grid(True, alpha=0.3)
        # 바닥선
        self.ax_side.axhline(0, color="brown", linewidth=1, linestyle="--", alpha=0.5, label="바닥")

        # 보정 기준: 실측 거리에 해당하는 호(arc) — 센서(0,0)에서 true_distance
        # 만큼 떨어진 지점들. 포인트가 이 호 위에 찍히면 레이더 보고값이
        # 실측과 일치한다는 뜻이고, 안/밖으로 어긋나면 그만큼 오차가 있다는 뜻.
        if self.true_distance is not None:
            angles = np.linspace(-np.pi / 2, np.pi / 2, 100)
            arc_x = self.true_distance * np.sin(angles)
            arc_y = self.true_distance * np.cos(angles)
            self.ax_top.plot(arc_x, arc_y, "g--", linewidth=1.5, alpha=0.7,
                              label=f"실측 {self.true_distance:.2f}m")
            self.ax_side.plot(arc_y, np.zeros_like(arc_y), "g--", linewidth=0.5, alpha=0.3)
            self.ax_side.axvline(self.true_distance, color="green", linestyle="--",
                                  linewidth=1.5, alpha=0.7, label=f"실측 {self.true_distance:.2f}m")

        self.ax_top.legend(loc="upper right", fontsize=8)
        self.ax_side.legend(loc="upper right", fontsize=8)

        self._sc_top = self.ax_top.scatter([], [], s=30, c=[], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        self._sc_side = self.ax_side.scatter([], [], s=30, c=[], cmap="coolwarm", vmin=-1.0, vmax=1.0)

        # 도플러 컬러바
        cb = self.fig.colorbar(self._sc_top, ax=self.ax_side, fraction=0.046, pad=0.04)
        cb.set_label("Doppler (m/s)", fontsize=8)

        self._txt_info  = self.fig.text(0.5, 0.04, "", ha="center", fontsize=9, color="gray")
        self._txt_calib = self.fig.text(0.5, 0.01, "", ha="center", fontsize=10,
                                         color="darkgreen", fontweight="bold")

        plt.ion()
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        plt.show()

    def _report_calibration(self, pc: PointCloud) -> None:
        """가장 가까운 클러스터의 좌표·거리를 실측값과 비교해 표시·로그로 남긴다."""
        clusters = cluster_points(pc, eps=self.cluster_eps, min_samples=self.cluster_min_samples)
        if not clusters:
            text = f"[보정] 실측={self.true_distance:.2f}m  비교할 클러스터 없음"
            self._txt_calib.set_text(text)
            return

        nearest = min(clusters, key=lambda c: float(np.linalg.norm(c.centroid())))
        cen = nearest.centroid()
        reported_range = float(np.linalg.norm(cen))
        reported_y = float(cen[1])
        err_range = reported_range - self.true_distance
        err_y = reported_y - self.true_distance

        text = (
            f"[보정] 실측={self.true_distance:.2f}m  "
            f"레이더 직선거리={reported_range:.2f}m (오차 {err_range:+.2f}m)  "
            f"Y거리={reported_y:.2f}m (오차 {err_y:+.2f}m)  "
            f"XYZ=({cen[0]:+.2f}, {cen[1]:.2f}, {cen[2]:+.2f})  n={len(nearest)}"
        )
        self._txt_calib.set_text(text)
        print(f"\r{text}", end="", flush=True)

    def update(self, frame: dict) -> None:
        pc_roi = PointCloud(frame["points"]).filter_snr(self._min_snr).filter_roi(
            x_range=X_LIM, y_range=self.y_range, z_range=Z_LIM
        )
        # 보정 비교는 항상 도플러 필터 이전 포인트로 한다 — 보정용 물체는
        # 보통 가만히 놓아두므로, 움직이는 포인트만 거르면 정작 비교 대상이
        # 사라져버린다.
        pc = pc_roi if self.all_points else filter_stationary(pc_roi, min_abs_doppler=0.05)

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

        if self.true_distance is not None:
            self._report_calibration(pc_roi)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_min_snr(self, v: float):
        self._min_snr = v

    def close(self):
        plt.close(self.fig)


def main() -> None:
    args = parse_args()
    monitor = Monitor(
        all_points=args.all_points,
        y_range=(args.y_min, args.y_max),
        true_distance=args.true_distance,
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples,
    )
    monitor.set_min_snr(args.min_snr)

    sensor = IWR6843Sensor(args.cli_port, args.data_port)
    sensor.configure(args.config)

    logger.info("모니터링 시작 (Ctrl+C로 종료)")
    if args.true_distance is not None:
        logger.info("보정 모드 — 실측 거리 %.2fm과 실시간 비교", args.true_distance)
    try:
        with sensor:
            while True:
                frame = sensor.read_frame()
                if frame is None:
                    continue
                monitor.update(frame)
    except KeyboardInterrupt:
        print()
        logger.info("종료")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
