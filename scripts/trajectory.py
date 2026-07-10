"""Z축 하강 궤적 실시간 시각화.

detect.py와 동일한 파이프라인(SNR→ROI→DBSCAN)을 사용하며,
클러스터 포인트를 직접 표시해 감지 상태를 시각적으로 확인한다.

  [Side View Y-Z]
    · 회색  : SNR·ROI 필터 통과 포인트 (레이더가 보는 전체)
    · 주황색: DBSCAN으로 선택된 추적 클러스터 포인트
    · 파랑/빨강 선: 무게중심 궤적 (낙하 여부로 색 구분)
    · ★     : 현재 무게중심

  [Height over time]
    추적 클러스터 무게중심의 Z 이력
"""

import argparse
import queue
import sys
import threading
from collections import deque
from pathlib import Path

import warnings
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points, select_target
from arda.detection import FallDetector
from arda.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"

# detect.py와 동일한 파라미터
MIN_SNR         = 6.0
CLUSTER_EPS     = 0.15
CLUSTER_MINSAMP = 2
Z_RANGE         = (-0.8, 0.8)
AIRBORNE_Z      = 0.40  # 공중 물체 판별 기준 (fall_detector.PEAK_Z_THRESHOLD와 동일)

TRAIL_LEN = 80


def parse_args():
    p = argparse.ArgumentParser(description="Z축 하강 궤적 시각화")
    p.add_argument("--cli-port",  default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--debug", action="store_true",
                   help="하강 중 매 프레임 [FD] 값을 터미널에 출력")
    return p.parse_args()


# ── 센서 스레드 ─────────────────────────────────────────────────────────────────

def sensor_thread(cli_port, data_port, config, frame_q, stop_evt):
    try:
        sensor = IWR6843Sensor(cli_port, data_port)
        print(f"[SENSOR] Configuring ({cli_port} / {data_port})...")
        sensor.configure(config)
        print("[SENSOR] Config done — opening data port")
        with sensor:
            print("[SENSOR] Streaming frames (close window or Ctrl+C to quit)")
            while not stop_evt.is_set():
                frame = sensor.read_frame()
                if frame is not None:
                    if frame_q.qsize() > 3:
                        try: frame_q.get_nowait()
                        except queue.Empty: pass
                    frame_q.put(frame)
    except Exception as e:
        logger.error("Sensor error: %s", e)
        frame_q.put({"error": str(e)})


# ── 시각화 ──────────────────────────────────────────────────────────────────────

class FallTrajectoryPlotter:
    def __init__(self, debug: bool = False):
        self._trail: deque[tuple[float, float, bool]] = deque(maxlen=TRAIL_LEN)
        self._frame_num = 0
        self._detector = FallDetector(debug=debug)
        self._paused = False
        self._prev_falling = False

        self.fig = plt.figure(figsize=(13, 5))
        self.fig.suptitle("Fall Trajectory  —  IWR6843AOPEVM", fontsize=12)

        # ── 왼쪽: 측면 뷰 Y-Z ─────────────────────────────────────
        self.ax_side = self.fig.add_subplot(1, 2, 1)
        self.ax_side.set_title("Side View  (Y - Z)")
        self.ax_side.set_xlabel("Y (m)  distance from sensor")
        self.ax_side.set_ylabel("Z (m)  height")
        self.ax_side.set_xlim(0.3, 2.5)
        self.ax_side.set_ylim(Z_RANGE[0] - 0.05, Z_RANGE[1] + 0.05)
        self.ax_side.set_aspect("equal")
        self.ax_side.grid(True, alpha=0.3)
        self.ax_side.axhline(0, color="brown", linewidth=1.2,
                              linestyle="--", alpha=0.6, label="floor")
        # Z ROI 경계
        self.ax_side.axhline(Z_RANGE[1], color="gray", linewidth=0.8,
                              linestyle=":", alpha=0.5, label=f"Z ROI {Z_RANGE}")

        # 전체 필터링 포인트 (회색) — DBSCAN 이전
        self._sc_all = self.ax_side.scatter([], [], c="lightgray", s=14,
                                             alpha=0.6, zorder=2, label="all pts")
        # 추적 클러스터 포인트 (주황) — DBSCAN 결과
        self._sc_cluster = self.ax_side.scatter([], [], c="orange", s=30,
                                                 alpha=0.8, zorder=4, label="cluster")
        # 무게중심 궤적: 정상(파랑), 낙하(빨강)
        self._sc_trail_ok   = self.ax_side.scatter([], [], c="steelblue", s=12,
                                                    alpha=0.5, zorder=3)
        self._sc_trail_fall = self.ax_side.scatter([], [], c="red", s=12,
                                                    alpha=0.7, zorder=5)
        # 현재 무게중심
        self._head = self.ax_side.scatter([], [], s=150, zorder=6, marker="*")
        self.ax_side.legend(fontsize=7, loc="upper right")

        # ── 오른쪽: Z-time 뷰 ─────────────────────────────────────
        self.ax_z = self.fig.add_subplot(1, 2, 2)
        self.ax_z.set_title("Height over time  (Z)")
        self.ax_z.set_xlabel("Frame")
        self.ax_z.set_ylabel("Z (m)  height")
        self.ax_z.set_ylim(Z_RANGE[0] - 0.05, Z_RANGE[1] + 0.05)
        self.ax_z.axhline(0, color="brown", linewidth=1.2,
                           linestyle="--", alpha=0.6)
        self.ax_z.grid(True, alpha=0.3)

        self._line_z_ok,   = self.ax_z.plot([], [], "steelblue", linewidth=1.5,
                                              label="stable")
        self._line_z_fall, = self.ax_z.plot([], [], "red", linewidth=2.5,
                                              label="falling")
        self._head_z = self.ax_z.scatter([], [], s=60, zorder=5)
        self.ax_z.legend(fontsize=8)

        self._status = self.fig.text(0.5, 0.01, "Waiting for sensor...",
                                     ha="center", fontsize=9, color="gray")

        # 낙하 감지 시 화면 중앙에 표시되는 오버레이
        self._overlay = self.fig.text(
            0.5, 0.5, "",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="red",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor="red", linewidth=2, alpha=0.92),
            visible=False, zorder=10,
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])

    @property
    def is_paused(self) -> bool:
        return self._paused

    def resume(self) -> None:
        """낙하 감지 일시정지 해제 — 이력을 초기화하고 관측을 재개한다."""
        self._paused = False
        self._prev_falling = False
        self._trail.clear()
        self._detector.reset()
        self._overlay.set_visible(False)
        # 산점도·라인 초기화
        for artist in (self._sc_all, self._sc_cluster,
                       self._sc_trail_ok, self._sc_trail_fall):
            artist.set_offsets(np.empty((0, 2)))
        self._head.set_offsets(np.empty((0, 2)))
        self._line_z_ok.set_data([], [])
        self._line_z_fall.set_data([], [])
        self._head_z.set_offsets(np.empty((0, 2)))
        self._status.set_text("재개 — 관측 중...")
        self._status.set_color("gray")

    def update(self, frame: dict):
        if self._paused:
            return

        if "error" in frame:
            self._status.set_text(f"Error: {frame['error']}")
            return

        self._frame_num = frame.get("frame_number", self._frame_num + 1)

        # ── detect.py와 동일한 파이프라인 ────────────────────────
        pc_all = (PointCloud(frame.get("points", []))
                  .filter_snr(MIN_SNR)
                  .filter_roi(z_range=Z_RANGE))

        clusters = cluster_points(pc_all, eps=CLUSTER_EPS, min_samples=CLUSTER_MINSAMP)
        target   = select_target(clusters, airborne_z=AIRBORNE_Z)

        is_falling = self._detector.update(target)
        centroid   = target.centroid()

        if centroid is not None:
            self._trail.append((centroid[1], centroid[2], is_falling))

        # 낙하 감지 첫 프레임(상승 에지)에서 일시정지
        if is_falling and not self._prev_falling:
            self._paused = True
            # 착지 소실 경우에도 마지막 위치를 사용
            c = self._detector.last_fall_centroid
            pos_str = (f"X={c[0]:+.2f} m  Y={c[1]:.2f} m"
                       if c is not None else "position unknown")
            self._overlay.set_text(
                f"FALL DETECTED\n{pos_str}\n\npress any key to resume"
            )
            self._overlay.set_visible(True)
        self._prev_falling = is_falling

        # ── 측면 뷰 업데이트 ──────────────────────────────────────

        # 전체 필터링 포인트
        if len(pc_all) > 0:
            self._sc_all.set_offsets(np.c_[pc_all.xyz[:, 1], pc_all.xyz[:, 2]])
        else:
            self._sc_all.set_offsets(np.empty((0, 2)))

        # 추적 클러스터 포인트
        if len(target) > 0:
            self._sc_cluster.set_offsets(np.c_[target.xyz[:, 1], target.xyz[:, 2]])
        else:
            self._sc_cluster.set_offsets(np.empty((0, 2)))

        # 무게중심 궤적
        if self._trail:
            ys  = np.array([p[0] for p in self._trail])
            zs  = np.array([p[1] for p in self._trail])
            fm  = np.array([p[2] for p in self._trail])

            ok_pts   = np.c_[ys[~fm], zs[~fm]] if (~fm).any() else np.empty((0, 2))
            fall_pts = np.c_[ys[fm],  zs[fm]]  if fm.any()   else np.empty((0, 2))
            self._sc_trail_ok.set_offsets(ok_pts)
            self._sc_trail_fall.set_offsets(fall_pts)

            head_color = "red" if is_falling else "steelblue"
            self._head.set_offsets([[ys[-1], zs[-1]]])
            self._head.set_color(head_color)

        # ── Z-time 뷰 업데이트 ────────────────────────────────────
        if len(self._trail) >= 2:
            n       = len(self._trail)
            n_start = max(0, self._frame_num - n + 1)
            frames  = np.arange(n_start, n_start + n)
            zs      = np.array([p[1] for p in self._trail])
            fm      = np.array([p[2] for p in self._trail])

            self._line_z_ok.set_data(frames, zs)

            fall_f = frames.astype(float).copy()
            fall_z = zs.copy()
            fall_f[~fm] = np.nan
            fall_z[~fm] = np.nan
            self._line_z_fall.set_data(fall_f, fall_z)

            self._head_z.set_offsets([[frames[-1], zs[-1]]])
            self._head_z.set_color("red" if is_falling else "steelblue")
            self.ax_z.set_xlim(frames[0], frames[0] + max(20, n))

        # ── 상태 텍스트 ───────────────────────────────────────────
        n_clusters = len(clusters)
        n_target   = len(target)
        z_vel      = self._detector.z_velocity()

        if centroid is not None:
            doppler = float(np.mean(target.doppler)) if n_target > 0 else 0.0
            state   = "*** FALLING ***" if is_falling else "stable"
            color   = "red" if is_falling else "gray"
            self._status.set_text(
                f"Frame #{self._frame_num}"
                f"  |  all={len(pc_all)}pts  clusters={n_clusters}"
                f"  |  target={n_target}pts"
                f"  |  Z={centroid[2]:.2f}m  dZ/dt={z_vel:+.2f}m/s"
                f"  |  [{state}]"
            )
        else:
            self._status.set_text(
                f"Frame #{self._frame_num}"
                f"  |  all={len(pc_all)}pts  clusters={n_clusters}"
                f"  |  no target"
            )
            color = "gray"
        self._status.set_color(color)

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ── 진입점 ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    frame_q  = queue.Queue()
    stop_evt = threading.Event()

    threading.Thread(
        target=sensor_thread,
        args=(args.cli_port, args.data_port, args.config, frame_q, stop_evt),
        daemon=True,
    ).start()

    plotter = FallTrajectoryPlotter(debug=args.debug)
    logger.info("Fall trajectory viewer started  (close window or Ctrl+C to quit)")

    def on_key(event):
        if plotter.is_paused:
            plotter.resume()

    plotter.fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        while plt.fignum_exists(plotter.fig.number):
            try:
                frame = frame_q.get(timeout=0.1)
                # 일시정지 중에도 큐를 소비해 적체를 막되, 화면은 갱신하지 않음
                if not plotter.is_paused:
                    plotter.update(frame)
            except queue.Empty:
                pass
            plotter.draw()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        plt.close("all")
        logger.info("Done")


if __name__ == "__main__":
    main()
