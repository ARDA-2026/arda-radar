"""5초 녹화 후 포인트·클러스터 궤적 시각화.

Left  : Z over time — 모든 프레임의 포인트·클러스터·무게중심 높이 변화
Right : 타겟 무게중심 Z / X / Y 시계열 (3개 패널, 위→아래)

실행:
    uv run scripts/record_and_view.py
    uv run scripts/record_and_view.py --duration 5
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore", message="Glyph.*missing from font")

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points
from arda.detection import FallDetector

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"
MIN_SNR         = 6.0
CLUSTER_EPS     = 0.15
CLUSTER_MINSAMP = 2
Z_RANGE         = (-0.8, 0.8)
AIRBORNE_Z      = 0.40
MAX_JUMP        = 0.5   # m — 직전 프레임 무게중심 대비 허용 최대 이동 거리 (노이즈 튐 방지)

# 클러스터별 구분 색상 (최대 5개 클러스터)
CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]


# ── 1단계: 녹화 ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="녹화 후 포인트·클러스터 궤적 시각화")
    p.add_argument("--cli-port",  default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--duration",  type=float, default=3.0,
                   help="녹화 시간 (초, 기본 3)")
    return p.parse_args()


def record(args) -> list[dict]:
    """센서에서 데이터를 수집해 프레임 리스트로 반환한다."""
    sensor   = IWR6843Sensor(args.cli_port, args.data_port)
    detector = FallDetector()
    sensor.configure(args.config)

    frames  = []
    t_start = None
    first_fall_t = None   # 최초 낙하 감지 시각

    print(f"\n[REC] {args.duration:.0f}s recording — drop objects now!\n")

    try:
        with sensor:
            while True:
                frame = sensor.read_frame()
                if frame is None:
                    continue

                now = time.monotonic()
                if t_start is None:
                    t_start = now
                elapsed = now - t_start

                if elapsed > args.duration:
                    break

                # 동일 파이프라인
                pc_all   = (PointCloud(frame.get("points", []))
                            .filter_snr(MIN_SNR)
                            .filter_roi(z_range=Z_RANGE))
                clusters = cluster_points(pc_all, eps=CLUSTER_EPS,
                                          min_samples=CLUSTER_MINSAMP)

                target = detector.choose_target(clusters, airborne_z=AIRBORNE_Z,
                                                max_jump=MAX_JUMP)

                # 최초 낙하 이후에는 판정 생략 (추적은 계속)
                if first_fall_t is None:
                    is_falling = detector.update(target)
                    if is_falling:
                        first_fall_t = elapsed
                else:
                    is_falling = False

                frames.append({
                    "t":          elapsed,
                    "all_xyz":    pc_all.xyz.copy() if len(pc_all) > 0
                                  else np.empty((0, 3)),
                    "clusters":   [c.xyz.copy() for c in clusters],
                    "target_xyz": target.xyz.copy() if len(target) > 0
                                  else np.empty((0, 3)),
                    "centroid":   target.centroid(),
                    "is_falling": bool(is_falling),
                })

                bar  = "#" * int(elapsed / args.duration * 40)
                if first_fall_t is not None:
                    fall = f"  [FALL @{first_fall_t:.2f}s — tracking]"
                else:
                    fall = ""
                print(f"\r[{bar:<40}] {elapsed:4.1f}s  "
                      f"pts={len(pc_all):3d}  clusters={len(clusters)}{fall}", end="")

    except KeyboardInterrupt:
        print("\n[REC] interrupted")

    print(f"\n[REC] {len(frames)} frames collected\n")
    return frames


# ── 2단계: 시각화 ─────────────────────────────────────────────────────────────

def visualize(frames: list[dict]):
    if not frames:
        print("[VIZ] No frames to visualize")
        return

    n  = len(frames)
    ts = [f["t"] for f in frames]

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"Recording  ({n} frames / {ts[-1]:.1f}s)", fontsize=12)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1],
                          hspace=0.55, wspace=0.32,
                          left=0.06, right=0.97, top=0.91, bottom=0.08)

    ax_zt = fig.add_subplot(gs[:, 0])          # 왼쪽: Z 공간 개요 (전체 높이)
    ax_z  = fig.add_subplot(gs[0, 1])          # 오른쪽 상: Z(t)
    ax_x  = fig.add_subplot(gs[1, 1])          # 오른쪽 중: X(t)
    ax_y  = fig.add_subplot(gs[2, 1])          # 오른쪽 하: Y(t)

    # 오른쪽 3개 패널 x축 공유
    ax_x.sharex(ax_z)
    ax_y.sharex(ax_z)

    fall_frames = [f for f in frames if f["is_falling"]]
    fall_ts     = [f["t"] for f in fall_frames]

    def _apply_time_ticks(ax):
        """x축을 0.1s(프레임) 단위 세밀 눈금으로 설정."""
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        ax.grid(True, which="major", alpha=0.30, lw=0.8)
        ax.grid(True, which="minor", alpha=0.10, lw=0.5)

    # ── 왼쪽: Z over time ────────────────────────────────────────────────────
    ax_zt.set_title("Z over time  (gray=pts / color=cluster / orange=target)")
    ax_zt.set_xlabel("Time (s)")
    ax_zt.set_ylabel("Z  height (m)")
    ax_zt.set_xlim(ts[0], ts[-1])
    ax_zt.set_ylim(Z_RANGE[0] - 0.05, Z_RANGE[1] + 0.05)
    ax_zt.axhline(0,          color="brown",     lw=1.2, ls="--", alpha=0.5, label="floor")
    ax_zt.axhline(AIRBORNE_Z, color="steelblue", lw=0.8, ls=":",  alpha=0.5,
                  label=f"airborne z={AIRBORNE_Z}m")
    _apply_time_ticks(ax_zt)

    for f in frames:
        pts = f["all_xyz"]
        if len(pts):
            ax_zt.scatter(np.full(len(pts), f["t"]), pts[:, 2],
                          c="lightgray", s=7, alpha=0.5, zorder=2)

    for f in frames:
        for ci, cxyz in enumerate(f["clusters"]):
            if len(cxyz):
                ax_zt.scatter(np.full(len(cxyz), f["t"]), cxyz[:, 2],
                              c=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)],
                              s=14, alpha=0.7, zorder=3)

    cent_tz = [(f["t"], f["centroid"][2]) for f in frames if f["centroid"] is not None]
    if cent_tz:
        ct, cz = zip(*cent_tz)
        ax_zt.plot(ct, cz, color="orange", lw=1.5, zorder=4, label="target centroid")
        ax_zt.scatter(ct, cz, c="orange", s=20, zorder=5)

    if fall_ts:
        fz = [f["centroid"][2] if f["centroid"] is not None else 0.0 for f in fall_frames]
        ax_zt.scatter(fall_ts, fz, c="red", s=100, marker="v", zorder=6, label="FALL")

    ax_zt.legend(fontsize=7, loc="upper right")

    # ── 오른쪽: Z / X / Y 시계열 ─────────────────────────────────────────────
    cent_txyz = [(f["t"], f["centroid"]) for f in frames if f["centroid"] is not None]

    def _plot_axis(ax, coord_idx, label, color, ylim=None, hlines=None):
        """공통 시계열 패널 그리기."""
        # 전체 필터링 포인트 (회색)
        for f in frames:
            pts = f["all_xyz"]
            if len(pts):
                ax.scatter(np.full(len(pts), f["t"]), pts[:, coord_idx],
                           c="lightgray", s=5, alpha=0.4, zorder=2)

        # 타겟 무게중심 (주황)
        if cent_txyz:
            ct2 = [r[0] for r in cent_txyz]
            cv  = [float(r[1][coord_idx]) for r in cent_txyz]
            ax.plot(ct2, cv, color=color, lw=1.8, zorder=4)
            ax.scatter(ct2, cv, c=color, s=18, zorder=5)

        # 낙하 감지 마커
        if fall_ts:
            fv = [float(f["centroid"][coord_idx]) if f["centroid"] is not None else np.nan
                  for f in fall_frames]
            ax.scatter(fall_ts, fv, c="red", s=80, marker="v", zorder=6)

        # 수평 보조선
        if hlines:
            for y_val, ls, alpha, lbl in hlines:
                ax.axhline(y_val, color="brown", lw=1.0, ls=ls, alpha=alpha)

        # 낙하 감지 수직선
        for ft in fall_ts:
            ax.axvline(ft, color="red", lw=0.8, ls="--", alpha=0.5)

        ax.set_ylabel(label)
        ax.set_xlim(ts[0], ts[-1])
        if ylim:
            ax.set_ylim(*ylim)
        _apply_time_ticks(ax)

    _plot_axis(ax_z, 2, "Z  height (m)",         "orange",
               ylim=(Z_RANGE[0] - 0.05, Z_RANGE[1] + 0.05),
               hlines=[(0, "--", 0.5, "floor"), (AIRBORNE_Z, ":", 0.4, "")])
    ax_z.set_title("Target centroid  Z / X / Y  over time")

    _plot_axis(ax_x, 0, "X  left-right (m)",      "steelblue")
    ax_x.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)

    _plot_axis(ax_y, 1, "Y  distance (m)",         "seagreen")
    ax_y.set_xlabel("Time (s)")

    # 맨 위·중간 패널 x축 눈금 숨김
    plt.setp(ax_z.get_xticklabels(), visible=False)
    plt.setp(ax_x.get_xticklabels(), visible=False)

    plt.show()


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    frames = record(args)
    visualize(frames)


if __name__ == "__main__":
    main()
