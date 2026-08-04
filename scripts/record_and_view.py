"""레이더 데이터 녹화 + 궤적 시각화 (다중 추적) — record.py/record_raw.py를
이 스크립트 하나로 통합했다.

기본 동작: 녹화 후 현재 config(SNR/ROI/클러스터링) 파이프라인 그대로
화면에 시각화만 하고 끝난다 — 아무것도 저장하지 않는다.

--label을 지정하면 라벨링 작업(예: data/labeling_worksheet.csv)에 바로
쓸 수 있게, 필터 이전 원시 포인트(json)와 시각화 이미지(png)를 함께
data/raw/<라벨>/ 아래 저장한다 — 이전엔 record_raw.py로 녹화하고 시각화는
별도 스크립트로 다시 만들어야 했는데, 한 번에 끝난다. 저장된 raw json은
scripts/analyze_drops.py로 나중에 다른 필터 설정으로도 재생할 수 있다
(스키마: {"meta": {...}, "frames": [{"t", "frame", "points"}, ...]}).

Left  : Z over time — 모든 프레임의 포인트·트랙별 높이 변화 (색=트랙 ID)
Right : 트랙별 무게중심 Z / X / Y 시계열 (3개 패널, 위→아래)

실행:
    uv run scripts/record_and_view.py                    # 3초 녹화, 화면 표시만
    uv run scripts/record_and_view.py --duration 10       # 녹화 시간 조절
    uv run scripts/record_and_view.py --label cover_test  # 원시 데이터+이미지 저장
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
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
from arda.utils import load_processing_config

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"

# detect.py와 동일하게 config/settings.yaml의 processing: 섹션에서 읽어온다.
_cfg = load_processing_config()
MIN_SNR         = _cfg["min_snr"]
CLUSTER_EPS     = _cfg["cluster_eps"]
CLUSTER_MINSAMP = _cfg["cluster_min_samples"]
ROI_X           = _cfg["roi_x"]
ROI_Y           = _cfg["roi_y"]
Z_RANGE         = _cfg["roi_z"]
MAX_JUMP        = _cfg["max_jump"]

# 클러스터별 구분 색상 (최대 5개 클러스터)
CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]


# ── 1단계: 녹화 ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="레이더 녹화 + 궤적 시각화 (--label 지정 시 원시 데이터·이미지도 저장)")
    p.add_argument("--cli-port",  default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--duration",  type=float, default=3.0,
                   help="녹화 시간 (초, 기본 3)")
    p.add_argument("--label",     default=None,
                   help="지정하면 data/raw/<라벨>/record_raw_YYYYMMDD_HHMMSS.{json,png}로 "
                        "원시 포인트와 시각화 이미지를 저장한다 (미지정 시 화면 표시만 하고 저장 안 함)")
    return p.parse_args()


def record(args) -> tuple[list[dict], list[dict]]:
    """센서에서 데이터를 수집한다.

    반환값 (frames, raw_records):
      - frames: 현재 config로 필터·클러스터링·추적까지 마친, 시각화용 프레임 리스트
      - raw_records: 필터 이전 원시 포인트 리스트 (--label 저장용 —
        기존 record_raw.py와 동일 스키마)
    """
    sensor   = IWR6843Sensor(args.cli_port, args.data_port)
    detector = FallDetector(max_jump=MAX_JUMP)
    sensor.configure(args.config)

    frames      = []
    raw_records = []
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

                points = frame.get("points", [])

                # 동일 파이프라인
                pc_all   = (PointCloud(points)
                            .filter_snr(MIN_SNR)
                            .filter_roi(x_range=ROI_X, y_range=ROI_Y, z_range=Z_RANGE))
                clusters = cluster_points(pc_all, eps=CLUSTER_EPS,
                                          min_samples=CLUSTER_MINSAMP)

                # 다중 추적: 클러스터마다 독립 트랙으로 갱신
                if first_fall_t is None:
                    is_falling = detector.update(clusters)
                    if is_falling:
                        first_fall_t = elapsed
                else:
                    # 최초 낙하 이후에는 판정 생략 (추적은 계속)
                    detector.update(clusters)
                    is_falling = False

                # 트랙별 스냅샷 — 어느 트랙이 어느 클러스터를 물고 있는지,
                # 그 트랙이 낙하로 확정됐는지까지 프레임마다 기록해둔다
                # (시각화에서 트랙별로 일관된 색으로 이어 그리기 위함).
                track_snapshots = [
                    {
                        "id":         t.id,
                        "xyz":        t.last_cluster.xyz.copy(),
                        "centroid":   t.last_centroid.copy(),
                        "fell":       t.fell,
                        "confidence": t.confidence,
                    }
                    for t in detector.tracks
                    if t.last_cluster is not None and len(t.last_cluster) > 0
                    and t.last_centroid is not None
                ]

                frames.append({
                    "t":          elapsed,
                    "all_xyz":    pc_all.xyz.copy() if len(pc_all) > 0
                                  else np.empty((0, 3)),
                    "clusters":   [c.xyz.copy() for c in clusters],
                    "tracks":     track_snapshots,
                    "is_falling": bool(is_falling),
                })

                # 필터 이전 원시 포인트 — --label 저장용 (record_raw.py와 동일 스키마)
                raw_records.append({
                    "t":      round(elapsed, 4),
                    "frame":  frame.get("frame_number", len(raw_records)),
                    "points": [
                        {"x": round(p["x"], 3), "y": round(p["y"], 3),
                         "z": round(p["z"], 3), "doppler": round(p["doppler"], 3),
                         "snr": round(p["snr"], 1)}
                        for p in points
                    ],
                })

                bar  = "#" * int(elapsed / args.duration * 40)
                if first_fall_t is not None:
                    fall = f"  [FALL @{first_fall_t:.2f}s — tracking]"
                else:
                    fall = ""
                print(f"\r[{bar:<40}] {elapsed:4.1f}s  "
                      f"pts={len(pc_all):3d}  clusters={len(clusters)}  "
                      f"tracks={len(detector.tracks)}{fall}", end="")

    except KeyboardInterrupt:
        print("\n[REC] interrupted")

    print(f"\n[REC] {len(frames)} frames collected\n")
    return frames, raw_records


def save_raw(raw_records: list[dict], duration: float, label: str) -> Path:
    """원시 포인트를 record_raw.py와 동일한 형식으로 data/raw/<라벨>/ 아래 저장."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("data/raw") / label / f"record_raw_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "duration_s": round(duration, 2),
                "n_frames":   len(raw_records),
                "filtered":   False,
            },
            "frames": raw_records,
        }, f, indent=2)
    return out_path


# ── 2단계: 시각화 ─────────────────────────────────────────────────────────────

def visualize(frames: list[dict], save_path: Path | None = None):
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

    track_ids = sorted({tr["id"] for f in frames for tr in f["tracks"]})

    def _track_color(tid: int) -> str:
        return CLUSTER_COLORS[tid % len(CLUSTER_COLORS)]

    # 트랙별 "낙하로 확정된 첫 프레임"의 위치에 마커를 찍는다 — track.fell은
    # 한 번 확정되면 계속 True(래치)라서, 매 프레임 다시 표시하면 같은
    # 트랙에 마커가 중복으로 쌓여 정작 "몇 개의 서로 다른 트랙이 낙하로
    # 잡혔는지"가 안 보인다. record()가 최초 낙하 이후 프레임 단위
    # is_falling을 False로 고정해버리는 것과 무관하게, 트랙별 스냅샷
    # (track_snapshots)은 매 프레임 정확한 fell 상태를 담고 있으므로 여기서
    # 직접 훑는다. confidence는 확정 이후에도(반등 감지 등으로) 몇 프레임
    # 더 바뀔 수 있어(Track._check_post_trigger_rebound 참고), 마커 위치는
    # 첫 확정 프레임에 고정하되 라벨에 표시하는 신뢰도는 그 트랙에서 마지막
    # 으로 관측된(=반등 검사까지 끝난 뒤 안정화된) 값을 쓴다.
    fall_events: dict[int, dict] = {}
    for f in frames:
        for tr in f["tracks"]:
            if tr["fell"]:
                if tr["id"] not in fall_events:
                    fall_events[tr["id"]] = {"t": f["t"], "centroid": tr["centroid"],
                                              "confidence": tr.get("confidence", 0.0)}
                else:
                    fall_events[tr["id"]]["confidence"] = tr.get("confidence", fall_events[tr["id"]]["confidence"])

    def _draw_fall_markers(ax, coord_idx: int, marker_size: float):
        """트랙마다 낙하 확정 첫 시점에, 그 트랙과 같은 색으로 마커 + "T{id} nn%" 라벨을 찍는다.

        검은 테두리(marker="v", edgecolors="black")로 일반 트랙 점(테두리 없음)과
        구분되게 하고, 채움색은 트랙 색과 맞춰서 어느 트랙이 낙하로 확정됐는지
        바로 보이게 한다. 라벨의 퍼센트는 Track.confidence(0~1, 노이즈성 오판정과
        진짜 낙하를 사후에 가르는 학습된 신뢰도 — 통계적으로 완벽히 캘리브레이션된
        확률은 아님) 참고.
        """
        for tid, ev in fall_events.items():
            t, centroid, conf = ev["t"], ev["centroid"], ev["confidence"]
            color = _track_color(tid)
            ax.scatter([t], [centroid[coord_idx]], c=color, s=marker_size, marker="v",
                       edgecolors="black", linewidths=1.3, zorder=7)
            ax.annotate(f"T{tid} {conf:.0%}", (t, centroid[coord_idx]), textcoords="offset points",
                        xytext=(5, 6), fontsize=7.5, fontweight="bold", color=color, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor=color, linewidth=0.8, alpha=0.85))
            ax.axvline(t, color=color, lw=0.9, ls="--", alpha=0.55, zorder=1)

    def _apply_time_ticks(ax):
        """x축을 0.1s(프레임) 단위 세밀 눈금으로 설정."""
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        ax.grid(True, which="major", alpha=0.30, lw=0.8)
        ax.grid(True, which="minor", alpha=0.10, lw=0.5)

    # ── 왼쪽: Z over time ────────────────────────────────────────────────────
    # 색상은 DBSCAN 클러스터의 프레임별 임의 인덱스가 아니라 트랙 ID로 고정
    # 한다 — 그래야 같은 물체가 여러 프레임에 걸쳐 같은 색으로 이어지고,
    # 트랙이 잘못 갈아타는 문제(다중 추적으로 없앤 하이재킹)가 있었다면
    # 색이 끊기는 것으로 바로 드러난다.
    ax_zt.set_title("Z over time  (gray=pts / color=track id / black-edge ▼=FALL, T{id} nn% confidence)")
    ax_zt.set_xlabel("Time (s)")
    ax_zt.set_ylabel("Z  height (m)")
    ax_zt.set_xlim(ts[0], ts[-1])
    ax_zt.set_ylim(Z_RANGE[0] - 0.05, Z_RANGE[1] + 0.05)
    ax_zt.axhline(0, color="brown", lw=1.2, ls="--", alpha=0.5, label="floor")
    _apply_time_ticks(ax_zt)

    for f in frames:
        pts = f["all_xyz"]
        if len(pts):
            ax_zt.scatter(np.full(len(pts), f["t"]), pts[:, 2],
                          c="lightgray", s=7, alpha=0.5, zorder=2)

    for f in frames:
        for tr in f["tracks"]:
            xyz = tr["xyz"]
            ax_zt.scatter(np.full(len(xyz), f["t"]), xyz[:, 2],
                          c=_track_color(tr["id"]), s=14, alpha=0.7, zorder=3)

    for tid in track_ids:
        pts_t = [(f["t"], tr["centroid"][2]) for f in frames for tr in f["tracks"] if tr["id"] == tid]
        if pts_t:
            ct, cz = zip(*pts_t)
            ax_zt.plot(ct, cz, color=_track_color(tid), lw=1.5, zorder=4, label=f"track#{tid}")
            ax_zt.scatter(ct, cz, c=_track_color(tid), s=20, zorder=5)

    _draw_fall_markers(ax_zt, coord_idx=2, marker_size=160)

    handles, labels = ax_zt.get_legend_handles_labels()
    if fall_events:
        fall_proxy = plt.Line2D([], [], marker="v", markersize=8, linestyle="none",
                                 markerfacecolor="lightgray", markeredgecolor="black",
                                 markeredgewidth=1.3, label="FALL (black edge, fill=track color)")
        handles.append(fall_proxy)
    ax_zt.legend(handles=handles, fontsize=7, loc="upper right")

    # ── 오른쪽: Z / X / Y 시계열 ─────────────────────────────────────────────

    def _plot_axis(ax, coord_idx, label, ylim=None, hlines=None):
        """공통 시계열 패널 그리기 — 트랙마다 고정된 색으로 이어 그린다."""
        # 전체 필터링 포인트 (회색)
        for f in frames:
            pts = f["all_xyz"]
            if len(pts):
                ax.scatter(np.full(len(pts), f["t"]), pts[:, coord_idx],
                           c="lightgray", s=5, alpha=0.4, zorder=2)

        # 트랙별 무게중심
        for tid in track_ids:
            pts_t = [(f["t"], tr["centroid"]) for f in frames for tr in f["tracks"] if tr["id"] == tid]
            if pts_t:
                ct2 = [r[0] for r in pts_t]
                cv  = [float(r[1][coord_idx]) for r in pts_t]
                ax.plot(ct2, cv, color=_track_color(tid), lw=1.5, zorder=4)
                ax.scatter(ct2, cv, c=_track_color(tid), s=14, zorder=5)

        _draw_fall_markers(ax, coord_idx=coord_idx, marker_size=110)

        # 수평 보조선
        if hlines:
            for y_val, ls, alpha, lbl in hlines:
                ax.axhline(y_val, color="brown", lw=1.0, ls=ls, alpha=alpha)

        ax.set_ylabel(label)
        ax.set_xlim(ts[0], ts[-1])
        if ylim:
            ax.set_ylim(*ylim)
        _apply_time_ticks(ax)

    _plot_axis(ax_z, 2, "Z  height (m)",
               ylim=(Z_RANGE[0] - 0.05, Z_RANGE[1] + 0.05),
               hlines=[(0, "--", 0.5, "floor")])
    ax_z.set_title("Track centroids  Z / X / Y  over time  (color=track id)")

    _plot_axis(ax_x, 0, "X  left-right (m)")
    ax_x.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)

    _plot_axis(ax_y, 1, "Y  distance (m)")
    ax_y.set_xlabel("Time (s)")

    # 맨 위·중간 패널 x축 눈금 숨김
    plt.setp(ax_z.get_xticklabels(), visible=False)
    plt.setp(ax_x.get_xticklabels(), visible=False)

    if save_path is not None:
        fig.savefig(save_path, dpi=140)

    plt.show()


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    frames, raw_records = record(args)

    save_path = None
    if args.label:
        duration = frames[-1]["t"] if frames else 0.0
        json_path = save_raw(raw_records, duration, args.label)
        save_path = json_path.with_name(json_path.stem + "_view.png")
        print(f"[SAVE] 원시 데이터: {json_path}")

    visualize(frames, save_path=save_path)

    if save_path is not None:
        print(f"[SAVE] 시각화 이미지: {save_path}")


if __name__ == "__main__":
    main()
