"""저장된 녹화 하나를 다시 불러와, 낙하 물체라고 생각되는 궤적을 마우스로
직접 그려서 트랙을 만드는 라벨링 보정 도구.

왜 필요한가: 자동 다중 추적(FallDetector)이 낙하 물체를 여러 트랙으로
쪼개거나(dropball 등) 도중에 다른 클러스터로 갈아타는(하이재킹) 경우,
자동 트랙 경계만으로는 그 사건의 올바른 전체 궤적을 하나로 얻을 수 없다
(예: covered/trial12에서 track3이 낙하 확정 후 다른 노이즈에 흡수되고
실제 낙하 물체는 track4가 되어버린 경우). 이 도구는 그런 사례를 사람이
직접 봐가며 보정해서, 깨끗한 학습 데이터를 추가로 얻기 위한 것이다.

동작 방식:
  1. Z(t) 그래프에 프레임별 전체 포인트(회색)와 DBSCAN 클러스터 중심점을
     띄운다.
  2. 마우스로 낙하 물체라고 생각되는 궤적을 드래그해서 그린다.
  3. 그린 궤적의 각 프레임 시각에서, 그 프레임의 클러스터 "중심점" 중
     Z가 가장 가까운 것을 골라 매칭한다 — X/Y가 이미 뭉쳐있는 실제
     클러스터 후보 안에서만 고르므로, raw 포인트 중에서 고르는 것보다
     공간적으로 말이 안 되는 매칭(엉뚱한 노이즈를 줍는 것)을 줄인다.
  4. 매칭된 클러스터 시퀀스를 실제 프로덕션 Track 클래스에 그대로
     흘려보낸다(FallDetector.update()가 매 프레임 하는 것과 동일하게
     track.last_cluster + track.update(centroid)를 호출) — 학습용 특징을
     별도 코드로 재계산하지 않고 실제 배포 코드를 그대로 태우는 이유는,
     이전에 한 번 겪은 학습/서빙 특징 시점 불일치 버그(fall_detector.py
     _MODEL_FEATURE_ORDER 주석 참고)를 다시 만들지 않기 위해서다.
  5. 결과 특징·신뢰도를 출력하고, data/labeling_worksheet.csv에 한 행을
     추가한다 (label/label_note는 기존 워크플로우와 동일하게 비워두고
     사람이 직접 채운다).

조작:
  드래그      궤적 그리기 (다시 드래그하면 이전 궤적을 덮어씀)
  r           지우고 다시 그리기
  Enter       확정 — 저장하고 종료
  q / 창 닫기  취소 — 아무것도 저장하지 않음

실행:
    uv run scripts/retrace_track.py data/raw/covered/record_raw_20260728_154137.json --batch covered --trial 3
"""

import argparse
import csv
import sys
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points
from arda.detection.fall_detector import (
    Track, FRAME_DT, _MODEL_FEATURE_ORDER, POST_TRIGGER_CHECK_FRAMES, TRACK_MAX_MISSES,
)
from arda.utils import load_processing_config

_cfg = load_processing_config()
MIN_SNR         = _cfg["min_snr"]
CLUSTER_EPS     = _cfg["cluster_eps"]
CLUSTER_MINSAMP = _cfg["cluster_min_samples"]
ROI_X           = _cfg["roi_x"]
ROI_Y           = _cfg["roi_y"]
Z_RANGE         = _cfg["roi_z"]

CSV_COLUMNS = [
    "batch", "trial", "file", "track_id", "view_png", "reason",
    "total_obs_count", "peak_z_smoothed", "net_drop_from_peak", "avg_descent_speed",
    "recent_velocity", "recent_accel", "z_max_ever", "z_min_ever", "avg_pts", "max_pts",
    "rebound_penalized", "suggested_label", "suggested_note", "label", "label_note",
]


# ── 1단계: 녹화 로드 + 재생 ───────────────────────────────────────────────────

def load_frames(path: Path, near_y_max: float | None = None,
                 min_doppler_mag: float | None = None) -> list[dict]:
    """analyze_drops.py의 replay()와 동일한 전처리(SNR/ROI 필터 → DBSCAN)로
    프레임별 전체 포인트·클러스터 후보를 뽑는다. near_y_max/min_doppler_mag는
    analyze_drops.py와 동일한 의미(사람 등 배경 후보 사전 배제용)."""
    data = json.load(path.open())
    frames = []
    for fr in data["frames"]:
        pc = (PointCloud(fr["points"])
              .filter_snr(MIN_SNR)
              .filter_roi(x_range=ROI_X, y_range=ROI_Y, z_range=Z_RANGE))
        clusters = cluster_points(pc, eps=CLUSTER_EPS, min_samples=CLUSTER_MINSAMP)

        candidates = clusters
        if near_y_max is not None:
            candidates = [c for c in candidates if c.centroid()[1] <= near_y_max]
        if min_doppler_mag is not None:
            candidates = [c for c in candidates
                          if abs(float(np.mean(c.doppler))) >= min_doppler_mag]

        frames.append({
            "t": fr["t"],
            "all_xyz": pc.xyz.copy() if len(pc) > 0 else np.empty((0, 3)),
            "clusters": candidates,
        })
    return frames


# ── 2단계: 마우스로 궤적 그리기 ────────────────────────────────────────────────

class TraceTool:
    """Z(t) 그래프 위에서 마우스 드래그로 궤적을 그리고, 프레임별 DBSCAN
    클러스터 중심점에 매칭해 보여주는 인터랙티브 도구."""

    def __init__(self, frames: list[dict], title: str, save_path: Path):
        self.frames = frames
        self.save_path = save_path
        self.drawn: list[tuple[float, float]] = []
        self.drawing = False
        self.matched: list[tuple[int, PointCloud | None]] | None = None
        self.confirmed = False

        self.fig, self.ax = plt.subplots(figsize=(15, 7))
        self._plot_background(title)

        self.draw_line,  = self.ax.plot([], [], color="black", lw=1.8, alpha=0.8, zorder=6,
                                          label="손으로 그린 궤적")
        self.match_line, = self.ax.plot([], [], "o-", color="#e74c3c", lw=2.4, ms=7, zorder=7,
                                          label="매칭된 트랙 (클러스터 중심)")
        self.ax.legend(fontsize=8, loc="upper right")

        self.status = self.ax.text(
            0.01, 0.99, "", transform=self.ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9), zorder=10)
        self._set_status("드래그로 낙하 물체 궤적을 그리세요  (Enter=확정 / r=다시 그리기 / q=취소)")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _plot_background(self, title: str):
        for f in self.frames:
            pts = f["all_xyz"]
            if len(pts):
                self.ax.scatter(np.full(len(pts), f["t"]), pts[:, 2],
                                 c="lightgray", s=8, alpha=0.5, zorder=2)
        for f in self.frames:
            for c in f["clusters"]:
                cen = c.centroid()
                if cen is not None:
                    self.ax.scatter([f["t"]], [cen[2]], c="#2c3e50", s=45,
                                     marker="o", edgecolors="white", linewidths=0.6, zorder=4)
        self.ax.axhline(0, color="brown", lw=1, ls="--", alpha=0.5)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Z (m)")
        self.ax.set_title(f"{title}  (회색=전체 포인트, 남색 원=DBSCAN 클러스터 중심)")
        self.ax.grid(alpha=0.3)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self.drawing = True
        self.drawn = [(event.xdata, event.ydata)]
        self._redraw_stroke()

    def _on_move(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.drawn.append((event.xdata, event.ydata))
        self._redraw_stroke()

    def _on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        self._match()

    def _redraw_stroke(self):
        if self.drawn:
            ts, zs = zip(*self.drawn)
            self.draw_line.set_data(ts, zs)
        self.fig.canvas.draw_idle()

    def _match(self):
        """그려진 궤적의 각 프레임 시각에서 가장 가까운 클러스터 중심을 고른다."""
        if len(self.drawn) < 2:
            self._set_status("궤적이 너무 짧습니다 — 다시 드래그해서 그려주세요")
            self.fig.canvas.draw_idle()
            return

        pts = sorted(self.drawn, key=lambda p: p[0])
        ts_drawn = [p[0] for p in pts]
        zs_drawn = [p[1] for p in pts]
        t_min, t_max = ts_drawn[0], ts_drawn[-1]

        matched: list[tuple[int, PointCloud | None]] = []
        for idx, f in enumerate(self.frames):
            t = f["t"]
            if t < t_min or t > t_max or not f["clusters"]:
                matched.append((idx, None))
                continue
            z_interp = float(np.interp(t, ts_drawn, zs_drawn))
            nearest = min(f["clusters"], key=lambda c: abs(float(c.centroid()[2]) - z_interp))
            matched.append((idx, nearest))

        self.matched = matched
        mt = [self.frames[idx]["t"] for idx, c in matched if c is not None]
        mz = [float(c.centroid()[2]) for _, c in matched if c is not None]
        self.match_line.set_data(mt, mz)
        self._set_status(f"매칭 {len(mt)}프레임 — Enter로 확정 / r로 다시 그리기 / q로 취소")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "r":
            self.drawn = []
            self.matched = None
            self.draw_line.set_data([], [])
            self.match_line.set_data([], [])
            self._set_status("드래그로 다시 그리세요")
            self.fig.canvas.draw_idle()
        elif event.key == "enter":
            if self.matched is None:
                self._set_status("먼저 궤적을 그려주세요")
                self.fig.canvas.draw_idle()
                return
            self.confirmed = True
            # plt.close()가 창을 파괴하기 전에, 확정된 화면 그대로 저장해둔다
            # (닫힌 뒤에는 캔버스가 이미 해체돼 savefig가 안전하지 않다).
            self.fig.savefig(self.save_path, dpi=140)
            plt.close(self.fig)
        elif event.key == "q":
            self.confirmed = False
            plt.close(self.fig)

    def _set_status(self, msg: str):
        self.status.set_text(msg)

    def run(self) -> list[tuple[int, PointCloud | None]] | None:
        plt.show()
        return self.matched if self.confirmed else None


# ── 3단계: 매칭 결과 → 실제 Track 클래스로 특징 계산 ──────────────────────────

def build_manual_track(frames: list[dict],
                        matched: list[tuple[int, PointCloud | None]]) -> tuple[Track | None, list[float] | None]:
    """매칭된 클러스터 시퀀스를 FallDetector.update()가 매 프레임 하는 것과
    똑같이 실제 Track 인스턴스에 흘려보낸다 — 학습 특징을 별도 코드로
    재계산하지 않고 배포 코드 그대로 태워서, 학습/서빙 시점 불일치를
    원천적으로 피한다.

    특징 스냅샷은 반드시 "확정되는 바로 그 프레임"에서 떠야 한다
    (_MODEL_FEATURE_ORDER 위 주석 참고 — 트리거 순간과 다른 시점 값을 섞으면
    학습/서빙이 어긋난다). 트리거 이후에도 반등 체크(POST_TRIGGER_CHECK_FRAMES)를
    태우려고 몇 프레임 더 먹이는데, _height_history/_raw_height_history는
    maxlen=history_window(10)짜리 deque라 트리거 이후 미매칭(None) 프레임을
    계속 먹이면 원래 있던 값이 뒤로 밀려 지워진다 — 그래서 여기서 직접
    _unified_features()를 "지금 당장" 호출해 스냅샷을 떠 두고, 이후 몇 프레임을
    더 먹이더라도 이 스냅샷은 절대 다시 계산하지 않는다. 트리거가 아예 안 된
    경우(자동 판정 로직이 이 궤적에서 낙하를 확정 못한 진단용 사례)엔
    반등 체크가 의미가 없으니, 프로덕션이면 이미 트랙을 삭제했을 시점
    (TRACK_MAX_MISSES 연속 미매칭)에서 멈추고 그 시점 값을 스냅샷한다.
    """
    start = next((i for i, (_, c) in enumerate(matched) if c is not None), None)
    if start is None:
        return None, None

    track = Track(track_id=0)
    trigger_feats: list[float] | None = None
    consecutive_misses = 0

    for _, c in matched[start:]:
        if c is not None:
            track.last_cluster = c
            fell = track.update(c.centroid())
            consecutive_misses = 0
        else:
            track.last_cluster = None
            fell = track.update(None)
            consecutive_misses += 1

        if fell and trigger_feats is None:
            trigger_feats = track._unified_features()

        if trigger_feats is not None:
            if track._post_trigger_count >= POST_TRIGGER_CHECK_FRAMES or consecutive_misses > TRACK_MAX_MISSES:
                break
        elif consecutive_misses > TRACK_MAX_MISSES:
            break

    feats = trigger_feats if trigger_feats is not None else track._unified_features()
    return track, feats


def append_csv(csv_path: Path, batch: str, trial: str, file_name: str, view_png: str,
                reason: str, feats: list[float], fell: bool, likely_real: bool):
    (total_obs, peak_z, net_drop, avg_desc, recent_v, recent_a,
     z_max, z_min, avg_pts, max_pts, rebound) = feats

    if fell:
        suggested_label = "1" if likely_real else "0"
        suggested_note = ""
    else:
        suggested_label = ""
        suggested_note = "자동 판정 로직이 이 수동 궤적에서 낙하를 확정하지 못함 — 임계값 재검토 여지"

    row = {
        "batch": batch, "trial": trial, "file": file_name, "track_id": "manual",
        "view_png": view_png, "reason": reason,
        "total_obs_count": int(total_obs), "peak_z_smoothed": round(peak_z, 4),
        "net_drop_from_peak": round(net_drop, 4), "avg_descent_speed": round(avg_desc, 4),
        "recent_velocity": round(recent_v, 4), "recent_accel": round(recent_a, 4),
        "z_max_ever": round(z_max, 4), "z_min_ever": round(z_min, 4),
        "avg_pts": round(avg_pts, 4), "max_pts": int(max_pts),
        "rebound_penalized": bool(rebound),
        "suggested_label": suggested_label, "suggested_note": suggested_note,
        "label": "", "label_note": "",
    }

    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ── 진입점 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="녹화 재생 위에 마우스로 궤적을 그려 수동 트랙을 만드는 라벨링 보정 도구")
    p.add_argument("file", help="data/raw/<배치>/record_raw_*.json 경로 (한 번에 파일 하나)")
    p.add_argument("--batch", default=None, help="labeling_worksheet.csv의 batch 값 (기본: 파일의 상위 폴더명)")
    p.add_argument("--trial", default="", help="labeling_worksheet.csv의 trial 값 (기본: 비움 — 직접 채워도 됨)")
    p.add_argument("--csv", default="data/labeling_worksheet.csv", help="행을 추가할 CSV 경로")
    p.add_argument("--near-y-max", type=float, default=None,
                   help="이 거리(m)보다 먼 클러스터는 후보에서 제외 (analyze_drops.py와 동일)")
    p.add_argument("--min-doppler-mag", type=float, default=None,
                   help="|도플러|가 이보다 작은 클러스터는 후보에서 제외 (analyze_drops.py와 동일)")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.file)
    if not src.exists():
        print(f"[오류] 파일 없음: {src}")
        return

    frames = load_frames(src, near_y_max=args.near_y_max, min_doppler_mag=args.min_doppler_mag)
    batch = args.batch or src.parent.name
    save_name = f"{src.stem}_manual_track.png"
    save_path = src.parent / save_name

    tool = TraceTool(frames, title=f"{batch}/{src.name}", save_path=save_path)
    matched = tool.run()
    if matched is None:
        print("[취소] 아무것도 저장하지 않았습니다.")
        return

    track, feats = build_manual_track(frames, matched)
    if track is None:
        print("[오류] 매칭된 클러스터가 하나도 없습니다 — 클러스터가 있는 구간을 다시 그려주세요.")
        return

    if track.fell:
        confidence = track.confidence
        reason = track._trigger_reason
    else:
        confidence = track._model_confidence()
        reason = ""

    print(f"\n[결과] fell={track.fell}  confidence={confidence:.2f}"
          f"{'  (likely_real)' if track.fell and track.likely_real else ''}")
    if reason:
        print(f"  reason: {reason}")
    else:
        print("  (현재 임계값으로는 자동 확정되지 않음 — 참고용 특징만 계산됨)")
    for name, val in zip(_MODEL_FEATURE_ORDER, feats):
        print(f"  {name:>18s} = {val:.4f}")

    append_csv(Path(args.csv), batch=batch, trial=args.trial, file_name=src.name,
               view_png=f"{batch}/{save_name}", reason=reason, feats=feats,
               fell=track.fell, likely_real=track.likely_real)

    print(f"\n[저장] 이미지: {save_path}")
    print(f"[저장] {args.csv}에 행 추가 (track_id=manual) — label/label_note는 직접 채워주세요")


if __name__ == "__main__":
    main()
