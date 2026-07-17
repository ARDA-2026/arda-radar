"""녹화된 낙하 데이터 여러 개를 일괄 재생·비교 분석하는 스크립트.

record.py로 녹화한 JSON 파일들(같은 물체를 여러 번 떨어뜨린 녹화)을 모아
'현재' 파이프라인(SNR/ROI 필터 → DBSCAN → choose_target → FallDetector)으로
다시 돌려, 시행별 궤적·판정 결과를 하나의 그래프로 비교하고 요약 통계를
출력한다. record.py가 녹화 당시 저장해둔 is_falling 값에 의존하지 않고
원시 포인트(points)를 다시 재생하므로, 이후 알고리즘을 바꿔도 같은
녹화로 재검증할 수 있다.

실행:
    uv run scripts/analyze_drops.py data/raw/drop_test_20260715
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points
from arda.detection import FallDetector
from arda.detection.fall_detector import (
    PEAK_Z_THRESHOLD, PEAK_DROP_THRESHOLD, MIN_DESCENT_FRAMES, RISING_TOLERANCE,
    FREEFALL_MIN_FRAMES, FREEFALL_ACCEL_MIN, FREEFALL_ACCEL_MAX, FRAME_DT,
)
from arda.utils import load_processing_config

# detect.py 등 실시간 스크립트와 동일하게 config/settings.yaml의
# processing: 섹션에서 읽어온다.
_cfg = load_processing_config()
MIN_SNR         = _cfg["min_snr"]
CLUSTER_EPS     = _cfg["cluster_eps"]
CLUSTER_MINSAMP = _cfg["cluster_min_samples"]
ROI_X           = _cfg["roi_x"]
ROI_Y           = _cfg["roi_y"]
Z_RANGE         = _cfg["roi_z"]
AIRBORNE_Z      = _cfg["airborne_z"]
MAX_JUMP        = _cfg["max_jump"]

TRIAL_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]


def replay(path: Path, near_y_max: float | None = None,
           min_doppler_mag: float | None = None) -> dict:
    """녹화 파일 하나를 현재 파이프라인으로 재생해 궤적/판정 결과를 반환한다.

    near_y_max가 주어지면, 타겟 선택(choose_target) 직전에 그보다 먼
    클러스터를 후보에서 제외한다 — 사람처럼 항상 더 먼 거리(Y)에서
    움직이는 물체가 근처에 같이 있을 때, 그 사람을 근본적으로 후보군에서
    배제하기 위한 것이다(단순 Z/도플러 기준보다 거리 기준이 훨씬 안정적
    으로 구분되는 경우가 있음 — data/raw/person_plus_drop_20260715 참고).

    min_doppler_mag가 주어지면, |도플러|가 그보다 작은(=거의 정지한)
    클러스터를 후보에서 제외한다 — 사람이 물체 바로 옆(비슷한 거리)에
    서 있어서 거리로는 구분이 안 될 때, 서 있는 사람은 도플러가 거의
    0에 가깝고 낙하 물체는 훨씬 크다는 점으로 구분한다
    (data/raw/person_close_drop_20260715 참고).

    두 필터 모두 시각화(all_pts_z/cluster_pts_z)에는 영향을 주지 않고
    원본 그대로 보여준다 — 무엇이 걸러졌는지 비교할 수 있게.
    """
    data = json.load(path.open())
    detector = FallDetector()

    ts, zs, falling = [], [], []
    fall_events = []
    all_pts_z    = []   # 프레임별 SNR/ROI 필터 통과 포인트의 Z 배열 (회색 산점도용)
    cluster_pts_z = []  # 프레임별 [클러스터별 Z 배열] (색상 산점도용)

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

        target = detector.choose_target(candidates, airborne_z=AIRBORNE_Z, max_jump=MAX_JUMP)
        fell = detector.update(target)

        # 이번 프레임에 실제 관측이 있었을 때만 z를 기록한다 — last_centroid는
        # 코스팅(예측만) 구간에도 마지막 값을 그대로 들고 있어 값을 그대로
        # 쓰면 "관측 없음"과 "정지 상태"를 구분할 수 없다.
        z = float(target.centroid()[2]) if len(target) > 0 else None
        ts.append(fr["t"])
        zs.append(z)
        falling.append(fell)
        if fell:
            fall_events.append(fr["t"])

        all_pts_z.append(pc.xyz[:, 2] if len(pc) > 0 else np.empty(0))
        cluster_pts_z.append([c.xyz[:, 2] for c in clusters])

    return {"name": path.stem, "ts": ts, "zs": zs, "falling": falling, "fall_events": fall_events,
            "all_pts_z": all_pts_z, "cluster_pts_z": cluster_pts_z}


def summarize(result: dict) -> dict:
    valid = [(t, z) for t, z in zip(result["ts"], result["zs"]) if z is not None]
    if not valid:
        return {"peak_z": None, "min_z": None, "drop": None, "n_active": 0,
                 "detected": bool(result["fall_events"]), "first_fall_t": None}
    zs_only = [z for _, z in valid]
    peak, trough = max(zs_only), min(zs_only)
    return {
        "peak_z": peak, "min_z": trough, "drop": peak - trough,
        "n_active": len(valid),
        "detected": bool(result["fall_events"]),
        "first_fall_t": result["fall_events"][0] if result["fall_events"] else None,
    }


def _diagnose_peak_drop(valid: list[tuple[int, float]]) -> str:
    """경로 1/2(피크-하강) 미감지 사유."""
    peak_pos   = max(range(len(valid)), key=lambda k: valid[k][1])
    peak_z     = valid[peak_pos][1]
    peak_frame = valid[peak_pos][0]
    if peak_z < PEAK_Z_THRESHOLD:
        return f"피크({peak_z:.2f}m) < 임계값({PEAK_Z_THRESHOLD}m)"

    post_peak = [z for i, z in valid if i > peak_frame]
    if len(post_peak) < MIN_DESCENT_FRAMES:
        return f"피크 이후 유효 프레임 {len(post_peak)}개 < {MIN_DESCENT_FRAMES}개"

    if any(z > peak_z for z in post_peak):
        return "피크 이후 반등(피크 초과) 있음"

    if len(post_peak) >= 2 and (post_peak[-1] - post_peak[-2]) > RISING_TOLERANCE:
        return "마지막 구간이 상승 중"

    drop = peak_z - post_peak[-1]
    if drop < PEAK_DROP_THRESHOLD:
        return f"하락폭({drop:.2f}m) < 임계값({PEAK_DROP_THRESHOLD}m)"

    return ""  # 경로 1/2 조건 충족 — 미감지 사유 아님


def _diagnose_freefall(valid: list[tuple[int, float]]) -> str:
    """경로 3(자유낙하) 미감지 사유 — fall_detector._freefall_check와 동일 로직."""
    if len(valid) < FREEFALL_MIN_FRAMES:
        return f"유효 프레임 {len(valid)}개 < {FREEFALL_MIN_FRAMES}개(자유낙하 판정 최소치)"

    recent = valid[-FREEFALL_MIN_FRAMES:]
    velocities, midpoints = [], []
    for (i0, z0), (i1, z1) in zip(recent, recent[1:]):
        dt = (i1 - i0) * FRAME_DT
        if dt <= 0:
            return "프레임 순서 이상(dt<=0)"
        velocities.append((z1 - z0) / dt)
        midpoints.append((i0 + i1) / 2.0 * FRAME_DT)

    if any(v2 >= v1 for v1, v2 in zip(velocities, velocities[1:])):
        return "구간 사이 가속이 끊김(반등/정체)"

    total_dt = midpoints[-1] - midpoints[0]
    accel = (velocities[-1] - velocities[0]) / total_dt if total_dt > 0 else 0.0
    if not (-FREEFALL_ACCEL_MAX <= accel <= -FREEFALL_ACCEL_MIN):
        return f"가속도({accel:+.1f}m/s²)가 자유낙하 범위(-{FREEFALL_ACCEL_MAX}~-{FREEFALL_ACCEL_MIN}) 밖"

    return ""  # 경로 3 조건 충족 — 미감지 사유 아님


def diagnose(result: dict) -> str:
    """미감지 사유를 경로 1/2(피크-하강)·경로 3(자유낙하) 둘 다 근사 진단한다.

    실제 FallDetector는 칼만 평활화된 위치로 판정하므로 여기(원시 target
    centroid 기준)와 미세하게 다를 수 있지만, 어느 게이트에서 막혔는지
    가늠하는 데는 충분하다.
    """
    valid = [(i, z) for i, z in enumerate(result["zs"]) if z is not None]
    if len(valid) < 2:
        return "유효 관측 2프레임 미만"

    peak_drop_reason = _diagnose_peak_drop(valid)
    freefall_reason   = _diagnose_freefall(valid)

    if not peak_drop_reason or not freefall_reason:
        return "조건 충족 (다른 원인으로 미감지 — 코드 직접 확인 필요)"

    return f"[피크-하강] {peak_drop_reason}  /  [자유낙하] {freefall_reason}"


def plot_points_clusters(results: list[dict], folder: Path) -> Path:
    """시행별로 시간에 따른 전체 포인트·DBSCAN 클러스터·타겟 궤적을 그린다.

    record_and_view.py의 "Z over time (gray=pts / color=cluster / orange=target)"
    패널을 여러 녹화 파일에 대해 재생 결과로 재현한 것 — 최종 타겟 Z(t) 하나만
    보여주는 analysis.png와 달리, 매 프레임 DBSCAN이 실제로 무엇을 봤는지(회색
    전체 포인트, 색상별 클러스터)까지 그대로 보여준다.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        for t, pz in zip(r["ts"], r["all_pts_z"]):
            if len(pz):
                ax.scatter(np.full(len(pz), t), pz, c="lightgray", s=10, alpha=0.5, zorder=2)

        for t, clist in zip(r["ts"], r["cluster_pts_z"]):
            for ci, cz in enumerate(clist):
                if len(cz):
                    ax.scatter(np.full(len(cz), t), cz,
                               c=TRIAL_COLORS[ci % len(TRIAL_COLORS)], s=18, alpha=0.8, zorder=3)

        valid_t = [t for t, z in zip(r["ts"], r["zs"]) if z is not None]
        valid_z = [z for z in r["zs"] if z is not None]
        ax.plot(valid_t, valid_z, "o-", color="orange", lw=1.5, ms=5, zorder=4, label="target")

        for ft in r["fall_events"]:
            ax.axvline(ft, color="red", ls="--", alpha=0.6)

        ax.axhline(0, color="brown", ls="--", lw=1, alpha=0.5)
        ax.axhline(PEAK_Z_THRESHOLD, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_title(f"{r['name']}  (회색=전체 포인트 / 색상=클러스터 / 주황=타겟, 빨간 점선=낙하 판정)",
                     fontsize=9)
        ax.set_ylabel("Z (m)")
        ax.set_ylim(-0.9, 0.9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"시행별 포인트·클러스터 시계열 — {folder.name}")
    plt.tight_layout()

    out_path = folder / "points_clusters.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="녹화된 낙하 데이터 일괄 재생·비교 분석")
    p.add_argument("folder", help="record.py로 녹화한 .json 파일들이 있는 폴더")
    p.add_argument("--near-y-max", type=float, default=None,
                   help="이 거리(m)보다 먼 클러스터는 타겟 후보에서 제외 "
                        "(사람처럼 항상 더 먼 곳에서 움직이는 대상이 같이 잡힐 때 사용)")
    p.add_argument("--min-doppler-mag", type=float, default=None,
                   help="|도플러|가 이보다 작은(거의 정지한) 클러스터는 타겟 후보에서 제외 "
                        "(물체 바로 옆에 사람이 서 있어 거리로는 구분 안 될 때 사용)")
    args = p.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"[분석] {folder}에 json 파일이 없습니다.")
        return

    results = [replay(f, near_y_max=args.near_y_max, min_doppler_mag=args.min_doppler_mag)
               for f in files]
    summaries = []

    fig, (ax_z, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"낙하 데이터 비교 분석 — {folder.name} ({len(files)}회, 현재 알고리즘으로 재생)")

    for i, r in enumerate(results):
        color = TRIAL_COLORS[i % len(TRIAL_COLORS)]
        valid_t = [t for t, z in zip(r["ts"], r["zs"]) if z is not None]
        valid_z = [z for z in r["zs"] if z is not None]
        ax_z.plot(valid_t, valid_z, "o-", color=color, label=f"trial {i + 1}", alpha=0.85)
        for ft in r["fall_events"]:
            ax_z.axvline(ft, color=color, ls="--", alpha=0.5)
        s = summarize(r)
        s["trial"] = i + 1
        summaries.append(s)

    ax_z.axhline(0, color="brown", ls="--", lw=1, alpha=0.5, label="floor")
    ax_z.axhline(0.37, color="gray", ls=":", lw=1, alpha=0.6, label="airborne(0.37m)")
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Filtered target Z (m)")
    ax_z.set_title("시행별 Z(t) 궤적 (점선 = 낙하 판정 시점)")
    ax_z.legend(fontsize=8)
    ax_z.grid(alpha=0.3)

    trials = [s["trial"] for s in summaries]
    peaks  = [s["peak_z"] if s["peak_z"] is not None else 0 for s in summaries]
    drops  = [s["drop"] if s["drop"] is not None else 0 for s in summaries]
    x = np.arange(len(trials))
    ax_bar.bar(x - 0.2, peaks, width=0.4, label="peak Z (m)", color="#3498db")
    ax_bar.bar(x + 0.2, drops, width=0.4, label="drop (m)", color="#e74c3c")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"#{t}" for t in trials])
    ax_bar.axhline(0.37, color="gray", ls=":", alpha=0.6)
    ax_bar.set_title("시행별 피크 높이 / 하락폭 (O/X = 판정 결과)")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(alpha=0.3, axis="y")
    for i, s in enumerate(summaries):
        mark = "O" if s["detected"] else "X"
        y = max(peaks[i], drops[i]) + 0.05
        ax_bar.text(i, y, mark, ha="center", fontsize=14,
                    color="green" if s["detected"] else "red", fontweight="bold")

    out_path = folder / "analysis.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[분석] 그래프 저장: {out_path}")

    points_path = plot_points_clusters(results, folder)
    print(f"[분석] 포인트·클러스터 시계열 저장: {points_path}")

    print("\n=== 시행별 요약 ===")
    for s, r in zip(summaries, results):
        status = "감지됨" if s["detected"] else "미감지"
        peak = f"{s['peak_z']:.2f}m" if s["peak_z"] is not None else "N/A"
        drop = f"{s['drop']:.2f}m" if s["drop"] is not None else "N/A"
        extra = f"  (t={s['first_fall_t']:.2f}s)" if s["first_fall_t"] is not None else ""
        reason = "" if s["detected"] else f"  — {diagnose(r)}"
        print(f"  #{s['trial']}: peak={peak}  drop={drop}  "
              f"유효프레임={s['n_active']}  판정={status}{extra}{reason}")

    n_detected = sum(1 for s in summaries if s["detected"])
    print(f"\n총 {len(summaries)}회 중 {n_detected}회 감지")


if __name__ == "__main__":
    main()
