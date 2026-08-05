"""data/labeling_worksheet.csv로 낙하 신뢰도 모델(로지스틱 회귀)을 학습하고,
arda/detection/fall_detector.py에 그대로 붙여넣을 수 있는 계수를 출력한다.

배치 단위 leave-one-out으로 검증한다(무작위 분할이 아님) — 같은 낙하 사건
에서 자동 추적이 조각낸 여러 트랙이 train/test 양쪽에 걸쳐 사실상 같은
정보를 중복 제공하면 검증 성능이 실제보다 부풀려지는데, 배치(녹화 세션)
전체를 통째로 남겨두면 이런 누수를 막을 수 있다.

이 스크립트는 fall_detector.py를 직접 수정하지 않는다. 어떤 계수를 실제로
배포할지, LIKELY_REAL_THRESHOLD를 얼마로 둘지는 재현율/정밀도 트레이드오프를
보고 사람이 판단해야 하는 결정이라(낙하 감지는 "놓치는 것"이 "오탐"보다
훨씬 치명적이라는 정책이 이미 fall_detector.py에 있다), 출력된 계수를 검토한
뒤 직접 붙여넣고 왜 바꿨는지 그 위 주석에 남기길 권한다(_MODEL_* 주석들의
기존 형식 참고 — 배치 구성, 검증 지표, 임계값을 그렇게 정한 이유).

실행:
    uv run scripts/train_confidence_model.py
    uv run scripts/train_confidence_model.py --threshold 0.24   # 그 임계값으로 in-sample 재현 검증까지
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# fall_detector.py의 _MODEL_FEATURE_ORDER와 순서가 반드시 같아야 한다 — 거기는
# Track 내부 변수명 기준 축약형(dwell/peak_z/net_drop/recent_v/recent_a/...)을
# 쓰고, 여기는 CSV 컬럼명을 그대로 쓴다는 것만 다르다.
FEATURE_COLS = [
    "total_obs_count", "peak_z_smoothed", "net_drop_from_peak", "avg_descent_speed",
    "recent_velocity", "recent_accel", "z_max_ever", "z_min_ever", "avg_pts", "max_pts",
    "rebound_penalized",
]

DEFAULT_CSV = "data/labeling_worksheet.csv"
THRESHOLD_SWEEP = [0.50, 0.36, 0.24, 0.15, 0.10, 0.05]


def load_labeled_rows(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """label이 0/1로 채워진 행만 특징 행렬로 뽑는다. label_note만 채워진
    애매한 행(빈 label)은 자동으로 제외된다.

    encoding="utf-8-sig": 엑셀 등으로 CSV를 저장하면 파일 앞에 BOM이 붙어
    csv.DictReader가 첫 컬럼명을 "batch"가 아니라 "﻿batch"로 읽는 문제가
    있었다 — utf-8-sig는 BOM이 있든 없든 안전하게 벗겨낸다.
    """
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    labeled = [r for r in rows if r["label"] in ("0", "1")]
    if not labeled:
        raise SystemExit(f"{csv_path}에 label(0/1)이 채워진 행이 없습니다.")

    X = np.array([
        [float(r[c]) if c != "rebound_penalized" else (1.0 if r[c] == "True" else 0.0)
         for c in FEATURE_COLS]
        for r in labeled
    ])
    y = np.array([int(r["label"]) for r in labeled])
    batches = np.array([r["batch"] for r in labeled])
    return X, y, batches, labeled


def batch_loo_probabilities(X: np.ndarray, y: np.ndarray, batches: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """배치를 하나씩 통째로 빼고 나머지로 학습 → 뺀 배치를 예측, 이걸 모든
    배치에 대해 반복한다. 반환값은 전체 배치의 held-out 예측 확률을 이어붙인
    것 — 이걸로 하나의 threshold sweep 표를 계산한다."""
    all_true, all_prob = [], []
    for b in sorted(set(batches)):
        train_mask = batches != b
        test_mask = ~train_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[train_mask]), y[train_mask])
        prob = clf.predict_proba(scaler.transform(X[test_mask]))[:, 1]
        all_true.extend(y[test_mask])
        all_prob.extend(prob)
    return np.array(all_true), np.array(all_prob)


def metrics_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sigmoid_confidence(x: list[float], mean: list[float], scale: list[float],
                        coef: list[float], intercept: float) -> float:
    """fall_detector.py의 Track._model_confidence()와 정확히 같은 계산 —
    학습 후 뽑은 계수를 그대로 붙여넣었을 때 실제 배포 코드와 같은 결과가
    나오는지 검증하는 데 쓴다(계수를 옮겨적다 생기는 오타를 잡기 위함)."""
    z = intercept
    for xi, m, s, c in zip(x, mean, scale, coef):
        z += c * (xi - m) / s
    return 1.0 / (1.0 + math.exp(-z))


def main():
    p = argparse.ArgumentParser(
        description="data/labeling_worksheet.csv로 낙하 신뢰도 모델을 학습하고 배포용 계수를 출력한다")
    p.add_argument("--csv", default=DEFAULT_CSV, help="라벨링 워크시트 경로")
    p.add_argument("--threshold", type=float, default=None,
                   help="지정하면 그 임계값으로 최종(전체 데이터) 모델의 in-sample 재현 검증까지 출력")
    args = p.parse_args()

    csv_path = Path(args.csv)
    X, y, batches, labeled = load_labeled_rows(csv_path)
    n_real, n_noise = int((y == 1).sum()), int((y == 0).sum())
    print(f"[데이터] {csv_path}  라벨링 {len(y)}건 (진짜 {n_real} / 노이즈 {n_noise}), "
          f"배치 {len(set(batches))}개\n")

    # ── 배치 단위 leave-one-out 검증 ──────────────────────────────────────
    loo_true, loo_prob = batch_loo_probabilities(X, y, batches)
    print("=== 배치 단위 leave-one-out 검증 (threshold sweep) ===")
    for th in THRESHOLD_SWEEP:
        m = metrics_at(loo_true, loo_prob, th)
        print(f"  th={m['threshold']:.2f}  acc={m['accuracy']:.2f}  recall={m['recall']:.2f}  "
              f"precision={m['precision']:.2f}  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    print("\n=== 재현율 우선 임계값 참고 — FN(놓침) 최소 지점 ===")
    best_fn1 = best_fn0 = None
    for th in sorted(set(np.round(loo_prob, 3)), reverse=True):
        m = metrics_at(loo_true, loo_prob, th)
        if best_fn1 is None and m["fn"] <= 1:
            best_fn1 = m
        if best_fn0 is None and m["fn"] == 0:
            best_fn0 = m
        if best_fn1 is not None and best_fn0 is not None:
            break
    if best_fn1:
        print(f"  FN<=1 최초 지점: threshold={best_fn1['threshold']:.3f}  "
              f"recall={best_fn1['recall']:.2f}  precision={best_fn1['precision']:.2f}")
    if best_fn0:
        print(f"  FN=0  최초 지점: threshold={best_fn0['threshold']:.3f}  "
              f"recall={best_fn0['recall']:.2f}  precision={best_fn0['precision']:.2f}")

    # ── 전체 라벨링 데이터로 최종 모델 학습 (배포용) ─────────────────────
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)

    mean, scale = scaler.mean_.round(6).tolist(), scaler.scale_.round(6).tolist()
    coef, intercept = clf.coef_[0].round(6).tolist(), round(float(clf.intercept_[0]), 6)

    print("\n=== fall_detector.py에 붙여넣을 계수 (전체 데이터로 재학습) ===")
    print(f"_MODEL_MEAN = {mean}")
    print(f"_MODEL_SCALE = {scale}")
    print(f"_MODEL_COEF = {coef}")
    print(f"_MODEL_INTERCEPT = {intercept}")

    # ── 위 계수로 실제 배포 시그모이드를 그대로 재현해 검증 ─────────────
    if args.threshold is not None:
        n_match = 0
        mismatches = []
        for xi, yi, row in zip(X, y, labeled):
            conf = sigmoid_confidence(xi, mean, scale, coef, intercept)
            pred = 1 if conf >= args.threshold else 0
            if pred == yi:
                n_match += 1
            else:
                mismatches.append((row["batch"], row["trial"], row["track_id"], yi, round(conf, 3)))

        print(f"\n=== threshold={args.threshold}  in-sample 재현 검증 ===")
        print(f"  {n_match}/{len(y)} 일치 ({n_match / len(y):.1%})")
        if mismatches:
            print("  불일치 (라벨=1인데 놓친 FN 방향은 특히 주의):")
            for batch, trial, track_id, label, conf in mismatches:
                tag = "FN(놓침!)" if label == 1 else "FP(오탐)"
                print(f"    {tag}  {batch} trial={trial} track={track_id}  conf={conf}")


if __name__ == "__main__":
    main()
