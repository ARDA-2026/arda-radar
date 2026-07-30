"""ROI/SNR 필터 없이 레이더가 잡는 모든 원시 포인트를 그대로 녹화한다.

시각화·클러스터링·낙하 판정 없이 저장만 한다 — ROI를 아직 못 정했거나
(예: 레이돔/덮개로 인한 감쇠 진단처럼) 필터링 이전 원본 데이터 자체를
봐야 할 때 쓴다. 저장 스키마는 record.py/analyze_drops.py와 동일
(`{"meta": {...}, "frames": [{"t", "frame", "points"}, ...]}`)하므로,
필요하면 이후 scripts/analyze_drops.py로도 재생할 수 있다 — 그때는
analyze_drops.py가 현재 config/settings.yaml의 ROI/SNR을 적용해서
재생하므로, 녹화 시점 ROI와 무관하게 여러 필터 설정을 사후에 실험해볼
수 있다.

실행:
    uv run scripts/record_raw.py                       # 10초 녹화
    uv run scripts/record_raw.py --duration 15
    uv run scripts/record_raw.py --label cover_test     # data/raw/cover_test/ 아래 저장
    uv run scripts/record_raw.py --out data/raw/foo.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"


def parse_args():
    p = argparse.ArgumentParser(description="ROI/필터 없이 레이더 원시 포인트 전체 녹화 (저장만, 시각화 없음)")
    p.add_argument("--cli-port",  default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--duration",  type=float, default=10.0,
                   help="녹화 시간 (초, 기본 10)")
    p.add_argument("--label",     default=None,
                   help="같은 라벨끼리 data/raw/<라벨>/ 폴더에 모아 저장 "
                        "(예: --label cover_test → data/raw/cover_test/record_raw_YYYYMMDD_HHMMSS.json)")
    p.add_argument("--out",       default=None,
                   help="저장 경로 직접 지정 (기본: --label 있으면 data/raw/<라벨>/record_raw_YYYYMMDD_HHMMSS.json, "
                        "없으면 data/record_raw_YYYYMMDD_HHMMSS.json)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.label:
            # 같은 라벨끼리 한 폴더에 모이도록 data/raw/<라벨>/ 아래 저장한다
            # (record.py로 정리해온 배치 폴더 관례와 동일).
            out_path = Path("data/raw") / args.label / f"record_raw_{ts}.json"
        else:
            out_path = Path("data") / f"record_raw_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sensor = IWR6843Sensor(args.cli_port, args.data_port)
    sensor.configure(args.config)

    records: list[dict] = []
    t_start = None
    elapsed = 0.0

    print(f"\n[RECORD-RAW] {args.duration:.0f}초 녹화 시작 — ROI/필터 없이 전체 포인트 저장\n")

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
                fn = frame.get("frame_number", len(records))

                bar = "#" * int(elapsed / args.duration * 30)
                print(f"\r[{bar:<30}] {elapsed:5.1f}s  points={len(points):3d}", end="")

                records.append({
                    "t":      round(elapsed, 4),
                    "frame":  fn,
                    "points": [
                        {"x": round(p["x"], 3), "y": round(p["y"], 3),
                         "z": round(p["z"], 3), "doppler": round(p["doppler"], 3),
                         "snr": round(p["snr"], 1)}
                        for p in points
                    ],
                })

    except KeyboardInterrupt:
        print("\n[RECORD-RAW] 사용자 중단")

    print(f"\n\n[RECORD-RAW] {len(records)}프레임 수집 완료")

    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "duration_s": round(elapsed, 2),
                "n_frames":   len(records),
                "filtered":   False,
            },
            "frames": records,
        }, f, indent=2)

    n_pts = [len(r["points"]) for r in records]
    print("\n=== 요약 ===")
    print(f"  전체 프레임     : {len(records)}")
    if n_pts:
        print(f"  프레임당 포인트 : 평균 {sum(n_pts) / len(n_pts):.1f}개, "
              f"최대 {max(n_pts)}개, 최소 {min(n_pts)}개")
    print(f"  저장 경로       : {out_path}\n")


if __name__ == "__main__":
    main()
