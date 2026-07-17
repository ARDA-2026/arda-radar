"""레이더 데이터 녹화 스크립트.

지정한 시간(기본 10초) 동안 프레임을 수집하고 JSON으로 저장한다.
저장된 파일을 분석해 낙하 감지 임계값 튜닝에 활용한다.

실행:
    uv run scripts/record.py              # 10초 녹화
    uv run scripts/record.py --duration 15
    uv run scripts/record.py --out data/test.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points
from arda.detection import FallDetector
from arda.utils import get_logger, load_processing_config

logger = get_logger(__name__)

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"

# detect.py와 동일하게 config/settings.yaml의 processing: 섹션에서 읽어온다.
_cfg = load_processing_config()
MIN_SNR         = _cfg["min_snr"]
CLUSTER_EPS     = _cfg["cluster_eps"]
CLUSTER_MINSAMP = _cfg["cluster_min_samples"]
ROI_X           = _cfg["roi_x"]
ROI_Y           = _cfg["roi_y"]
Z_RANGE         = _cfg["roi_z"]
AIRBORNE_Z      = _cfg["airborne_z"]
MAX_JUMP        = _cfg["max_jump"]


def parse_args():
    p = argparse.ArgumentParser(description="레이더 데이터 녹화")
    p.add_argument("--cli-port",  default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--duration",  type=float, default=10.0,
                   help="녹화 시간 (초, 기본 10)")
    p.add_argument("--out",       default=None,
                   help="저장 경로 (기본: data/record_YYYYMMDD_HHMMSS.json)")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out) if args.out else (
        Path("data") / f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sensor   = IWR6843Sensor(args.cli_port, args.data_port)
    detector = FallDetector(debug=False)

    sensor.configure(args.config)

    records = []
    t_start = None
    elapsed = 0.0

    print(f"\n[RECORD] {args.duration:.0f}초 녹화 시작 — 물체를 떨어뜨려 주세요\n")

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

                fn = frame.get("frame_number", len(records))

                # detect.py와 동일한 파이프라인
                pc_all   = (PointCloud(frame.get("points", []))
                            .filter_snr(MIN_SNR)
                            .filter_roi(x_range=ROI_X, y_range=ROI_Y, z_range=Z_RANGE))
                clusters = cluster_points(pc_all, eps=CLUSTER_EPS,
                                          min_samples=CLUSTER_MINSAMP)
                target   = detector.choose_target(clusters, airborne_z=AIRBORNE_Z, max_jump=MAX_JUMP)

                is_falling = detector.update(target)
                centroid   = target.centroid()
                z_vel      = detector.z_velocity()

                # 진행 표시
                bar  = "#" * int(elapsed / args.duration * 30)
                info = (f"Z={centroid[2]:.2f}m  dZ/dt={z_vel:+.2f}m/s"
                        if centroid is not None else "no target        ")
                fall_mark = "  *** FALL ***" if is_falling else "              "
                print(f"\r[{bar:<30}] {elapsed:5.1f}s  {info}{fall_mark}", end="")

                records.append({
                    "t":          round(elapsed, 4),
                    "frame":      fn,
                    "n_raw":      len(frame.get("points", [])),
                    "n_filtered": len(pc_all),
                    "n_clusters": len(clusters),
                    "n_target":   len(target),
                    "centroid_z": round(float(centroid[2]), 4) if centroid is not None else None,
                    "centroid_y": round(float(centroid[1]), 4) if centroid is not None else None,
                    "z_velocity": round(float(z_vel), 4),
                    "doppler":    round(float(target.doppler.mean()), 4) if len(target) > 0 else None,
                    "is_falling": bool(is_falling),
                    # 원시 포인트 (필터 이전 전체 포함)
                    "points": [
                        {"x": round(p["x"], 3), "y": round(p["y"], 3),
                         "z": round(p["z"], 3), "doppler": round(p["doppler"], 3),
                         "snr": round(p["snr"], 1)}
                        for p in frame.get("points", [])
                    ],
                })

    except KeyboardInterrupt:
        print("\n[RECORD] 사용자 중단")

    print(f"\n\n[RECORD] {len(records)}프레임 수집 완료")

    # 저장
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "duration_s":      round(elapsed, 2),
                "n_frames":        len(records),
                "min_snr":         MIN_SNR,
                "cluster_eps":     CLUSTER_EPS,
                "cluster_minsamp": CLUSTER_MINSAMP,
                "z_range":         list(Z_RANGE),
            },
            "frames": records,
        }, f, indent=2)

    # 간단 요약
    fall_frames = [r for r in records if r["is_falling"]]
    active      = [r for r in records if r["centroid_z"] is not None]
    z_vels      = [r["z_velocity"] for r in active]

    print(f"\n=== 요약 ===")
    print(f"  전체 프레임     : {len(records)}")
    print(f"  물체 감지 프레임: {len(active)}")
    print(f"  낙하 판정 프레임: {len(fall_frames)}")
    if z_vels:
        print(f"  Z속도 범위      : {min(z_vels):+.3f} ~ {max(z_vels):+.3f} m/s")
    print(f"  저장 경로       : {out_path}\n")


if __name__ == "__main__":
    main()
