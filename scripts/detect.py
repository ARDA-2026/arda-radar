"""낙하 감지 전용 스크립트 — DBSCAN 클러스터링 + 다중 추적.

파이프라인:
    RAW 포인트 → SNR 필터 → ROI 필터 → DBSCAN 클러스터링
    → 클러스터마다 독립 트랙으로 추적(FallDetector) → 트랙별 낙하 판정

실행:
    uv run scripts/detect.py
    uv run scripts/detect.py --debug   # 하강 중 프레임별 [FD] 값 출력
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.processing.pointcloud import PointCloud
from arda.processing.clustering import cluster_points
from arda.detection import FallDetector
from arda.utils import get_logger, load_processing_config

logger = get_logger(__name__)

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_short_range.cfg"

# 포인트 필터·타겟 선택 튜닝값 — config/settings.yaml의 processing: 섹션에서 읽어온다.
_cfg = load_processing_config()
MIN_SNR         = _cfg["min_snr"]
CLUSTER_EPS     = _cfg["cluster_eps"]
CLUSTER_MINSAMP = _cfg["cluster_min_samples"]
ROI_X           = _cfg["roi_x"]
ROI_Y           = _cfg["roi_y"]
Z_RANGE         = _cfg["roi_z"]
MAX_JUMP        = _cfg["max_jump"]


def parse_args():
    p = argparse.ArgumentParser(description="낙하 감지 전용 — DBSCAN 물체 추적")
    p.add_argument("--cli-port",  default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--debug", action="store_true",
                   help="하강 중 매 프레임 [FD] + 클러스터 정보 출력")
    return p.parse_args()


def main():
    args = parse_args()

    sensor   = IWR6843Sensor(args.cli_port, args.data_port)
    detector = FallDetector(debug=args.debug, max_jump=MAX_JUMP)

    sensor.configure(args.config)

    print("\n[ARDA] 낙하 감지 시작 (DBSCAN 물체 추적) — Ctrl+C로 종료\n")

    frame_count     = 0
    fall_count      = 0
    last_fall_frame = -999

    try:
        with sensor:
            while True:
                frame = sensor.read_frame()
                if frame is None:
                    continue

                frame_count += 1
                fn = frame.get("frame_number", frame_count)

                # 1) 필터링
                pc = (PointCloud(frame.get("points", []))
                      .filter_snr(MIN_SNR)
                      .filter_roi(x_range=ROI_X, y_range=ROI_Y, z_range=Z_RANGE))

                # 2) DBSCAN 클러스터링
                clusters = cluster_points(pc, eps=CLUSTER_EPS,
                                          min_samples=CLUSTER_MINSAMP)

                # 3) 클러스터마다 독립 트랙으로 추적하며 낙하 판정 (다중 추적)
                is_falling = detector.update(clusters)

                # 4) 디버그 출력 — 클러스터가 보일 때
                if args.debug and clusters:
                    primary = detector.primary_track
                    z_vel = detector.z_velocity()
                    z = primary.last_centroid[2] if primary is not None and primary.last_centroid is not None else float("nan")
                    print(
                        f"[CLUSTER] frame={fn}"
                        f"  clusters={len(clusters)}"
                        f"  tracks={len(detector.tracks)}"
                        f"  Z={z:.2f}m  dZ/dt={z_vel:+.2f}m/s"
                        + ("  *** FALLING ***" if is_falling else "")
                    )

                # 5) 낙하 이벤트 출력
                if is_falling and fn - last_fall_frame > 3:
                    fall_count += 1
                    c = detector.last_fall_centroid   # 착지 소실 시에도 마지막 위치 사용
                    pos = (f"X={c[0]:+.2f}m  Y={c[1]:.2f}m"
                           if c is not None else "X=?  Y=?")
                    print(
                        f"\n*** FALL #{fall_count:03d} ***  "
                        f"frame={fn}  {pos}"
                    )
                    last_fall_frame = fn

    except KeyboardInterrupt:
        elapsed = frame_count / 10  # 10fps (100ms 프레임 주기)
        print(
            f"\n[ARDA] 종료 — 총 {frame_count}프레임 ({elapsed:.0f}초),"
            f" 낙하 감지 {fall_count}회"
        )


if __name__ == "__main__":
    main()
