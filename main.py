"""ARDA 메인 엔트리포인트 — IWR6843AOPEVM 낙하 감지."""

import argparse
from pathlib import Path

from arda.radar import IWR6843Sensor
from arda.processing import PointCloud, cluster_points
from arda.processing.filters import filter_stationary
from arda.detection import FallDetector
from arda.visualization import RealtimePlotter
from arda.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/profiles/xwr68xx_AOP_profile_2026_06_28T01_40_17_736.cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARDA — 레이더 기반 낙하 감지")
    parser.add_argument("--cli-port", default="/dev/ttyUSB0", help="CLI 포트 (기본: /dev/ttyUSB0)")
    parser.add_argument("--data-port", default="/dev/ttyUSB1", help="데이터 포트 (기본: /dev/ttyUSB1)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="레이더 설정 파일 경로")
    parser.add_argument("--no-viz", action="store_true", help="시각화 비활성화")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sensor = IWR6843Sensor(args.cli_port, args.data_port)
    sensor.configure(args.config)

    detector = FallDetector()
    plotter = RealtimePlotter() if not args.no_viz else None

    logger.info("ARDA 시작 — 낙하 감지 모니터링 중")
    try:
        with sensor:
            while True:
                frame = sensor.read_frame()
                if frame is None:
                    continue

                pc = PointCloud(frame["points"])
                pc = pc.filter_snr(min_snr=8.0).filter_roi()
                pc = filter_stationary(pc)

                clusters = cluster_points(pc)
                target = max(clusters, key=len) if clusters else pc

                fell = detector.update(target)

                if plotter and len(target) > 0:
                    plotter.update(target.xyz, fall_detected=fell)

    except KeyboardInterrupt:
        logger.info("사용자 중단")
    finally:
        if plotter:
            plotter.close()


if __name__ == "__main__":
    main()
