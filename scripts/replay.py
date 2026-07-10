"""녹화된 데이터를 재생하며 낙하 감지를 검증하는 스크립트."""

import argparse
import json
import time
from pathlib import Path

from arda.processing import PointCloud, cluster_points
from arda.processing.filters import filter_stationary
from arda.detection import FallDetector
from arda.visualization import RealtimePlotter
from arda.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="녹화 데이터 재생")
    parser.add_argument("input", help="JSONL 녹화 파일 경로")
    parser.add_argument("--speed", type=float, default=1.0, help="재생 배속 (기본 1.0)")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    detector = FallDetector()
    plotter = RealtimePlotter() if not args.no_viz else None
    fall_count = 0

    with Path(args.input).open() as f:
        for line in f:
            frame = json.loads(line)
            pc = PointCloud(frame["points"]).filter_snr().filter_roi()
            pc = filter_stationary(pc)
            clusters = cluster_points(pc)
            target = max(clusters, key=len) if clusters else pc

            fell = detector.update(target)
            if fell:
                fall_count += 1

            if plotter and len(target) > 0:
                plotter.update(target.xyz, fall_detected=fell)

            time.sleep(0.1 / args.speed)

    logger.info("재생 완료 — 낙하 감지 횟수: %d", fall_count)
    if plotter:
        plotter.close()


if __name__ == "__main__":
    main()
