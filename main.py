"""ARDA 메인 엔트리포인트 — IWR6843AOPEVM 낙하 감지."""

import argparse
from pathlib import Path

import yaml

from arda.radar import IWR6843Sensor
from arda.processing import PointCloud, cluster_points
from arda.processing.filters import filter_stationary
from arda.detection import FallDetector
from arda.visualization import RealtimePlotter
from arda.utils import CoordSender, get_logger, to_site_coords

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/profiles/xwr68xx_AOP_profile_2026_06_28T01_40_17_736.cfg")
DEFAULT_SETTINGS = Path("config/settings.yaml")


def load_settings(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARDA — 레이더 기반 낙하 감지")
    parser.add_argument("--cli-port", default="/dev/ttyUSB0", help="CLI 포트 (기본: /dev/ttyUSB0)")
    parser.add_argument("--data-port", default="/dev/ttyUSB1", help="데이터 포트 (기본: /dev/ttyUSB1)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="레이더 설정 파일 경로")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS), help="사이트 설치 좌표 등 설정 파일 경로")
    parser.add_argument("--no-viz", action="store_true", help="시각화 비활성화")
    parser.add_argument("--no-servo-out", action="store_true", help="서보 제어기로의 좌표 UDP 전송 비활성화")
    parser.add_argument("--servo-host", default="127.0.0.1", help="서보 제어기(arda-servo) UDP 호스트")
    parser.add_argument("--servo-port", type=int, default=9999, help="서보 제어기(arda-servo) UDP 포트")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.settings))
    site_cfg = settings.get("site", {})
    site_origin = [site_cfg.get("x", 0.0), site_cfg.get("y", 0.0), site_cfg.get("z", 0.0)]

    sensor = IWR6843Sensor(args.cli_port, args.data_port)
    sensor.configure(args.config)

    detector = FallDetector()
    plotter = RealtimePlotter() if not args.no_viz else None
    sender = None if args.no_servo_out else CoordSender(args.servo_host, args.servo_port)
    if sender:
        logger.info("서보 좌표 전송 활성화 — UDP %s:%d", args.servo_host, args.servo_port)

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

                if sender and detector.last_centroid is not None:
                    sender.send(detector.last_centroid, fall=fell)

                if fell and detector.last_centroid is not None:
                    # 확정 시점의 실측 Z는 바닥 접촉 높이가 아니므로(피크 대비
                    # 일정량만 하강한 순간일 뿐) 사용하지 않는다. X,Y만
                    # 실측값을 site_origin으로 변환하고, Z는 site.z가 "바닥
                    # 기준 센서 설치 높이"로 정의되어 있으므로 항상 바닥(0)이다.
                    x, y, _ = detector.last_centroid
                    site_x, site_y, _ = to_site_coords([x, y, 0.0], site_origin)
                    logger.warning("낙하 위치(설치 좌표계) X=%.2f Y=%.2f Z=0.00(바닥)", site_x, site_y)

                if plotter and len(target) > 0:
                    plotter.update(target.xyz, fall_detected=fell)

    except KeyboardInterrupt:
        logger.info("사용자 중단")
    finally:
        if plotter:
            plotter.close()
        if sender:
            sender.close()


if __name__ == "__main__":
    main()
