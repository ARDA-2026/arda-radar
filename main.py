"""ARDA 메인 엔트리포인트 — IWR6843AOPEVM 낙하 감지."""

import argparse
import time
from pathlib import Path

from arda.radar import IWR6843Sensor
from arda.processing import PointCloud, cluster_points
from arda.processing.filters import filter_stationary
from arda.detection import FallDetector
from arda.visualization import RealtimePlotter
from arda.utils import (
    CoordSender,
    ThermalTriggerSender,
    ThermalVerdictReceiver,
    get_logger,
    load_processing_config,
    load_settings,
    to_site_coords,
)

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/profiles/xwr68xx_AOP_profile_short_range.cfg")
DEFAULT_SETTINGS = Path("config/settings.yaml")


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
    parser.add_argument(
        "--thermal-gate", action="store_true",
        help="낙하 확정 시 열화상(arda-thermal-test) 판정을 기다렸다가 사람으로 "
             "확인된 경우에만 낙하 위치를 로그로 남김 (기본: 비활성화, 기존처럼 즉시 로그)",
    )
    parser.add_argument("--thermal-host", default="127.0.0.1", help="열화상 판정기 UDP 호스트 (트리거 전송용)")
    parser.add_argument("--thermal-port", type=int, default=9998, help="열화상 판정기 UDP 포트 (트리거 전송용)")
    parser.add_argument("--thermal-verdict-port", type=int, default=9997, help="열화상 판정 결과 수신 포트")
    parser.add_argument(
        "--thermal-pending-timeout", type=float, default=6.0,
        help="열화상 판정 응답을 기다리는 최대 시간(초) — 이 시간 안에 회신이 없으면 보류 처리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.settings))
    site_cfg = settings.get("site", {})
    site_origin = [site_cfg.get("x", 0.0), site_cfg.get("y", 0.0), site_cfg.get("z", 0.0)]

    cfg = load_processing_config(Path(args.settings))

    sensor = IWR6843Sensor(args.cli_port, args.data_port)
    sensor.configure(args.config)

    detector = FallDetector(max_jump=cfg["max_jump"])
    plotter = RealtimePlotter() if not args.no_viz else None
    sender = None if args.no_servo_out else CoordSender(args.servo_host, args.servo_port)
    if sender:
        logger.info("서보 좌표 전송 활성화 — UDP %s:%d", args.servo_host, args.servo_port)

    thermal_sender = None
    thermal_receiver = None
    if args.thermal_gate:
        thermal_sender = ThermalTriggerSender(args.thermal_host, args.thermal_port)
        thermal_receiver = ThermalVerdictReceiver(port=args.thermal_verdict_port)
        logger.info(
            "열화상 게이트 활성화 — 트리거 전송 UDP %s:%d, 판정 수신 포트 %d",
            args.thermal_host, args.thermal_port, args.thermal_verdict_port,
        )
    # 열화상 게이트 대기 중인 낙하 1건의 실좌표(X,Y)와 트리거 전송 시각.
    # 응답이 오거나 pending-timeout이 지나면 None으로 비운다 — 대기 중에는
    # 새 낙하가 확정돼도 중복으로 트리거를 보내지 않는다(한 번에 하나만 판정).
    pending_site_xy = None
    pending_since = 0.0

    logger.info("ARDA 시작 — 낙하 감지 모니터링 중")
    try:
        with sensor:
            while True:
                frame = sensor.read_frame()
                if frame is None:
                    continue

                pc = PointCloud(frame["points"])
                pc = (pc.filter_snr(min_snr=cfg["min_snr"])
                        .filter_roi(x_range=cfg["roi_x"], y_range=cfg["roi_y"], z_range=cfg["roi_z"]))
                pc = filter_stationary(pc, min_abs_doppler=cfg["min_abs_doppler"])

                clusters = cluster_points(pc, eps=cfg["cluster_eps"], min_samples=cfg["cluster_min_samples"])

                fell = detector.update(clusters)

                if fell and detector.last_fall_centroid is not None:
                    # 서보는 fall=true 좌표만 반응하고(홈 대기 → 낙하 시 이동)
                    # 그 외 좌표는 전부 무시하므로, 매 프레임이 아니라 낙하가
                    # 확정된 이 순간에만 보낸다.
                    if sender:
                        sender.send(detector.last_fall_centroid, fall=True)

                    # 확정 시점의 실측 Z는 바닥 접촉 높이가 아니므로(피크 대비
                    # 일정량만 하강한 순간일 뿐) 사용하지 않는다. X,Y만
                    # 실측값을 site_origin으로 변환하고, Z는 site.z가 "바닥
                    # 기준 센서 설치 높이"로 정의되어 있으므로 항상 바닥(0)이다.
                    x, y, _ = detector.last_fall_centroid
                    site_x, site_y, _ = to_site_coords([x, y, 0.0], site_origin)

                    # fall_detector.py의 Track이 이미 "FALL DETECTED [track#N ...]"를
                    # 찍지만, run_all.sh처럼 여러 로그가 섞여 나올 때 놓치기 쉬워
                    # 레이더가 낙하로 판단한 순간을 여기서 한 번 더 눈에 띄게 남긴다.
                    logger.warning("*" * 50)
                    logger.warning("레이더 낙하 판단 — 로컬 X=%.2f Y=%.2f", x, y)
                    logger.warning("*" * 50)

                    if not args.thermal_gate:
                        logger.warning("낙하 위치(설치 좌표계) X=%.2f Y=%.2f Z=0.00(바닥)", site_x, site_y)
                    elif pending_site_xy is None:
                        # 이미 판정 대기 중인 낙하가 있으면 새 트리거를 또
                        # 보내지 않는다 — 노이즈로 fall이 반복돼도 한 번에
                        # 하나만 열화상에 판정을 맡긴다. 카메라가 서보에 고정
                        # 장착돼 서보가 향한 곳을 그대로 보므로, 좌표가 아니라
                        # "지금 관찰 시작"이라는 신호만 보낸다.
                        thermal_sender.send()
                        pending_site_xy = (site_x, site_y)
                        pending_since = time.time()
                        logger.info(
                            "열화상 판정 요청 전송 — X=%.2f Y=%.2f, 회신 대기 중", site_x, site_y,
                        )

                if thermal_receiver:
                    verdict = thermal_receiver.recv()
                    if verdict is not None and pending_site_xy is not None:
                        vx, vy = pending_site_xy
                        if verdict.person:
                            logger.warning("낙하 위치(설치 좌표계) X=%.2f Y=%.2f Z=0.00(바닥) — 열화상 확인됨", vx, vy)
                        else:
                            logger.info("낙하 판정 기각 — 열화상에서 사람 미확인 (X=%.2f Y=%.2f)", vx, vy)
                        pending_site_xy = None
                    elif (
                        pending_site_xy is not None
                        and (time.time() - pending_since) > args.thermal_pending_timeout
                    ):
                        logger.warning(
                            "열화상 판정 응답 없음(%.1fs 경과) — 낙하 확정 보류",
                            time.time() - pending_since,
                        )
                        pending_site_xy = None

                primary = detector.primary_track
                if plotter and primary is not None and primary.last_cluster is not None:
                    plotter.update(primary.last_cluster.xyz, fall_detected=fell)

    except KeyboardInterrupt:
        logger.info("사용자 중단")
    finally:
        if plotter:
            plotter.close()
        if sender:
            sender.close()
        if thermal_sender:
            thermal_sender.close()
        if thermal_receiver:
            thermal_receiver.close()


if __name__ == "__main__":
    main()
