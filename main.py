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
    ThermalEngaged,
    ThermalTriggerSender,
    ThermalVerdict,
    ThermalVerdictReceiver,
    get_logger,
    load_processing_config,
    load_settings,
    local_to_latlon,
    send_fall_report,
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
        "--thermal-pending-timeout", type=float, default=10.0,
        help="열화상 판정 응답을 기다리는 최대 시간(초) — 이 시간 안에 회신이 없으면 보류 처리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.settings))
    site_cfg = settings.get("site", {})
    site_lat = site_cfg.get("lat", 0.0)
    site_lon = site_cfg.get("lon", 0.0)
    site_heading_deg = site_cfg.get("heading_deg", 0.0)
    report_url = site_cfg.get("report_url", "")
    if report_url:
        logger.info("웹 전송 활성화 — 열화상 확인된 낙하만 %s로 POST", report_url)

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
    # 열화상 게이트 대기 중인 낙하 1건의 위도/경도·신뢰도와 트리거 전송 시각.
    # 응답이 오거나 pending-timeout이 지나면 None으로 비운다 — 대기 중에는
    # 새 낙하가 확정돼도 기본적으로 중복 트리거를 보내지 않는다(한 번에
    # 하나만 판정). 다만 새로 확정된 낙하의 confidence가 지금 대기 중인
    # 후보보다 높으면 예외적으로 기존 대기를 취소하고 새 후보로 즉시
    # 대체한다 — 단, 열화상이 이번 대기에서 이미 열원을 검출해 engaged
    # 신호를 보내온 뒤(pending_engaged)라면 confidence와 무관하게 이 선점을
    # 하지 않는다. arda_servo.ServoController._thermal_engaged와 같은 원칙:
    # 열화상이 실제 열원을 붙잡아 추적을 시작한 순간부터는 열화상이 우선권을
    # 갖는다 — 그래야 서보가 실제로 보고 있는 지점과 열화상이 판정하는
    # 지점이 어긋나지 않는다. (구버전 thermal-camera는 engaged 신호를 보내지
    # 않으므로 pending_engaged가 항상 False로 남아 기존처럼 confidence만으로
    # 선점된다.)
    pending_latlon = None
    pending_confidence = 0.0
    pending_since = 0.0
    pending_engaged = False
    # FallDetector.update()는 한 번 확정된 트랙에 대해 계속 True를 반환하므로
    # (래치), 마지막으로 반응(서보 전송·로그·열화상 트리거)한 트랙 id를
    # 기억해 "새로 확정된 낙하"일 때만 반응하고 같은 낙하가 계속 보고되는
    # 동안은 반복하지 않는다.
    last_reacted_track_id = None

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

                if (
                    fell
                    and detector.last_fall_centroid is not None
                    and detector.last_fall_track_id != last_reacted_track_id
                ):
                    last_reacted_track_id = detector.last_fall_track_id

                    # 서보는 fall=true 좌표만 반응하고(홈 대기 → 낙하 시 이동)
                    # 그 외 좌표는 전부 무시하므로, 매 프레임이 아니라 낙하가
                    # 확정된 이 순간에만 보낸다. confidence도 함께 보내
                    # 서보가 dwell 중이라도 더 유력한 후보가 오면 즉시 그
                    # 방향으로 전환(선점)할 수 있게 한다.
                    if sender:
                        sender.send(
                            detector.last_fall_centroid, fall=True,
                            confidence=detector.last_fall_confidence,
                        )

                    # X,Y(레이더 기준 좌우/정면 거리)만 실좌표 변환에 쓴다 —
                    # 확정 시점의 실측 Z는 바닥 접촉 높이가 아니라 피크 대비
                    # 일정량만 하강한 순간의 값이라 신뢰할 수 없어 애초에 쓰지
                    # 않는다. site_heading_deg만큼 회전시켜 동/북 변위로 바꾼
                    # 뒤 설치 위경도에 더한다 — 기기가 어느 방향을 보고
                    # 설치되든 이 회전 덕분에 부호가 자동으로 맞는다.
                    x, y, _ = detector.last_fall_centroid
                    lat, lon = local_to_latlon(x, y, site_lat, site_lon, site_heading_deg)

                    if not args.thermal_gate:
                        logger.warning(
                            "낙하 위치(GPS) lat=%.6f lon=%.6f confidence=%.2f",
                            lat, lon, detector.last_fall_confidence,
                        )
                    elif pending_latlon is None:
                        # 대기 중인 낙하가 없으면 바로 트리거. 카메라가 서보에
                        # 고정 장착돼 서보가 향한 곳을 그대로 보므로, 좌표가
                        # 아니라 "지금 관찰을 시작하라"는 신호만 보낸다.
                        thermal_sender.send()
                        pending_latlon = (lat, lon)
                        pending_confidence = detector.last_fall_confidence
                        pending_since = time.time()
                        pending_engaged = False
                        logger.info(
                            "열화상 판정 요청 전송 — lat=%.6f lon=%.6f confidence=%.2f, 회신 대기 중",
                            lat, lon, pending_confidence,
                        )
                    elif pending_engaged:
                        # 열화상이 이미 대기 중인 낙하의 열원을 붙잡아 추적
                        # 중이다 — confidence와 무관하게 선점하지 않는다.
                        logger.info(
                            "[제어권 유지] 더 높은 확률의 낙하 후보(%.2f > %.2f) 발견 — 열화상이 "
                            "이미 열원을 추적 중이라 무시함",
                            detector.last_fall_confidence, pending_confidence,
                        )
                    elif detector.last_fall_confidence > pending_confidence:
                        # 이미 판정 대기 중인 낙하보다 이번에 확정된 낙하의
                        # confidence가 더 높다 — 기존 대기(및 그 판정 결과)는
                        # 포기하고 이 후보로 즉시 대체한다. arda-thermal-test는
                        # 새 트리거를 받으면 진행 중이던 관찰을 중단하고 이
                        # 트리거로 즉시 재시작한다(단, 그쪽도 이미 engaged면
                        # 무시하고 계속 관찰함 — thermal_main.py 참고).
                        logger.info(
                            "[제어권 이동] 더 높은 확률의 낙하 후보 발견(%.2f > %.2f) — 기존 판정 "
                            "대기 취소, 새 트리거 전송 lat=%.6f lon=%.6f",
                            detector.last_fall_confidence, pending_confidence, lat, lon,
                        )
                        thermal_sender.send()
                        pending_latlon = (lat, lon)
                        pending_confidence = detector.last_fall_confidence
                        pending_since = time.time()
                        pending_engaged = False

                if thermal_receiver:
                    result = thermal_receiver.recv()
                    if isinstance(result, ThermalEngaged) and pending_latlon is not None:
                        pending_engaged = True
                    elif isinstance(result, ThermalVerdict) and pending_latlon is not None:
                        verdict = result
                        vlat, vlon = pending_latlon
                        if verdict.person:
                            logger.warning("낙하 위치(GPS) lat=%.6f lon=%.6f — 열화상 확인됨", vlat, vlon)
                            if report_url:
                                send_fall_report(report_url, vlat, vlon)
                        else:
                            logger.info("낙하 판정 기각 — 열화상에서 사람 미확인 (lat=%.6f lon=%.6f)", vlat, vlon)
                        pending_latlon = None
                        pending_confidence = 0.0
                        pending_engaged = False
                    elif (
                        pending_latlon is not None
                        and (time.time() - pending_since) > args.thermal_pending_timeout
                    ):
                        logger.warning(
                            "열화상 판정 응답 없음(%.1fs 경과) — 낙하 확정 보류",
                            time.time() - pending_since,
                        )
                        pending_latlon = None
                        pending_confidence = 0.0
                        pending_engaged = False

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
