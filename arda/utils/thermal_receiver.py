"""열화상 판정기(arda-thermal-test)로부터 사람 판정 결과를 UDP로 수신."""

import json
import socket
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ThermalVerdict:
    person: bool
    ts: float


@dataclass
class ThermalEngaged:
    """열화상이 지금 관찰에서 발열 영역을 처음 검출했다는 신호(판정 아님).

    이 신호가 오면(판정 회신 전까지) confidence가 더 높은 새 낙하 후보가
    와도 지금 대기 중인 낙하를 선점하지 않아야 한다 — 서보(arda_servo.
    ServoController._thermal_engaged)와 같은 원칙을 레이더 쪽 판정 대기
    상태에도 적용하기 위함."""

    ts: float


class ThermalVerdictReceiver:
    """열화상(판정 결과·engaged 신호) UDP 수신기.

    메인 루프(센서 프레임 읽기)를 막지 않도록 아주 짧은 타임아웃을 두고
    논블로킹으로 폴링한다.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9997, timeout: float = 0.01):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(timeout)

    def recv(self) -> ThermalVerdict | ThermalEngaged | None:
        """1건 수신. 타임아웃 내 수신 실패 또는 잘못된 패킷이면 None.

        `{"person": ...}` 패킷은 ThermalVerdict(최종 판정), `{"engaged": true}`
        패킷은 ThermalEngaged(관찰 중 열원 최초 검출 알림)로 구분해 반환한다 —
        구버전 thermal-camera는 engaged 신호를 보내지 않으므로 항상 ThermalVerdict나
        None만 오고, 그 경우 호출부는 기존처럼 confidence만으로 선점 여부를 판단한다.
        """
        try:
            data, _ = self._sock.recvfrom(4096)
        except socket.timeout:
            return None

        try:
            obj = json.loads(data.decode("utf-8"))
            if "engaged" in obj:
                return ThermalEngaged(ts=float(obj.get("ts", 0.0)))
            return ThermalVerdict(person=bool(obj["person"]), ts=float(obj.get("ts", 0.0)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("잘못된 열화상 패킷 수신: %s", e)
            return None

    def close(self) -> None:
        self._sock.close()
