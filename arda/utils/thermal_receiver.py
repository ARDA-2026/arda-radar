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


class ThermalVerdictReceiver:
    """열화상 판정 결과 UDP 수신기.

    메인 루프(센서 프레임 읽기)를 막지 않도록 아주 짧은 타임아웃을 두고
    논블로킹으로 폴링한다.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9997, timeout: float = 0.01):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(timeout)

    def recv(self) -> ThermalVerdict | None:
        """판정 결과 1건 수신. 타임아웃 내 수신 실패 또는 잘못된 패킷이면 None."""
        try:
            data, _ = self._sock.recvfrom(4096)
        except socket.timeout:
            return None

        try:
            obj = json.loads(data.decode("utf-8"))
            return ThermalVerdict(person=bool(obj["person"]), ts=float(obj.get("ts", 0.0)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("잘못된 열화상 판정 패킷 수신: %s", e)
            return None

    def close(self) -> None:
        self._sock.close()
