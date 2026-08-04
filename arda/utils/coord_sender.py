"""좌표 UDP 송신 — 외부 프로세스(예: arda-servo)로 타겟 좌표를 전달한다."""

import json
import socket
import time

import numpy as np


class CoordSender:
    """탐지된 타겟 무게중심을 UDP로 브로드캐스트하는 얇은 래퍼."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, xyz: np.ndarray, fall: bool = False, confidence: float = 0.0) -> None:
        """confidence는 FallDetector.last_fall_confidence(0~1, 로지스틱 회귀 낙하
        신뢰도)를 그대로 전달한다 — arda-servo가 dwell 중 더 높은 확률의 새
        낙하 후보를 받으면 즉시 그쪽으로 전환하는 선점 판단에 쓴다."""
        payload = json.dumps({
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
            "fall": bool(fall),
            "confidence": float(confidence),
            "ts": time.time(),
        }).encode("utf-8")
        self._sock.sendto(payload, self._addr)

    def close(self) -> None:
        self._sock.close()
