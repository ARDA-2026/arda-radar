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

    def send(self, xyz: np.ndarray, fall: bool = False) -> None:
        payload = json.dumps({
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
            "fall": bool(fall),
            "ts": time.time(),
        }).encode("utf-8")
        self._sock.sendto(payload, self._addr)

    def close(self) -> None:
        self._sock.close()
