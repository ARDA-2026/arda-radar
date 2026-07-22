"""열화상 판정기(arda-thermal-test)로 "지금 관찰을 시작하라"는 신호만 전달.

카메라가 서보에 고정 장착되어 서보가 향한 곳을 그대로 보므로, 어디를
보라는 좌표는 필요 없다 — 언제 관찰을 시작할지만 알려주면 된다.
"""

import json
import socket
import time


class ThermalTriggerSender:
    """열화상 관찰 시작 트리거 UDP 송신기."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9998):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self) -> None:
        payload = json.dumps({"ts": time.time()}).encode("utf-8")
        self._sock.sendto(payload, self._addr)

    def close(self) -> None:
        self._sock.close()
