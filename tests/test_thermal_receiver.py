"""ThermalVerdictReceiver 단위 테스트."""

import json
import socket

from arda.utils.thermal_receiver import ThermalVerdictReceiver


def _send(port: int, payload: dict) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", port))
    sock.close()


def test_recv_parses_valid_verdict():
    receiver = ThermalVerdictReceiver(host="127.0.0.1", port=0, timeout=1.0)
    port = receiver._sock.getsockname()[1]

    _send(port, {"person": True, "ts": 123.0})
    verdict = receiver.recv()

    assert verdict is not None
    assert verdict.person is True

    receiver.close()


def test_recv_times_out_when_no_data():
    receiver = ThermalVerdictReceiver(host="127.0.0.1", port=0, timeout=0.05)
    assert receiver.recv() is None
    receiver.close()


def test_recv_ignores_malformed_packet():
    receiver = ThermalVerdictReceiver(host="127.0.0.1", port=0, timeout=1.0)
    port = receiver._sock.getsockname()[1]

    _send(port, {"not_person": True})
    assert receiver.recv() is None

    receiver.close()
