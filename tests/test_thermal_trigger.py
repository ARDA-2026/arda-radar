"""ThermalTriggerSender 단위 테스트."""

import json
import socket

from arda.utils.thermal_trigger import ThermalTriggerSender


def test_send_emits_udp_json_with_timestamp():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(1.0)
    port = listener.getsockname()[1]

    sender = ThermalTriggerSender(host="127.0.0.1", port=port)
    sender.send()

    data, _ = listener.recvfrom(4096)
    payload = json.loads(data.decode("utf-8"))

    assert "ts" in payload

    sender.close()
    listener.close()
