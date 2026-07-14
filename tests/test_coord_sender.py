"""CoordSender 단위 테스트."""

import json
import socket

import numpy as np

from arda.utils.coord_sender import CoordSender


def test_send_emits_udp_json_payload():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(1.0)
    port = listener.getsockname()[1]

    sender = CoordSender(host="127.0.0.1", port=port)
    sender.send(np.array([0.5, 1.2, 0.3]), fall=True)

    data, _ = listener.recvfrom(4096)
    payload = json.loads(data.decode("utf-8"))

    assert payload["x"] == 0.5
    assert payload["y"] == 1.2
    assert payload["z"] == 0.3
    assert payload["fall"] is True
    assert "ts" in payload

    sender.close()
    listener.close()
