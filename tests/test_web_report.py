"""send_fall_report() 단위 테스트."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from arda.utils.web_report import send_fall_report


class _CapturingHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.received.append(json.loads(body.decode("utf-8")))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass  # 테스트 출력에 접속 로그가 섞이지 않게 함


def _start_server():
    _CapturingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_send_fall_report_posts_minimal_json():
    server = _start_server()
    try:
        port = server.server_address[1]
        ok = send_fall_report(f"http://127.0.0.1:{port}/", lat=37.5665, lon=126.9780)

        assert ok is True
        assert len(_CapturingHandler.received) == 1
        payload = _CapturingHandler.received[0]
        assert payload["lat"] == 37.5665
        assert payload["lon"] == 126.9780
        assert set(payload.keys()) == {"lat", "lon", "timestamp"}
        # 한국시간(+09:00) ISO8601 형식인지만 확인 — 정확한 현재 시각까지는 검증하지 않음.
        assert payload["timestamp"].endswith("+09:00")
    finally:
        server.shutdown()


def test_send_fall_report_returns_false_on_connection_failure():
    # 아무도 듣고 있지 않은 포트로 보내면 연결 자체가 실패해야 한다.
    ok = send_fall_report("http://127.0.0.1:1/", lat=0.0, lon=0.0, timeout=1.0)
    assert ok is False
