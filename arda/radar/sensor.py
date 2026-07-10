"""IWR6843AOPEVM 센서 추상화 레이어."""

import select
import serial
import time
from pathlib import Path
from .parser import FrameParser
from ..utils import get_logger

logger = get_logger(__name__)

# IWR6843AOP 기본 시리얼 설정
CLI_BAUD = 115200
DATA_BAUD = 921600


class IWR6843Sensor:
    """IWR6843AOPEVM 레이더 센서 인터페이스."""

    def __init__(self, cli_port: str, data_port: str):
        self.cli_port = cli_port
        self.data_port = data_port
        self._cli_serial: serial.Serial | None = None
        self._data_serial: serial.Serial | None = None
        self._parser = FrameParser()

    def configure(self, config_path: str | Path) -> None:
        """CLI 포트를 통해 .cfg 파일을 센서에 전송한다."""
        config_path = Path(config_path)
        commands = [
            line.strip()
            for line in config_path.read_text().splitlines()
            if line.strip() and not line.startswith("%")
        ]
        with serial.Serial(self.cli_port, CLI_BAUD, timeout=0) as cli:
            # running 상태(이전 세션 잔존)를 정규화: sensorStop은 멱등성 보장
            print("[CFG] Sending sensorStop to normalize sensor state...")
            cli.reset_input_buffer()
            cli.write(b"sensorStop\n")
            self._read_until_done(cli, timeout=0.5)  # 응답 무시 (실패해도 무관)
            cli.reset_input_buffer()

            if not self._wait_for_ready(cli):
                print("[WARN] Sensor CLI unresponsive — attempting config anyway")

            empty_streak = 0
            for cmd in commands:
                cli.write((cmd + "\n").encode())
                wait = 1.0 if "sensorStart" in cmd else 0.6
                resp_bytes = self._read_until_done(cli, timeout=wait)
                resp_str = resp_bytes.decode(errors="replace").strip()

                if not resp_str:
                    empty_streak += 1
                    if empty_streak >= 5:
                        raise RuntimeError(
                            "Sensor CLI not responding (5 consecutive empty replies).\n"
                            "  1) Unplug sensor, let it cool for 10+ seconds, reconnect\n"
                            "  2) Windows: usbipd detach -> usbipd attach --wsl"
                        )
                else:
                    empty_streak = 0

                level = "ERROR" if "error" in resp_str.lower() else "OK"
                print(f"[CFG {level}] {cmd!r:40s} -> {resp_str!r}")

        logger.info("Configuration sent from %s", config_path)

    @staticmethod
    def _wait_for_ready(port: serial.Serial, timeout: float = 5.0) -> bool:
        """펌웨어 CLI가 응답할 때까지 대기한다. 응답 여부를 bool로 반환.

        sensorStop 이후에 호출하므로 센서는 stopped 상태.
        'version' 명령에 응답이 오면 준비 완료로 판단한다.
        """
        t_start = time.monotonic()
        deadline = t_start + timeout

        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            # 잔류 데이터 flush 후 version 전송
            port.reset_input_buffer()
            port.write(b"version\n")

            # 응답 대기: 최대 1.5s
            buf = b""
            t_end = time.monotonic() + 1.5
            while time.monotonic() < t_end:
                ready, _, _ = select.select([port], [], [], 0.05)
                if ready:
                    buf += port.read(port.in_waiting or 1)
                    if buf:
                        break

            if buf:
                elapsed = time.monotonic() - t_start
                logger.info("Sensor ready (attempt=%d, elapsed=%.1fs)", attempt, elapsed)
                port.reset_input_buffer()
                return True

            logger.debug("Waiting for sensor CLI... (attempt %d)", attempt)

        logger.warning(
            "Sensor CLI did not respond within %.0fs — proceeding anyway. "
            "If config fails, power-cycle the sensor and reattach USB.",
            timeout,
        )
        return False

    @staticmethod
    def _read_until_done(port: serial.Serial, timeout: float = 0.4) -> bytes:
        """Done 또는 Error가 올 때까지 읽거나 timeout 후 반환.

        마지막 수신 후 50ms 이상 추가 데이터가 없으면 종료한다.
        (청크 분할 응답 대응 — 이전의 '첫 청크 후 즉시 종료' 버그 수정)
        """
        buf = b""
        deadline = time.monotonic() + timeout
        last_rx = None
        while time.monotonic() < deadline:
            ready, _, _ = select.select([port], [], [], 0.01)
            if ready:
                chunk = port.read(port.in_waiting or 1)
                if chunk:
                    buf += chunk
                    last_rx = time.monotonic()
                    if b"Done" in buf or b"Error" in buf or b"error" in buf:
                        break
            elif last_rx is not None and (time.monotonic() - last_rx) > 0.05:
                # 마지막 수신 후 50ms 조용하면 응답 완료로 간주
                break
        return buf

    def open(self) -> None:
        self._data_serial = serial.Serial(self.data_port, DATA_BAUD, timeout=1)
        logger.info("Data port %s opened", self.data_port)

    def close(self) -> None:
        if self._data_serial and self._data_serial.is_open:
            self._data_serial.close()

    def read_frame(self, poll_timeout: float = 1.0) -> dict | None:
        """데이터 포트에서 한 프레임을 읽어 파싱된 dict를 반환한다.

        WSL2의 cp210x blocking 문제를 피하기 위해 select() + in_waiting 방식 사용.
        """
        if self._data_serial is None:
            raise RuntimeError("Sensor not opened. Call open() first.")

        ready, _, _ = select.select([self._data_serial], [], [], poll_timeout)
        if not ready:
            return None

        available = self._data_serial.in_waiting
        if available == 0:
            return None

        raw = self._data_serial.read(min(available, 65536))
        if not raw:
            return None
        return self._parser.parse(raw)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
