"""포트 진단 스크립트 — 센서 연결 없이 raw 바이트 확인."""

import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).parents[1]))

CLI_PORT  = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
DATA_PORT = sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyUSB1"
CLI_BAUD  = 115200
DATA_BAUD = 921600
MAGIC     = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

print(f"=== 포트 진단 ===")
print(f"CLI  port: {CLI_PORT}  baud={CLI_BAUD}")
print(f"Data port: {DATA_PORT}  baud={DATA_BAUD}")
print()

# 1) 데이터 포트에서 5초간 raw 바이트 수신 확인
print("[1] 데이터 포트 raw 수신 테스트 (5초)...")
try:
    with serial.Serial(DATA_PORT, DATA_BAUD, timeout=0.5) as s:
        total = 0
        magic_found = False
        for _ in range(10):
            chunk = s.read(4096)
            total += len(chunk)
            if not magic_found and MAGIC in chunk:
                magic_found = True
                idx = chunk.index(MAGIC)
                print(f"  magic word 발견! offset={idx}  chunk_size={len(chunk)}")
            else:
                print(f"  chunk: {len(chunk)} bytes  total={total}"
                      + (f"  first8={chunk[:8].hex()}" if chunk else "  (empty)"))
        print(f"  => 5초간 수신 합계: {total} bytes  magic_found={magic_found}")
except serial.SerialException as e:
    print(f"  [FAIL] 데이터 포트 열기 실패: {e}")

print()

# 2) CLI 포트에 sensorStop / sensorStart 전송 후 데이터 확인
print("[2] CLI 포트로 sensorStop -> sensorStart 전송...")
try:
    with serial.Serial(CLI_PORT, CLI_BAUD, timeout=1) as cli:
        for cmd in ["sensorStop", "sensorStart"]:
            cli.write((cmd + "\n").encode())
            time.sleep(0.1)
            resp = cli.read(cli.in_waiting or 1)
            print(f"  {cmd!r} -> {resp.decode(errors='replace').strip()!r}")
except serial.SerialException as e:
    print(f"  [FAIL] CLI 포트 열기 실패: {e}")

print()

# 3) sensorStart 후 데이터 포트 재확인
print("[3] sensorStart 후 데이터 포트 재확인 (3초)...")
try:
    with serial.Serial(DATA_PORT, DATA_BAUD, timeout=0.5) as s:
        total = 0
        magic_found = False
        for _ in range(6):
            chunk = s.read(4096)
            total += len(chunk)
            if not magic_found and MAGIC in chunk:
                magic_found = True
                idx = chunk.index(MAGIC)
                print(f"  magic word 발견! offset={idx}  chunk_size={len(chunk)}")
            else:
                print(f"  chunk: {len(chunk)} bytes" + (f"  first8={chunk[:8].hex()}" if chunk else "  (empty)"))
        print(f"  => 3초간 수신 합계: {total} bytes  magic_found={magic_found}")
except serial.SerialException as e:
    print(f"  [FAIL] 데이터 포트 열기 실패: {e}")
