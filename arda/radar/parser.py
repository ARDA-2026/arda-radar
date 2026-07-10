"""TI mmWave SDK UART 프레임 파서 (IWR6843AOP)."""

import struct
from dataclasses import dataclass, field
import numpy as np

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

# TLV 타입 상수 (mmWave SDK 3.x)
TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_AZIMUTH_STATIC_HEAT_MAP = 4
TLV_RANGE_DOPPLER_HEAT_MAP = 5
TLV_STATS = 6
TLV_DETECTED_POINTS_SIDE_INFO = 7


@dataclass
class DetectedPoint:
    x: float
    y: float
    z: float
    doppler: float
    snr: float = 0.0
    noise: float = 0.0


@dataclass
class RadarFrame:
    frame_number: int
    timestamp: float
    points: list[DetectedPoint] = field(default_factory=list)
    num_tlvs: int = 0


class FrameParser:
    """바이트 스트림에서 mmWave 프레임을 파싱한다."""

    def __init__(self):
        self._buffer = b""

    def parse(self, data: bytes) -> dict | None:
        self._buffer += data

        idx = self._buffer.find(MAGIC_WORD)
        if idx == -1:
            if len(self._buffer) > 100:
                self._buffer = b""
            return None
        self._buffer = self._buffer[idx:]

        if len(self._buffer) < 40:  # 최소 헤더 크기
            return None

        # 헤더 파싱 (TI SDK 표준 구조)
        header_fmt = "<8sIIIIIIII"
        header_size = struct.calcsize(header_fmt)
        magic, version, total_len, platform, frame_num, time_cpu, num_detected, num_tlvs, subframe = struct.unpack(
            header_fmt, self._buffer[:header_size]
        )

        if len(self._buffer) < total_len:
            return None  # 아직 전체 프레임 미도착

        frame_bytes = self._buffer[:total_len]
        self._buffer = self._buffer[total_len:]

        frame = RadarFrame(frame_number=frame_num, timestamp=time_cpu / 1e6, num_tlvs=num_tlvs)
        offset = header_size
        side_info: list[tuple[float, float]] = []
        rd_heatmap_raw: np.ndarray | None = None

        for _ in range(num_tlvs):
            if offset + 8 > len(frame_bytes):
                break
            tlv_type, tlv_len = struct.unpack("<II", frame_bytes[offset : offset + 8])
            offset += 8
            payload = frame_bytes[offset : offset + tlv_len]
            offset += tlv_len

            if tlv_type == TLV_DETECTED_POINTS:
                point_size = 16  # x, y, z, doppler (4 floats)
                for i in range(len(payload) // point_size):
                    x, y, z, d = struct.unpack("<ffff", payload[i * point_size : (i + 1) * point_size])
                    frame.points.append(DetectedPoint(x=x, y=y, z=z, doppler=d))

            elif tlv_type == TLV_DETECTED_POINTS_SIDE_INFO:
                info_size = 4  # snr, noise (2 uint16)
                for i in range(len(payload) // info_size):
                    snr, noise = struct.unpack("<HH", payload[i * info_size : (i + 1) * info_size])
                    side_info.append((snr * 0.1, noise * 0.1))

            elif tlv_type == TLV_RANGE_DOPPLER_HEAT_MAP:
                # uint16 배열: [numRangeBins × numDopplerBins], row-major (range 우선)
                n_vals = len(payload) // 2
                rd_heatmap_raw = np.frombuffer(payload, dtype=np.uint16).reshape(-1, n_vals) if n_vals else None
                # reshape는 호출측에서 (numRangeBins, numDopplerBins)로 수행
                rd_heatmap_raw = np.frombuffer(payload, dtype=np.uint16).astype(np.float32)

        for i, pt in enumerate(frame.points):
            if i < len(side_info):
                pt.snr, pt.noise = side_info[i]

        result: dict = {
            "frame_number": frame.frame_number,
            "timestamp": frame.timestamp,
            "points": [
                {"x": p.x, "y": p.y, "z": p.z, "doppler": p.doppler, "snr": p.snr, "noise": p.noise}
                for p in frame.points
            ],
        }
        if rd_heatmap_raw is not None:
            result["rd_heatmap"] = rd_heatmap_raw
        return result
