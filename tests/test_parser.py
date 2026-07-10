"""FrameParser 단위 테스트."""

import struct
import pytest
from arda.radar.parser import FrameParser, MAGIC_WORD, TLV_DETECTED_POINTS


def _build_frame(points: list[tuple[float, float, float, float]]) -> bytes:
    """테스트용 최소 mmWave 프레임을 조립한다."""
    tlv_payload = b"".join(struct.pack("<ffff", *p) for p in points)
    tlv = struct.pack("<II", TLV_DETECTED_POINTS, len(tlv_payload)) + tlv_payload

    header_fmt = "<8sIIIIIIII"
    num_tlvs = 1
    total_len = struct.calcsize(header_fmt) + len(tlv)
    header = struct.pack(
        header_fmt,
        MAGIC_WORD, 0x01020304, total_len, 0xA5430000, 1, 0, len(points), num_tlvs, 0
    )
    return header + tlv


def test_parse_single_frame():
    parser = FrameParser()
    pts = [(1.0, 2.0, 0.5, -0.3)]
    raw = _build_frame(pts)
    result = parser.parse(raw)

    assert result is not None
    assert result["frame_number"] == 1
    assert len(result["points"]) == 1
    assert abs(result["points"][0]["x"] - 1.0) < 1e-4


def test_parse_empty_points():
    parser = FrameParser()
    raw = _build_frame([])
    result = parser.parse(raw)
    assert result is not None
    assert result["points"] == []


def test_incomplete_frame_returns_none():
    parser = FrameParser()
    result = parser.parse(b"\x02\x01\x04\x03")
    assert result is None
