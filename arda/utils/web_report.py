"""열화상까지 확인을 마친 최종 낙하 위치를 웹 엔드포인트로 전송.

레이더 낙하 판단, 열화상 판정 요청/기각 같은 중간 과정은 로그로만 남기고,
여기서는 "열화상이 사람으로 확인한" 최종 결과만 아래 최소 포맷으로 보낸다:

{"lat": <위도>, "lon": <경도>, "timestamp": "<한국시간 ISO8601>"}

image_jpeg가 주어지면(arda-raset처럼 열화상을 같은 프로세스에서 통합 실행할
때만 해당 — UDP 기반 arda-radar 단독 실행은 이 인자를 안 씀) base64로 인코딩해
"thermal_image_base64"/"confirmed" 필드를 추가로 실어 보낸다. 웹이 아직
개발 중이라 이 두 필드명은 확정 스펙이 아니라 잠정 값 — 백엔드와 맞춰볼 것.
"""

import base64
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from .logger import get_logger

logger = get_logger(__name__)

KST = timezone(timedelta(hours=9))


def send_fall_report(
    url: str,
    lat: float,
    lon: float,
    image_jpeg: bytes | None = None,
    confirmed: bool = False,
    timeout: float = 3.0,
) -> bool:
    """{"lat", "lon", "timestamp"}(한국시간) JSON을 url로 POST한다.

    네트워크 문제로 감지 루프가 죽으면 안 되므로, 실패해도 예외를 던지지
    않고 False만 반환한다.
    """
    payload_dict = {
        "lat": lat,
        "lon": lon,
        "timestamp": datetime.now(KST).isoformat(),
    }
    if image_jpeg is not None:
        payload_dict["thermal_image_base64"] = base64.b64encode(image_jpeg).decode("ascii")
        payload_dict["confirmed"] = confirmed
    payload = json.dumps(payload_dict).encode("utf-8")

    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ok = 200 <= response.status < 300
            if not ok:
                logger.warning("낙하 위치 전송 실패 — HTTP %d", response.status)
            return ok
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        logger.warning("낙하 위치 전송 실패 — %s", e)
        return False
