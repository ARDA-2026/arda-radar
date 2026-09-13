"""열화상까지 확인을 마친 최종 낙하 위치를 웹 엔드포인트로 전송.

레이더 낙하 판단, 열화상 판정 요청/기각 같은 중간 과정은 로그로만 남기고,
여기서는 "열화상이 사람으로 확인한" 최종 결과만 아래 최소 포맷으로 보낸다:

{"lat": <위도>, "lon": <경도>, "timestamp": "<한국시간 ISO8601>"}

image_jpeg가 주어지면(arda-raset처럼 열화상을 같은 프로세스에서 통합 실행할
때만 해당 — UDP 기반 arda-radar 단독 실행은 이 인자를 안 씀) base64로 인코딩해
"thermal_image_base64"/"confirmed" 필드를 추가로 실어 보낸다.

image_jpeg_yolo도 같이 주어지면(실제 온도 그대로인 image_jpeg와 별개로,
YOLO가 보는 배경-상대 보정 이미지) "thermal_image_yolo_base64" 필드로 같이
실어 보낸다 — 웹 쪽이 두 이미지를 한 번에 받아서 그 자리에서 토글로
선택해 볼 수 있게 하기 위함(재요청 없이 클라이언트에서 바로 전환).

report_url을 arda-algo_general(한강 표류 예측 서버, 보통 노트북에서 별도
실행)의 POST /report 엔드포인트로 설정하면, 이 함수가 이미 보내는
payload(이미지 없는 호출은 "낙하 확정" 신호, 이미지 있는 호출은 열화상
스트리밍)를 그대로 받아 시뮬레이션 시작/재수렴 + 열화상 시각화에 쓴다 —
별도 브릿지 함수 없이 report_url 하나로 웹 리포트와 algo_general 연동을
겸한다.
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
    image_jpeg_yolo: bytes | None = None,
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
    if image_jpeg_yolo is not None:
        payload_dict["thermal_image_yolo_base64"] = base64.b64encode(image_jpeg_yolo).decode("ascii")
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
