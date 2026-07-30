"""설치 위치(위도/경도)와 정면 방위각 기준 실좌표(GPS) 변환."""

import math

# 위도 1도당 거리(m). 지구를 구로 근사한 값 — 위도에 따라 실제로는 미세하게
# 다르지만(적도 110.57km ~ 극지 111.69km) 이 프로젝트의 탐지 범위(수 m)에서는
# 그 차이가 무시할 수준이라 하나의 상수로 충분하다.
METERS_PER_DEG_LAT = 111_320.0


def local_to_latlon(
    x: float,
    y: float,
    site_lat: float,
    site_lon: float,
    heading_deg: float = 0.0,
) -> tuple[float, float]:
    """센서 기준 로컬 좌표(X: 좌우 +우측, Y: 정면 거리, 단위 m)를 설치 지점의
    위도/경도와 정면 방위각(heading_deg)을 이용해 실제 위도/경도로 변환한다.

    heading_deg는 센서가 정면(Y축, boresight)으로 바라보는 나침반 방위각이다
    (정북 0°, 시계방향으로 증가 — 동 90°, 남 180°, 서 270°). 로컬 좌표를 이
    각도만큼 회전시켜 동/북 방향 변위로 바꾸므로, 기기가 어느 방향을 보고
    설치되든(정북이 아니어도) 그 방향에 맞는 부호·축으로 자동 계산된다 —
    예를 들어 정동(90°)을 보고 있으면 센서의 "정면(Y)"이 실제로는 동쪽
    이동으로, "우측(X)"은 남쪽 이동으로 계산된다.
    """
    heading_rad = math.radians(heading_deg)

    # 방위각 θ 방향의 단위 벡터는 (East=sin θ, North=cos θ). 정면(Y)의 방위각은
    # heading_deg 그 자체, 우측(X)은 그보다 시계방향으로 90도 회전한 방위각이다.
    east_m = x * math.cos(heading_rad) + y * math.sin(heading_rad)
    north_m = -x * math.sin(heading_rad) + y * math.cos(heading_rad)

    delta_lat = north_m / METERS_PER_DEG_LAT
    delta_lon = east_m / (METERS_PER_DEG_LAT * math.cos(math.radians(site_lat)))

    return site_lat + delta_lat, site_lon + delta_lon
