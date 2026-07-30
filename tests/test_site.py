"""local_to_latlon() 단위 테스트."""

import math

import pytest

from arda.utils.site import METERS_PER_DEG_LAT, local_to_latlon


def test_zero_offset_returns_site_latlon_unchanged():
    lat, lon = local_to_latlon(0.0, 0.0, site_lat=37.5, site_lon=127.0, heading_deg=0.0)
    assert lat == pytest.approx(37.5)
    assert lon == pytest.approx(127.0)


def test_heading_zero_forward_moves_north_right_moves_east():
    # 정북(heading=0)을 보고 있으면 정면(y)은 북쪽, 우측(x)은 동쪽으로 이동해야 한다.
    lat_fwd, lon_fwd = local_to_latlon(0.0, METERS_PER_DEG_LAT, site_lat=0.0, site_lon=0.0, heading_deg=0.0)
    assert lat_fwd == pytest.approx(1.0)  # 위도 1도만큼 북쪽
    assert lon_fwd == pytest.approx(0.0, abs=1e-9)

    lat_right, lon_right = local_to_latlon(METERS_PER_DEG_LAT, 0.0, site_lat=0.0, site_lon=0.0, heading_deg=0.0)
    assert lat_right == pytest.approx(0.0, abs=1e-9)
    assert lon_right == pytest.approx(1.0)  # 적도(lat=0)라 경도 1도 = 위도 1도와 같은 거리


def test_heading_90_east_forward_moves_east_right_moves_south():
    # 정동(heading=90)을 보고 있으면 정면(y)은 동쪽, 우측(x)은 남쪽으로 이동해야 한다
    # — 부호가 heading에 따라 자동으로 바뀌는지 확인하는 핵심 테스트.
    lat_fwd, lon_fwd = local_to_latlon(0.0, METERS_PER_DEG_LAT, site_lat=0.0, site_lon=0.0, heading_deg=90.0)
    assert lat_fwd == pytest.approx(0.0, abs=1e-9)
    assert lon_fwd == pytest.approx(1.0)

    lat_right, lon_right = local_to_latlon(METERS_PER_DEG_LAT, 0.0, site_lat=0.0, site_lon=0.0, heading_deg=90.0)
    assert lat_right == pytest.approx(-1.0)  # 우측(x)이 남쪽 = 위도 감소
    assert lon_right == pytest.approx(0.0, abs=1e-9)


def test_heading_180_south_flips_both_signs():
    # 정남(heading=180)을 보고 있으면 정면(y)은 남쪽, 우측(x)은 서쪽 —
    # heading=0 대비 부호가 둘 다 반대로 바뀌어야 한다.
    lat, lon = local_to_latlon(
        METERS_PER_DEG_LAT, METERS_PER_DEG_LAT, site_lat=0.0, site_lon=0.0, heading_deg=180.0
    )
    assert lat == pytest.approx(-1.0)
    assert lon == pytest.approx(-1.0)


def test_longitude_compresses_at_higher_latitude():
    # 경도 1도의 실제 거리는 고위도로 갈수록 짧아지므로(위선이 좁아짐),
    # 같은 동쪽 이동 거리(m)에 대해 고위도에서는 경도 변화(도)가 더 커야 한다.
    site_lat = 60.0
    _, lon_high = local_to_latlon(METERS_PER_DEG_LAT, 0.0, site_lat=site_lat, site_lon=0.0, heading_deg=0.0)
    _, lon_low = local_to_latlon(METERS_PER_DEG_LAT, 0.0, site_lat=0.0, site_lon=0.0, heading_deg=0.0)

    delta_high = abs(lon_high - 0.0)
    delta_low = abs(lon_low - 0.0)
    assert delta_high > delta_low
    assert delta_high == pytest.approx(1.0 / math.cos(math.radians(site_lat)))
