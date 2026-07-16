"""to_site_coords 단위 테스트."""

import numpy as np

from arda.utils.site import to_site_coords


def test_zero_origin_returns_unchanged_coords():
    local = np.array([0.5, 1.2, 0.3])
    site = to_site_coords(local, [0.0, 0.0, 0.0])
    assert np.allclose(site, local)


def test_nonzero_origin_offsets_coords():
    local = np.array([0.5, 1.2, 0.3])
    origin = [10.0, 20.0, 0.0]
    site = to_site_coords(local, origin)
    assert np.allclose(site, [10.5, 21.2, 0.3])


def test_accepts_plain_list_inputs():
    site = to_site_coords([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert np.allclose(site, [2.0, 2.0, 2.0])


def test_negative_local_z_below_sensor_subtracts_correctly():
    # 센서 아래에서 탐지된 포인트는 로컬 Z가 이미 음수이므로,
    # 더하기만 해도 site.z보다 낮은 실좌표가 자연스럽게 나온다.
    local = np.array([0.0, 1.0, -0.8])
    origin = [0.0, 0.0, 1.1]
    site = to_site_coords(local, origin)
    assert np.allclose(site, [0.0, 1.0, 0.3])
