"""설치 위치 기준 실좌표 변환."""

import numpy as np


def to_site_coords(local_xyz: np.ndarray, site_origin: np.ndarray) -> np.ndarray:
    """센서 기준 로컬 좌표(X,Y,Z, m)를 설치 위치(site_origin) 기준 실좌표로 변환한다.

    site_origin은 이 레이더가 설치된 지점의 고정 실좌표(m)다 — 평행이동만
    적용하며, 센서 장착 방향(heading) 회전은 보정하지 않는다.

    local_xyz의 Z는 TI mmWave SDK가 원래 부호를 갖고 보내는 값이다(센서
    아래는 음수, 위는 양수) — 별도의 부호 반전 없이 그대로 더하면 된다.
    """
    return np.asarray(local_xyz, dtype=np.float64) + np.asarray(site_origin, dtype=np.float64)
