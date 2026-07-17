"""config/settings.yaml 로더.

detect.py/trajectory.py/record.py/record_and_view.py/main.py가 각자
파일 상단에 중복 정의하던 전처리 튜닝 상수(MIN_SNR, CLUSTER_EPS 등)를
config/settings.yaml의 processing: 섹션 한 곳에서 읽어오도록 공유한다.
"""

from pathlib import Path

import yaml

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")


def load_settings(path: Path | str = DEFAULT_SETTINGS_PATH) -> dict:
    """settings.yaml을 그대로 파싱해 반환한다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processing_config(path: Path | str = DEFAULT_SETTINGS_PATH) -> dict:
    """processing: 섹션을 읽어온다. 없는 키는 기존 기본값으로 채운다.

    반환 키: min_snr, roi_x, roi_y, roi_z, min_abs_doppler,
             cluster_eps, cluster_min_samples, airborne_z, max_jump
    """
    proc = load_settings(path).get("processing", {})
    roi  = proc.get("roi", {})
    return {
        "min_snr":             proc.get("min_snr", 8.0),
        "roi_x":               tuple(roi.get("x", (-1.5, 1.5))),
        "roi_y":               tuple(roi.get("y", (0.3, 2.5))),
        "roi_z":               tuple(roi.get("z", (-0.2, 2.2))),
        "min_abs_doppler":     proc.get("min_abs_doppler", 0.1),
        "cluster_eps":         proc.get("cluster_eps", 0.5),
        "cluster_min_samples": proc.get("cluster_min_samples", 3),
        "airborne_z":          proc.get("airborne_z", 0.40),
        "max_jump":            proc.get("max_jump", 0.5),
    }
