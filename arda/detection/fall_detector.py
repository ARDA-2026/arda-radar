"""낙하 감지 메인 로직."""

from collections import deque
import numpy as np

from ..processing.pointcloud import PointCloud
from ..utils import get_logger

logger = get_logger(__name__)

# 낙하 판정 임계값
# 실험 데이터(10s, 낙하 3회) 분석 결과:
#   - 실제 낙하 시 피크 Z: 0.57 ~ 0.69m
#   - 비낙하(노이즈) 최대 Z: 0.36m
#   → 0.40m 기준으로 완전 분리 가능
HEIGHT_DROP_THRESHOLD  = 0.4   # m   — 조건 A: 윈도우 전체 높이 변화량
FALL_DOPPLER_THRESHOLD = -0.3  # m/s — 조건 A: 하향 도플러 보조 조건
PEAK_Z_THRESHOLD       = 0.40  # m   — 조건 B: 이력 내 최고 Z가 이 이상이어야 "공중에 있었음"
PEAK_DROP_THRESHOLD    = 0.30  # m   — 조건 B: 피크 대비 현재 Z 하락폭
CONFIRM_FRAMES = 1             # 낙하는 1프레임(100ms)에 완료 → 즉시 감지
HISTORY_WINDOW = 6             # 프레임 수 (100ms × 6 = 0.6초)
FRAME_DT = 0.10                # 초 (100ms 프레임 주기, 16루프)


class FallDetector:
    """피크 Z 기반 낙하 감지기.

    두 가지 조건 중 하나라도 충족하면 낙하로 판정한다.
    - 조건 A (이력 기반): 윈도우 전체 높이 하락 >= HEIGHT_DROP_THRESHOLD
                         AND 평균 도플러 <= FALL_DOPPLER_THRESHOLD
    - 조건 B (피크 낙하): 이력 내 최고 Z >= PEAK_Z_THRESHOLD (공중에 있었음)
                         AND 현재 Z가 피크 대비 >= PEAK_DROP_THRESHOLD 하락
                         → 물체가 공중에 나타났다가 급락하는 패턴 감지.
                           단조 하강 불필요 — 물체는 1프레임 안에 낙하 완료.
    """

    def __init__(self, history_window: int = HISTORY_WINDOW, debug: bool = False,
                 confirm_frames: int = CONFIRM_FRAMES):
        self._height_history: deque[float | None] = deque(maxlen=history_window)
        self._doppler_history: deque[float] = deque(maxlen=history_window)
        self._fall_triggered = False
        self._trigger_reason = ""
        self._candidate_frames = 0       # 연속 낙하 후보 프레임 카운터
        self._confirm_frames = confirm_frames
        self._debug = debug

    def update(self, pc: PointCloud) -> bool:
        """새 프레임을 입력받아 낙하 여부를 반환한다."""
        centroid = pc.centroid()
        if centroid is None:
            self._height_history.append(None)
            self._doppler_history.append(0.0)
            self._candidate_frames = 0
            return False

        height = float(centroid[2])
        mean_doppler = float(np.mean(pc.doppler))
        self._height_history.append(height)
        self._doppler_history.append(mean_doppler)

        candidate, reason = self._check_fall(debug=self._debug)
        if candidate:
            self._candidate_frames += 1
        else:
            self._candidate_frames = 0

        # confirm_frames 연속으로 낙하 후보여야 실제 낙하로 확정
        fell = self._candidate_frames >= self._confirm_frames

        if fell and not self._fall_triggered:
            self._fall_triggered = True
            self._trigger_reason = reason
            logger.warning("FALL DETECTED [%s] — Z=%.2f m, doppler=%.2f m/s",
                           reason, height, mean_doppler)
        elif not fell:
            self._fall_triggered = False

        return fell

    def _check_fall(self, debug: bool = False) -> tuple[bool, str]:
        valid_heights = [h for h in self._height_history if h is not None]
        if len(valid_heights) < 2:
            if debug:
                print(f"[FD] valid_heights={len(valid_heights)} — not enough data")
            return False, ""

        recent_doppler = list(self._doppler_history)[-3:]
        mean_downward  = float(np.mean(recent_doppler))
        height_drop    = valid_heights[0] - valid_heights[-1]
        peak_z         = max(valid_heights)
        peak_drop      = peak_z - valid_heights[-1]

        if debug:
            print(f"[FD] peak_z={peak_z:+.3f}  cur_z={valid_heights[-1]:+.3f}"
                  f"  peak_drop={peak_drop:.3f}m  drop={height_drop:.3f}m"
                  f"  doppler={mean_downward:+.3f}  cand={self._candidate_frames}")

        # 조건 A: 이력 전체 높이 하락 + 도플러 (느린 낙하 보조)
        if height_drop >= HEIGHT_DROP_THRESHOLD and mean_downward <= FALL_DOPPLER_THRESHOLD:
            return True, f"drop={height_drop:.2f}m"

        # 조건 B: 피크 Z 기반 낙하 감지
        # 이력 내 최고점이 충분히 높고(공중에 있었음), 현재 Z가 그로부터 급락했으면 낙하
        if peak_z >= PEAK_Z_THRESHOLD and peak_drop >= PEAK_DROP_THRESHOLD:
            return True, f"peak_drop={peak_drop:.2f}m(from {peak_z:.2f}m)"

        return False, ""

    def z_velocity(self) -> float:
        """직전 두 프레임의 Z 순간 속도 (m/s). 시각화용."""
        valid = [h for h in self._height_history if h is not None]
        if len(valid) < 2:
            return 0.0
        return (valid[-1] - valid[-2]) / FRAME_DT

    def reset(self) -> None:
        self._height_history.clear()
        self._doppler_history.clear()
        self._fall_triggered = False
        self._candidate_frames = 0
