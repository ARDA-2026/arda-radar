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
PEAK_Z_THRESHOLD   = 0.40  # m — 공중 판별 최소 높이
PEAK_DROP_THRESHOLD = 0.35  # m — 피크 대비 최소 하락폭 (낙하 확정)
MIN_DESCENT_FRAMES  = 5     # 피크 이후 연속 하강 최소 프레임 수 (500ms)

CONFIRM_FRAMES = 1          # 낙하 조건 충족 즉시 확정
HISTORY_WINDOW = 10         # 프레임 수 (100ms × 10 = 1.0초)
FRAME_DT       = 0.10       # 초 (100ms 프레임 주기)


class FallDetector:
    """Z축 하강 궤적 기반 낙하 감지기.

    두 가지 경로로 낙하를 감지한다.

    경로 1 — 착지 후 소실 (주 경로):
        물체가 피크에서 MIN_DESCENT_FRAMES 프레임 이상 연속 하강한 뒤
        레이더에서 사라지면 (클러스터 미형성) 착지로 판단한다.
        바닥 근처에서 포인트가 1개 이하로 줄어 클러스터가 사라지는
        실제 낙하 패턴을 포착한다.

    경로 2 — 저 Z 직접 감지 (보조 경로):
        물체가 레이더에 보이는 상태로 충분히 낮은 위치까지 하강한 경우.

    last_fall_centroid: 감지 직전 마지막으로 알려진 무게중심 (X, Y, Z).
    """

    def __init__(self, history_window: int = HISTORY_WINDOW, debug: bool = False,
                 confirm_frames: int = CONFIRM_FRAMES):
        self._height_history: deque[float | None] = deque(maxlen=history_window)
        self._doppler_history: deque[float] = deque(maxlen=history_window)
        self._fall_triggered  = False
        self._trigger_reason  = ""
        self._candidate_frames = 0
        self._confirm_frames   = confirm_frames
        self._debug            = debug

        self._last_centroid      = None   # 마지막으로 본 무게중심
        self.last_fall_centroid  = None   # 낙하 감지 시 저장된 위치

    # ── 메인 업데이트 ────────────────────────────────────────────────────────

    def update(self, pc: PointCloud) -> bool:
        """새 프레임을 입력받아 낙하 여부를 반환한다."""
        centroid = pc.centroid()

        if centroid is None:
            return self._update_no_target()

        self._last_centroid = centroid
        height       = float(centroid[2])
        mean_doppler = float(np.mean(pc.doppler))
        self._height_history.append(height)
        self._doppler_history.append(mean_doppler)

        candidate, reason = self._check_fall_visible(debug=self._debug)
        if candidate:
            self._candidate_frames += 1
        else:
            self._candidate_frames = 0

        fell = self._candidate_frames >= self._confirm_frames
        if fell and not self._fall_triggered:
            self._fall_triggered    = True
            self._trigger_reason    = reason
            self.last_fall_centroid = centroid
            logger.warning("FALL DETECTED [%s] — Z=%.2f m", reason, height)
        elif not fell:
            self._fall_triggered = False

        return fell

    def _update_no_target(self) -> bool:
        """타겟 클러스터가 없을 때 — 착지 소실 패턴 확인 후 이력 업데이트."""
        # 착지 소실 감지: 히스토리에 충분한 하강 궤적이 있으면 낙하로 판정
        fell = False
        if not self._fall_triggered:
            fell, reason = self._check_fall_on_disappear()
            if fell:
                self._fall_triggered    = True
                self._trigger_reason    = reason
                self.last_fall_centroid = self._last_centroid
                z = self._last_centroid[2] if self._last_centroid is not None else float("nan")
                logger.warning("FALL DETECTED (landing disappearance) [%s] — last Z=%.2f m",
                               reason, z)

        self._height_history.append(None)
        self._doppler_history.append(0.0)
        self._candidate_frames = 0

        # 소실 후 fall_triggered 유지: 다음 valid 프레임이 올 때 해제
        return fell

    # ── 낙하 조건 체크 ───────────────────────────────────────────────────────

    def _check_fall_visible(self, debug: bool = False) -> tuple[bool, str]:
        """경로 2: 물체가 보이는 상태에서의 직접 감지."""
        history = list(self._height_history)
        valid   = [(i, h) for i, h in enumerate(history) if h is not None]

        if len(valid) < 2:
            return False, ""

        result, reason = self._trajectory_check(valid)
        if debug and valid:
            peak_z = max(h for _, h in valid)
            cur_z  = valid[-1][1]
            print(f"[FD] peak_z={peak_z:.3f}  cur_z={cur_z:.3f}"
                  f"  cand={self._candidate_frames}  {'HIT' if result else ''}")
        return result, reason

    def _check_fall_on_disappear(self) -> tuple[bool, str]:
        """경로 1: 물체 소실 시 직전 히스토리로 낙하 완료 판단."""
        history = list(self._height_history)
        valid   = [(i, h) for i, h in enumerate(history) if h is not None]
        if len(valid) < 2:
            return False, ""
        return self._trajectory_check(valid)

    def _trajectory_check(self, valid: list[tuple[int, float]]) -> tuple[bool, str]:
        """히스토리 내 하강 궤적 공통 판정 로직.

        - 피크 Z >= PEAK_Z_THRESHOLD
        - 피크 이후 유효 프레임이 MIN_DESCENT_FRAMES 이상 존재
        - 피크 이후 모든 값이 피크 이하 (반등 없음)
        - 마지막 유효 Z가 피크 대비 PEAK_DROP_THRESHOLD 이상 하락
        """
        peak_pos   = max(range(len(valid)), key=lambda k: valid[k][1])
        peak_z     = valid[peak_pos][1]
        peak_frame = valid[peak_pos][0]

        if peak_z < PEAK_Z_THRESHOLD:
            return False, ""

        post_peak = [h for i, h in valid if i > peak_frame]

        if len(post_peak) < MIN_DESCENT_FRAMES:
            return False, ""

        if any(h > peak_z for h in post_peak):
            return False, ""

        last_z    = post_peak[-1]
        peak_drop = peak_z - last_z

        if peak_drop >= PEAK_DROP_THRESHOLD:
            reason = (f"peak={peak_z:.2f}m"
                      f"  descent={len(post_peak)}f"
                      f"  drop={peak_drop:.2f}m")
            return True, reason

        return False, ""

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    def z_velocity(self) -> float:
        """직전 두 프레임의 Z 순간 속도 (m/s). 시각화용."""
        valid = [h for h in self._height_history if h is not None]
        if len(valid) < 2:
            return 0.0
        return (valid[-1] - valid[-2]) / FRAME_DT

    def reset(self) -> None:
        self._height_history.clear()
        self._doppler_history.clear()
        self._fall_triggered   = False
        self._candidate_frames = 0
        self._last_centroid    = None
        self.last_fall_centroid = None
