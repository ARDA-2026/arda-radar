"""낙하 감지 메인 로직."""

from collections import deque
import numpy as np

from ..processing.clustering import select_target
from ..processing.pointcloud import PointCloud
from ..utils import get_logger
from .tracker import KalmanTracker

logger = get_logger(__name__)

# 낙하 판정 임계값
# 실험 데이터(10s, 낙하 3회) 분석 결과:
#   - 실제 낙하 시 피크 Z: 0.57 ~ 0.69m
#   - 비낙하(노이즈) 최대 Z: 0.36m
# data/failDetecting.png 재현 결과(낮고 빠른 실제 낙하): 피크 Z ≈ 0.38m,
# 소실 전까지 유효 프레임 3개뿐 — 기존 값(0.40m / 5프레임)으로는 궤적이
# 깨끗한 단조 하강이어도 두 게이트 모두에서 막혔다. 물리적으로도 피크
# 0.4m 근처에서 자유낙하하면 바닥까지 채 3프레임(300ms)이 안 걸리므로,
# 5프레임(500ms) 요구는 애초에 낮은 낙하에서는 충족 불가능하다.
# data/raw/drop_test_20260715 5회 재생 분석 결과: 실제 낙하는 피크 이후
# 유효 프레임이 1~2개에서 소실되는 경우가 대부분이라(레이더 반사가
# 바닥 근처에서 급격히 희박해짐), 3프레임 요구로는 5회 중 1회만 감지됐다.
PEAK_Z_THRESHOLD   = 0.37  # m — 공중 판별 최소 높이 (노이즈 최대 0.36m보다는 높게 유지)
PEAK_DROP_THRESHOLD = 0.35  # m — 피크 대비 최소 하락폭 (낙하 확정)
MIN_DESCENT_FRAMES  = 2     # 피크 이후 연속 하강 최소 프레임 수 (200ms)

# 피크 대비 하락폭만 보면, 바닥을 찍고 다시 위로 올라가는 중(바운스,
# 노이즈로 무게중심이 끌려 올라가는 경우 등)에도 "피크보다는 여전히
# 낮다"는 이유로 낙하로 오판할 수 있다. 그래서 마지막 두 유효 프레임 사이
# 높이가 이 값 넘게 상승했으면(=현재 궤적이 위로 향함) 후보에서 제외한다.
# 칼만 필터의 속도 추정치(z_velocity)는 등속도 모델 특성상 방향 전환에
# 여러 프레임 지연되어 반응하므로, 방향 전환 감지에는 위치 자체의 최근
# 변화량을 직접 보는 편이 더 즉각적이다.
RISING_TOLERANCE = 0.03  # m — 프레임 간 이 값 넘게 상승하면 하강 중이 아님

# data/failTracking.png 재현 결과: 실제로 추적 중이던 물체의 클러스터가
# 딱 한두 프레임 형성되지 않은 순간(포인트 부족 등), select_target이
# 근접 게이팅을 포기하고 전역 재탐색으로 넘어가면서 근처에 있던 무관한
# 정지 클러스터를 그대로 "재포착"으로 착각해 무게중심이 튀었다. 게이팅
# 실패가 이 프레임 수 이내면 재탐색을 미루고 칼만 예측만으로 버틴다.
REACQUIRE_GRACE_FRAMES = 2  # 프레임 — 이 이상 연속으로 근처에 후보가 없어야 전역 재탐색 허용

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

    내부적으로 칼만 필터(KalmanTracker, 등속도 모델)를 태워 무게중심을
    평활화한다. 궤적 판정(_height_history 등)은 원시 관측치가 아니라 이
    필터링된 위치를 사용한다 — 한 프레임의 클러스터 선택이 잘못되어도
    무게중심이 통째로 튀지 않고 예측과 블렌딩된 만큼만 움직이게 된다.

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

        self._tracker   = KalmanTracker(dt=FRAME_DT)
        self._tracking  = False           # 칼만 필터가 최소 1회 이상 보정됐는지
        self._gate_miss_streak = 0        # 근접 게이팅이 연속으로 실패한 프레임 수

        self._last_centroid      = None   # 마지막으로 본(필터링된) 무게중심
        self.last_fall_centroid  = None   # 낙하 감지 시 저장된 위치

    # ── 메인 업데이트 ────────────────────────────────────────────────────────

    def choose_target(self, clusters: list[PointCloud],
                       airborne_z: float = 0.40, fall_doppler: float = -0.1,
                       max_jump: float = 0.5, max_rise: float = 0.05) -> PointCloud:
        """DBSCAN 클러스터 목록에서 추적할 타겟을 고른다 (select_target 래퍼).

        칼만 예측 위치로 근접 게이팅하되(select_target), 게이팅 실패가
        REACQUIRE_GRACE_FRAMES를 넘기 전까지는 전역 재탐색을 미루고 빈
        PointCloud를 반환한다 — 그래야 실제 물체의 클러스터가 잠깐(1~2
        프레임) 형성되지 않았을 때 근처의 무관한 클러스터로 즉시
        "재포착"되어 무게중심이 튀는 일을 막을 수 있다. 빈 PointCloud가
        반환되면 update()가 알아서 착지 소실 경로로 처리하며 칼만 필터는
        예측만으로 코스팅한다.

        max_jump는 연속으로 게이팅에 실패한 프레임 수(_gate_miss_streak)에
        비례해 넓힌다 — 추적이 뜨문뜨문 이어질 때는 칼만 속도 추정이 아직
        실제 낙하 속도를 못 따라잡아 예측 위치가 실제 물체보다 훨씬 위에
        머물러 있을 수 있는데, 고정된 max_jump로는 그 사이 이미 한참
        떨어진 실제 물체를 거리 초과로 놓치게 된다(data/raw/drop_test_
        20260715 재생 중 실측).
        """
        predicted = self.predicted_centroid()
        strict = predicted is not None and self._gate_miss_streak < REACQUIRE_GRACE_FRAMES
        effective_max_jump = max_jump * (1 + self._gate_miss_streak)

        target = select_target(clusters, airborne_z=airborne_z, fall_doppler=fall_doppler,
                                last_centroid=predicted, max_jump=effective_max_jump,
                                max_rise=max_rise, strict=strict)

        if len(target) > 0:
            self._gate_miss_streak = 0
        elif predicted is not None:
            self._gate_miss_streak += 1

        return target

    def predicted_centroid(self) -> np.ndarray | None:
        """다음 관측이 위치할 것으로 예상되는 지점 (칼만 예측, 게이팅 기준점).

        내부 상태를 변경하지 않는 미리보기(peek)다 — 실제 시간 전진은
        update()/_update_no_target() 호출 시 이루어진다. 아직 트랙이
        시작되지 않았으면(첫 관측 전) None을 반환해 게이팅 없이 전역
        탐색하도록 한다.
        """
        if not self._tracking:
            return None
        return (self._tracker.F @ self._tracker.x)[:3].flatten()

    def update(self, pc: PointCloud) -> bool:
        """새 프레임을 입력받아 낙하 여부를 반환한다."""
        centroid = pc.centroid()

        if centroid is None:
            return self._update_no_target()

        if self._tracking:
            self._tracker.predict()
        self._tracker.update(np.asarray(centroid, dtype=np.float64))
        self._tracking = True
        centroid = self._tracker.x[:3].flatten()   # 평활화된 위치로 대체

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
        # 관측 없음 — 보정 없이 예측만으로 코스팅 (짧은 소실 대비)
        if self._tracking:
            self._tracker.predict()

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
        - 마지막 두 유효 프레임 사이 궤적이 상승 중이 아님 (바운스/재상승 제외)
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

        # 피크 대비 하락폭만으로는 "바닥 찍고 다시 올라가는 중"을 구분할 수
        # 없다 — 최근 궤적이 실제로 상승 중이면 낙하 후보에서 제외한다.
        if len(post_peak) >= 2 and (post_peak[-1] - post_peak[-2]) > RISING_TOLERANCE:
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

    @property
    def last_centroid(self) -> np.ndarray | None:
        """가장 최근에 관측된 무게중심. select_target()의 근접 게이팅용."""
        return self._last_centroid

    def z_velocity(self) -> float:
        """Z 속도 (m/s). 시각화용 — 칼만 필터의 평활화된 속도 추정치를 쓴다."""
        if self._tracking:
            return float(self._tracker.x[5, 0])
        return 0.0

    def reset(self) -> None:
        self._height_history.clear()
        self._doppler_history.clear()
        self._fall_triggered   = False
        self._candidate_frames = 0
        self._tracker           = KalmanTracker(dt=FRAME_DT)
        self._tracking          = False
        self._gate_miss_streak  = 0
        self._last_centroid    = None
        self.last_fall_centroid = None
