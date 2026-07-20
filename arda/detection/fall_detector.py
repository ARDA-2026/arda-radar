"""낙하 감지 메인 로직 — 클러스터별 독립 트랙(다중 추적) 기반.

이전엔 매 프레임 여러 DBSCAN 클러스터 중 하나를 "그 물체"로 미리 확정한
뒤(select_target/choose_target) 그 하나의 이력만으로 낙하를 판정했다.
문제는 씬에 클러스터가 여러 개 동시에 있을 때(사람 + 낙하 물체 등) 잘못된
클러스터 하나를 고르는 순간 전체 판정이 그 오답을 따라간다는 것이었다
(data/reference/wrongChoice.png — 근접 게이팅이 2프레임 이상 실패하면
절대 기준만으로 전역 재탐색하다 엉뚱한 클러스터를 하이재킹).

그래서 지금은 반대로 접근한다: 클러스터마다 독립된 트랙(Track)을 만들어
계속 따라가고, 트랙마다 "이 궤적이 낙하답게 생겼는가"만 독립적으로
판정한다. 사람이든 노이즈든 자기 트랙 안에서 그냥 추적만 되고, 실제로
피크-하강이나 자유낙하 패턴을 보이는 트랙만 낙하로 확정된다 — 어느 하나를
잘못 골라도 다른 트랙까지 오염되지 않는다.
"""

from collections import deque
import numpy as np

from ..processing.pointcloud import PointCloud
from ..utils import get_logger
from .tracker import GRAVITY, KalmanTracker

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

# 경로 3 — 자유낙하 궤적: PEAK_Z_THRESHOLD 같은 절대 높이 기준 없이, 최근
# 궤적 자체가 "자유낙하답게" 가속하며 떨어지고 있으면 어디서 처음
# 포착됐든 낙하로 본다. narrow-ROI 테스트에서 물체가 저고도에서 처음
# 잡히거나(피크 자체가 임계값 미만) 근접장 잡음에 앵커링되는 경우처럼,
# 시작 위치에 의존하는 기존 경로 1/2가 놓치는 케이스를 보완한다.
#
# 중력을 아는 값으로 써서 궤적에 포물선을 맞추는 방식(2점으로 초기속도
# 추정 후 투영, 혹은 전체 창 최소제곱 적합)도 시도해봤지만, 실측 배치
# 전반(person_plus_drop_20260715 등, 이전엔 3/3이던 것들 포함)에서 오히려
# 더 나빠졌다 — 창 안에 착지/반등 이후 데이터가 섞이면 "하강 방향이
# 아님"으로 통째로 걸러지는 등, 기존 방식보다 실측 데이터에 덜 안정적
# 이었다. 그래서 아래의 (다소 투박하지만 실측으로 검증된) 방식을 유지한다.
#
# data/raw/drop_test_20260717_freefall_validation 재생 결과: 원시 관측치
# 기준으로는 근접장·타겟 선택 잡음, 소실 구간으로 인한 오차 때문에 실제
# 가속도 추정치가 중력(9.8)보다 상당히 작게(-0.3~-2.5) 또는 크게(-16.1)
# 나오는 경우가 흔했다 — 여유를 넉넉히 뒀다.
FREEFALL_MIN_FRAMES = 3     # 자유낙하로 볼 최소 연속 유효 프레임 수 (속도 2개 비교 필요)
FREEFALL_ACCEL_MIN  = 3.0   # m/s² — 최소 이만큼은 가속해야 자유낙하 (공기저항 등 여유)
FREEFALL_ACCEL_MAX  = 18.0  # m/s² — 이보다 크면 센서 노이즈/점프로 보고 배제 (중력 9.8 기준 여유)

CONFIRM_FRAMES = 1          # 낙하 조건 충족 즉시 확정
HISTORY_WINDOW = 10         # 프레임 수 (100ms × 10 = 1.0초)
FRAME_DT       = 0.10       # 초 (100ms 프레임 주기)

# 다중 추적 — 트랙 매칭/생명주기
MAX_JUMP = 0.5             # m — 트랙 예측 위치 기준, 클러스터를 그 트랙으로 매칭할 최대 거리
TRACK_MAX_MISSES = 5       # 프레임 — 이 이상 연속으로 매칭 안 되면 트랙 삭제 (500ms)

# data/reference/wrongChoice.png 재분석(narrow_roi 재생) 결과 확인된 하이재킹
# (정지 트랙이 몇 프레임 미매칭 뒤 넓어진 max_jump 반경 안에 들어온 무관한
# 물체를 "같은 물체의 연속"으로 흡수)을 막아보려고, 매칭이 요구하는
# 가속도(트랙의 최근 원시 속도 → 후보가 암시하는 속도로의 변화량 /
# 경과시간)가 일정 상한을 넘으면 거부하는 게이트를 시도했었다.
#
# 그런데 실측으로 계산해보니 근본적으로 성립하지 않는 접근이었다:
# data/failDetecting.png 재현 케이스(연속 프레임에서 -1.3→-4.7m/s로
# 급가속하는 정상적인 실제 낙하)는 요구 가속도가 34m/s²인 반면, 실제
# wrongChoice 하이재킹 사례(정지 트랙이 1프레임 미매칭 후 흡수)는 요구
# 가속도가 약 11.75m/s²로 오히려 더 낮았다 — 경과시간이 길수록 같은
# 속도 변화도 더 작은 가속도로 계산되기 때문에, "가속도가 큰 매칭을
# 거부"하는 방식으로는 두 경우를 구분할 수 없다(정상 사례를 걷어내지
# 않을 만큼 느슨하게 잡으면 하이재킹 사례는 자동으로 통과됨). 실제로
# 7개 배치로 검증했을 때도 순개선 없이 freefall_validation만 5/5→3/5로
# 후퇴시켰다. 그래서 이 접근은 폐기했다 — wrongChoice류 하이재킹은
# 운동학(속도·가속도) 정보만으로는 원리적으로 못 잡고, 별도 신호(예:
# 클러스터 크기/모양 일관성)가 필요해 보인다.


class Track:
    """단일 클러스터 계열을 추적하는 트랙.

    자체 칼만 필터(등속도 모델)와 낙하 판정 이력(height_history,
    raw_height_history)을 갖는다 — FallDetector가 여러 Track을 동시에
    관리하며, 클러스터 매칭 결과에 따라 각 트랙을 독립적으로 갱신한다.

    피크-하강(경로 1/2)은 칼만 평활화된 높이를, 자유낙하(경로 3)는 원시
    높이를 쓴다 — 이유는 FallDetector 클래스 docstring 참고.

    한 번 낙하로 확정되면(_fall_triggered) 그 트랙이 살아있는 동안은
    다시 판정하지 않고 계속 True만 반환한다(래치). 원래는 매 프레임
    조건을 다시 평가해서 조건을 못 만족하면 즉시 미확정으로 되돌렸는데,
    착지 후 바닥 근처 센서 노이즈로 높이가 미세하게 오르내리며 반등/정체
    조건에 반복해서 걸렸다 풀렸다 하는 바람에, 같은 낙하 사건 하나가
    한 녹화 안에서 "새로 감지됨"으로 여러 번 재발화하는 문제가 있었다
    (narrow_roi 재생 시 한 트랙에서만 5번 재발화 확인).
    """

    def __init__(self, track_id: int, history_window: int = HISTORY_WINDOW,
                 debug: bool = False, confirm_frames: int = CONFIRM_FRAMES):
        self.id = track_id
        self.misses = 0                            # 연속 미매칭 프레임 수
        self.last_cluster: PointCloud | None = None  # 이번 프레임에 매칭된 원시 클러스터 (시각화용)
        self.last_centroid: np.ndarray | None = None  # 평활화된 위치
        self.last_fall_centroid: np.ndarray | None = None  # 낙하 확정 시점 위치

        self._height_history: deque[float | None] = deque(maxlen=history_window)
        self._raw_height_history: deque[float | None] = deque(maxlen=history_window)
        self._fall_triggered  = False
        self._trigger_reason  = ""
        self._candidate_frames = 0
        self._confirm_frames   = confirm_frames
        self._debug             = debug

        self._tracker  = KalmanTracker(dt=FRAME_DT)
        self._tracking = False   # 칼만 필터가 최소 1회 이상 보정됐는지

    # ── 메인 업데이트 ────────────────────────────────────────────────────────

    def predicted_centroid(self) -> np.ndarray | None:
        """다음 관측이 위치할 것으로 예상되는 지점 (칼만 예측, 매칭 기준점).

        내부 상태를 변경하지 않는 미리보기(peek)다. 아직 트랙에 관측이
        한 번도 반영되지 않았으면 None을 반환한다.

        Z축에는 등속도 예측 위에 중력가속도 한 프레임분(½g·dt²)을 추가로
        더한다 — 자유낙하 중인 물체는 등속도 가정만으로는 예측이 실제
        낙하 속도를 못 따라잡아 매칭이 계속 뒤처지는데, Z만은 가속도를
        아는 값(중력)으로 보정할 수 있다. 이 보정은 여기 미리보기에만 쓰고
        self._tracker.x(영속 상태)에는 반영하지 않는다 — 정지·상승 중인
        물체까지 매 프레임 아래로 편향시키면 안 되기 때문이다.
        """
        if not self._tracking:
            return None
        predicted = self._tracker.F @ self._tracker.x
        predicted[2, 0] -= 0.5 * GRAVITY * FRAME_DT ** 2
        return predicted[:3].flatten()

    def update(self, centroid: np.ndarray | None) -> bool:
        """이번 프레임에 매칭된 관측(없으면 None)으로 트랙을 갱신한다.

        한 번 낙하로 확정된 트랙은 이후 계속 True를 반환한다(래치) — 아래
        _fall_triggered 관련 설명 참고.
        """
        if centroid is None:
            return self._update_no_target()

        raw_xyz = np.asarray(centroid, dtype=np.float64)
        raw_height = float(raw_xyz[2])  # 경로 3(자유낙하 가속도 계산)용 원시 높이

        if self._tracking:
            self._tracker.predict()
        self._tracker.update(raw_xyz)
        self._tracking = True
        smoothed = self._tracker.x[:3].flatten()   # 평활화된 위치

        self.last_centroid = smoothed
        height = float(smoothed[2])
        self._height_history.append(height)
        self._raw_height_history.append(raw_height)

        if self._fall_triggered:
            return True

        candidate, reason = self._check_fall_visible()
        if candidate:
            self._candidate_frames += 1
        else:
            self._candidate_frames = 0

        fell = self._candidate_frames >= self._confirm_frames
        if fell:
            self._fall_triggered    = True
            self._trigger_reason    = reason
            self.last_fall_centroid = smoothed
            logger.warning("FALL DETECTED [track#%d %s] — Z=%.2f m", self.id, reason, height)

        return fell

    def _update_no_target(self) -> bool:
        """이번 프레임에 매칭된 클러스터가 없을 때 — 착지 소실 패턴 확인 후 이력 업데이트."""
        if self._tracking:
            self._tracker.predict()

        self._height_history.append(None)
        self._raw_height_history.append(None)
        self._candidate_frames = 0

        if self._fall_triggered:
            return True

        fell, reason = self._check_fall_on_disappear()
        if fell:
            self._fall_triggered    = True
            self._trigger_reason    = reason
            self.last_fall_centroid = self.last_centroid
            z = self.last_centroid[2] if self.last_centroid is not None else float("nan")
            logger.warning("FALL DETECTED (landing disappearance) [track#%d %s] — last Z=%.2f m",
                           self.id, reason, z)

        return fell

    # ── 낙하 조건 체크 ───────────────────────────────────────────────────────

    def _check_fall_visible(self) -> tuple[bool, str]:
        """경로 2: 물체가 보이는 상태에서의 직접 감지."""
        history = list(self._height_history)
        valid   = [(i, h) for i, h in enumerate(history) if h is not None]

        if len(valid) < 2:
            return False, ""

        result, reason = self._trajectory_check(valid)
        if self._debug and valid:
            peak_z = max(h for _, h in valid)
            cur_z  = valid[-1][1]
            print(f"[FD track#{self.id}] peak_z={peak_z:.3f}  cur_z={cur_z:.3f}"
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
        """히스토리 내 하강 궤적 판정 — 경로 2/1(피크-하강) 또는 경로 3(자유낙하).

        피크-하강(valid)은 칼만 평활화된 높이를, 자유낙하는 원시 높이
        (_raw_height_history)를 쓴다 — 이유는 클래스 docstring 참고.
        """
        result, reason = self._peak_drop_check(valid)
        if result:
            return result, reason

        raw_history = list(self._raw_height_history)
        raw_valid = [(i, h) for i, h in enumerate(raw_history) if h is not None]
        return self._freefall_check(raw_valid)

    def _peak_drop_check(self, valid: list[tuple[int, float]]) -> tuple[bool, str]:
        """경로 1/2 공용 — 피크 대비 하강폭 기반 판정.

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

    def _freefall_check(self, valid: list[tuple[int, float]]) -> tuple[bool, str]:
        """경로 3 — 시작 높이 무관, 최근 궤적이 자유낙하 패턴인지만 본다.

        최근 FREEFALL_MIN_FRAMES개 유효 프레임 구간에서
        - 창(window) 전체로 봤을 때 실제로 순하강(마지막 높이 < 첫 높이)이고
        - 프레임별 속도를 구해, 매 구간 속도가 계속 더 음수로(가속) 바뀌고
          (반등/정체 없음)
        - 그 가속도가 중력 근방(FREEFALL_ACCEL_MIN~MAX)이면
        낙하로 판정한다. PEAK_Z_THRESHOLD 등 절대 높이 조건은 보지 않는다.
        """
        if len(valid) < FREEFALL_MIN_FRAMES:
            return False, ""

        recent = valid[-FREEFALL_MIN_FRAMES:]

        # data/raw/drop_test_20260720_mot_review 재생 분석 결과 발견된 버그:
        # "속도가 점점 더 음수로 바뀌는 것"과 "실제로 하강 중인 것"은 다른데,
        # 원래 아래 모노토닉 체크만으로는 이 둘을 구분 못 했다 — 위로
        # 올라가는 중이지만 상승 속도 자체가 둔화되는 트랙도(예: +4.29 →
        # +0.78 m/s) "속도가 계속 더 음수로 바뀐다"는 조건과 가속도 범위
        # 조건을 그대로 통과해버려, 계속 상승만 하는 트랙이 자유낙하로
        # 오판정됐다(신규 시행 전부에서 T1이 이 패턴). 그래서 창(window) 전체
        # 순변위(첫 프레임 대비 마지막 프레임 높이)가 실제로 하강인지부터
        # 본다 — 구간별 속도 하나하나가 전부 음수여야 한다는 더 엄격한
        # 조건도 시도해봤지만, 근접장 바닥 근처의 실측 노이즈로 중간에 한
        # 구간만 살짝 양수(예: 반사점 흔들림)인 진짜 낙하까지 걸러내
        # 버려서(data/raw/drop_test_20260715_baseline 등 기존 배치 회귀
        # 확인) 폐기했다. 창 전체의 순하강만 요구하면 중간의 짧은 노이즈성
        # 흔들림은 통과시키면서도, 계속 위로만 가는 궤적은 여전히 걸러낸다.
        if recent[-1][1] >= recent[0][1]:
            return False, ""

        velocities = []
        midpoints  = []  # 각 속도가 대표하는 시각 (등가속 구간의 평균속도 = 중점 순간속도)
        for (i0, z0), (i1, z1) in zip(recent, recent[1:]):
            dt = (i1 - i0) * FRAME_DT
            if dt <= 0:
                return False, ""
            velocities.append((z1 - z0) / dt)
            midpoints.append((i0 + i1) / 2.0 * FRAME_DT)

        # 매 구간 계속 더 음수로 가속해야 한다 (반등/정체 시 탈락)
        if any(v2 >= v1 for v1, v2 in zip(velocities, velocities[1:])):
            return False, ""

        # 가속도 = 속도 변화량 / (중점 시각 간격). 구간 시작~끝 전체 폭을
        # 쓰면 프레임 간격이 고르지 않을 때(소실 구간 등) 실제보다 훨씬
        # 작은 가속도로 과소평가된다 — 등가속도 구간의 평균속도는 그
        # 구간 중점에서의 순간속도와 같다는 성질을 이용해 보정한다.
        total_dt = midpoints[-1] - midpoints[0]
        if total_dt <= 0:
            return False, ""
        accel = (velocities[-1] - velocities[0]) / total_dt

        if not (-FREEFALL_ACCEL_MAX <= accel <= -FREEFALL_ACCEL_MIN):
            return False, ""

        reason = (f"freefall {len(recent)}f"
                  f"  accel={accel:+.1f}m/s²"
                  f"  v={velocities[-1]:+.2f}m/s")
        return True, reason

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    @property
    def fell(self) -> bool:
        """이 트랙이 현재 낙하로 확정된 상태인지."""
        return self._fall_triggered

    def z_velocity(self) -> float:
        """Z 속도 (m/s). 시각화용 — 칼만 필터의 평활화된 속도 추정치를 쓴다."""
        if self._tracking:
            return float(self._tracker.x[5, 0])
        return 0.0


class FallDetector:
    """여러 클러스터를 동시에 독립 트랙으로 추적하며 낙하를 감지한다 (다중 추적).

    매 프레임 다음을 수행한다:
      1. 기존 트랙마다 칼만 예측 위치를 구한다(Track.predicted_centroid()).
      2. 이번 프레임의 클러스터들과 트랙들을 최근접 거리로 매칭한다(그리디,
         거리 오름차순으로 하나씩 확정) — 매칭 거리 상한은 max_jump에
         연속 미매칭 횟수(track.misses)를 곱해 넓힌다. 추적이 뜨문뜨문
         이어질 때 칼만 속도 추정이 아직 실제 낙하 속도를 못 따라잡아
         예측 위치가 실제 물체보다 위에 남아있는 문제를 보완한다.

         (매칭에 가속도 타당성 체크를 추가해 정지 트랙이 무관한 물체를
         흡수하는 걸 막아보려 했지만 — data/reference/wrongChoice.png
         하이재킹 사례가 요구하는 가속도가, 정상적인 빠른 낙하 사례보다
         오히려 낮게 계산돼(경과시간이 길수록 같은 속도변화도 더 작은
         가속도로 나타나므로) 둘을 구분할 수 없었고 실측 검증에서 순개선도
         없어 제거했다 — 위 TRACK_MAX_MISSES 아래 관련 설명 참고.)
      3. 매칭된 트랙은 관측을 반영(Track.update), 매칭 안 된 트랙은 예측만
         하고 코스팅한다. 이 상태에서도 트랙마다 기존 낙하 판정 로직
         (_trajectory_check)이 독립적으로 돈다.
      4. 매칭 안 된 클러스터는 새 트랙으로 시작한다.
      5. TRACK_MAX_MISSES 프레임 넘게 연속 미매칭인 트랙은 삭제한다.

    아무 트랙이나 낙하로 판정되면 그 트랙의 결과를 낙하로 보고한다. 어느
    트랙을 클러스터에 매칭할지는 근접성만으로 정하고 "낙하답게 생겼는가"는
    전혀 보지 않는다 — 그 판단은 전적으로 트랙별 궤적 이력(_trajectory_check)
    몫이다. 그래서 사람이나 노이즈처럼 낙하와 무관한 클러스터가 동시에
    있어도 각자 자기 트랙에서 조용히 추적만 될 뿐, 실제 자유낙하/피크-하강
    패턴을 보이는 트랙만 낙하로 확정된다.

    내부적으로 트랙마다 칼만 필터(KalmanTracker, 등속도 모델)를 태워
    무게중심을 평활화한다. 경로 1/2(_height_history)는 원시 관측치가
    아니라 이 필터링된 위치를 사용한다 — 한 프레임의 매칭이 잘못되어도
    무게중심이 통째로 튀지 않고 예측과 블렌딩된 만큼만 움직이게 된다.

    경로 3(자유낙하, _raw_height_history)만은 예외로 원시 관측치를 쓴다 —
    칼만 필터의 등속도 모델은 두 보정 사이(특히 소실 구간)에는 "가속이
    없다"고 가정하고 코스팅하므로, 정확히 자유낙하가 있는지를 보려는
    가속도 계산에는 오히려 실제 가속을 과소평가시키는 방향으로 편향된다.

    last_fall_centroid: 낙하로 확정된 트랙의, 확정 시점 무게중심(X, Y, Z).
    last_centroid: 현재 "주 트랙"(primary_track)의 최신 무게중심 — 매
        프레임 갱신되며, 서보 좌표 전송처럼 낙하 확정 여부와 무관하게
        연속적인 대표 위치가 필요한 호출부용 편의 속성이다.
    """

    def __init__(self, history_window: int = HISTORY_WINDOW, debug: bool = False,
                 confirm_frames: int = CONFIRM_FRAMES, max_jump: float = MAX_JUMP,
                 max_track_misses: int = TRACK_MAX_MISSES):
        self._history_window   = history_window
        self._debug             = debug
        self._confirm_frames    = confirm_frames
        self._max_jump          = max_jump
        self._max_track_misses  = max_track_misses

        self._tracks: list[Track] = []
        self._next_id = 1

        self.last_fall_centroid: np.ndarray | None = None

    # ── 메인 업데이트 ────────────────────────────────────────────────────────

    @property
    def tracks(self) -> list[Track]:
        """현재 살아있는 트랙 목록 (시각화·디버그용, 읽기 전용 사본)."""
        return list(self._tracks)

    @property
    def primary_track(self) -> Track | None:
        """대표 트랙 — 방금 낙하가 확정된 트랙이 있으면 그것을, 없으면 가장
        오래(안정적으로) 추적된 트랙을 반환한다. 서보 좌표 전송·단일 궤적
        시각화처럼 "하나만" 필요한 호출부를 위한 편의 속성이며, 낙하 판정
        자체는 이 값과 무관하게 트랙마다 독립적으로 이뤄진다.
        """
        if not self._tracks:
            return None
        triggered = [t for t in self._tracks if t._fall_triggered]
        if triggered:
            return max(triggered, key=lambda t: len(t._height_history))
        return max(self._tracks, key=lambda t: len(t._height_history))

    @property
    def last_centroid(self) -> np.ndarray | None:
        track = self.primary_track
        return track.last_centroid if track is not None else None

    def z_velocity(self) -> float:
        track = self.primary_track
        return track.z_velocity() if track is not None else 0.0

    def update(self, clusters: list[PointCloud]) -> bool:
        """이번 프레임의 DBSCAN 클러스터 목록으로 모든 트랙을 갱신하고,
        아무 트랙이나 낙하로 판정됐는지 반환한다."""
        centroids = [c.centroid() for c in clusters]

        # 1) 트랙별 예측 위치 (매칭 기준점)
        predictions = {track.id: track.predicted_centroid() for track in self._tracks}

        # 2) 트랙↔클러스터 최근접 매칭 (그리디: 거리 오름차순으로 확정)
        pairs = []  # (거리, 트랙, 클러스터 인덱스)
        for track in self._tracks:
            predicted = predictions[track.id]
            if predicted is None:
                continue
            effective_max_jump = self._max_jump * (1 + track.misses)
            for i, cen in enumerate(centroids):
                if cen is None:
                    continue
                dist = float(np.linalg.norm(cen - predicted))
                if dist <= effective_max_jump:
                    pairs.append((dist, track, i))
        pairs.sort(key=lambda p: p[0])

        matched_cluster_of: dict[int, int] = {}
        used_clusters: set[int] = set()
        used_tracks: set[int] = set()
        for dist, track, i in pairs:
            if track.id in used_tracks or i in used_clusters:
                continue
            matched_cluster_of[track.id] = i
            used_tracks.add(track.id)
            used_clusters.add(i)

        # 3) 트랙 갱신 — 매칭 성공 시 관측 반영, 실패 시 예측만 하고 코스팅
        fell = False
        fall_track: Track | None = None
        surviving: list[Track] = []
        for track in self._tracks:
            if track.id in used_tracks:
                idx = matched_cluster_of[track.id]
                track.last_cluster = clusters[idx]
                track_fell = track.update(centroids[idx])
                track.misses = 0
            else:
                track.last_cluster = None
                track_fell = track.update(None)
                track.misses += 1

            if track_fell and fall_track is None:
                fell = True
                fall_track = track

            if track.misses <= self._max_track_misses:
                surviving.append(track)
        self._tracks = surviving

        # 4) 매칭 안 된 클러스터 → 새 트랙 시작
        for i, cen in enumerate(centroids):
            if i in used_clusters or cen is None:
                continue
            new_track = Track(self._next_id, history_window=self._history_window,
                               debug=self._debug, confirm_frames=self._confirm_frames)
            self._next_id += 1
            new_track.last_cluster = clusters[i]
            new_track.update(cen)
            self._tracks.append(new_track)

        if fall_track is not None:
            self.last_fall_centroid = fall_track.last_fall_centroid

        return fell

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
        self.last_fall_centroid = None
