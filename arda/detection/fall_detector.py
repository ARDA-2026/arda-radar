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
import math
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

# 경로 1/2(피크-하강)는 원래 속도를 전혀 안 봤다 — "한 번 높이 찍혔다가
# 충분히 떨어졌나"만 보므로, 봉지처럼 공기저항을 크게 받아 몇 초에 걸쳐
# 천천히 내려오는 물체도 그대로 낙하로 판정돼버렸다(0.6m→0.05m를 1초에
# 걸쳐 등속 0.55m/s로 내려가는 시나리오로 실제 재현 확인). 최종 목적이
# 사람의 낙하(자유낙하에 준하게 빠름)를 감지하는 것이므로, 자유낙하가
# 아닌 느린 하강까지 낙하로 잡으면 안 된다. 그래서 피크~마지막 프레임
# 사이의 "평균" 하강 속도에 최소 기준을 둔다 — 경로 3처럼 프레임별
# 가속도 모양(모노토닉·범위)까지 엄격히 보진 않아, 노이즈 낀 실제
# 빠른 낙하에 대한 경로 1/2 본연의 관용성은 유지하면서 명백히 느린
# 하강만 걸러낸다. 값은 자유낙하 최소 낙하폭(PEAK_DROP_THRESHOLD)을
# 자유낙하로 낙하할 때의 평균속도(√(g·d/2) ≈ 1.3m/s, d=0.35m 기준)보다
# 낮게 잡아 여유를 뒀다.
MIN_AVG_DESCENT_SPEED = 1.0  # m/s — 피크~마지막 프레임 평균 하강 속도 최소 기준

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

# data/raw/covered/record_raw_20260728_154201 trial5 track#1 재현 결과 발견된
# 버그: 3초 내내 z=0.04~0.07m에 정지해 있던 클러스터가, 클러스터 중심점
# 계산 노이즈로 인한 ±1.5cm 수준의 짧은 상하 흔들림(0.047→0.062→0.046m)만으로
# 자유낙하로 오판정됐다 — 순변위 부호(음수)와 속도 단조 감소, 가속도 범위
# 조건을 전부 "우연히" 만족했지만(v=-0.16m/s, accel=-3.1m/s²로 문턱을 근소하게
# 통과), 실제 이동 속도 자체는 노이즈 수준이었다. 문제는 이 세 조건 중
# 어느 것도 속도·변위의 절대적 크기를 보지 않는다는 것 — 부호와 상대적
# 순서, 그리고 아주 짧은(2구간) 기준선에서 계산한 가속도만 본다. 실제
# 자유낙하라면 이 시점(경로 3 판정 시점)의 속도 자체가 이미 노이즈보다
# 뚜렷하게 커야 하므로, 마지막 구간 속도의 절대값에도 최소 기준을 둔다.
# 실측 배치 전반의 진짜 자유낙하 감지 이벤트들의 최소 속도가 0.83m/s였고
# (그 아래로는 이번에 발견된 노이즈성 오판정 사례의 0.16m/s뿐), 그 사이
# 여유를 두고 0.5m/s로 잡았다.
FREEFALL_MIN_TRIGGER_SPEED = 0.5  # m/s — 마지막 구간 속도 절대값 최소 기준

# 낙하 판정(_fall_triggered)은 이진값으로 남겨둔다 — 노이즈로 인한 오판정을
# 하드 게이트로 걸러보려는 시도(자유낙하 창 확장, 연속 프레임 확정, 낙하
# 깊이 확인, ROI 축소, 도플러 일관성 등)를 전부 71개 실측 배치로 검증해봤지만
# 예외 없이 진짜 낙하를 대거 같이 잃었다 — 문턱을 어느 쪽으로 옮겨도 노이즈와
# 진짜 낙하가 겹치는 회색지대가 있어서였다. 그래서 판정 자체를 더 엄격하게
# 만드는 대신, 이미 확정된 낙하에 "이게 노이즈보다는 진짜에 가까운 특징을
# 얼마나 갖췄는지"를 보조적으로 알려주는 신뢰도 점수를 덧붙인다.
#
# 처음엔 손으로 조합한 휴리스틱(문턱 초과 여유 + dwell + 반등 감점)이었는데,
# data/labeling_worksheet.csv에 손으로 라벨링한 71개 배치의 확정 이벤트
# 61건(진짜 28 / 노이즈 33, 애매한 11건 제외)으로 검증해보니 정확도 0.72
# 수준이었다. 같은 라벨로 로지스틱 회귀를 학습(배치 단위 leave-one-out
# 검증: 정확도 0.89, 재현율 0.89, 정밀도 0.86)했더니 손튜닝 휴리스틱보다
# 뚜렷이 나아서, 아래 _MODEL_* 계수로 교체했다 — 학습에 쓴 특징은
# _unified_features() 참고. 낙하 감지에서는 "놓치는 것"이 "오탐"보다
# 훨씬 치명적이라(재현율 우선), 기본 0.5 대신 LIKELY_REAL_THRESHOLD=0.24를
# 쓴다 — 임계값을 낮출수록 재현율은 오르지만 대가로 오탐도 늘어나는데,
# 완전한 재현율(FN=0)까지 밀어붙이면 정밀도가 0.61까지 떨어져 원래
# 휴리스틱과 다를 바 없어지므로, FN 1건까지만 허용하는 지점(0.24)에서
# 절충했다.
#
# 학습 특징은 반드시 "확정되는 바로 그 순간"의 값으로 뽑아야 한다 — 처음
# CSV를 뽑을 때 z_max_ever/avg_pts 등 일부를 트랙이 소멸할 때까지의
# 최종값으로 잘못 뽑았었는데(경로 1/2의 peak_z 등은 확정 순간 값이라
# 서로 시점이 어긋남), 실제 배포 코드(_model_confidence, 확정 즉시 1회
# 호출)는 오직 확정 순간의 값만 알 수 있어 학습·서빙 시점이 안 맞았다
# (covered trial1/track#2처럼 명백한 진짜 낙하가 실시간 재생에서 0.15로
# 나오는 등, 재검증 중 발견). CSV 전체를 확정 순간 기준으로 다시 뽑아
# 재학습했다 — rebound_penalized는 확정 그 순간엔 항상 False라 학습
# 데이터에 분산이 없어 계수가 0으로 나왔지만(반등 여부 자체는 다른
# 특징들이 사후 재계산 시점엔 자연히 함께 갱신되어 있어 간접적으로 반영됨),
# 나머지 계수는 재학습 후에도 검증 지표가 거의 그대로였다.
#
# sphere_person(사람이 옆에 서 있는 상태에서 공 낙하, 10건 모두 진짜 낙하)
# 배치 추가 후 71건(진짜 38 / 노이즈 33)으로 재학습 — scripts/retrace_track.py
# (마우스로 궤적을 그려 자동 추적이 조각내거나 하이재킹한 트랙을 사람이
# 직접 보정하는 도구)로 라벨링했다. 처음 뽑은 값은 트리거 이후에도 미매칭
# 프레임을 녹화 끝까지 계속 먹이는 바람에 maxlen=10짜리 _height_history/
# _raw_height_history가 전부 None으로 밀려나 peak_z/net_drop/avg_descent_speed/
# recent_v/recent_a 5개가 0으로 깨져 있었다(재현 중 발견) — build_manual_track()이
# 트리거되는 바로 그 프레임에서 즉시 스냅샷 뜨도록 고치고 10건 다시 그려
# 받았다. 배치 단위 leave-one-out 재검증 결과 LIKELY_REAL_THRESHOLD=0.24에서
# 재현율이 0.96(FN=1)에서 1.00(FN=0)으로 개선됐고 정밀도는 0.79로 그대로라
# (기존엔 FN=0까지 밀면 정밀도가 0.61까지 떨어졌었다), 임계값은 그대로 두고
# 계수만 교체한다.
_MODEL_FEATURE_ORDER = [
    "dwell", "peak_z", "net_drop", "avg_descent_speed", "recent_v", "recent_a",
    "z_max_ever", "z_min_ever", "avg_pts", "max_pts", "rebound_penalized",
]
_MODEL_MEAN = [3.929577, 0.297815, 0.49178, 2.164382, -2.729538, -11.108708,
               0.3064, -0.252424, 8.314639, 14.802817, 0.0]
_MODEL_SCALE = [1.817938, 0.395246, 0.442052, 2.20304, 2.426245, 13.533177,
                0.39503, 0.336519, 5.117823, 9.624821, 1.0]
_MODEL_COEF = [-0.93064, 0.139807, 0.631744, 1.229086, -1.905325, 0.635923,
               0.128977, 0.460901, 1.051285, -0.333993, 0.0]
_MODEL_INTERCEPT = 1.137596
LIKELY_REAL_THRESHOLD = 0.24  # confidence가 이 이상이면 노이즈보다 진짜에 가깝다고 본다

# covered trial12/track1(장기 정지 후 단발 노이즈 딥) 재현 결과: 확정 시점
# z=-0.069m였는데 바로 다음 프레임 z=+0.133m로 확정 높이보다도 더 위로
# 반등했다 — 진짜 낙하라면 물리적으로 이렇게 짧은 시간에 다시 올라올 수
# 없다(바운스라 해도 확정 시점보다 높이 튀는 경우는 드물다). 반대로
# covered trial7·uncovered trial1/track2 같은 검증된 진짜 낙하는 확정
# 직후 레이더 반사가 사라져(바닥 근처 소실) 이 체크 자체가 아예 안 걸린다
# — 그래서 이 체크는 노이즈만 골라내고 진짜 낙하는 건드리지 않는다.
# rebound_penalized는 그 자체로 모델 입력 특징이라, 반등이 확인되면 페널티를
# 곱하는 대신 그 특징값을 반영해 모델을 다시 계산한다.
POST_TRIGGER_CHECK_FRAMES     = 3    # 확정 후 이 프레임까지만 반등 여부 관찰
POST_TRIGGER_REBOUND_MARGIN   = 0.10 # m — 확정 시점 높이보다 이만큼 넘게 오르면 반등

# data/raw/uncovered/record_raw_20260728_155515 trial1 track#2 재현 결과
# 발견된 문제: 명백히 실제 낙하로 보이는 궤적(0.306→0.190→0.070→-0.272m,
# 0.3초 만에 순하강 0.58m)인데도 감지가 안 됐다 — FREEFALL_MIN_FRAMES(3)
# 프레임만으로 계산하면 속도 2개(인접 구간)뿐이라 한쪽 구간의 무게중심
# 노이즈(클러스터 매칭이 흔들려 순간적으로 튐)가 두 속도 추정 모두에
# 걸쳐 가속도를 왜곡시킨다 — 이 사례는 앞 구간 accel=-0.4(너무 느림),
# 바로 다음 구간 accel=-22.2(너무 빠름)로 오히려 문턱 양쪽 다 걸려 탈락
# 했다. 유효 프레임이 더 있으면(4개 이상) 그만큼 창을 넓혀 속도를 더
# 여러 구간에서 뽑으면, 한 구간의 순간 노이즈가 희석돼 전체 가속도
# 추정이 훨씬 안정적이다(같은 사례 재계산 시 accel=-11.3으로 정상 범위
# 진입). 다만 너무 넓히면 낙하 시작 훨씬 전의 무관한 이력까지 끌어와
# 가속도를 과소평가시킬 수 있어 상한을 둔다.
FREEFALL_WINDOW_MAX = 5     # 프레임 — 자유낙하 가속도 계산에 쓰는 최근 유효 프레임 상한

CONFIRM_FRAMES = 1          # 낙하 조건 충족 즉시 확정
HISTORY_WINDOW = 10         # 프레임 수 (100ms × 10 = 1.0초)
FRAME_DT       = 0.10       # 초 (100ms 프레임 주기)

# 다중 추적 — 트랙 매칭/생명주기
MAX_JUMP = 0.5             # m — 트랙 예측 위치 기준, 클러스터를 그 트랙으로 매칭할 최대 거리
TRACK_MAX_MISSES = 5       # 프레임 — 이 이상 연속으로 매칭 안 되면 트랙 삭제 (500ms)

# data/raw/dropball처럼 max_jump보다 빠르게 하강하는 물체가 매 프레임
# 새 트랙으로 쪼개지는 문제(재현 시 0.5초 안에 트랙 14개 생성) 대응으로,
# 이미 하강 중인 트랙의 Z축 매칭 반경만 속도(칼만 평활화 속도·원시
# 프레임 간 속도 둘 다 시도)에 비례해 넓히는 방식을 시도했었다. 그런데
# 둘 다 트랙 생성 직후 1~2프레임은 속도 정보 자체가 없어 정작 가장
# 심하게 쪼개지는 구간에는 효과가 없었고(dropball 최악 사례는 그대로
# 트랙 14개), 오히려 다중 트랙이 클러스터를 두고 경쟁하는 상황에서
# 그리디 매칭 우선순위가 바뀌어 기존에 잘 되던 실제 낙하 몇 건을
# 깨뜨렸다(실측 71개 배치 회귀 57/71→55~56/71, humanfall_Rside
# 3/3→2/3). 순효과가 마이너스라 폐기했다.

# 트랙 분류 — 사람 vs 낙하 물체
PERSON_Z_RANGE_MIN  = 1.1  # m — 사람 z_range 기준 (두 발~머리 수직 범위)
PERSON_MIN_DURATION = 1.5  # s — 장기 트랙 기준 (실제 자유낙하 ROI 통과 ≪ 0.5s)
FALL_Z_DROP_MIN     = 0.30 # m — 장기 fell 트랙에서 실제 net 하강 최소치

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
                 debug: bool = False, confirm_frames: int = CONFIRM_FRAMES,
                 frame_dt: float = FRAME_DT,
                 person_z_range_min: float = PERSON_Z_RANGE_MIN,
                 person_min_duration: float = PERSON_MIN_DURATION,
                 fall_z_drop_min: float = FALL_Z_DROP_MIN):
        self.id = track_id
        self.misses = 0                            # 연속 미매칭 프레임 수
        self.last_cluster: PointCloud | None = None  # 이번 프레임에 매칭된 원시 클러스터 (시각화용)
        self.last_centroid: np.ndarray | None = None  # 평활화된 위치
        self.last_fall_centroid: np.ndarray | None = None  # 낙하 확정 시점 위치

        self._height_history: deque[float | None] = deque(maxlen=history_window)
        self._raw_height_history: deque[float | None] = deque(maxlen=history_window)
        self._n_pts_history: deque[int] = deque(maxlen=history_window)  # 모델 특징(avg_pts/max_pts)용
        self._fall_triggered  = False
        self._trigger_reason  = ""
        self._trigger_confidence = 0.0
        self._trigger_raw_height: float | None = None
        self._post_trigger_count = 0
        self._rebound_penalized  = False
        self._candidate_frames = 0
        self._confirm_frames   = confirm_frames
        self._debug             = debug
        self._frame_dt         = frame_dt

        # 분류용 누적 통계 (rolling window 밖 과거도 반영)
        self._person_z_range_min  = person_z_range_min
        self._person_min_duration = person_min_duration
        self._fall_z_drop_min     = fall_z_drop_min
        self._total_obs_count     = 0
        self._z_max_ever          = float("-inf")
        self._z_min_ever          = float("inf")
        self._z_first_obs: float | None = None   # 첫 관측 z (z_drop 기준점)

        self._tracker  = KalmanTracker(dt=frame_dt)
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
        predicted[2, 0] -= 0.5 * GRAVITY * self._frame_dt ** 2
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
        self._n_pts_history.append(len(self.last_cluster) if self.last_cluster is not None else 0)

        # 분류용 누적 통계 갱신
        self._total_obs_count += 1
        if raw_height > self._z_max_ever:
            self._z_max_ever = raw_height
        if raw_height < self._z_min_ever:
            self._z_min_ever = raw_height
        if self._z_first_obs is None:
            self._z_first_obs = raw_height

        if self._fall_triggered:
            self._check_post_trigger_rebound(raw_height)
            return True

        candidate, reason = self._check_fall_visible()
        if candidate:
            self._candidate_frames += 1
        else:
            self._candidate_frames = 0

        fell = self._candidate_frames >= self._confirm_frames
        if fell:
            self._fall_triggered      = True
            self._trigger_reason      = reason
            self._trigger_confidence  = self._model_confidence()
            self._trigger_raw_height  = raw_height
            self.last_fall_centroid   = smoothed
            logger.warning("FALL DETECTED [track#%d %s  conf=%.2f] — Z=%.2f m",
                            self.id, reason, self._trigger_confidence, height)

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
            self._fall_triggered      = True
            self._trigger_reason      = reason
            self._trigger_confidence  = self._model_confidence()
            last_valid_raw = next((h for h in reversed(self._raw_height_history) if h is not None), None)
            self._trigger_raw_height = last_valid_raw
            self.last_fall_centroid   = self.last_centroid
            z = self.last_centroid[2] if self.last_centroid is not None else float("nan")
            logger.warning("FALL DETECTED (landing disappearance) [track#%d %s  conf=%.2f] — last Z=%.2f m",
                           self.id, reason, self._trigger_confidence, z)

        return fell

    def _unified_features(self) -> list[float]:
        """경로(피크-하강/자유낙하) 무관 공통 특징 벡터 — _MODEL_FEATURE_ORDER 순서.

        data/labeling_worksheet.csv 라벨링에 쓴 것과 동일한 계산 방식이어야
        학습된 계수가 의미가 있다. peak_z/net_drop/avg_descent_speed는
        칼만 평활화 높이(_height_history) 기준, recent_v/recent_a는 원시
        높이(_raw_height_history) 마지막 최대 3프레임 기준 — 각각 경로
        1/2, 경로 3이 원래 쓰던 것과 같다(클래스 docstring 참고).
        """
        smoothed_valid = [(i, h) for i, h in enumerate(self._height_history) if h is not None]
        if smoothed_valid:
            peak_i = max(range(len(smoothed_valid)), key=lambda k: smoothed_valid[k][1])
            peak_z = smoothed_valid[peak_i][1]
            net_drop = peak_z - smoothed_valid[-1][1]
            if peak_i < len(smoothed_valid) - 1:
                elapsed = (smoothed_valid[-1][0] - smoothed_valid[peak_i][0]) * self._frame_dt
                avg_descent_speed = net_drop / elapsed if elapsed > 0 else 0.0
            else:
                avg_descent_speed = 0.0
        else:
            peak_z = net_drop = avg_descent_speed = 0.0

        raw_valid = [(i, h) for i, h in enumerate(self._raw_height_history) if h is not None]
        recent = raw_valid[-3:] if len(raw_valid) >= 3 else raw_valid
        velocities = []
        for (i0, z0), (i1, z1) in zip(recent, recent[1:]):
            dt = (i1 - i0) * self._frame_dt
            if dt > 0:
                velocities.append((z1 - z0) / dt)
        recent_v = velocities[-1] if velocities else 0.0
        recent_a = ((velocities[-1] - velocities[0]) / ((len(velocities) - 1) * self._frame_dt)
                    if len(velocities) >= 2 else 0.0)

        n_pts = list(self._n_pts_history) or [0]

        return [
            float(self._total_obs_count), peak_z, net_drop, avg_descent_speed,
            recent_v, recent_a, self._z_max_ever, self._z_min_ever,
            sum(n_pts) / len(n_pts), float(max(n_pts)),
            1.0 if self._rebound_penalized else 0.0,
        ]

    def _model_confidence(self) -> float:
        """학습된 로지스틱 회귀로 신뢰도(0~1)를 계산한다 — 모듈 docstring의
        _MODEL_* 주석 참고."""
        z = _MODEL_INTERCEPT
        for x, mean, scale, coef in zip(self._unified_features(), _MODEL_MEAN, _MODEL_SCALE, _MODEL_COEF):
            z += coef * (x - mean) / scale
        return 1.0 / (1.0 + math.exp(-z))

    def _check_post_trigger_rebound(self, raw_height: float) -> None:
        """확정 직후 몇 프레임 동안 확정 시점보다 다시 위로 반등하는지 본다
        — POST_TRIGGER_* 주석 참고. 반등이 확인되면 rebound_penalized 특징을
        반영해 모델 신뢰도를 다시 계산한다(한 번만)."""
        if (self._rebound_penalized
                or self._post_trigger_count >= POST_TRIGGER_CHECK_FRAMES
                or self._trigger_raw_height is None):
            return
        self._post_trigger_count += 1
        if raw_height - self._trigger_raw_height > POST_TRIGGER_REBOUND_MARGIN:
            self._rebound_penalized  = True
            self._trigger_confidence = self._model_confidence()

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
        - 피크~마지막 프레임 사이 평균 하강 속도가 MIN_AVG_DESCENT_SPEED 이상
          (자유낙하가 아닌 느린 하강 배제 — MIN_AVG_DESCENT_SPEED 주석 참고)

        """
        peak_pos   = max(range(len(valid)), key=lambda k: valid[k][1])
        peak_z     = valid[peak_pos][1]
        peak_frame = valid[peak_pos][0]

        if peak_z < PEAK_Z_THRESHOLD:
            return False, ""

        post_peak = [(i, h) for i, h in valid if i > peak_frame]

        if len(post_peak) < MIN_DESCENT_FRAMES:
            return False, ""

        post_peak_heights = [h for _, h in post_peak]
        if any(h > peak_z for h in post_peak_heights):
            return False, ""

        # 피크 대비 하락폭만으로는 "바닥 찍고 다시 올라가는 중"을 구분할 수
        # 없다 — 최근 궤적이 실제로 상승 중이면 낙하 후보에서 제외한다.
        if len(post_peak) >= 2 and (post_peak_heights[-1] - post_peak_heights[-2]) > RISING_TOLERANCE:
            return False, ""

        last_frame, last_z = post_peak[-1]
        peak_drop = peak_z - last_z

        if peak_drop < PEAK_DROP_THRESHOLD:
            return False, ""

        elapsed = (last_frame - peak_frame) * self._frame_dt
        avg_speed = peak_drop / elapsed if elapsed > 0 else float("inf")
        if avg_speed < MIN_AVG_DESCENT_SPEED:
            return False, ""

        reason = (f"peak={peak_z:.2f}m"
                  f"  descent={len(post_peak)}f"
                  f"  drop={peak_drop:.2f}m"
                  f"  avg_v={avg_speed:.2f}m/s")
        return True, reason

    def _freefall_check(self, valid: list[tuple[int, float]]) -> tuple[bool, str]:
        """경로 3 — 시작 높이 무관, 최근 궤적이 자유낙하 패턴인지만 본다.

        가장 좁은 창(FREEFALL_MIN_FRAMES)부터 시도해, 실패하면 한 프레임씩
        창을 넓혀가며(최대 FREEFALL_WINDOW_MAX) 다시 시도한다 — 좁은 창을
        먼저 보는 이유는 data/raw/uncovered/record_raw_20260728_155515
        trial1 track#2 재현 결과 확인됨: 실제 낙하가 이미 좁은 창에서
        깨끗하게 잡히는 경우(예: drop_test_20260717_freefall_validation
        trial1 — 평평하게 시작하다 마지막 구간에서 급가속) 넓은 창을 쓰면
        그 평평했던 앞부분 때문에 오히려 모노토닉 조건에서 탈락한다. 반대로
        uncovered trial1 track#2처럼 정확히 좁은 창(3프레임)의 마지막 구간
        하나가 클러스터 매칭 흔들림으로 튀어(가속도가 허용범위를 벗어나게)
        실패하는 경우엔, 한 프레임 더 넓혀 그 튄 구간을 다른 구간과 섞어
        평균 내면 전체 가속도가 다시 정상 범위로 들어온다. 그래서 "좁은
        창부터 시도, 실패 시 확장"이 두 경우 모두를 살린다.
        """
        if len(valid) < FREEFALL_MIN_FRAMES:
            return False, ""

        max_window = min(len(valid), FREEFALL_WINDOW_MAX)
        for window_size in range(FREEFALL_MIN_FRAMES, max_window + 1):
            result, reason = self._freefall_window_check(valid[-window_size:])
            if result:
                return result, reason
        return False, ""

    def _freefall_window_check(self, recent: list[tuple[int, float]]) -> tuple[bool, str]:
        """_freefall_check의 한 창 크기에 대한 실제 판정 로직."""
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
            dt = (i1 - i0) * self._frame_dt
            if dt <= 0:
                return False, ""
            velocities.append((z1 - z0) / dt)
            midpoints.append((i0 + i1) / 2.0 * FRAME_DT)

        # 정지 클러스터의 위치 노이즈만으로도 순변위 부호·속도 단조감소·
        # 가속도 범위를 전부 우연히 만족할 수 있다 (FREEFALL_MIN_TRIGGER_SPEED
        # 주석의 trial5 재현 사례 참고) — 마지막 구간 속도가 노이즈 수준이면
        # 애초에 자유낙하 후보로 보지 않는다.
        if abs(velocities[-1]) < FREEFALL_MIN_TRIGGER_SPEED:
            return False, ""

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

    @property
    def confidence(self) -> float:
        """낙하 확정 시점의 신뢰도(0~1) — 모듈 docstring의 _MODEL_* 주석 참고.

        data/labeling_worksheet.csv에 라벨링한 71개 배치 61건(진짜 28 /
        노이즈 33)으로 학습한 로지스틱 회귀 출력이다. 확정 직후 최대
        POST_TRIGGER_CHECK_FRAMES 프레임 동안 반등 여부가 반영돼 값이
        한 번 더 바뀔 수 있다(_check_post_trigger_rebound 참고). 아직
        확정 전이면 0.0.
        """
        return self._trigger_confidence

    @property
    def likely_real(self) -> bool:
        """confidence가 LIKELY_REAL_THRESHOLD 이상인지 — 낙하 감지는 놓치는
        게 오탐보다 치명적이라 재현율을 우선해 기본 0.5보다 낮게 잡았다
        (LIKELY_REAL_THRESHOLD 주석 참고)."""
        return self._trigger_confidence >= LIKELY_REAL_THRESHOLD

    @property
    def track_class(self) -> str:
        """트랙 분류: 'PERSON' / 'FALLING_OBJECT' / 'OTHER'.

        scan_person.py의 배치 분류 로직을 실시간으로 근사한다.
          - z_max / z_rng : 전체 관측 기간 누적값 (_z_max_ever 등)
          - t_span        : 실제 관측 횟수 × frame_dt
          - z_drop        : 최초 관측 z(_z_first_obs) - 최근 1/3 구간 평균
        """
        if self._total_obs_count == 0 or self._z_first_obs is None:
            return "OTHER"

        z_max  = self._z_max_ever
        z_rng  = self._z_max_ever - self._z_min_ever
        t_span = self._total_obs_count * self._frame_dt

        # 1. 사람 메인 트랙: z_range(키 수준) 충분 + 장기 지속
        if z_rng >= self._person_z_range_min and t_span >= self._person_min_duration:
            return "PERSON"

        # 2. 단명 fell: 빠른 낙하 물체 (음수 z 전용 클러스터는 바닥 반사)
        if self._fall_triggered and t_span < self._person_min_duration:
            return "FALLING_OBJECT" if z_max > 0.0 else "OTHER"

        # 3. 장기 fell: net 하강 여부로 물체/사람 서브-트랙 구분
        if self._fall_triggered:
            raw_valid = [h for h in self._raw_height_history if h is not None]
            n3 = max(1, len(raw_valid) // 3)
            z_recent = sum(raw_valid[-n3:]) / n3 if raw_valid else z_max
            z_drop = self._z_first_obs - z_recent
            if z_drop >= self._fall_z_drop_min and z_max > 0.0:
                return "FALLING_OBJECT"
            return "PERSON"

        # 4. 장기 지속 (fell 없음): 사람 몸 부위
        if t_span >= self._person_min_duration:
            return "PERSON"

        return "OTHER"

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
    last_fall_track_id: 그 낙하를 확정한 트랙의 id. update()는 한 번 확정된
        트랙에 대해 계속 True를 반환하므로(래치), 호출부에서 "새로 확정된
        낙하인지, 같은 낙하가 계속 보고되는 중인지"를 구분하려면 이 값이
        이전 프레임과 달라졌는지 비교하면 된다.
    last_fall_confidence: 그 낙하의 신뢰도(0~1) — Track.confidence 참고.
        판정 자체(이진값)는 바꾸지 않고, 노이즈로 인한 오판정과 진짜
        낙하를 사후에 구분하는 보조 지표로 쓴다.
    last_centroid: 현재 "주 트랙"(primary_track)의 최신 무게중심 — 매
        프레임 갱신되며, 서보 좌표 전송처럼 낙하 확정 여부와 무관하게
        연속적인 대표 위치가 필요한 호출부용 편의 속성이다.
    """

    def __init__(self, history_window: int = HISTORY_WINDOW, debug: bool = False,
                 confirm_frames: int = CONFIRM_FRAMES, max_jump: float = MAX_JUMP,
                 max_track_misses: int = TRACK_MAX_MISSES,
                 frame_dt: float = FRAME_DT,
                 person_z_range_min: float = PERSON_Z_RANGE_MIN,
                 person_min_duration: float = PERSON_MIN_DURATION,
                 fall_z_drop_min: float = FALL_Z_DROP_MIN):
        self._history_window      = history_window
        self._debug               = debug
        self._confirm_frames      = confirm_frames
        self._max_jump            = max_jump
        self._max_track_misses    = max_track_misses
        self._frame_dt            = frame_dt
        self._person_z_range_min  = person_z_range_min
        self._person_min_duration = person_min_duration
        self._fall_z_drop_min     = fall_z_drop_min

        self._tracks: list[Track] = []
        self._next_id = 1

        self.last_fall_centroid: np.ndarray | None = None
        self.last_fall_track_id: int | None = None
        self.last_fall_confidence: float = 0.0

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
                               debug=self._debug, confirm_frames=self._confirm_frames,
                               frame_dt=self._frame_dt,
                               person_z_range_min=self._person_z_range_min,
                               person_min_duration=self._person_min_duration,
                               fall_z_drop_min=self._fall_z_drop_min)
            self._next_id += 1
            new_track.last_cluster = clusters[i]
            new_track.update(cen)
            self._tracks.append(new_track)

        if fall_track is not None:
            self.last_fall_centroid   = fall_track.last_fall_centroid
            self.last_fall_track_id   = fall_track.id
            self.last_fall_confidence = fall_track.confidence

        return fell

    @property
    def has_falling_object(self) -> bool:
        """낙하 물체(FALLING_OBJECT)로 분류된 트랙이 하나라도 있으면 True."""
        return any(t.track_class == "FALLING_OBJECT" for t in self._tracks)

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
        self.last_fall_centroid = None
        self.last_fall_track_id = None
        self.last_fall_confidence = 0.0
