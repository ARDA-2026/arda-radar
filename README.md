# ARDA — Automated Radar-based Detection & Alert

TI IWR6843AOPEVM 레이더를 이용한 실시간 물체 낙하 감지 시스템.

## 디렉토리 구조

```
ARDA/
├── config/
│   └── profiles/          # 레이더 .cfg 설정 파일
├── arda/
│   ├── radar/             # 시리얼 통신 & 프레임 파서
│   ├── processing/        # 포인트 클라우드 전처리 & 클러스터링
│   ├── detection/         # 낙하 감지 알고리즘
│   ├── visualization/     # 실시간 3D 플롯
│   └── utils/             # 로거 등 공통 유틸
├── data/
│   ├── raw/               # 녹화된 원시 프레임 (JSONL)
│   ├── processed/         # 전처리 결과
│   └── logs/              # 이벤트 로그
├── models/                # 학습된 ML 모델 (옵션)
├── scripts/               # 각 스크립트 역할은 "스크립트 설명" 절 참고
├── tests/                 # pytest 단위 테스트
└── main.py                # 실시간 감지 실행
```

## 사전 준비

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (권장) 또는 pip
- IWR6843AOPEVM 레이더 보드 (실시간 감지 시 필요, 녹화 재생/테스트만 할 경우 불필요)

## 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/ARDA-2026/arda-radar.git
cd arda-radar

# 의존성 설치
uv sync   # 또는 pip install -e .

# 테스트 (하드웨어 없이 실행 가능)
pytest

# 센서 설정 & 실시간 감지 실행 (하드웨어 연결 필요)
python main.py --cli-port /dev/ttyUSB0 --data-port /dev/ttyUSB1

# 낙하가 확정되는 순간의 좌표만 UDP(127.0.0.1:9999)로 전송되어 sibling
# 프로젝트인 arda-servo(서보 모터 제어기)가 소비한다. 비활성화하려면 --no-servo-out.

# 데이터 녹화 (60초, 하드웨어 연결 필요)
python scripts/record.py --output data/raw/session1.jsonl --duration 60

# 녹화 재생 (하드웨어 없이 검증 가능)
python scripts/replay.py data/raw/session1.jsonl
```

레이더 하드웨어가 없는 팀원은 `pytest`와 `scripts/replay.py`로 파이프라인 동작을 확인할 수 있습니다. (재생용 샘플 `.jsonl` 파일은 `data/raw/`에 없으므로, 녹화가 있는 팀원에게 공유받아 넣어주세요.)

## 하드웨어 연결

| 포트 | 역할 | 기본값 |
|------|------|--------|
| CLI port | 설정 명령 전송 | `/dev/ttyUSB0` |
| Data port | 포인트 클라우드 수신 | `/dev/ttyUSB1` |

Windows에서는 `COM3` / `COM4` 형식으로 지정.

## 좌표 기준(원점) 및 유효 범위 수정

레이더가 출력하는 `(x, y, z)`는 **센서 자체를 원점(0,0,0)으로 하는 좌표**입니다.

| 축 | 의미 | 단위 |
|----|------|------|
| `x` | 센서 정면 기준 좌우 (우측이 +) | m |
| `y` | 센서 정면 거리 (센서 바로 앞이 0, 항상 양수) | m |
| `z` | 센서 기준 높이 (센서와 같은 높이가 0) | m |

**원점 자체(0,0,0)는 소프트웨어에서 옮길 수 없습니다** — TI mmWave 하드웨어의
안테나 기준점으로 고정되어 있어, 원점을 바꾸려면 센서를 물리적으로
재장착/재조준해야 합니다. 소프트웨어에서 조정 가능한 건 "이 원점을 기준으로
어느 범위까지를 유효한 타겟으로 볼지"(ROI)입니다.

**ROI를 바꾸려면 여기를 고치세요**: [`arda/processing/pointcloud.py`](arda/processing/pointcloud.py)의
`PointCloud.filter_roi()` 기본 인자(`x_range`, `y_range`, `z_range`, 단위 m).
`main.py`의 감지 루프(`pc.filter_roi()`)가 인자 없이 호출하므로 이 기본값이
그대로 적용됩니다. `config/settings.yaml`의 `processing.roi`는 코드에서
읽지 않는 문서용 값이라 여기를 고쳐도 반영되지 않습니다 (아래 "낙하 감지
설정값" 섹션의 주의사항 참고).

> arda-servo와 연계할 때는 이 원점·좌표축이 곧 서보 각도 계산의 기준이
> 됩니다 — 레이더와 서보가 물리적으로 같은 위치·같은 정면 방향에 있다고
> 가정하므로, 실제로 떨어져 있거나 방향이 어긋나 있다면 arda-servo 쪽
> README의 "좌표 기준(원점) 보정" 섹션을 참고하세요.

### 설치 위치 기준 실좌표 변환 (`site`)

낙하가 확정되면, 센서 기준 로컬 좌표를 **이 레이더가 실제로 설치된
지점의 고정 실좌표(시설/도면 좌표계 등)** 로 변환해 로그에 남깁니다.
위 ROI와 달리 이 값은 **실제로 코드에서 읽어서 사용**합니다.

**설정 위치**: `config/settings.yaml`의 `site.x` / `site.y` / `site.z`
(단위 m, 설치 시 1회 실측해서 채워 넣는 값). 설정 파일 경로는
`--settings`로 바꿀 수 있습니다 (`--config`는 레이더 칩 자체의 `.cfg`
프로파일 경로라 서로 다른 옵션입니다).

```yaml
site:
  x: 12.4   # 시설 도면 기준 X (m)
  y: 3.1    # 시설 도면 기준 Y (m)
  z: 1.1    # 이 레이더가 설치된 "바닥 기준" 높이 (m) — 이 좌표계는 바닥을 Z=0으로 둔다
```

**이 좌표계는 바닥을 Z=0으로 정의합니다** — `site.z`는 그 자체로 "바닥에서
센서까지의 높이"입니다. X, Y는 로컬 좌표에 `site.x`/`site.y`를 더하는
평행이동으로 변환합니다 ([`arda/utils/site.py`](arda/utils/site.py)의
`to_site_coords()`). 센서가 정면으로 보는 방향(heading)이 시설 좌표계
축과 다르면 그 회전은 보정하지 않으므로, 센서를 시설 좌표계 축에
맞춰(예: 정북 방향 등) 장착하거나 별도 회전 보정이 필요합니다.

**Z는 실측값을 쓰지 않고 항상 0(바닥)으로 보고합니다.** 낙하가 확정되는
순간의 실측 로컬 Z(`last_centroid[2]`)는 "바닥에 닿은 높이"가 아니라
"피크보다 `PEAK_DROP_THRESHOLD`(기본 0.35m) 이상 떨어진 순간"의
높이라 아직 완전히 쓰러지기 전일 수 있어 신뢰할 수 없습니다. 대신 낙하는
바닥에서 일어난다고 간주하고, 위 좌표계 정의(바닥=Z 0)를 그대로
사용합니다 — `site.z`가 어떤 값이든 상관없이 항상 실좌표 Z=0으로
보고됩니다 (`site.z`가 그 자체로 "바닥 기준 높이"이므로, 바닥의 실좌표는
정의상 언제나 0입니다). 변환 결과는 낙하 확정 시 `logger.warning`으로
남을 뿐 아직 서버로 전송하지는 않습니다 — 서버 연동은 이후 단계입니다.

## 알고리즘 흐름

```
레이더 프레임 → SNR/ROI 필터 → DBSCAN 클러스터링 → 타겟 클러스터 선택(select_target)
              → FallDetector.update() → 낙하 여부
```

1. **필터링**: 노이즈(낮은 SNR) 및 관심 영역(ROI) 밖 포인트 제거
2. **클러스터링 & 타겟 선택**: DBSCAN으로 포인트를 묶은 뒤, `select_target()`이 우선순위대로 추적 대상을 고름 — ①공중+하향 이동 ②공중 ③하향 이동 ④가장 큰 클러스터(fallback) 순
3. **낙하 판정** (`FallDetector`, 매 프레임 타겟 중심 높이(Z)를 이력에 누적): 아래 세 경로 중 하나라도 만족하면 낙하로 판정
   - **경로 1 (착지 후 소실, 주 경로)**: 피크(공중) 이후 일정 프레임 이상 연속 하강하다 클러스터가 레이더에서 사라지면(바닥 근처 소실) 착지로 판단
   - **경로 2 (저 Z 직접 감지, 보조 경로)**: 물체가 레이더에 계속 보이는 상태로, 피크 대비 충분히 하강한 경우 직접 감지
   - **경로 3 (자유낙하 궤적, 시작 위치 무관)**: `PEAK_Z_THRESHOLD` 같은 절대 높이 기준 없이, 최근 프레임들의 속도가 중력 근방으로 계속 가속하며 떨어지는 패턴이면 어디서 처음 포착됐든 낙하로 판정

## 낙하 감지 설정값 (`config/settings.yaml`)

`config/settings.yaml`의 `processing:` 섹션은 `arda.utils.load_processing_config()`를
통해 `main.py`와 `scripts/detect.py`·`trajectory.py`·`record.py`·
`record_and_view.py`·`analyze_drops.py`가 공통으로 읽습니다 — 이 파일 한 곳만
고치면 전부에 반영됩니다 (이전엔 각 스크립트 상단에 동일한 값이 중복 하드코딩돼
있었습니다).

| 섹션 | 항목 | 설명 | 기본값 |
|------|------|------|--------|
| processing | `min_snr` | 노이즈 포인트 제거용 최소 SNR (근거리일수록 낮게) | `6.0` |
| processing | `roi.x / .y / .z` | 감지 대상으로 볼 관심 영역 범위 (m) | `[-1.0,1.0]` / `[0.1,1.0]` / `[-0.8,0.8]` |
| processing | `min_abs_doppler` | 정적(비이동) 포인트 제거 임계값 (main.py의 filter_stationary 전용) | `0.05` |
| processing | `cluster_eps` | DBSCAN 클러스터링 반경 (m) — 근거리일수록 줄임 | `0.15` |
| processing | `cluster_min_samples` | 클러스터로 인정할 최소 포인트 수 | `2` |
| processing | `airborne_z` | 공중 판별 최소 높이 (m) — `select_target`/`choose_target`용 | `0.40` |
| processing | `max_jump` | 직전 프레임 무게중심 대비 허용 최대 이동 거리 (m) — `choose_target`용 | `0.5` |
| detection | `history_window` | 낙하 판정에 사용하는 프레임 이력 개수 (100ms × N) | `6` |

실제 판정 임계값은 [`arda/detection/fall_detector.py`](arda/detection/fall_detector.py) 상단 상수로 관리됩니다.

| 상수 | 설명 | 기본값 |
|------|------|--------|
| `PEAK_Z_THRESHOLD` | 공중에 있었다고 판별할 최소 높이 (m) | `0.37` |
| `PEAK_DROP_THRESHOLD` | 피크 대비 최소 하락폭 — 이 이상 떨어지면 낙하 확정 (m) | `0.35` |
| `MIN_DESCENT_FRAMES` | 피크 이후 연속 하강해야 하는 최소 프레임 수 (100ms × N) | `2` |
| `HISTORY_WINDOW` | 판정에 사용하는 프레임 이력 개수 | `10` |
| `FREEFALL_MIN_FRAMES` | 경로 3: 자유낙하로 볼 최소 연속 유효 프레임 수 | `3` |
| `FREEFALL_ACCEL_MIN` / `_MAX` | 경로 3: 자유낙하로 인정할 가속도 범위 (m/s², 중력 9.8 기준 여유) | `4.0` / `15.0` |

> **주의**: `detection:` 섹션(`height_drop_threshold`, `fall_doppler_threshold`,
> `z_velocity_threshold`)은 여전히 이전 버전 알고리즘 기준으로 남아있는 문서용
> 값이며, 코드에서 읽어오지 않고 현재 로직과도 맞지 않습니다. 판정 임계값을
> 튜닝할 때는 `fall_detector.py` 상단 상수를 직접 수정하세요. (`history_window`만
> 예외로, `detection.history_window`는 `FallDetector` 생성 시 인자로 넘길 수
> 있으나 현재 어떤 진입점도 그렇게 연결해두진 않았습니다.)

## 스크립트 설명 (`scripts/`)

| 스크립트 | 역할 |
|----------|------|
| `check_ports.py` | 센서 연결 전 포트 진단 — CLI/Data 포트에서 raw 바이트가 들어오는지만 확인 |
| `record.py` | 지정 시간만큼 레이더 데이터를 녹화해 JSON으로 저장 (임계값 튜닝용 데이터 수집) |
| `record_and_view.py` | 짧게 녹화한 뒤 포인트·클러스터·타겟 무게중심의 Z(t)/X(t)/Y(t) 궤적을 그래프로 시각화 |
| `replay.py` | 녹화된 JSONL 파일을 재생하며 낙하 감지 로직을 검증 (하드웨어 불필요) |
| `detect.py` | 실시간 낙하 감지 실행 — SNR/ROI 필터 → DBSCAN → `select_target` → `FallDetector` |
| `trajectory.py` | 실시간으로 Z축 하강 궤적과 감지 상태를 시각화 (`detect.py`와 동일 파이프라인) |
| `monitor.py` | 낙하 감지 없이 원시 포인트 클라우드만 실시간 모니터링 |
| `rdmap.py` | Range-Doppler Map(거리·속도별 신호 세기) 실시간 시각화 |

## 커밋 규칙

- 작업 진행 중인 커밋은 커밋 메시지 맨 앞에 `[WIP]` 태그를 붙입니다. 예: `[WIP] 클러스터링 파라미터 튜닝`
- 회의/발표용으로 확정된 최종 코드는 `[Done]` 태그를 붙입니다. 예: `[Done] 낙하 감지 임계값 확정`
- 그 외 세부 파트별 작업은 각자 별도 저장소(ARDA-2026 조직 내)에서 자유롭게 진행합니다.
