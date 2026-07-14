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

# 타겟 좌표는 기본적으로 UDP(127.0.0.1:9999)로 전송되어 sibling 프로젝트인
# arda-servo(서보 모터 제어기)가 소비한다. 비활성화하려면 --no-servo-out.

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

## 알고리즘 흐름

```
레이더 프레임 → SNR/ROI 필터 → DBSCAN 클러스터링 → 타겟 클러스터 선택(select_target)
              → FallDetector.update() → 낙하 여부
```

1. **필터링**: 노이즈(낮은 SNR) 및 관심 영역(ROI) 밖 포인트 제거
2. **클러스터링 & 타겟 선택**: DBSCAN으로 포인트를 묶은 뒤, `select_target()`이 우선순위대로 추적 대상을 고름 — ①공중+하향 이동 ②공중 ③하향 이동 ④가장 큰 클러스터(fallback) 순
3. **낙하 판정** (`FallDetector`, 매 프레임 타겟 중심 높이(Z)를 이력에 누적): 아래 두 경로 중 하나라도 만족하면 낙하로 판정
   - **경로 1 (착지 후 소실, 주 경로)**: 피크(공중) 이후 일정 프레임 이상 연속 하강하다 클러스터가 레이더에서 사라지면(바닥 근처 소실) 착지로 판단
   - **경로 2 (저 Z 직접 감지, 보조 경로)**: 물체가 레이더에 계속 보이는 상태로, 피크 대비 충분히 하강한 경우 직접 감지

## 낙하 감지 설정값 (`config/settings.yaml`)

튜닝이 자주 발생하는 항목은 `config/settings.yaml`의 `processing:` / `detection:` 섹션에 모여 있습니다.

| 섹션 | 항목 | 설명 | 기본값 |
|------|------|------|--------|
| processing | `min_snr` | 노이즈 포인트 제거용 최소 SNR (근거리일수록 낮게) | `8.0` |
| processing | `roi.x / .y / .z` | 감지 대상으로 볼 관심 영역 범위 (m) | `[-1.5,1.5]` / `[0.3,2.5]` / `[-0.2,2.2]` |
| processing | `min_abs_doppler` | 정적(비이동) 포인트 제거 임계값 — 낮출수록 느린 낙하도 포착 | `0.05` |
| processing | `cluster_eps` | DBSCAN 클러스터링 반경 (m) — 근거리일수록 줄임 | `0.3` |
| processing | `cluster_min_samples` | 클러스터로 인정할 최소 포인트 수 | `2` |
| detection | `history_window` | 낙하 판정에 사용하는 프레임 이력 개수 (100ms × N) | `6` |

실제 판정 임계값은 [`arda/detection/fall_detector.py`](arda/detection/fall_detector.py) 상단 상수로 관리됩니다.

| 상수 | 설명 | 기본값 |
|------|------|--------|
| `PEAK_Z_THRESHOLD` | 공중에 있었다고 판별할 최소 높이 (m) | `0.40` |
| `PEAK_DROP_THRESHOLD` | 피크 대비 최소 하락폭 — 이 이상 떨어지면 낙하 확정 (m) | `0.35` |
| `MIN_DESCENT_FRAMES` | 피크 이후 연속 하강해야 하는 최소 프레임 수 (100ms × N) | `5` |
| `HISTORY_WINDOW` | 판정에 사용하는 프레임 이력 개수 | `10` |

> **주의**: `config/settings.yaml`의 `detection:` 섹션(`height_drop_threshold`, `fall_doppler_threshold`, `z_velocity_threshold`)은 코드에서 읽어오지 않는 문서용 값이며, 이전 버전 알고리즘 기준이라 현재 로직과도 맞지 않습니다. 임계값을 튜닝할 때는 `fall_detector.py`의 상수를 직접 수정하세요.

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
