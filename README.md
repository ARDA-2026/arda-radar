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
├── scripts/
│   ├── record.py          # 데이터 녹화
│   └── replay.py          # 녹화 재생 & 검증
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
레이더 프레임 → SNR/ROI 필터 → 정적 포인트 제거 → DBSCAN 클러스터링
              → 최대 클러스터 선택 → FallDetector.update() → 낙하 여부
```

1. **필터링**: 노이즈(낮은 SNR) 및 관심 영역(ROI) 밖 포인트 제거, 속도가 거의 없는 정적 포인트 제거
2. **클러스터링**: 남은 포인트를 DBSCAN으로 묶고, 가장 큰 클러스터를 감지 대상으로 선택
3. **낙하 판정** (`FallDetector`, 매 프레임 대상 클러스터의 중심 높이(Z)를 이력에 누적): 아래 두 조건 중 하나라도 만족하면 낙하로 판정
   - **조건 A (완만한 하강)**: 최근 프레임 이력 동안 높이가 임계값 이상 떨어졌고, 동시에 하향 도플러도 감지됨
   - **조건 B (급락)**: 이력 중 최고 높이가 일정 이상(공중에 있었음)이었다가, 현재 높이가 그 지점 대비 크게 떨어짐

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
| detection | `height_drop_threshold` | 조건 A: 이력 윈도우 내 높이 하락량 임계값 (m) | `0.3` |
| detection | `fall_doppler_threshold` | 조건 A: 하향 도플러 보조 조건 (m/s) | `-0.2` |
| detection | `z_velocity_threshold` | 조건 B: 프레임 간 Z 순간 속도 임계값, 자유낙하 1프레임 감지용 (m/s) | `-0.3` |

> **주의**: 현재 실제 감지 로직([`arda/detection/fall_detector.py`](arda/detection/fall_detector.py))은 위 값을 `settings.yaml`에서 읽지 않고, 파일 상단에 `HEIGHT_DROP_THRESHOLD`, `FALL_DOPPLER_THRESHOLD`, `PEAK_Z_THRESHOLD`, `PEAK_DROP_THRESHOLD` 등 별도 상수로 하드코딩되어 있습니다. 임계값을 튜닝할 때는 `settings.yaml`과 `fall_detector.py` 양쪽을 함께 확인해주세요.

## 커밋 규칙

- 작업 진행 중인 커밋은 커밋 메시지 맨 앞에 `[WIP]` 태그를 붙입니다. 예: `[WIP] 클러스터링 파라미터 튜닝`
- 회의/발표용으로 확정된 최종 코드는 `[Done]` 태그를 붙입니다. 예: `[Done] 낙하 감지 임계값 확정`
- 그 외 세부 파트별 작업은 각자 별도 저장소(ARDA-2026 조직 내)에서 자유롭게 진행합니다.
