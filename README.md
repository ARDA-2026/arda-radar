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

## 커밋 규칙

- 작업 진행 중인 커밋은 커밋 메시지 맨 앞에 `[WIP]` 태그를 붙입니다. 예: `[WIP] 클러스터링 파라미터 튜닝`
- 회의/발표용으로 확정된 최종 코드는 `[Done]` 태그를 붙입니다. 예: `[Done] 낙하 감지 임계값 확정`
- 그 외 세부 파트별 작업은 각자 별도 저장소(ARDA-2026 조직 내)에서 자유롭게 진행합니다.
