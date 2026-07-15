# data 폴더 구조

```
data/
├── raw/          # record.py로 녹화한 세션 (시나리오별 폴더, raw/README.md 참고)
├── reference/    # 특정 실패 사례를 보여주기 위해 남겨둔 참고용 캡처 이미지
└── logs/         # 실행 중 쌓이는 이벤트 로그 (arda.log)
```

- `raw/` 안의 각 폴더는 원본 `record_*.json`과, 그걸 재생해 만든 분석
  그래프(`analysis.png`, `points_clusters.png`)를 함께 둔다. 자세한 폴더별
  설명은 [raw/README.md](raw/README.md) 참고.
- `reference/`의 이미지들(`failDetecting.png`, `failTracking.png`,
  `wrongTracking.png`)은 각각 특정 실패 패턴(피크/프레임 부족으로 미감지,
  근접 노이즈로 타겟이 튐, 방향성 없는 재탐색으로 엉뚱한 클러스터를 물음)을
  논의하며 참고했던 캡처로, 대응하는 원본 녹화 데이터는 없다.
