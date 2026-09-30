# 공개 데이터 확장 실험

A사가 개인 고객 데이터와 판촉 집행 기록을 모은다면 무엇이 가능해지는지 공개 데이터로 보여주는 실험입니다. 회사 자료가 아니며, A사 자료에서는 권당 공헌이익(교보 분석 결과의 중앙값) 하나만 가정값으로 가져옵니다.

| 실험 | 데이터 | 내용 |
|---|---|---|
| A. 판촉 메시지 효과·업리프트·ROI | Hillstrom E-Mail Analytics Challenge (2008, 고객 무작위 배정) | 메일 발송 효과(ATE), 효과가 큰 고객을 고르는 업리프트 모델, 메시지 1건당 본전 비용 |
| B. 집계 페르소나 추천 vs 개인 이력 추천 | Book-Crossing (Ziegler 외, 2005) | 연령대 집계 기반 추천과 개인 이력 기반 협업 필터링의 적중률 비교 |

데이터 파일은 저장소에 포함하지 않습니다. `hillstrom/hillstrom.csv`, `bookcrossing/BX-*.csv`에 직접 받아 넣은 뒤 실행합니다.

```bash
cd 코드
python public_experiments.py
```

## 출처

- Hillstrom, K. (2008). MineThatData E-Mail Analytics and Data Mining Challenge. http://www.minethatdata.com
- Ziegler, C.-N., McNee, S. M., Konstan, J. A., & Lausen, G. (2005). Improving Recommendation Lists Through Topic Diversification. WWW 2005.

## 한계

공개 데이터는 국가·업종·시기가 달라 효과 크기를 A사에 그대로 적용하지 않으며, 가정 기반 시나리오로만 씁니다.
