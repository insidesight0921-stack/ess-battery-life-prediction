# DAY2 요구사항 정리 (Notion "DS, Mini Project" 기준, 2026-10-02 확인)

> 원문은 저작권 보호 대상이라 요지만 정리함. 원문: Notion DS Mini Project 페이지 DAY 2 / Deliverables

## 목표
- DAY1 모델 설계 전략을 코드로 구현 → 최적 모델 개발 → 논문 성능(Target)과 비교 → 시사점 도출
- 교수자 강조: 성능만으로 평가하지 않음 / AI 도구 활용 증명이 목적 아님 → **고민의 과정**이 보이게 / 예쁜 보고서는 평가 대상 아님

## Modeling
- 필수: Batch1 = 학습, Batch2 = 테스트
- 선택: 최적 모델로 Batch3 추가 검증 (배치 간 차이로 성능 하락 가능)
  - Batch3 주의점 (Notion 명시)
    - Cycle Life 분포가 배치별로 다름 (Notion: "Batch1/2는 유사, Batch3는 차이")
    - 충전 커브 시작 시점이 배치별로 달라 Qdlin 단순 비교 시 왜곡 가능
    - 배치 수집 사이 수개월 공백 → 일부 셀은 원논문에서도 제거됨

## Performance Reporting (회귀 포맷)
| 구분 | MAPE (%) | 비고 |
|---|---|---|
| Train (Batch1 CV) | | Batch1 내 CV 평균 |
| Valid (Batch1 Hold-out) | | Batch1 내 Hold-out |
| Test (Batch2) | | 최종 평가 |
| Gap (Train-Valid) | | (+) 과적합 의심 |
| Gap (Valid-Test) | | (+) 배치 간 일반화 저하 의심 |
| Gap (Target-Test) | | Target: 원논문 9.1% |
| Test (Batch3) — 선택 | | |
| Gap (Batch2-Batch3) — 선택 | | 크면 피처가 특정 배치에 과적합됐을 가능성 분석·원인 제시 |
| Gap (Target-Test) Batch3 — 선택 | | |

- **Valid를 Hold-out으로 쓰는 이유 (Notion 명시)**: 셀마다 충전 프로토콜이 다르므로 CV를 하면 같은 프로토콜 셀이 train/valid에 나뉘어 누수 위험 → Hold-out으로 셀 단위 분리를 명확히 보장

## 제출
- GitHub public 링크, **DAY2 16시** 마감, 반별 Slack 스레드
- README.md: Notion 샘플 구조
  - 프로젝트 개요 / 파일 구조 / 환경 설정
  - EDA (분포, 열화 곡선, ΔQ, C-rate, 추가 발견 — 각 "핵심 발견" 한 줄)
  - Modeling (피처 엔지니어링 전략·근거, 후보/최종 모델·선택 이유)
  - 성능 결과 (포맷), 원논문 대비 Gap(Target-Test)은 Batch2 대상
  - 오류 분석 (가장 크게 틀린 셀의 공통점, 원인 가설, 개선 방향)
  - ESS 도메인 해석 (BESS 적용 시 의사결정 활용, 한계와 실배포 필요사항)
  - 참고문헌, 팀 구성

## 평가 (DAY2, 100점)
| 항목 | 내용 | 배점 |
|---|---|---|
| 전략 → 구현 반영 | 전략 기반 Feature·모델 구현 | 20 |
| Pipeline 개발 | 개발 파이프라인, **데이터 분할 적절성**, 핵심 변수 구현 | 40 |
| 성능 리포팅 및 해석 | 포맷 기반 정리, 목표 대비 Gap 해석력 | 20 |
| 분석 결과 해석 | 도메인 관점 해석, 개발 한계점 | 20 |

## 현재 코드와 비교해 확인할 점
- [ ] Hold-out 분할: 지금은 무작위 → 같은 충전 정책 셀(대부분 2개씩)이 Train/Valid에 갈라질 수 있음 → **정책 단위 분할 필요** (Notion 누수 설명과 같은 이유)
- [ ] Train CV도 정책 단위(GroupKFold)로 할지 검토
- [ ] Notion은 "Batch1/2 분포 유사"라고 했는데 우리 EDA에선 크게 다름(중앙값 772 vs 472) → 원인 확인 필요 (미확인: 셀 정리 기준 차이 또는 Kaggle 버전 차이 가능성)
- [ ] Batch3 Qdlin 시작점 차이 → ΔQ 피처 왜곡 여부 확인
- [ ] README 작성, GitHub public 업로드
