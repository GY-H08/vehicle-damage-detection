# 차량 손상 탐지 시스템 (Vehicle Damage Detection)

YOLOv8 기반 2-stage 파이프라인으로 차량 부위를 탐지하고 손상 여부를 자동 판별하는 시스템.
AI Hub 차량 손상 데이터셋 약 56만 장을 직접 전처리하고, 총 9차례의 학습 실험을 통해 모델 성능을 개선하였음.

---

## 전체 파이프라인
[AI Hub 원본 데이터 ~56만 장]
↓
[1단계] 데이터 탐색 및 전처리
- JSON 구조 분석 및 이미지-라벨 매칭 검증
- JSON → YOLO 포맷(txt) 변환
- Segmentation 기반 Tight BBox 추출
- BBox 품질 필터링 (면적/비율/종횡비)
- 클래스 불균형 해소 (언더샘플링 + 오버샘플링)
- Train / Val 80:20 재분할
↓
[2단계] 반복 학습 실험 (총 9차)
- 다양한 전처리/모델/학습 전략 실험
- 실패 원인 분석 및 개선 반복
↓
[3단계] 2-Stage 추론 파이프라인
- Part Model: 차량 부위 탐지 (9종)
- Damage Model: 탐지된 부위 내 손상 탐지 (4종)
- ---

## 왜 2-Stage 파이프라인인가

단일 모델로 부위 탐지와 손상 탐지를 동시에 수행할 경우 part × damage = 88개 클래스가 필요하고, 클래스당 데이터 불균형이 극심해짐 (일부 클래스 수백 장 수준). 초기 88클래스 단일 모델 실험 결과 mAP50 최고 6.57%로 사실상 학습 실패.

→ Part 탐지와 Damage 탐지를 분리하는 2-stage 구조로 전환하여 클래스 수를 줄이고 각 모델의 학습 난이도를 낮춤.

---

## 데이터 전처리

### 클래스 불균형 문제

| 클래스 | 원본 수량 | 비율 |
|--------|----------|------|
| Scratched | 약 36만 건 | 62% |
| Separated | 약 10만 건 | 20% |
| Crushed | 약 7만 건 | 9% |
| Breakage | 약 7만 건 | 9% |

원본 불균형 비율 6.6:1 → 최종 2.2:1까지 해소 (언더샘플링 + 소수 클래스 선별 오버샘플링)

### Segmentation 기반 Tight BBox
원본 JSON bbox는 실제 손상 영역보다 훨씬 넓어 배경을 많이 포함함.
Segmentation 폴리곤에서 bbox를 직접 추출하고 5% padding 추가 → 손상 영역 집중 학습 가능.
이 전처리만으로 mAP50 +7%p 향상.

### BBox 품질 필터링
- 면적 1,500px 미만 제거 (손상 특징 학습 불가)
- 이미지 대비 면적 60% 초과 제거 (배경 노이즈)
- 종횡비 12 초과 제거 (비정상 박스)

### Part 클래스 통합 (32개 → 9개)
원본 데이터는 좌우/전후 분리로 32개 클래스, 일부 클래스는 수십 장 수준으로 학습 불가.
방향 정보를 제거하고 부위 단위로 통합:
- Front/Rear Fender LH/RH → Fender
- Front/Rear Door LH/RH → Door
- Head lights LH/RH → Head_lights
- 나머지 소수 클래스 → Other

---

## 반복 학습 실험 과정

### 1차 시도 — 88클래스 단일 모델 (실패)
part × damage 조합으로 88개 클래스 구성. 클래스 불균형 심각 (일부 수백 장). mAP50 최고 6.57%.
→ 2-stage 구조로 전환 결정.

### 2차 — Damage 기본 학습
JSON bbox 그대로 사용, 기본 하이퍼파라미터. mAP50 30.7%.
문제: bbox가 너무 넓어 배경 많이 포함, 모델이 객체를 background로 오인.

### 3차 — Segmentation Tight BBox + 필터링 적용
실제 손상 영역만 포함하도록 bbox 재추출. mAP50 37.9% (+7%p).
→ **모델 튜닝보다 데이터 전처리가 성능에 훨씬 결정적임을 확인.**

### 4차 — Active Learning (실패)
Stage 1 모델로 오답 샘플 27,486개 추출 후 집중 Fine-tuning.
결과: mAP50 36.7% (오히려 하락). 어려운 샘플만 반복 학습 → 다양성 감소 → 과적합.
→ Active Learning은 이 데이터셋 구조에서 효과 없음.

### 5차 — 소수 클래스 오버샘플링
소수 클래스 2배 복제. 불균형 비율 6.6:1 → 3.7:1. mAP50 40.8% (TTA 적용).

### 6차 — Part 모델 학습
32개 클래스를 9개로 통합, 소수 클래스 오버샘플링 적용. mAP50 약 80%, mAP50-95 약 73%. 35 epoch 조기 종료.

### 7차 — CBAM Attention 삽입 (실패)
YOLOv8 backbone에 CBAM 모듈 삽입 시도. Pretrained weights 매핑 실패로 처음부터 학습하는 상황 발생. 7 epoch 만에 포기.
→ 모델 구조 변경 시 pretrained 활용이 어려움.

### 8차 — 선별적 오버샘플링
Scratched only 파일 43,155개 제거 후 소수 클래스만 선별 복제. 불균형 비율 2.2:1, Scratched 비율 47.5%까지 해소.

### 9차 — Freeze Transfer Learning (실패)
Backbone 10개 레이어 고정 후 Head만 학습. mAP50 33.1% (Baseline 38.9%보다 낮음).
→ 손상은 형태가 아닌 텍스처 기반 탐지 → Backbone도 재학습 필요. Freeze 전략 부적합.

### 최종 — 8차 모델 추가 Fine-tuning
8차 모델에서 lr=0.0001로 20 epoch 추가 학습.
**최종 Damage mAP50: 40.8%, mAP50-95: 21.8%, Precision: 49.2%, Recall: 41.0%**

---

## 최종 모델 성능

| 모델 | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| Part Model | 약 80.5% | 약 73.9% | 75.7% | 74.9% |
| Damage Model | 약 40.8% | 약 21.8% | 49.2% | 41.0% |

**Damage Model 성능이 낮은 이유:**
손상은 크기가 작고 클래스 간 시각적 경계가 불명확함 (예: Scratched와 Crushed의 미세한 차이).
텍스처 기반 탐지의 본질적 어려움으로 한계 존재.

---

## 핵심 교훈

> **모델 튜닝(Active Learning, CBAM, Freeze)보다 데이터 전처리(Segmentation Tight BBox)가 성능에 훨씬 결정적인 영향을 미쳤음.**
> Segmentation 기반 BBox 전처리만으로 mAP50 +7%p 향상. 반면 Active Learning, CBAM, Freeze는 모두 실패 또는 효과 없음.

---

## 클래스 구성

**차량 부위 (9종):** Front bumper, Rear bumper, Fender, Door, Head lights, Rocker panel, Trunk lid, Bonnet, Other

**손상 유형 (4종):** Scratched(긁힘), Separated(분리), Crushed(찌그러짐), Breakage(파손)

---

## 한계점 및 개선 방향

- Damage Model 40%대 한계 — 세그멘테이션 기반 접근으로 전환 시 개선 가능성 있음
- 현재 2-stage 구조에서 Part conf가 낮으면 Damage 탐지 자체 불가 — 앙상블 구조 도입 검토 가능
- 보험/사고 처리 자동화, 모바일 실시간 손상 진단 서비스로 확장 가능

---

## 기술 스택

- Python, YOLOv8 (Ultralytics)
- Google Colab, Google Drive
- OpenCV, Shapely

---

## 실행 환경

Google Colab (GPU) 기반으로 구현. 노트북 파일(`vehicle_damage_detection.ipynb`) 참고.

```bash
pip install ultralytics shapely opencv-python
```
