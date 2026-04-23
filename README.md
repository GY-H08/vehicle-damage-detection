# 차량 손상 탐지 시스템 (Vehicle Damage Detection)

YOLOv8 기반 2-stage 파이프라인으로 차량 부위를 탐지하고 손상 여부를 자동 판별하는 시스템.
AI Hub 차량 손상 데이터셋 약 56만 장을 직접 전처리하고, Active Learning을 적용하여 모델 성능을 개선하였음.

---

## 전체 파이프라인
[AI Hub 원본 데이터 ~56만 장]
↓
[1단계] 데이터 탐색 및 전처리
- JSON 구조 분석 및 이미지-라벨 매칭 검증
- JSON → YOLO 포맷(txt) 변환
- BBox 품질 필터링
- 클래스 불균형 해소 (Scratched 언더샘플링)
- Train / Val 80:20 재분할
↓
[2단계] Active Learning
- 전체 데이터로 초기 학습 (YOLOv8s, 50 epochs)
- 오답 샘플(어려운 샘플) 추출
- 어려운 샘플 집중 Fine-tuning (낮은 학습률 적용)
↓
[3단계] 2-Stage 추론 파이프라인
- Part Model: 차량 부위 탐지
- Damage Model: 탐지된 부위 내 손상 탐지

  ---

## 왜 2-Stage 파이프라인인가

단일 모델로 부위 탐지와 손상 탐지를 동시에 수행할 경우, 전체 이미지에서 작은 손상을 탐지해야 하므로 정밀도가 낮아지는 문제가 있음. 2-stage 구조에서는 Part Model이 먼저 차량 부위를 탐지하고, 해당 영역만 크롭하여 Damage Model에 입력함으로써 탐지 난이도를 낮추고 정밀도를 높임.

---

## 데이터 전처리

### 클래스 불균형 문제
원본 데이터의 손상 클래스 분포가 심각하게 불균형하였음.

| 클래스 | 원본 수량 | 비율 |
|--------|----------|------|
| Scratched | 약 36만 건 | 69% |
| Separated | 약 10만 건 | 20% |
| Crushed | 약 7만 건 | 14% |
| Breakage | 약 7만 건 | 13% |

Scratched 클래스가 전체의 약 70%를 차지하여 모델이 Scratched에 편향되는 문제 발생.
→ Scratched 클래스 언더샘플링으로 불균형 해소

### BBox 필터링
학습 품질 향상을 위해 아래 기준으로 불량 BBox 제거:

- 면적 1,500px 미만 (너무 작은 객체 — 손상 특징 학습 불가)
- 이미지 대비 면적 비율 60% 초과 (너무 큰 객체 — 노이즈 가능성)
- 종횡비 12 초과 (비정상적으로 길쭉한 박스)

### JSON → YOLO 변환
AI Hub 원본 JSON 포맷을 YOLO txt 포맷으로 직접 변환 구현.
절대 좌표(x, y, w, h) → YOLO 정규화 좌표(x_center, y_center, w, h) 변환.

---

## Active Learning

단순 전체 데이터 학습이 아닌 2단계 학습 전략을 적용하여 성능 개선.

**1단계 - 초기 학습:**
전처리된 전체 데이터로 YOLOv8s 학습 (50 epochs, AdamW optimizer, Mosaic/Mixup/CopyPaste 증강 적용)

**2단계 - 어려운 샘플 추출:**
1단계 모델로 Train 이미지 전수 검증. GT 대비 탐지 수 차이가 50% 이상인 샘플을 오답(어려운 샘플)으로 분류. 전체 Train 중 약 27,000장을 어려운 샘플로 추출.

**3단계 - Fine-tuning:**
1단계 best.pt에서 시작, 어려운 샘플만으로 집중 재학습 (30 epochs, lr0=0.0001로 낮춰 기존 학습 보존).

### 학습 파라미터

| 파라미터 | 1단계 | Fine-tuning |
|---------|-------|-------------|
| epochs | 50 | 30 |
| batch | 24 | 24 |
| optimizer | AdamW | AdamW |
| lr0 | 0.001 | 0.0001 |
| mosaic | 1.0 | - |
| mixup | 0.2 | - |
| copy_paste | 0.2 | - |

---

## 2-Stage 추론 파이프라인
입력 이미지
↓
[Part Model] 차량 부위 탐지 (conf=0.15)
↓ 탐지된 부위 bbox를 margin=20px 확장하여 크롭
[Damage Model] 크롭 이미지 내 손상 탐지 (conf=0.25)
↓
결과: 부위별 손상 유형 시각화 출력

---

## 클래스 구성

**차량 부위 (9종)**

| ID | 클래스 |
|----|--------|
| 0 | Front bumper |
| 1 | Rear bumper |
| 2 | Fender |
| 3 | Door |
| 4 | Head lights |
| 5 | Rocker panel |
| 6 | Trunk lid |
| 7 | Bonnet |
| 8 | Other |

**손상 유형 (4종)**

| ID | 클래스 | 설명 |
|----|--------|------|
| 0 | Scratched | 긁힘 |
| 1 | Separated | 분리/박리 |
| 2 | Crushed | 찌그러짐 |
| 3 | Breakage | 파손 |

---

## 모델 성능

| 모델 | mAP50 |
|------|-------|
| Part Model (차량 부위 탐지) | 약 80% |
| Damage Model (손상 탐지) | 약 40.8% |

**Damage Model mAP50이 상대적으로 낮은 이유:**
손상은 부위에 비해 크기가 훨씬 작고, 동일 클래스 내에서도 외관 차이가 크게 나타남 (예: Scratched는 가는 선부터 넓은 면적까지 다양). 또한 클래스 불균형 해소 후에도 Scratched와 나머지 클래스 간의 시각적 경계가 불명확한 케이스가 많아 탐지 난이도가 높음.

---

## 한계점 및 개선 방향

- Damage Model 성능이 40%대에 머무는 근본 원인은 손상의 시각적 모호성. 세그멘테이션 기반 접근으로 전환하면 개선 가능성 있음.
- 데이터 전처리 시 Scratched 언더샘플링 비율 조정으로 추가 성능 개선 여지 있음.
- 현재 2-stage 구조에서 Part Model confidence가 낮을 경우 Damage 탐지 자체가 불가능 — 앙상블 구조 도입 검토 가능.

---

## 기술 스택

- Python, YOLOv8 (Ultralytics)
- Google Colab, Google Drive
- OpenCV, Shapely
- Active Learning

---

## 실행 환경

Google Colab (GPU) 기반으로 구현. 노트북 파일(`vehicle_damage_detection.ipynb`) 참고.

```bash
pip install ultralytics shapely opencv-python
```
