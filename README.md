# 차량 손상 탐지 시스템 (Vehicle Damage Detection)

## 프로젝트 개요
YOLOv8 기반 2-stage 파이프라인으로 차량 부위를 탐지하고 손상 여부를 자동 판별하는 시스템.
약 56만 장의 대규모 데이터셋을 전처리하고 Active Learning을 적용하여 모델 성능을 개선하였음.

## 주요 기능
- **Part Model**: 차량 부위(문, 보닛, 범퍼 등) 탐지
- **Damage Model**: 탐지된 부위 내 손상(찍힘, 긁힘 등) 탐지
- **2-stage Pipeline**: Part → Damage 순차 추론으로 정밀도 향상
- **Active Learning**: 불확실한 샘플 중심으로 재학습하여 성능 개선

## 기술 스택
- Python, YOLOv8 (Ultralytics)
- Google Colab, Google Drive
- Active Learning
- OpenCV

## 데이터셋
- 약 56만 장 차량 이미지
- 차량 부위 및 손상 라벨링 데이터
- Train / Validation 분리 구성

## 모델 성능
- Part Model: mAP50 약 80%
- Damage Model: mAP50 약 40.8%

## 실행 환경
Google Colab 기반으로 구현. 노트북 파일 참고.
