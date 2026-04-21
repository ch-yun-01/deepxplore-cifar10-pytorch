# DeepXplore CIFAR-10 (PyTorch)

> 신뢰할 수 있는 인공지능 Assignment 2

---

## 프로젝트 개요

본 프로젝트는 [DeepXplore](https://arxiv.org/abs/1705.06640)를 CIFAR-10 데이터셋과 PyTorch 환경에 맞게 구현한 것입니다.  
서로 다른 방식으로 학습된 세 개의 ResNet-50 모델에 대해 **difference-inducing input**을 생성하고, **neuron coverage**를 극대화하면서 모델 간 예측 불일치를 유발하는 입력을 탐색합니다.

`python test.py` 한 번으로 exp1~exp14 전체 실험이 자동으로 순차 실행됩니다.

---

## 환경 설치

```bash
pip install -r requirements.txt
```

주요 패키지 버전은 다음과 같습니다.

| 패키지 | 버전 |
|---|---|
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| numpy | 2.2.6 |
| pillow | 12.2.0 |
| opencv-python | 4.13.0.92 |

---

## 실행 방법

```bash
# 모든 실험 자동 실행 (exp1 ~ exp14)
python test.py
```

---

## 파일 구조

```
.
├── test.py                  # 전체 실험 자동 실행 (exp1~14)
├── gen_diff_cifar10.py      # difference-inducing input 생성 메인 스크립트
├── train_models.py          # 세 모델 학습 스크립트
├── models_torch.py          # CIFARResNet50 모델 정의
├── utils_torch.py           # neuron coverage, gradient 제약 등 유틸리티
├── configs.py               # 출력 색상 설정 (bcolors)
├── requirements.txt         # 패키지 의존성 목록
├── .gitignore
├── checkpoints/
│   ├── model1.pth           # scratch 학습 모델
│   ├── model2.pth           # pretrained full fine-tune 모델
│   └── model3.pth           # pretrained partial fine-tune 모델
├── data/                    # CIFAR-10 데이터셋 (자동 다운로드)
└── results/
    ├── exp1_light_baseline/
    ├── exp2_occl/
    ├── exp3_blackout/
    ├── exp4_high_weight_diff/
    ├── exp5_high_weight_nc/
    ├── exp6_target_model1/
    ├── exp7_target_model2/
    ├── exp8_small_step/
    ├── exp9_large_step/
    ├── exp10_occl_high_both/
    ├── exp11_low_threshold/
    ├── exp12_high_threshold/
    ├── exp13_seeds200/
    └── exp14_seeds300/
```

각 실험 폴더에는 disagreement를 유발한 이미지(`.png`)와 결과 요약(`result.json`)이 저장됩니다.

---

## 실험 모델

| 모델 | 초기화 방식 | 학습 방식 | 학습 레이어 | Data Augmentation | Test Accuracy |
|---|---|---|---|---|---|
| Model 1 | Scratch | Full Training | All Layers | RandomCrop + HorizontalFlip + ColorJitter | 87.35% |
| Model 2 | ImageNet Pretrained | Full Fine-tuning | All Layers | HorizontalFlip + ColorJitter + Rotation | 96.99% |
| Model 3 | ImageNet Pretrained | Partial Fine-tuning | layer3 + layer4 + fc | RandomCrop + HorizontalFlip | 90.34% |

---

## 실험 파라미터 설명

| 항목 | 설명 |
|---|---|
| Transformation | 입력 이미지에 적용한 perturbation 방식 (light, occl, blackout) |
| w_diff | 모델 간 prediction disagreement를 최대화하는 loss의 가중치 |
| w_nc | neuron coverage를 증가시키는 coverage loss의 가중치 |
| step | gradient ascent 시 한 번에 입력을 업데이트하는 크기 |
| seeds | 초기 seed input 개수 |
| grad_iter | 각 seed에 대해 수행한 gradient optimization 반복 횟수 |
| threshold | neuron이 activated로 판단되는 activation 기준값 |
| target | differential testing의 기준이 되는 target model 번호 |
| found | disagreement를 유발한 input의 개수 |
| avg_NC | 세 모델의 평균 neuron coverage 값 |

---

## 실험 결과

| Exp | Trans. | w_diff | w_nc | Step | Seeds | Grad Iter | Threshold | Target | Found | NC M1 | NC M2 | NC M3 | Avg NC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp1 | light | 1.0 | 0.1 | 0.05 | 100 | 100 | 0.5 | 0 | 19 | 0.9753 | 0.8493 | 0.8978 | 0.9075 |
| exp2 | occl | 1.0 | 0.1 | 0.05 | 100 | 100 | 0.5 | 0 | 27 | 0.9736 | 0.9046 | 0.9330 | 0.9371 |
| exp3 | blackout | 1.0 | 0.1 | 0.05 | 100 | 100 | 0.5 | 0 | 100 | 0.9978 | 0.9777 | 0.9894 | 0.9883 |
| exp4 | light | 2.0 | 0.1 | 0.05 | 100 | 100 | 0.5 | 0 | 17 | 0.9719 | 0.8422 | 0.8671 | 0.8938 |
| exp5 | light | 1.0 | 0.5 | 0.05 | 100 | 100 | 0.5 | 0 | 14 | 0.9730 | 0.8039 | 0.8362 | 0.8710 |
| exp6 | light | 1.0 | 0.1 | 0.05 | 100 | 100 | 0.5 | 1 | 17 | 0.9760 | 0.8661 | 0.8931 | 0.9117 |
| exp7 | light | 1.0 | 0.1 | 0.05 | 100 | 100 | 0.5 | 2 | 17 | 0.9644 | 0.8317 | 0.8622 | 0.8861 |
| exp8 | light | 1.0 | 0.1 | 0.01 | 100 | 100 | 0.5 | 0 | 18 | 0.9736 | 0.8683 | 0.9244 | 0.9221 |
| exp9 | light | 1.0 | 0.1 | 0.10 | 100 | 100 | 0.5 | 0 | 13 | 0.9402 | 0.7811 | 0.8525 | 0.8579 |
| exp10 | occl | 2.0 | 0.5 | 0.05 | 100 | 100 | 0.5 | 0 | 43 | 0.9931 | 0.9545 | 0.9699 | 0.9725 |
| exp11 | light | 1.0 | 0.1 | 0.05 | 100 | 100 | 0.3 | 0 | 13 | 1.0000 | 0.9044 | 0.9516 | 0.9520 |
| exp12 | light | 1.0 | 0.1 | 0.05 | 100 | 100 | 0.7 | 0 | 15 | 0.8399 | 0.6663 | 0.7387 | 0.7483 |
| exp13 | light | 1.0 | 0.1 | 0.05 | 200 | 100 | 0.5 | 0 | 33 | 0.9973 | 0.9338 | 0.9385 | 0.9565 |
| exp14 | light | 1.0 | 0.1 | 0.05 | 300 | 100 | 0.5 | 0 | 55 | 0.9960 | 0.9632 | 0.9766 | 0.9786 |

---

## 참조

본 프로젝트는 아래 원본 구현을 참조하여 PyTorch 및 CIFAR-10 환경에 맞게 재구현하였습니다.

- **Paper**: Pei et al., *DeepXplore: Automated Whitebox Testing of Deep Learning Systems*, SOSP 2017
- **Original Code**: [https://github.com/peikexin9/deepxplore](https://github.com/peikexin9/deepxplore)