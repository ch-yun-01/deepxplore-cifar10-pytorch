# DeepXplore CIFAR-10 (PyTorch)

> 신뢰할 수 있는 인공지능 Assignment 2

---

## 프로젝트 개요

본 프로젝트는 [DeepXplore](https://arxiv.org/abs/1705.06640)를 CIFAR-10 데이터셋과 PyTorch 환경에 맞게 구현한 것입니다.  
서로 다른 방식으로 학습된 세 개의 ResNet-50 모델에 대해 **difference-inducing input**을 생성하고, **neuron coverage**를 극대화하면서 모델 간 예측 불일치를 유발하는 입력을 탐색합니다.

`python test.py` 으로 exp1~exp14 전체 실험이 자동으로 순차 실행됩니다.

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

---

## 실험 결과 파일 구조

각 실험 폴더에는 다음 파일들이 저장됩니다.

| 파일명 | 설명 |
|---|---|
| `{transformation}_{seed}_{iter}_{p1}_{p2}_{p3}.png` | gradient로 수정된 이미지 (perturbed) |
| `{transformation}_{seed}_{iter}_{p1}_{p2}_{p3}_orig.png` | 수정 전 원본 이미지 |
| `already_{seed}_{p1}_{p2}_{p3}.png` | perturbation 없이 이미 disagreement가 발생한 이미지 |
| `result.json` | 실험 파라미터 및 최종 결과 요약 |

### already_ 이미지란?

`already_` 접두사가 붙은 이미지는 gradient 탐색 없이 **원본 seed 이미지만으로도** 세 모델이 서로 다른 예측을 내놓은 경우입니다. perturbation이 적용되지 않은 자연 입력에서 disagreement가 발생한 케이스이며, `result.json`의 `found` 수치에는 이 경우도 포함됩니다.

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
| found | disagreement를 유발한 input의 개수 (already_ 케이스 포함) |
| avg_NC | 세 모델의 평균 neuron coverage 값 |

---

## 실험 구성

| 실험 | 변환 | 주요 변경 파라미터 |
|---|---|---|
| exp1 | light | baseline (seeds=100) |
| exp2 | occl | - |
| exp3 | blackout | - |
| exp4 | light | weight_diff=2.0 |
| exp5 | light | weight_nc=0.5 |
| exp6 | light | target_model=1 |
| exp7 | light | target_model=2 |
| exp8 | light | step=0.01 |
| exp9 | light | step=0.10 |
| exp10 | occl | weight_diff=2.0, weight_nc=0.5 |
| exp11 | light | threshold=0.3 |
| exp12 | light | threshold=0.7 |
| exp13 | light | seeds=200 |
| exp14 | light | seeds=300 |

---

## 참조

본 프로젝트는 아래 원본 구현을 참조하여 PyTorch 및 CIFAR-10 환경에 맞게 재구현하였습니다.

- **Paper**: Pei et al., *DeepXplore: Automated Whitebox Testing of Deep Learning Systems*, SOSP 2017
- **Original Code**: [https://github.com/peikexin9/deepxplore](https://github.com/peikexin9/deepxplore)
