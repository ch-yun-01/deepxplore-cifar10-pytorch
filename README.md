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