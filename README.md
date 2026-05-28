# corrosion-cvae

전자제품 내 구리 소자의 부식 이미지를 생성하는 Conditional VAE 모델 구현 및 손실 함수 조합 비교 실험.

---

## 배경

전자제품 소형화로 구리 사용량이 늘어나면서 구리 부식으로 인한 시스템 불량 문제가 대두되고 있다.  
비파괴 검사(NDT)로 측정한 s-파라미터를 활용해 부식 정도를 이미지로 시각화할 수 있다면,  
실험에 오랜 시간이 걸리는 실제 부식 이미지 수집을 대체할 수 있다.

이 연구에서는 **부식률을 조건으로** 구리 부식 이미지를 생성하는 cVAE 모델을 설계하고,  
어떤 손실 함수 조합이 가장 적합한지를 실험으로 탐색했다.

---

## 데이터

- 구리 시편 56개를 MIL-STD-810G 규격 가속 온습도 시험으로 부식시킴 (5주기 × 24시간)
- 각 주기마다 이미지 및 s-파라미터 측정 → 총 **280장** 수집
- **마킹 이미지**: RGB 이미지에서 부식을 가장 잘 반영하는 R 채널 기반으로 생성, (256×256) 리사이즈
- **부식률**: 마킹된 R 채널 픽셀 평균값 (0~100%), 이미지 레이블 및 조건값으로 사용
- **s-파라미터**: RF프로브 + VNA로 측정 (S11, S21, Phase11, Phase21 각 201차원, 300kHz~14GHz)

---

## 모델 구조

```
Encoder: Marking Image + S-parameters (s11, s21, p11, p21) → μ, σ
Reparameterize: z ~ N(μ, σ²)
Decoder: z + 부식률(condition) → 생성 이미지
```

- Conv 기반 인코더 / ConvTranspose 기반 디코더 (num_layers로 깊이 조절)
- 활성화 함수: ReLU (인코더), Tanh (디코더 출력층)
- Optimizer: Adam (lr=1e-4), Batch size: 32, Latent dim: 50, Epochs: 2,000

---

## 손실 함수 실험

부식 이미지 생성에 적합한 손실 함수 조합을 탐색하기 위해 6가지 조합을 비교했다.  
평가 지표: **MSE** (조건 부식률과 생성 이미지 부식률 차이), **LPIPS** (시각적 유사도)

| 조합 | 손실 함수 | MSE | LPIPS |
|------|----------|-----|-------|
| 1 | L1 + KLD | 69.81 | 0.54 |
| 2 | L2 + KLD | 33.29 | 0.57 |
| 3 | L1 + L2 + KLD | 96.98 | 0.55 |
| 4 | L1 + VGG + KLD | 1340.96 | 0.57 |
| 5 | **L2 + VGG + KLD** | **24.88** | **0.54** |
| 6 | L1 + L2 + VGG + KLD | 144.12 | 0.55 |

**L2+VGG+KLD** 조합이 MSE, LPIPS 모두 가장 낮아 최적 조합으로 선정.

- L2는 부식률 정보를 생성 이미지에 담는 데 효과적
- VGG perceptual loss는 부식 패턴 학습에 긍정적 영향
- L1+VGG 조합은 오히려 성능을 저하시킴

---

## 실행 방법

```bash
pip install -r requirement.txt

# 논문 실험 설정으로 학습
python scripts/train.py --config configs/paper.yaml

# 데이터 증강 없이 학습
python scripts/train.py --config configs/no_aug.yaml

# 데이터 증강 적용
python scripts/train.py --config configs/with_aug.yaml
```

결과(체크포인트, 손실 로그)는 config의 `logging.output_dir`에 저장된다.

---

## 디렉토리 구조

```
.
├── configs/
│   ├── paper.yaml        # 논문 실험 설정
│   ├── no_aug.yaml       # 증강 없음
│   └── with_aug.yaml     # 증강 적용
├── scripts/
│   └── train.py          # 학습 스크립트
├── src/
│   ├── model.py          # CVAE 모델 정의
│   ├── dataset.py        # 데이터셋 및 DataLoader
│   ├── loss.py           # 손실 함수 (L1/L2/VGG/KLD)
│   ├── configs.py        # YAML config 파싱
│   └── utils.py          # 시드 고정, 체크포인트 저장 등
└── requirement.txt
```

---

## 기술 스택

`Python` `PyTorch` `torchvision` `Pillow` `pandas` `numpy`

---

## 참고 논문

김소정, 길하균, 강태엽, 홍승혁. "전자제품 사용자의 안전을 위한 조건부 VAE 기반 구리 부식 이미지 생성 및 손실 함수 최적화." 한국정보과학회 한국소프트웨어종합학술대회(KSC2024), 2024.
