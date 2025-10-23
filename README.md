# LiteVLM on Jetson

NVIDIA Jetson Orin에서 실행 가능한 경량 Vision-Language Model (VLM) 추론 시스템입니다.

## 🚀 주요 특징

- **저지연 추론**: FP8 양자화 및 TensorRT 최적화로 최대 3.2배 속도 향상
- **메모리 효율**: 50% 메모리 절감, 4GB VRAM에서도 1B 모델 실행 가능
- **3단계 최적화 파이프라인**:
  - Patch Selection Module: 중요 패치만 선택하여 인코더 연산량 3배 절감
  - Visual Token Compression: 특징 토큰 압축으로 LLM 입력 길이 감소
  - Speculative Decoding: 병렬 토큰 생성으로 2~3배 디코딩 속도 향상

## 📋 요구사항

### 하드웨어
- NVIDIA Jetson Orin (AGX Orin, Orin NX, Orin Nano)
- 최소 8GB RAM 권장
- 최소 32GB 저장공간

### 소프트웨어
- JetPack 5.1+ (CUDA 11.4+, TensorRT 8.5+)
- Python 3.8+
- PyTorch 2.0+

## 🛠️ 설치 방법

### 1. 저장소 클론 (Jetson에서 실행)

```bash
# Git Clone (권장)
git clone https://github.com/limchanggeon/LITEVLM_FORJETSOn.git
cd LITEVLM_FORJETSOn

# 또는 ZIP 다운로드
wget https://github.com/limchanggeon/LITEVLM_FORJETSOn/archive/refs/heads/main.zip
unzip main.zip
cd LITEVLM_FORJETSOn-main
```

💡 **자세한 설치 가이드**: [INSTALL_GUIDE.md](INSTALL_GUIDE.md) 참고

### 2. Conda 환경 설정

```bash
# Conda 환경 생성
conda create -n litevlm python=3.10
conda activate litevlm

# 또는 venv 사용
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
# PyTorch 및 기본 라이브러리 설치 (Jetson용)
pip install -r requirements.txt

# Jetson에서 PyTorch는 NVIDIA에서 제공하는 wheel 사용 권장
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

### 4. 모델 다운로드 및 변환

```bash
# 모델 다운로드
python scripts/download_models.py

# TensorRT로 변환 (FP8 양자화 포함)
python scripts/convert_to_tensorrt.py --fp8
```

## 💻 사용 방법

### 1. Web UI (가장 간단! 🎨)

```bash
# Gradio 기반 웹 인터페이스 실행
python webui.py

# 특정 포트 지정
python webui.py --port 8080

# 외부 접속 허용
python webui.py --host 0.0.0.0 --port 7860

# Public 공유 링크 생성
python webui.py --share
```

브라우저에서 `http://localhost:7860` 접속!

### 2. 기본 추론 (Python API)

```python
from litevlm import LiteVLM

# 모델 초기화
vlm = LiteVLM(
    vision_encoder="models/internvit_fp8.engine",
    text_decoder="models/qwen_fp8.engine",
    token_compression=True,
    speculative_decode=True
)

# 이미지 분석
result = vlm.chat(
    image="example.jpg", 
    prompt="이 사진을 설명해줘"
)
print(result)
```

### 3. 커맨드라인 인터페이스

```bash
python inference.py --image example.jpg --prompt "What is in this image?"
```

### 4. 벤치마크

```bash
# 추론 속도 및 메모리 사용량 측정
python benchmark.py --model_path models/ --num_runs 100
```

## 📁 프로젝트 구조

```
liteVLM_injetson/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
├── models/                  # 변환된 TensorRT 엔진 파일
├── litevlm/                 # 메인 패키지
│   ├── __init__.py
│   ├── model.py            # LiteVLM 클래스
│   ├── vision_encoder.py   # Vision 인코더 (InternViT)
│   ├── text_decoder.py     # Text 디코더 (Qwen2)
│   ├── token_compression.py # 토큰 압축 모듈
│   └── speculative_decode.py # Speculative decoding
├── scripts/
│   ├── download_models.py   # 모델 다운로드
│   ├── convert_to_tensorrt.py # TensorRT 변환
│   └── setup_jetson.sh      # Jetson 초기 설정
├── examples/
│   ├── basic_inference.py
│   └── batch_processing.py
├── inference.py             # CLI 추론 스크립트
└── benchmark.py             # 성능 벤치마크
```

## 🎯 Jetson Orin 최적화 팁

### 1. 전력 모드 설정
```bash
sudo nvpmodel -m 0  # MAX 성능 모드
sudo jetson_clocks   # 클럭 고정
```

### 2. Swap 메모리 확장 (메모리 부족 시)
```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 8G /mnt/8GB.swap
sudo mkswap /mnt/8GB.swap
sudo swapon /mnt/8GB.swap
```

### 3. TensorRT 최적화 레벨
- FP16: 높은 정확도, 중간 속도
- FP8: 균형잡힌 성능 (권장)
- INT8: 최고 속도, 약간의 정확도 손실

## 📊 성능 벤치마크

| 플랫폼 | 모델 크기 | 지연시간 | 메모리 사용량 |
|--------|----------|---------|-------------|
| Jetson Orin AGX | 1B (FP8) | ~45ms | 3.2GB |
| Jetson Orin NX | 1B (FP8) | ~68ms | 3.2GB |
| RTX 4070 | 1B (FP8) | ~28ms | 3.0GB |

*이미지 크기: 336×336, 텍스트 길이: 평균 50 토큰 기준

## 🔧 문제 해결

### CUDA Out of Memory
- 작은 모델 사용 (0.5B)
- 배치 크기 축소
- Swap 메모리 활성화

### TensorRT 변환 실패
- JetPack 버전 확인
- ONNX 모델 먼저 검증
- 로그 확인: `--verbose` 플래그 사용

## 📚 참고 자료

- [LiteVLM Paper](https://arxiv.org/abs/2501.xxxxx)
- [InternVL2.5 Documentation](https://github.com/OpenGVLab/InternVL)
- [Qwen2 Model Card](https://huggingface.co/Qwen)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## 📝 라이선스

MIT License

## 🤝 기여

이슈 및 풀 리퀘스트 환영합니다!

## 📧 문의

프로젝트에 대한 문의사항은 이슈를 통해 남겨주세요.
