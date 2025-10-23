# LiteVLM on Jetson - Quick Start Guide

## 🚀 빠른 시작 (Jetson에서)

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/liteVLM_injetson.git
cd liteVLM_injetson
```

### 2. 초기 설정

```bash
# Jetson 환경 설정 (처음 한 번만)
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

### 3. Python 환경 설정

```bash
# Conda 환경 생성
conda env create -f environment.yml
conda activate litevlm

# 또는 venv 사용
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. 모델 다운로드 및 변환

```bash
# HuggingFace에서 모델 다운로드
python scripts/download_models.py

# TensorRT로 변환 (FP8 양자화)
python scripts/convert_to_tensorrt.py --fp8
```

이 과정은 시간이 걸릴 수 있습니다 (약 30-60분).

### 5. 추론 실행

**방법 1: Web UI (추천! 🎨)**
```bash
# Gradio 웹 인터페이스 실행
python webui.py

# 브라우저에서 http://localhost:7860 접속
```

**방법 2: 커맨드라인**
```bash
# 기본 추론
python inference.py \
    --image path/to/image.jpg \
    --prompt "이 사진을 설명해줘" \
    --stats

# 예제 실행
python examples/basic_inference.py
```

### 6. 벤치마크

```bash
# 성능 측정
python benchmark.py --num-runs 100
```

## 📝 추가 설정

### Git 설정

```bash
# 사용자 정보 설정
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 원격 저장소 추가 (본인의 GitHub 저장소)
git remote set-url origin https://github.com/yourusername/liteVLM_injetson.git
```

### 전력 모드 최적화

```bash
# 최대 성능 모드
sudo nvpmodel -m 0
sudo jetson_clocks

# 전력 모드 확인
sudo nvpmodel -q
```

## 🔍 테스트

```bash
# Python import 테스트
python -c "from litevlm import LiteVLM; print('LiteVLM imported successfully!')"

# CUDA 테스트
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 예상 성능

| 플랫폼 | 지연시간 | 메모리 | 전력 |
|--------|---------|-------|------|
| Jetson Orin AGX | ~45ms | 3.2GB | ~25W |
| Jetson Orin NX | ~68ms | 3.2GB | ~20W |
| Jetson Orin Nano | ~95ms | 3.0GB | ~15W |

## ❓ 문제 해결

문제가 발생하면 README.md의 "문제 해결" 섹션을 참고하세요.

## 📚 다음 단계

- 자신의 이미지로 테스트
- 배치 처리 실험 (`examples/batch_processing.py`)
- 파라미터 튜닝 (temperature, top_p 등)
- 성능 최적화 실험
