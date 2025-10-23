# LiteVLM on Jetson - 프로젝트 요약

## 📦 생성된 프로젝트 구조

```
liteVLM_injetson/
├── README.md                    # 프로젝트 메인 문서
├── QUICKSTART.md                # 빠른 시작 가이드
├── DEVELOPMENT.md               # 개발자 가이드
├── LICENSE                      # MIT 라이선스
├── requirements.txt             # Python 의존성
├── environment.yml              # Conda 환경 설정
├── config.py                    # 설정 파일
├── .gitignore                   # Git 제외 파일
│
├── litevlm/                     # 메인 패키지
│   ├── __init__.py             # 패키지 초기화
│   ├── model.py                # LiteVLM 메인 클래스
│   ├── vision_encoder.py       # Vision Encoder (InternViT + Patch Selection)
│   ├── text_decoder.py         # Text Decoder (Qwen2)
│   ├── token_compression.py    # Visual Token Compression
│   └── speculative_decode.py   # Speculative Decoding
│
├── scripts/                     # 유틸리티 스크립트
│   ├── download_models.py      # 모델 다운로드
│   ├── convert_to_tensorrt.py  # TensorRT 변환
│   └── setup_jetson.sh         # Jetson 초기 설정
│
├── examples/                    # 사용 예제
│   ├── README.md
│   ├── basic_inference.py      # 기본 추론 예제
│   └── batch_processing.py     # 배치 처리 예제
│
├── models/                      # 모델 저장 디렉토리
│   └── README.md               # 모델 관리 가이드
│
├── inference.py                 # CLI 추론 스크립트
└── benchmark.py                 # 성능 벤치마크
```

## 🎯 주요 기능

### 1. **3단계 최적화 파이프라인**
- ✅ **Patch Selection**: 중요 패치만 선택하여 연산량 3배 절감
- ✅ **Token Compression**: Visual tokens 압축으로 LLM 입력 길이 감소
- ✅ **Speculative Decoding**: 병렬 토큰 생성으로 2-3배 디코딩 속도 향상

### 2. **TensorRT 최적화**
- ✅ FP8/FP16/INT8 양자화 지원
- ✅ Vision Encoder 및 Text Decoder TensorRT 변환
- ✅ Jetson Orin 최적화

### 3. **사용 편의성**
- ✅ 간단한 Python API
- ✅ CLI 인터페이스
- ✅ 배치 처리 지원
- ✅ 성능 벤치마크 도구

## 🚀 Jetson에서 사용 방법

### 1단계: 클론 및 설정
```bash
# GitHub에 푸시 후
git clone https://github.com/yourusername/liteVLM_injetson.git
cd liteVLM_injetson

# Jetson 환경 설정
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

### 2단계: Python 환경
```bash
# Conda 환경
conda env create -f environment.yml
conda activate litevlm

# 의존성 설치
pip install -r requirements.txt
```

### 3단계: 모델 준비
```bash
# 모델 다운로드
python scripts/download_models.py

# TensorRT 변환 (FP8)
python scripts/convert_to_tensorrt.py --fp8
```

### 4단계: 추론 실행
```bash
# 단일 이미지 추론
python inference.py \
    --image path/to/image.jpg \
    --prompt "이 사진을 설명해줘" \
    --stats

# 예제 실행
python examples/basic_inference.py

# 벤치마크
python benchmark.py --num-runs 100
```

## 📊 예상 성능 (Jetson Orin)

| 모델 | 정밀도 | 지연시간 | 메모리 | FPS |
|------|-------|---------|-------|-----|
| InternVL2-1B + Qwen2-1.5B | FP8 | ~45ms | 3.2GB | ~22 |
| InternVL2-1B + Qwen2-1.5B | FP16 | ~68ms | 4.8GB | ~15 |

*Jetson Orin AGX, 336×336 이미지, 50 토큰 생성 기준

## 🔧 주요 구성 요소

### LiteVLM 클래스 (model.py)
```python
from litevlm import LiteVLM

vlm = LiteVLM(
    vision_encoder="models/vision_encoder_fp8.engine",
    text_decoder="models/text_decoder_fp8.engine",
    token_compression=True,
    speculative_decode=True
)

result = vlm.chat(
    image="image.jpg",
    prompt="Describe this image"
)
```

### Vision Encoder (vision_encoder.py)
- InternViT 기반
- Patch Selection으로 중요 패치만 선택
- TensorRT 엔진 지원

### Text Decoder (text_decoder.py)
- Qwen2 기반
- Autoregressive 생성
- KV-cache 최적화

### Token Compression (token_compression.py)
- Adaptive pooling
- Attention-based selection
- 압축 비율 조정 가능

### Speculative Decoding (speculative_decode.py)
- Draft model로 후보 생성
- Target model로 검증
- 2-3배 속도 향상

## 📝 다음 단계

### macOS에서:
1. ✅ 프로젝트 완성 확인
2. GitHub 저장소 생성
3. 코드 푸시:
```bash
cd /Users/limchang-geon/Desktop/liteVLM_injetson
git init
git add .
git commit -m "Initial commit: LiteVLM for Jetson Orin"
git branch -M main
git remote add origin https://github.com/yourusername/liteVLM_injetson.git
git push -u origin main
```

### Jetson에서:
1. 저장소 클론
2. 환경 설정 (`./scripts/setup_jetson.sh`)
3. 모델 다운로드 및 변환
4. 추론 실험 시작!

## 🎓 학습 리소스

- **TensorRT 최적화**: `scripts/convert_to_tensorrt.py`
- **파이프라인 구조**: `litevlm/model.py`
- **Patch Selection**: `litevlm/vision_encoder.py`
- **Token Compression**: `litevlm/token_compression.py`
- **Speculative Decoding**: `litevlm/speculative_decode.py`

## 🐛 알려진 제한사항

1. **TensorRT 엔진**: Jetson Orin에서 빌드한 엔진은 다른 GPU에서 사용 불가
2. **메모리**: Jetson Orin Nano (8GB)에서는 swap 메모리 필요할 수 있음
3. **모델 크기**: 현재 1B 모델 기준, 더 작은 모델 지원 예정

## 💡 최적화 팁

1. **전력 모드**: `sudo nvpmodel -m 0` (MAX 성능)
2. **클럭 고정**: `sudo jetson_clocks`
3. **Swap 확장**: 메모리 부족 시 8GB swap 추가
4. **압축 비율**: `config.py`에서 조정 가능

## 📞 문제 해결

모든 문서는 다음 위치에서 확인 가능:
- 빠른 시작: `QUICKSTART.md`
- 개발 가이드: `DEVELOPMENT.md`
- 메인 README: `README.md`
- 모델 관리: `models/README.md`

---

**프로젝트 준비 완료!** 🎉

이제 GitHub에 푸시하고 Jetson에서 클론하여 실험을 시작하세요!
