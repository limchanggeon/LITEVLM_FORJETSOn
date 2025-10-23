# Models Directory

이 디렉토리는 LiteVLM 모델 파일들을 저장하는 곳입니다.

## 디렉토리 구조

```
models/
├── vision_encoder/          # 원본 Vision Encoder (InternVL2)
│   └── (Hugging Face 모델 파일들)
├── text_decoder/            # 원본 Text Decoder (Qwen2)
│   └── (Hugging Face 모델 파일들)
├── vision_encoder.onnx      # ONNX 변환 (중간 단계)
├── text_decoder.onnx        # ONNX 변환 (중간 단계)
├── vision_encoder_fp8.engine    # TensorRT 엔진 (최종, FP8)
├── text_decoder_fp8.engine      # TensorRT 엔진 (최종, FP8)
├── vision_encoder_fp16.engine   # TensorRT 엔진 (선택, FP16)
└── text_decoder_fp16.engine     # TensorRT 엔진 (선택, FP16)
```

## 모델 다운로드

```bash
# 1. Hugging Face에서 모델 다운로드
python scripts/download_models.py

# 특정 모델만 다운로드
python scripts/download_models.py --skip-text  # Text 모델 제외
python scripts/download_models.py --skip-vision  # Vision 모델 제외
```

## TensorRT 변환

```bash
# 2. TensorRT로 변환 (FP8 양자화)
python scripts/convert_to_tensorrt.py --fp8

# FP16 양자화 (더 높은 정확도)
python scripts/convert_to_tensorrt.py --precision fp16

# 특정 모델만 변환
python scripts/convert_to_tensorrt.py --skip-text
```

## 모델 크기

| 모델 | 원본 크기 | ONNX | TensorRT (FP16) | TensorRT (FP8) |
|------|----------|------|----------------|---------------|
| Vision Encoder (InternVL2-1B) | ~2.0 GB | ~2.0 GB | ~1.0 GB | ~0.5 GB |
| Text Decoder (Qwen2-1.5B) | ~3.0 GB | ~3.0 GB | ~1.5 GB | ~0.75 GB |
| **Total** | **~5.0 GB** | **~5.0 GB** | **~2.5 GB** | **~1.25 GB** |

## 권장 설정

### Jetson Orin AGX (32GB)
- Precision: FP8 또는 FP16
- 메모리 사용량: ~3.2 GB
- 추론 속도: ~45ms

### Jetson Orin NX (16GB)
- Precision: FP8 (권장)
- 메모리 사용량: ~3.2 GB
- 추론 속도: ~68ms

### Jetson Orin Nano (8GB)
- Precision: FP8 (필수)
- Token compression: 활성화
- 메모리 사용량: ~3.0 GB
- 추론 속도: ~95ms

## 문제 해결

### CUDA Out of Memory

```bash
# 1. Swap 메모리 활성화
sudo fallocate -l 8G /mnt/8GB.swap
sudo mkswap /mnt/8GB.swap
sudo swapon /mnt/8GB.swap

# 2. 더 작은 모델 사용
# InternVL2-0.5B + Qwen2-0.5B
```

### TensorRT 변환 실패

```bash
# JetPack 버전 확인
dpkg -l | grep nvidia-jetpack

# TensorRT 설치 확인
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 재설치
sudo apt-get install --reinstall python3-libnvinfer-dev
```

## 사전 변환된 모델 (선택사항)

사전 변환된 TensorRT 엔진을 다운로드하려면:

```bash
# TODO: 사전 변환 모델 배포 시 링크 추가
wget https://example.com/litevlm_jetson_fp8.tar.gz
tar -xzf litevlm_jetson_fp8.tar.gz -C models/
```

## 참고 사항

- `.engine` 파일은 특정 GPU 아키텍처에 맞춰 빌드됩니다
- Jetson Orin에서 생성한 엔진은 다른 GPU에서 작동하지 않을 수 있습니다
- 모델 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다
