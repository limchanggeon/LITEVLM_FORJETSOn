# LiteVLM 개발 가이드

## 프로젝트 구조 설명

### 핵심 모듈 (`litevlm/`)

#### 1. `model.py` - 메인 LiteVLM 클래스
전체 VLM 파이프라인을 통합하는 메인 클래스입니다.

주요 메서드:
- `chat()`: 이미지와 프롬프트를 받아 응답 생성
- `batch_chat()`: 여러 이미지 배치 처리
- `benchmark()`: 성능 측정

#### 2. `vision_encoder.py` - Vision Encoder
InternViT 기반 이미지 인코더로, Patch Selection 기능 포함.

특징:
- 중요 패치만 선택하여 연산량 3배 절감
- TensorRT 엔진 지원
- PyTorch 폴백 지원

#### 3. `text_decoder.py` - Text Decoder
Qwen2 기반 텍스트 생성 모델.

특징:
- Autoregressive 디코딩
- KV-cache 최적화
- TensorRT 엔진 지원

#### 4. `token_compression.py` - Token Compression
Visual tokens 압축으로 LLM 입력 길이 감소.

압축 방법:
- Adaptive pooling
- Attention-based selection
- Learned compression

#### 5. `speculative_decode.py` - Speculative Decoding
병렬 토큰 생성으로 디코딩 속도 2-3배 향상.

작동 원리:
1. Draft model로 후보 토큰 생성
2. Target model로 검증
3. 승인된 토큰만 accept

## 커스터마이징

### 새로운 Vision Encoder 추가

```python
# litevlm/vision_encoder.py 수정
class CustomVisionEncoder:
    def __init__(self, model_path, **kwargs):
        # 커스텀 모델 로드
        pass
    
    def __call__(self, image):
        # 이미지 인코딩 로직
        return visual_tokens
```

### 새로운 압축 방법 추가

```python
# litevlm/token_compression.py 수정
def compress_custom(self, tokens):
    # 커스텀 압축 로직
    compressed = your_compression_function(tokens)
    return compressed
```

### 설정 변경

`config.py`에서 설정 수정:

```python
OPTIMIZATION = {
    "compression_ratio": 0.3,  # 압축률 조정
    "num_speculative_tokens": 6,  # Speculative tokens 수 조정
}
```

## 성능 최적화 팁

### 1. 메모리 최적화
- FP8 양자화 사용
- Token compression 활성화
- Batch size 조정

### 2. 속도 최적화
- Speculative decoding 활성화
- TensorRT 엔진 사용
- Jetson clocks 활성화

### 3. 정확도 최적화
- FP16 대신 FP8 사용 시 약간의 정확도 손실
- Temperature 조정
- Compression ratio 조정

## 테스트

```bash
# 유닛 테스트 (TODO: 추가 예정)
pytest tests/

# 통합 테스트
python examples/basic_inference.py
```

## 기여 가이드

1. Fork 생성
2. Feature branch 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. Branch에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 알려진 이슈

- [ ] TensorRT 엔진 초기화 시간 최적화 필요
- [ ] Batch inference 구현 개선
- [ ] INT8 양자화 지원 추가
- [ ] 더 많은 Vision Encoder 지원

## 로드맵

### v0.2.0
- [ ] 추가 Vision Encoder 지원 (CLIP, DINOv2)
- [ ] Batch inference 최적화
- [ ] Web UI 추가

### v0.3.0
- [ ] 스트리밍 응답 지원
- [ ] 멀티 GPU 지원
- [ ] 더 작은 모델 (0.5B) 지원

## 참고 자료

- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [Jetson Documentation](https://docs.nvidia.com/jetson/)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [Qwen2 GitHub](https://github.com/QwenLM/Qwen2)
