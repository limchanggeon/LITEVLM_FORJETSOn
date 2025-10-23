# Sample Images

이 디렉토리에 테스트용 샘플 이미지를 추가하세요.

## 사용 예제

```bash
# 샘플 이미지로 추론
python inference.py \
    --image examples/sample.jpg \
    --prompt "Describe this image"
```

## 권장 이미지 형식

- **포맷**: JPG, PNG
- **크기**: 336x336 또는 448x448 (자동 리사이즈됨)
- **용량**: <10MB

## 테스트 이미지 다운로드

```bash
# COCO 데이터셋에서 샘플 이미지
wget http://images.cocodataset.org/val2017/000000039769.jpg -O examples/sample.jpg

# 또는 자신의 이미지 복사
cp ~/Pictures/test.jpg examples/sample.jpg
```

## 배치 처리

여러 이미지를 한 번에 처리하려면:

```bash
# 이 디렉토리에 여러 이미지 추가
cp ~/Pictures/*.jpg examples/

# 배치 처리 실행
python examples/batch_processing.py
```
