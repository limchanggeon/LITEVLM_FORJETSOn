"""
기본 추론 예제
LiteVLM을 사용하여 이미지 분석하기
"""

from litevlm import LiteVLM
from PIL import Image
import numpy as np


def main():
    # 1. 모델 초기화
    print("Loading LiteVLM...")
    
    vlm = LiteVLM(
        vision_encoder="models/vision_encoder_fp8.engine",
        text_decoder="models/text_decoder_fp8.engine",
        token_compression=True,
        speculative_decode=True,
        device="cuda",
        verbose=True
    )
    
    print("Model loaded!\n")
    
    # 2. 테스트 이미지 생성 (실제 사용 시에는 실제 이미지 사용)
    print("Creating test image...")
    test_image = Image.fromarray(
        np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
    )
    test_image.save("test_image.jpg")
    
    # 3. 다양한 프롬프트로 추론
    prompts = [
        "이 사진을 설명해줘",
        "What objects do you see in this image?",
        "Describe the scene in detail.",
        "이미지에서 가장 두드러지는 특징은 무엇인가요?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}/{len(prompts)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        # 추론 실행
        result = vlm.chat(
            image="test_image.jpg",
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
            return_stats=True
        )
        
        # 결과 출력
        print(f"Response:\n{result['text']}")
        print(f"\nLatency: {result['stats']['total_time']*1000:.1f}ms")
    
    # 4. 벤치마크 (선택사항)
    print(f"\n{'='*60}")
    print("Running benchmark...")
    print(f"{'='*60}")
    
    benchmark_results = vlm.benchmark(num_runs=50)
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
