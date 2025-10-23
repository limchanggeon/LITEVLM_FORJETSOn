"""
배치 처리 예제
여러 이미지를 한 번에 처리하기
"""

from litevlm import LiteVLM
from PIL import Image
import numpy as np
from pathlib import Path
import time


def create_test_images(num_images: int = 10, output_dir: str = "test_images"):
    """테스트 이미지 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        # 랜덤 이미지 생성
        img = Image.fromarray(
            np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
        )
        img_path = output_path / f"test_{i:03d}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
    
    return image_paths


def main():
    # 1. 모델 로드
    print("Loading LiteVLM...")
    
    vlm = LiteVLM(
        vision_encoder="models/vision_encoder_fp8.engine",
        text_decoder="models/text_decoder_fp8.engine",
        token_compression=True,
        speculative_decode=True,
        device="cuda",
        verbose=False
    )
    
    print("Model loaded!\n")
    
    # 2. 테스트 이미지 생성
    num_images = 20
    print(f"Creating {num_images} test images...")
    image_paths = create_test_images(num_images)
    
    # 3. 배치 처리
    prompts = ["Describe this image."] * num_images
    
    print(f"\n{'='*60}")
    print(f"Processing {num_images} images...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # 순차 처리
    results = vlm.batch_chat(
        images=image_paths,
        prompts=prompts,
        max_new_tokens=50,
        return_stats=True
    )
    
    total_time = time.time() - start_time
    
    # 4. 결과 출력
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/num_images*1000:.1f}ms")
    print(f"Throughput: {num_images/total_time:.2f} images/sec")
    
    # 개별 결과 샘플
    print(f"\n{'='*60}")
    print("Sample Results (first 3 images)")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results[:3]):
        print(f"Image {i+1}: {image_paths[i]}")
        if isinstance(result, dict):
            print(f"Response: {result['text'][:100]}...")
            print(f"Latency: {result['stats']['total_time']*1000:.1f}ms\n")
        else:
            print(f"Response: {result[:100]}...\n")
    
    # 5. 통계
    if isinstance(results[0], dict):
        latencies = [r['stats']['total_time'] * 1000 for r in results]
        
        print(f"{'='*60}")
        print("Latency Statistics")
        print(f"{'='*60}")
        print(f"Min:  {min(latencies):.1f}ms")
        print(f"Max:  {max(latencies):.1f}ms")
        print(f"Avg:  {sum(latencies)/len(latencies):.1f}ms")
        print(f"Med:  {sorted(latencies)[len(latencies)//2]:.1f}ms")
    
    print("\nBatch processing completed!")


if __name__ == "__main__":
    main()
