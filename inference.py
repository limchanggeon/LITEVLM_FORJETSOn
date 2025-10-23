#!/usr/bin/env python3
"""
LiteVLM 추론 스크립트
커맨드라인에서 이미지와 프롬프트를 입력받아 VLM 추론 실행
"""

import argparse
import time
from pathlib import Path

from litevlm import LiteVLM


def main():
    parser = argparse.ArgumentParser(
        description="LiteVLM Inference - Vision-Language Model on Jetson"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for the model"
    )
    parser.add_argument(
        "--vision-encoder",
        type=str,
        default="models/vision_encoder_fp8.engine",
        help="Path to vision encoder model"
    )
    parser.add_argument(
        "--text-decoder",
        type=str,
        default="models/text_decoder_fp8.engine",
        help="Path to text decoder model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable token compression"
    )
    parser.add_argument(
        "--no-speculative",
        action="store_true",
        help="Disable speculative decoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # 이미지 파일 확인
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    # 모델 초기화
    print("Loading LiteVLM...")
    start_time = time.time()
    
    try:
        vlm = LiteVLM(
            vision_encoder=args.vision_encoder,
            text_decoder=args.text_decoder,
            token_compression=not args.no_compression,
            speculative_decode=not args.no_speculative,
            device=args.device,
            verbose=args.verbose
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s\n")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure models are downloaded:")
        print("   python scripts/download_models.py")
        print("2. Convert to TensorRT:")
        print("   python scripts/convert_to_tensorrt.py --fp8")
        return
    
    # 추론 실행
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    
    try:
        result = vlm.chat(
            image=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            return_stats=args.stats
        )
        
        if args.stats:
            text = result['text']
            stats = result['stats']
            
            print("\nGenerated Text:")
            print("-" * 60)
            print(text)
            print("-" * 60)
            
            print("\nPerformance Statistics:")
            print(f"  Preprocessing:  {stats['preprocess_time']*1000:>6.1f} ms")
            print(f"  Encoding:       {stats['encoding_time']*1000:>6.1f} ms")
            print(f"  Generation:     {stats['generation_time']*1000:>6.1f} ms")
            print(f"  Total:          {stats['total_time']*1000:>6.1f} ms")
            print(f"  Visual Tokens:  {stats['num_visual_tokens']:>6}")
            
        else:
            print("\nGenerated Text:")
            print("-" * 60)
            print(result)
            print("-" * 60)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
