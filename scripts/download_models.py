#!/usr/bin/env python3
"""
모델 다운로드 스크립트
InternVL2.5 및 Qwen2 모델을 Hugging Face에서 다운로드
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download_model(
    model_name: str,
    output_dir: Path,
    token: str = None
):
    """
    Hugging Face Hub에서 모델 다운로드
    
    Args:
        model_name: 모델 이름 (예: "OpenGVLab/InternVL2.5-1B")
        output_dir: 저장 디렉토리
        token: HF 토큰 (private 모델인 경우)
    """
    print(f"\n[Download] {model_name}")
    print(f"[Output] {output_dir}")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            token=token,
            resume_download=True,
            max_workers=4,
        )
        print(f"[✓] Downloaded {model_name}")
        
    except Exception as e:
        print(f"[✗] Error downloading {model_name}: {e}")
        print(f"[!] You may need to login: huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download VLM models")
    parser.add_argument(
        "--vision-model",
        type=str,
        default="OpenGVLab/InternVL2-1B",
        help="Vision model name from Hugging Face"
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Text model name from Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for models"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for private models"
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip downloading vision model"
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip downloading text model"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Vision model 다운로드
    if not args.skip_vision:
        vision_dir = output_dir / "vision_encoder"
        vision_dir.mkdir(exist_ok=True)
        download_model(args.vision_model, vision_dir, args.token)
    
    # Text model 다운로드
    if not args.skip_text:
        text_dir = output_dir / "text_decoder"
        text_dir.mkdir(exist_ok=True)
        download_model(args.text_model, text_dir, args.token)
    
    print("\n[✓] All models downloaded successfully!")
    print(f"\nNext steps:")
    print(f"1. Convert models to TensorRT:")
    print(f"   python scripts/convert_to_tensorrt.py")
    print(f"2. Run inference:")
    print(f"   python inference.py --image example.jpg --prompt 'Describe this image'")


if __name__ == "__main__":
    main()
