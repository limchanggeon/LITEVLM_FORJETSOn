#!/usr/bin/env python3
"""
TensorRT 변환 스크립트
PyTorch 모델을 TensorRT 엔진으로 변환 (FP8/INT8 양자화 포함)
"""

import os
import argparse
from pathlib import Path
import torch


def export_to_onnx(
    model_path: Path,
    onnx_path: Path,
    model_type: str = "vision",
    opset_version: int = 17
):
    """
    PyTorch 모델을 ONNX로 변환
    
    Args:
        model_path: PyTorch 모델 경로
        onnx_path: ONNX 저장 경로
        model_type: 'vision' or 'text'
        opset_version: ONNX opset 버전
    """
    print(f"\n[1/3] Exporting {model_type} model to ONNX...")
    
    try:
        from transformers import AutoModel, AutoModelForCausalLM
        
        # 모델 로드
        if model_type == "vision":
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            # 더미 입력
            dummy_input = torch.randn(1, 3, 336, 336)
            input_names = ["image"]
            output_names = ["visual_tokens"]
            dynamic_axes = {
                "image": {0: "batch_size"},
                "visual_tokens": {0: "batch_size", 1: "sequence_length"}
            }
        else:  # text
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            dummy_input = torch.randint(0, 32000, (1, 100))
            input_names = ["input_ids"]
            output_names = ["logits"]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        
        model.eval()
        
        # ONNX 변환
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        
        print(f"[✓] ONNX model saved to {onnx_path}")
        return True
        
    except Exception as e:
        print(f"[✗] Error exporting to ONNX: {e}")
        return False


def convert_to_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp8",
    max_batch_size: int = 1,
    workspace_size: int = 4
):
    """
    ONNX 모델을 TensorRT 엔진으로 변환
    
    Args:
        onnx_path: ONNX 모델 경로
        engine_path: TensorRT 엔진 저장 경로
        precision: 'fp32', 'fp16', 'fp8', 'int8'
        max_batch_size: 최대 배치 크기
        workspace_size: 작업 메모리 크기 (GB)
    """
    print(f"\n[2/3] Converting ONNX to TensorRT ({precision})...")
    
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # ONNX 파싱
        print(f"[*] Parsing ONNX model...")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"[✗] {parser.get_error(error)}")
                return False
        
        # 빌더 설정
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size * (1 << 30)  # GB to bytes
        )
        
        # Precision 설정
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "fp8":
            config.set_flag(trt.BuilderFlag.FP8)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # INT8 calibration 필요 (여기서는 생략)
        
        # 엔진 빌드
        print(f"[*] Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("[✗] Failed to build TensorRT engine")
            return False
        
        # 엔진 저장
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"[✓] TensorRT engine saved to {engine_path}")
        return True
        
    except ImportError:
        print("[✗] TensorRT not installed!")
        print("[!] On Jetson, TensorRT is included in JetPack")
        print("[!] Try: sudo apt-get install python3-libnvinfer-dev")
        return False
        
    except Exception as e:
        print(f"[✗] Error converting to TensorRT: {e}")
        return False


def verify_engine(engine_path: Path):
    """TensorRT 엔진 검증"""
    print(f"\n[3/3] Verifying TensorRT engine...")
    
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine:
            print(f"[✓] Engine is valid")
            print(f"    Max batch size: {engine.max_batch_size}")
            print(f"    Num bindings: {engine.num_bindings}")
            return True
        else:
            print("[✗] Engine is invalid")
            return False
            
    except Exception as e:
        print(f"[✗] Error verifying engine: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert models to TensorRT")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Directory containing downloaded models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for TensorRT engines"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp8",
        choices=["fp32", "fp16", "fp8", "int8"],
        help="Quantization precision"
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip converting vision model"
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip converting text model"
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Use FP8 precision (shorthand)"
    )
    
    args = parser.parse_args()
    
    if args.fp8:
        args.precision = "fp8"
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Vision model 변환
    if not args.skip_vision:
        print("\n" + "="*60)
        print("Converting Vision Encoder")
        print("="*60)
        
        vision_model = model_dir / "vision_encoder"
        vision_onnx = output_dir / "vision_encoder.onnx"
        vision_engine = output_dir / f"vision_encoder_{args.precision}.engine"
        
        if export_to_onnx(vision_model, vision_onnx, "vision"):
            if convert_to_tensorrt(vision_onnx, vision_engine, args.precision):
                verify_engine(vision_engine)
    
    # Text model 변환
    if not args.skip_text:
        print("\n" + "="*60)
        print("Converting Text Decoder")
        print("="*60)
        
        text_model = model_dir / "text_decoder"
        text_onnx = output_dir / "text_decoder.onnx"
        text_engine = output_dir / f"text_decoder_{args.precision}.engine"
        
        if export_to_onnx(text_model, text_onnx, "text"):
            if convert_to_tensorrt(text_onnx, text_engine, args.precision):
                verify_engine(text_engine)
    
    print("\n" + "="*60)
    print("[✓] Conversion complete!")
    print("="*60)
    print(f"\nOptimized models saved to: {output_dir}")
    print(f"\nNext step:")
    print(f"  python inference.py --image example.jpg --prompt 'Describe this'")


if __name__ == "__main__":
    main()
