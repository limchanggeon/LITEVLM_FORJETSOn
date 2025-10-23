#!/usr/bin/env python3
"""
LiteVLM 벤치마크 스크립트
성능 측정 및 비교
"""

import argparse
import time
from pathlib import Path
import json

import torch
import psutil
import pynvml

from litevlm import LiteVLM


class SystemMonitor:
    """시스템 리소스 모니터링"""
    
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_gpu = True
        except:
            self.has_gpu = False
    
    def get_gpu_memory(self):
        """GPU 메모리 사용량 (MB)"""
        if not self.has_gpu:
            return 0
        
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return info.used / 1024 / 1024  # MB
        except:
            return 0
    
    def get_cpu_memory(self):
        """CPU 메모리 사용량 (MB)"""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def get_gpu_power(self):
        """GPU 전력 소비 (W)"""
        if not self.has_gpu:
            return 0
        
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power / 1000  # mW to W
        except:
            return 0
    
    def __del__(self):
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def benchmark_inference(
    vlm: LiteVLM,
    image_path: str,
    prompt: str,
    num_runs: int = 100,
    warmup: int = 10,
):
    """추론 성능 벤치마크"""
    
    monitor = SystemMonitor()
    
    print(f"\n{'='*60}")
    print(f"Running benchmark: {num_runs} iterations")
    print(f"{'='*60}\n")
    
    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        _ = vlm.chat(image_path, prompt, max_new_tokens=50)
    
    # 초기 메모리
    initial_gpu_mem = monitor.get_gpu_memory()
    initial_cpu_mem = monitor.get_cpu_memory()
    
    # 벤치마크
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    preprocess_times = []
    encoding_times = []
    generation_times = []
    
    for i in range(num_runs):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_runs}")
        
        result = vlm.chat(
            image_path,
            prompt,
            max_new_tokens=50,
            return_stats=True
        )
        
        stats = result['stats']
        times.append(stats['total_time'])
        preprocess_times.append(stats['preprocess_time'])
        encoding_times.append(stats['encoding_time'])
        generation_times.append(stats['generation_time'])
    
    # 최종 메모리
    final_gpu_mem = monitor.get_gpu_memory()
    final_cpu_mem = monitor.get_cpu_memory()
    avg_power = monitor.get_gpu_power()
    
    # 통계 계산
    import numpy as np
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    p50_time = np.percentile(times, 50) * 1000
    p95_time = np.percentile(times, 95) * 1000
    p99_time = np.percentile(times, 99) * 1000
    fps = 1000 / avg_time
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}\n")
    
    print("Latency (ms):")
    print(f"  Average:     {avg_time:>8.2f} ± {std_time:.2f}")
    print(f"  Min:         {min_time:>8.2f}")
    print(f"  Max:         {max_time:>8.2f}")
    print(f"  P50:         {p50_time:>8.2f}")
    print(f"  P95:         {p95_time:>8.2f}")
    print(f"  P99:         {p99_time:>8.2f}")
    print(f"\nThroughput:    {fps:>8.2f} FPS")
    
    print(f"\nBreakdown (average):")
    print(f"  Preprocess:  {np.mean(preprocess_times)*1000:>8.2f} ms")
    print(f"  Encoding:    {np.mean(encoding_times)*1000:>8.2f} ms")
    print(f"  Generation:  {np.mean(generation_times)*1000:>8.2f} ms")
    
    print(f"\nMemory Usage:")
    print(f"  GPU Memory:  {final_gpu_mem:>8.1f} MB")
    print(f"  CPU Memory:  {final_cpu_mem:>8.1f} MB")
    print(f"  GPU Power:   {avg_power:>8.1f} W")
    
    # JSON 결과
    results = {
        'latency': {
            'average_ms': float(avg_time),
            'std_ms': float(std_time),
            'min_ms': float(min_time),
            'max_ms': float(max_time),
            'p50_ms': float(p50_time),
            'p95_ms': float(p95_time),
            'p99_ms': float(p99_time),
        },
        'throughput': {
            'fps': float(fps),
        },
        'breakdown': {
            'preprocess_ms': float(np.mean(preprocess_times) * 1000),
            'encoding_ms': float(np.mean(encoding_times) * 1000),
            'generation_ms': float(np.mean(generation_times) * 1000),
        },
        'memory': {
            'gpu_mb': float(final_gpu_mem),
            'cpu_mb': float(final_cpu_mem),
        },
        'power': {
            'gpu_watts': float(avg_power),
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="LiteVLM Benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/",
        help="Path to model directory"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Test image path (optional)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # 모델 로드
    print("Loading LiteVLM...")
    model_path = Path(args.model_path)
    
    vlm = LiteVLM(
        vision_encoder=model_path / "vision_encoder_fp8.engine",
        text_decoder=model_path / "text_decoder_fp8.engine",
        token_compression=True,
        speculative_decode=True,
        device=args.device,
        verbose=False
    )
    
    # 테스트 이미지
    if args.image:
        image_path = args.image
    else:
        # 더미 이미지 생성
        print("No image provided, using dummy image")
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
        )
        image_path = "/tmp/dummy_test_image.jpg"
        dummy_img.save(image_path)
    
    prompt = "Describe this image in detail."
    
    # 벤치마크 실행
    results = benchmark_inference(
        vlm,
        image_path,
        prompt,
        num_runs=args.num_runs,
        warmup=args.warmup,
    )
    
    # 결과 저장
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
