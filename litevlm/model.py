"""
LiteVLM Main Model Class
통합된 Vision-Language Model 추론 파이프라인
"""

import os
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from .vision_encoder import VisionEncoder
from .text_decoder import TextDecoder
from .token_compression import TokenCompression
from .speculative_decode import SpeculativeDecoder


class LiteVLM:
    """
    LiteVLM 통합 추론 클래스
    
    Features:
    - Patch Selection Module (3배 연산량 절감)
    - Visual Token Compression (입력 길이 감소)
    - Speculative Decoding (2-3배 디코딩 속도 향상)
    """
    
    def __init__(
        self,
        vision_encoder: Union[str, Path],
        text_decoder: Union[str, Path],
        tokenizer: Optional[str] = "Qwen/Qwen2-1.5B-Instruct",
        token_compression: bool = True,
        speculative_decode: bool = True,
        compression_ratio: float = 0.5,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        Args:
            vision_encoder: Vision encoder 모델 경로 (.engine 또는 .pth)
            text_decoder: Text decoder 모델 경로 (.engine 또는 .pth)
            tokenizer: Tokenizer 모델 이름 또는 경로
            token_compression: 토큰 압축 활성화
            speculative_decode: Speculative decoding 활성화
            compression_ratio: 토큰 압축 비율 (0.5 = 50% 압축)
            device: 실행 디바이스 ('cuda' 또는 'cpu')
            verbose: 상세 로그 출력
        """
        self.device = device
        self.verbose = verbose
        
        if self.verbose:
            print(f"[LiteVLM] Initializing on {device}...")
        
        # Vision Encoder 초기화
        self.vision_encoder = VisionEncoder(
            model_path=vision_encoder,
            device=device
        )
        
        # Text Decoder 초기화
        self.text_decoder = TextDecoder(
            model_path=text_decoder,
            device=device
        )
        
        # Tokenizer 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            trust_remote_code=True
        )
        
        # Token Compression Module
        self.use_compression = token_compression
        if self.use_compression:
            self.token_compressor = TokenCompression(
                compression_ratio=compression_ratio,
                device=device
            )
        
        # Speculative Decoding Module
        self.use_speculative = speculative_decode
        if self.use_speculative:
            self.speculative_decoder = SpeculativeDecoder(
                draft_model=None,  # 간단한 draft model 사용
                target_model=self.text_decoder,
                device=device
            )
        
        if self.verbose:
            print(f"[LiteVLM] ✓ Vision Encoder loaded")
            print(f"[LiteVLM] ✓ Text Decoder loaded")
            print(f"[LiteVLM] ✓ Token Compression: {token_compression}")
            print(f"[LiteVLM] ✓ Speculative Decoding: {speculative_decode}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """이미지 전처리"""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # InternVL 기본 전처리
        # 336x336 또는 448x448 크기로 리사이즈
        image = image.resize((336, 336), Image.Resampling.BICUBIC)
        
        # Tensor로 변환
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """이미지를 visual tokens으로 인코딩"""
        with torch.no_grad():
            visual_tokens = self.vision_encoder(image)
        
        # Token Compression 적용
        if self.use_compression:
            visual_tokens = self.token_compressor(visual_tokens)
        
        return visual_tokens
    
    def generate_text(
        self,
        visual_tokens: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Visual tokens와 텍스트 프롬프트를 결합하여 응답 생성"""
        
        # 프롬프트 토크나이징
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)
        
        # Visual tokens와 text tokens 결합
        # [visual_tokens] + [prompt_tokens]
        input_embeds = torch.cat([visual_tokens, prompt_tokens], dim=1)
        
        # 텍스트 생성
        if self.use_speculative:
            output_ids = self.speculative_decoder.generate(
                input_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            output_ids = self.text_decoder.generate(
                input_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def chat(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_stats: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        이미지와 프롬프트를 입력받아 응답 생성
        
        Args:
            image: 입력 이미지 (경로 또는 PIL Image)
            prompt: 텍스트 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: Nucleus sampling threshold
            return_stats: 통계 정보 반환 여부
            
        Returns:
            생성된 텍스트 또는 (텍스트, 통계) 딕셔너리
        """
        stats = {}
        
        # 1. 이미지 전처리
        t0 = time.time()
        image_tensor = self.preprocess_image(image)
        stats['preprocess_time'] = time.time() - t0
        
        # 2. 이미지 인코딩 (Vision Encoder)
        t0 = time.time()
        visual_tokens = self.encode_image(image_tensor)
        stats['encoding_time'] = time.time() - t0
        stats['num_visual_tokens'] = visual_tokens.shape[1]
        
        # 3. 텍스트 생성 (Text Decoder)
        t0 = time.time()
        generated_text = self.generate_text(
            visual_tokens=visual_tokens,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        stats['generation_time'] = time.time() - t0
        
        # 총 처리 시간
        stats['total_time'] = (
            stats['preprocess_time'] +
            stats['encoding_time'] +
            stats['generation_time']
        )
        
        if self.verbose:
            print(f"\n[LiteVLM] Inference Stats:")
            print(f"  Preprocess: {stats['preprocess_time']*1000:.1f}ms")
            print(f"  Encoding: {stats['encoding_time']*1000:.1f}ms")
            print(f"  Generation: {stats['generation_time']*1000:.1f}ms")
            print(f"  Total: {stats['total_time']*1000:.1f}ms")
            print(f"  Visual Tokens: {stats['num_visual_tokens']}")
        
        if return_stats:
            return {
                'text': generated_text,
                'stats': stats
            }
        
        return generated_text
    
    def batch_chat(
        self,
        images: list,
        prompts: list,
        **kwargs
    ) -> list:
        """배치 추론"""
        assert len(images) == len(prompts), "Images and prompts must have same length"
        
        results = []
        for image, prompt in zip(images, prompts):
            result = self.chat(image, prompt, **kwargs)
            results.append(result)
        
        return results
    
    @torch.no_grad()
    def benchmark(self, num_runs: int = 100, image_size: tuple = (336, 336)):
        """성능 벤치마크"""
        print(f"\n[LiteVLM] Running benchmark ({num_runs} runs)...")
        
        # 더미 이미지 생성
        dummy_image = torch.randn(1, 3, *image_size).to(self.device)
        dummy_prompt = "Describe this image."
        
        # Warmup
        for _ in range(10):
            visual_tokens = self.encode_image(dummy_image)
            _ = self.generate_text(visual_tokens, dummy_prompt, max_new_tokens=50)
        
        # 벤치마크
        times = []
        for _ in range(num_runs):
            t0 = time.time()
            visual_tokens = self.encode_image(dummy_image)
            _ = self.generate_text(visual_tokens, dummy_prompt, max_new_tokens=50)
            times.append(time.time() - t0)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        fps = 1000 / avg_time
        
        print(f"\n[Benchmark Results]")
        print(f"  Average Latency: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput: {fps:.2f} FPS")
        print(f"  Min: {min(times)*1000:.2f} ms")
        print(f"  Max: {max(times)*1000:.2f} ms")
        
        return {
            'avg_latency_ms': avg_time,
            'std_latency_ms': std_time,
            'fps': fps,
            'min_latency_ms': min(times) * 1000,
            'max_latency_ms': max(times) * 1000,
        }
