"""
Token Compression Module
Visual tokens을 압축하여 LLM 입력 길이 감소
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCompression:
    """
    Visual Token Compression Module
    
    Features:
    - Adaptive pooling으로 토큰 개수 감소
    - Attention-based selection
    - 압축 비율 조정 가능
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.5,
        compression_method: str = "adaptive",  # 'adaptive', 'attention', 'linear'
        device: str = "cuda",
    ):
        """
        Args:
            compression_ratio: 압축 비율 (0.5 = 50% 압축)
            compression_method: 압축 방법
            device: 실행 디바이스
        """
        self.compression_ratio = compression_ratio
        self.compression_method = compression_method
        self.device = device
        
        # Attention-based compression을 위한 레이어
        if compression_method == "attention":
            self.attention = None  # 필요시 초기화
    
    def compress_adaptive(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Pooling 기반 압축
        단순하지만 효과적
        """
        B, N, D = tokens.shape
        
        # 목표 토큰 개수
        target_n = int(N * self.compression_ratio)
        
        # Adaptive average pooling
        # [B, N, D] -> [B, D, N] -> pool -> [B, D, target_n] -> [B, target_n, D]
        tokens = tokens.transpose(1, 2)  # [B, D, N]
        compressed = F.adaptive_avg_pool1d(tokens, target_n)  # [B, D, target_n]
        compressed = compressed.transpose(1, 2)  # [B, target_n, D]
        
        return compressed
    
    def compress_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Attention 기반 압축
        중요한 토큰만 선택
        """
        B, N, D = tokens.shape
        target_n = int(N * self.compression_ratio)
        
        # Self-attention으로 중요도 계산
        # Q = K = V = tokens
        attention_scores = torch.matmul(tokens, tokens.transpose(1, 2))  # [B, N, N]
        attention_scores = attention_scores.mean(dim=2)  # [B, N] - 평균 attention
        
        # Top-K 선택
        _, top_indices = torch.topk(attention_scores, target_n, dim=1)  # [B, target_n]
        top_indices = top_indices.sort(dim=1)[0]  # 순서 유지
        
        # 선택된 토큰만 추출
        compressed = torch.gather(
            tokens,
            1,
            top_indices.unsqueeze(2).expand(-1, -1, D)
        )  # [B, target_n, D]
        
        return compressed
    
    def compress_linear(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Linear projection 기반 압축
        학습 가능한 압축
        """
        B, N, D = tokens.shape
        target_n = int(N * self.compression_ratio)
        
        # Linear projection (간단한 구현)
        # 실제로는 학습된 projection 사용
        compressed = tokens[:, :target_n, :]  # 단순히 앞부분만 선택
        
        return compressed
    
    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Visual tokens 압축
        
        Args:
            tokens: [B, N, D] visual tokens
            
        Returns:
            compressed_tokens: [B, N', D] 압축된 tokens (N' < N)
        """
        if self.compression_method == "adaptive":
            return self.compress_adaptive(tokens)
        elif self.compression_method == "attention":
            return self.compress_attention(tokens)
        elif self.compression_method == "linear":
            return self.compress_linear(tokens)
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")


class LearnedTokenCompression(nn.Module):
    """
    학습 가능한 토큰 압축 모듈
    더 높은 성능이 필요할 때 사용
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        
        self.compression_ratio = compression_ratio
        
        # Query for compression
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Cross-attention for compression
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projection
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]
            
        Returns:
            compressed: [B, N', D]
        """
        B, N, D = tokens.shape
        target_n = int(N * self.compression_ratio)
        
        # Learnable queries
        queries = self.query.expand(B, target_n, -1)  # [B, target_n, D]
        
        # Cross-attention: queries attend to tokens
        compressed, _ = self.cross_attn(
            query=queries,
            key=tokens,
            value=tokens
        )  # [B, target_n, D]
        
        # Project and normalize
        compressed = self.proj(compressed)
        compressed = self.norm(compressed)
        
        return compressed
