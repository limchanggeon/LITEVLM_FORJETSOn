"""
LiteVLM: Lightweight Vision-Language Model for Jetson
Optimized for low-latency inference with TensorRT and FP8 quantization
"""

__version__ = "0.1.0"
__author__ = "LiteVLM Team"

from .model import LiteVLM
from .vision_encoder import VisionEncoder
from .text_decoder import TextDecoder
from .token_compression import TokenCompression
from .speculative_decode import SpeculativeDecoder

__all__ = [
    "LiteVLM",
    "VisionEncoder",
    "TextDecoder",
    "TokenCompression",
    "SpeculativeDecoder",
]
