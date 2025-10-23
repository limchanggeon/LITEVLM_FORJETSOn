"""
Configuration file for LiteVLM
사용자 정의 설정을 여기에 저장
"""

# Model Paths
MODEL_CONFIG = {
    "vision_encoder": "models/vision_encoder_fp8.engine",
    "text_decoder": "models/text_decoder_fp8.engine",
    "tokenizer": "Qwen/Qwen2-1.5B-Instruct",
}

# Optimization Settings
OPTIMIZATION = {
    "token_compression": True,
    "compression_ratio": 0.5,
    "compression_method": "adaptive",  # 'adaptive', 'attention', 'linear'
    "speculative_decode": True,
    "num_speculative_tokens": 4,
    "use_kv_cache": True,
}

# Inference Settings
INFERENCE = {
    "device": "cuda",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

# Image Processing
IMAGE = {
    "size": 336,  # 336 or 448
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
}

# Jetson Specific
JETSON = {
    "power_mode": 0,  # 0: MAXN, 1: 30W, 2: 15W
    "enable_jetson_clocks": True,
    "swap_size_gb": 8,
}

# Logging
LOGGING = {
    "verbose": True,
    "log_file": "litevlm.log",
    "benchmark_output": "benchmark_results.json",
}
