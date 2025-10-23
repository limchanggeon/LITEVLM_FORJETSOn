#!/usr/bin/env python3
"""
LiteVLM Web UI with Gradio
간단한 웹 인터페이스로 이미지 업로드 및 추론 실행
"""

import gradio as gr
import os
import time
from pathlib import Path
from PIL import Image

from litevlm import LiteVLM


class LiteVLMWebUI:
    """LiteVLM Web Interface"""
    
    def __init__(self, model_path="models"):
        self.model_path = Path(model_path)
        self.vlm = None
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            print("Loading LiteVLM model...")
            
            vision_encoder = self.model_path / "vision_encoder_fp8.engine"
            text_decoder = self.model_path / "text_decoder_fp8.engine"
            
            # 모델 파일 확인
            if not vision_encoder.exists():
                return "❌ Vision encoder not found. Please run: python scripts/convert_to_tensorrt.py"
            
            if not text_decoder.exists():
                return "❌ Text decoder not found. Please run: python scripts/convert_to_tensorrt.py"
            
            self.vlm = LiteVLM(
                vision_encoder=str(vision_encoder),
                text_decoder=str(text_decoder),
                token_compression=True,
                speculative_decode=True,
                device="cuda",
                verbose=False
            )
            
            print("✓ Model loaded successfully!")
            return "✓ Model loaded"
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            print(error_msg)
            return error_msg
    
    def inference(
        self,
        image,
        prompt,
        max_tokens,
        temperature,
        top_p,
        use_compression,
        use_speculative
    ):
        """추론 실행"""
        
        if self.vlm is None:
            return None, "❌ Model not loaded. Please check the model files.", ""
        
        if image is None:
            return None, "❌ Please upload an image.", ""
        
        if not prompt.strip():
            return None, "❌ Please enter a prompt.", ""
        
        try:
            # 최적화 설정 업데이트
            self.vlm.use_compression = use_compression
            self.vlm.use_speculative = use_speculative
            
            # 추론 실행
            start_time = time.time()
            result = self.vlm.chat(
                image=image,
                prompt=prompt,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                return_stats=True
            )
            total_time = time.time() - start_time
            
            # 결과 추출
            generated_text = result['text']
            stats = result['stats']
            
            # 통계 포맷팅
            stats_text = f"""
📊 **Performance Statistics**

⏱️ **Timing**
- Preprocessing: {stats['preprocess_time']*1000:.1f} ms
- Encoding: {stats['encoding_time']*1000:.1f} ms
- Generation: {stats['generation_time']*1000:.1f} ms
- **Total: {stats['total_time']*1000:.1f} ms**

🎯 **Efficiency**
- Visual Tokens: {stats['num_visual_tokens']}
- Throughput: {1000/stats['total_time']:.2f} FPS

⚙️ **Settings**
- Token Compression: {'✓' if use_compression else '✗'}
- Speculative Decoding: {'✓' if use_speculative else '✗'}
- Max Tokens: {max_tokens}
- Temperature: {temperature}
- Top-p: {top_p}
"""
            
            return image, generated_text, stats_text
            
        except Exception as e:
            error_msg = f"❌ Error during inference: {str(e)}"
            print(error_msg)
            return image, error_msg, ""


def create_ui():
    """Gradio UI 생성"""
    
    # LiteVLM 인스턴스 생성
    ui_handler = LiteVLMWebUI()
    
    # 예제 프롬프트
    example_prompts = [
        "이 사진을 자세히 설명해줘",
        "Describe this image in detail.",
        "What objects do you see in this image?",
        "이미지에서 가장 눈에 띄는 것은 무엇인가요?",
        "What is happening in this scene?",
    ]
    
    # Gradio Interface
    with gr.Blocks(
        title="LiteVLM on Jetson",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 1200px !important}
        #component-0 {height: 600px !important}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🚀 LiteVLM on Jetson Orin
        
        **Lightweight Vision-Language Model** with 3-stage optimization pipeline
        
        - 🎯 Patch Selection (3x computation reduction)
        - 🗜️ Token Compression (reduced LLM input)
        - ⚡ Speculative Decoding (2-3x faster)
        """)
        
        with gr.Row():
            # 왼쪽: 입력
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Input")
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=400
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your question about the image...",
                    lines=3,
                    value="이 사진을 자세히 설명해줘"
                )
                
                gr.Markdown("### ⚙️ Settings")
                
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=512,
                        value=256,
                        step=1,
                        label="Max Tokens"
                    )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p"
                    )
                
                with gr.Row():
                    use_compression = gr.Checkbox(
                        label="Token Compression",
                        value=True
                    )
                    
                    use_speculative = gr.Checkbox(
                        label="Speculative Decoding",
                        value=True
                    )
                
                generate_btn = gr.Button(
                    "🚀 Generate",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### 💡 Example Prompts")
                example_buttons = []
                for prompt in example_prompts:
                    btn = gr.Button(prompt, size="sm")
                    btn.click(
                        fn=lambda p=prompt: p,
                        outputs=prompt_input
                    )
            
            # 오른쪽: 출력
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Output")
                
                output_image = gr.Image(
                    label="Processed Image",
                    height=400
                )
                
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=8,
                    show_copy_button=True
                )
                
                stats_text = gr.Markdown(
                    label="Statistics",
                    value="📊 Statistics will appear here after inference..."
                )
        
        # 이벤트 핸들러
        generate_btn.click(
            fn=ui_handler.inference,
            inputs=[
                image_input,
                prompt_input,
                max_tokens,
                temperature,
                top_p,
                use_compression,
                use_speculative
            ],
            outputs=[
                output_image,
                output_text,
                stats_text
            ]
        )
        
        # 예제 섹션
        gr.Markdown("""
        ---
        ### 📚 Quick Guide
        
        1. **Upload an image** - Click or drag & drop
        2. **Enter a prompt** - Ask questions about the image
        3. **Adjust settings** - Tune performance/quality
        4. **Click Generate** - Get AI-powered description
        
        ### ⚡ Optimization Tips
        
        - **Token Compression**: Reduces memory usage, slight quality tradeoff
        - **Speculative Decoding**: 2-3x faster generation
        - **Temperature**: Lower = focused, Higher = creative
        - **Top-p**: Controls diversity of outputs
        
        ### 🔧 Troubleshooting
        
        - If model not loaded, run: `python scripts/convert_to_tensorrt.py --fp8`
        - For best performance: Enable both optimizations
        - Adjust max tokens based on memory availability
        """)
    
    return demo


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LiteVLM Web UI")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models",
        help="Path to model directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    
    args = parser.parse_args()
    
    # UI 생성 및 실행
    demo = create_ui()
    
    print("\n" + "="*60)
    print("🚀 Starting LiteVLM Web UI")
    print("="*60)
    print(f"\n📍 Local URL: http://localhost:{args.port}")
    print(f"📍 Network URL: http://{args.host}:{args.port}")
    
    if args.share:
        print(f"📍 Public URL: (will be generated)")
    
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
