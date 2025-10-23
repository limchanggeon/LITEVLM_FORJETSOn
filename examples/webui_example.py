"""
Gradio Web UI 예제
간단한 웹 인터페이스로 LiteVLM 사용하기
"""

import subprocess
import sys

def main():
    print("="*60)
    print("LiteVLM Web UI Example")
    print("="*60)
    print()
    print("이 예제는 Gradio 기반 웹 인터페이스를 실행합니다.")
    print()
    print("사용 방법:")
    print("1. 웹 브라우저가 자동으로 열립니다")
    print("2. 이미지를 업로드하고 프롬프트를 입력하세요")
    print("3. 'Generate' 버튼을 클릭하여 추론을 실행하세요")
    print()
    print("="*60)
    print()
    
    # Web UI 실행
    try:
        subprocess.run([
            sys.executable,
            "../webui.py",
            "--port", "7860"
        ])
    except KeyboardInterrupt:
        print("\n\nWeb UI가 종료되었습니다.")
    except Exception as e:
        print(f"Error: {e}")
        print("\n대신 다음 명령어로 실행해보세요:")
        print("  python webui.py")


if __name__ == "__main__":
    main()
