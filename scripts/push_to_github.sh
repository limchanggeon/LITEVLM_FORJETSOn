#!/bin/bash
###############################################################################
# GitHub에 프로젝트 푸시 가이드
# 이 스크립트는 프로젝트를 GitHub에 업로드하는 과정을 안내합니다
###############################################################################

echo "========================================="
echo "LiteVLM - GitHub Upload Guide"
echo "========================================="
echo ""

# 색상
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Git 저장소 초기화
echo -e "${GREEN}Step 1: Git 저장소 초기화${NC}"
echo "다음 명령어를 실행하세요:"
echo ""
echo "  cd /Users/limchang-geon/Desktop/liteVLM_injetson"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial commit: LiteVLM for Jetson Orin'"
echo ""
read -p "위 명령어를 실행했나요? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "먼저 Git 저장소를 초기화해주세요."
    exit 1
fi

# 2. GitHub 저장소 생성 안내
echo ""
echo -e "${GREEN}Step 2: GitHub에서 새 저장소 생성${NC}"
echo "1. https://github.com/new 에 접속"
echo "2. Repository name: liteVLM_injetson"
echo "3. Description: Lightweight Vision-Language Model for NVIDIA Jetson Orin"
echo "4. Public/Private 선택"
echo "5. README, .gitignore, license 추가 체크 해제 (이미 있음)"
echo "6. 'Create repository' 클릭"
echo ""
read -p "GitHub 저장소를 생성했나요? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "GitHub 저장소를 먼저 생성해주세요."
    exit 1
fi

# 3. GitHub username 입력
echo ""
echo -e "${GREEN}Step 3: GitHub 정보 입력${NC}"
read -p "GitHub username을 입력하세요: " github_username

# 4. 원격 저장소 추가 및 푸시
echo ""
echo -e "${GREEN}Step 4: 원격 저장소 연결 및 푸시${NC}"
echo "다음 명령어를 실행하세요:"
echo ""
echo "  git remote add origin https://github.com/${github_username}/liteVLM_injetson.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo -e "${YELLOW}참고: GitHub 로그인이 필요할 수 있습니다.${NC}"
echo ""

# 5. 완료
echo ""
echo "========================================="
echo -e "${GREEN}✓ GitHub 업로드 가이드 완료!${NC}"
echo "========================================="
echo ""
echo "다음 단계:"
echo "1. Jetson에서 클론:"
echo "   git clone https://github.com/${github_username}/liteVLM_injetson.git"
echo ""
echo "2. 환경 설정:"
echo "   cd liteVLM_injetson"
echo "   ./scripts/setup_jetson.sh"
echo ""
echo "3. 모델 다운로드:"
echo "   python scripts/download_models.py"
echo ""
echo "4. TensorRT 변환:"
echo "   python scripts/convert_to_tensorrt.py --fp8"
echo ""
echo "5. 추론 실행:"
echo "   python inference.py --image test.jpg --prompt 'Describe this image'"
echo ""
