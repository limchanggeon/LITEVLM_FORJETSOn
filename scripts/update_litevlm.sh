#!/bin/bash
###############################################################################
# LiteVLM 자동 업데이트 스크립트
# Jetson에서 최신 버전을 자동으로 다운로드합니다
###############################################################################

set -e

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}LiteVLM Update Script${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# 현재 버전 백업
if [ -d "liteVLM_injetson" ]; then
    echo -e "${YELLOW}📦 Backing up current version...${NC}"
    rm -rf liteVLM_injetson_backup
    mv liteVLM_injetson liteVLM_injetson_backup
    echo -e "${GREEN}✓ Backup created: liteVLM_injetson_backup${NC}"
fi

# 최신 버전 다운로드
echo ""
echo -e "${YELLOW}⬇️  Downloading latest version...${NC}"
wget https://github.com/limchanggeon/LITEVLM_FORJETSOn/archive/refs/heads/main.zip -O litevlm.zip

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Download failed!${NC}"
    echo "Please check your internet connection and try again."
    exit 1
fi

# 압축 해제
echo ""
echo -e "${YELLOW}📂 Extracting files...${NC}"
unzip -q litevlm.zip

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Extraction failed!${NC}"
    exit 1
fi

# 폴더 이름 변경
mv LITEVLM_FORJETSOn-main liteVLM_injetson

# 임시 파일 삭제
rm litevlm.zip

# 완료
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ Update complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "📁 Project location: ${YELLOW}./liteVLM_injetson${NC}"

if [ -d "liteVLM_injetson_backup" ]; then
    echo -e "📁 Backup location: ${YELLOW}./liteVLM_injetson_backup${NC}"
fi

echo ""
echo "Next steps:"
echo "  cd liteVLM_injetson"
echo "  python webui.py"
echo ""
