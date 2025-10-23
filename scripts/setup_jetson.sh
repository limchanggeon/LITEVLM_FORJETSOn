#!/bin/bash
###############################################################################
# Jetson Orin 초기 설정 스크립트
# JetPack 설치 후 실행하여 LiteVLM 실행 환경 구성
###############################################################################

set -e

echo "========================================="
echo "LiteVLM Jetson Setup Script"
echo "========================================="

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 시스템 정보 확인
echo -e "\n${GREEN}[1/7] Checking system info...${NC}"
echo "Jetson Model: $(cat /proc/device-tree/model)"
echo "JetPack Version: $(dpkg -l | grep nvidia-jetpack | awk '{print $3}')"
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}')"

# 2. 시스템 업데이트
echo -e "\n${GREEN}[2/7] Updating system...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# 3. 필수 패키지 설치
echo -e "\n${GREEN}[3/7] Installing dependencies...${NC}"
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran

# 4. PyTorch 설치 (Jetson 전용)
echo -e "\n${GREEN}[4/7] Installing PyTorch for Jetson...${NC}"
if ! python3 -c "import torch" &> /dev/null; then
    echo "Downloading PyTorch wheel for Jetson..."
    # PyTorch 2.0.0 for JetPack 5.1+
    wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl -O torch-2.0.0-cp310-cp310-linux_aarch64.whl
    pip3 install torch-2.0.0-cp310-cp310-linux_aarch64.whl
    rm torch-2.0.0-cp310-cp310-linux_aarch64.whl
    echo -e "${GREEN}PyTorch installed!${NC}"
else
    echo -e "${YELLOW}PyTorch already installed${NC}"
fi

# 5. TorchVision 설치
echo -e "\n${GREEN}[5/7] Installing TorchVision...${NC}"
if ! python3 -c "import torchvision" &> /dev/null; then
    sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
    git clone --branch v0.15.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.15.0
    python3 setup.py install --user
    cd ..
    rm -rf torchvision
    echo -e "${GREEN}TorchVision installed!${NC}"
else
    echo -e "${YELLOW}TorchVision already installed${NC}"
fi

# 6. 전력 모드 설정
echo -e "\n${GREEN}[6/7] Setting power mode...${NC}"
echo -e "${YELLOW}Available power modes:${NC}"
sudo nvpmodel -q
echo -e "\nSetting to MAX performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# 7. Swap 메모리 확장 (선택사항)
echo -e "\n${GREEN}[7/7] Configuring swap memory...${NC}"
SWAP_SIZE="8G"
if [ ! -f /mnt/swapfile ]; then
    echo "Creating ${SWAP_SIZE} swap file..."
    sudo fallocate -l ${SWAP_SIZE} /mnt/swapfile
    sudo chmod 600 /mnt/swapfile
    sudo mkswap /mnt/swapfile
    sudo swapon /mnt/swapfile
    echo '/mnt/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo -e "${GREEN}Swap configured!${NC}"
else
    echo -e "${YELLOW}Swap already configured${NC}"
fi

echo ""
echo "========================================="
echo -e "${GREEN}Jetson setup complete!${NC}"
echo "========================================="
echo ""
echo "System Info:"
free -h
echo ""
echo "GPU Info:"
tegrastats --interval 1000 --stop

echo ""
echo "Next steps:"
echo "1. Install Python dependencies:"
echo "   pip3 install -r requirements.txt"
echo "2. Download models:"
echo "   python3 scripts/download_models.py"
echo "3. Convert to TensorRT:"
echo "   python3 scripts/convert_to_tensorrt.py --fp8"
echo ""
