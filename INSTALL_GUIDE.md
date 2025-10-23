# Jetson에서 프로젝트 설치 가이드

GitHub에서 프로젝트를 받아오는 여러 가지 방법을 안내합니다.

## 방법 1: Git Clone (권장)

가장 간단하고 업데이트가 쉬운 방법입니다.

```bash
# 프로젝트 클론
git clone https://github.com/limchanggeon/LITEVLM_FORJETSOn.git

# 폴더로 이동
cd LITEVLM_FORJETSOn

# 이후 업데이트
git pull
```

### ⚠️ 인증 에러가 발생하는 경우

```bash
# Git credential 캐시 삭제
git credential reject https://github.com
# (엔터 두 번)

# Git 설정 초기화
git config --global --unset credential.helper

# 다시 클론 시도
git clone https://github.com/limchanggeon/LITEVLM_FORJETSOn.git
```

---

## 방법 2: ZIP 다운로드 + Git 연결

Git clone이 안 되는 경우 ZIP으로 다운로드 후 Git 저장소로 변환합니다.

### 단계별 가이드

```bash
# 1. ZIP 파일 다운로드
wget https://github.com/limchanggeon/LITEVLM_FORJETSOn/archive/refs/heads/main.zip

# 2. 압축 해제
unzip main.zip

# 3. 폴더 이름 변경
mv LITEVLM_FORJETSOn-main liteVLM_injetson

# 4. 폴더로 이동
cd liteVLM_injetson

# 5. Git 저장소로 변환
git init
git remote add origin https://github.com/limchanggeon/LITEVLM_FORJETSOn.git
git fetch origin
git reset --hard origin/main
git branch --set-upstream-to=origin/main main

# 6. 확인
git status
git log --oneline -3

# 7. 이제 git pull 사용 가능!
git pull
```

### ✅ 성공 확인

```bash
# Git 상태 확인
git status
# 출력: On branch main, Your branch is up to date with 'origin/main'

# 원격 저장소 확인
git remote -v
# 출력: origin https://github.com/limchanggeon/LITEVLM_FORJETSOn.git (fetch)
#       origin https://github.com/limchanggeon/LITEVLM_FORJETSOn.git (push)
```

---

## 방법 3: 자동 업데이트 스크립트

Git 없이 매번 최신 버전을 받고 싶다면 자동화 스크립트를 사용하세요.

### 스크립트 생성

```bash
# update_litevlm.sh 생성
cat > update_litevlm.sh << 'EOF'
#!/bin/bash

echo "========================================="
echo "LiteVLM Update Script"
echo "========================================="

# 현재 디렉토리 백업
if [ -d "liteVLM_injetson" ]; then
    echo "📦 Backing up current version..."
    rm -rf liteVLM_injetson_backup
    mv liteVLM_injetson liteVLM_injetson_backup
fi

# 최신 버전 다운로드
echo "⬇️  Downloading latest version..."
wget https://github.com/limchanggeon/LITEVLM_FORJETSOn/archive/refs/heads/main.zip -O litevlm.zip

# 압축 해제
echo "📂 Extracting files..."
unzip -q litevlm.zip

# 폴더 이름 변경
mv LITEVLM_FORJETSOn-main liteVLM_injetson

# 임시 파일 삭제
rm litevlm.zip

echo "✅ Update complete!"
echo "📁 Project location: ./liteVLM_injetson"
echo "📁 Backup location: ./liteVLM_injetson_backup"
echo "========================================="
EOF

# 실행 권한 부여
chmod +x update_litevlm.sh
```

### 스크립트 사용

```bash
# 최신 버전으로 업데이트
./update_litevlm.sh

# 업데이트 후 이동
cd liteVLM_injetson
```

---

## 방법 비교

| 방법 | 장점 | 단점 | 추천 |
|------|------|------|------|
| **Git Clone** | - 간단함<br>- `git pull`로 쉬운 업데이트<br>- Git 기능 활용 가능 | - 인증 문제 가능 | ⭐⭐⭐⭐⭐ |
| **ZIP + Git 연결** | - 초기 다운로드 확실<br>- 이후 Git 사용 가능 | - 초기 설정 복잡 | ⭐⭐⭐⭐ |
| **자동 스크립트** | - 인증 불필요<br>- 자동화 가능 | - 매번 전체 다운로드<br>- Git 기능 없음 | ⭐⭐⭐ |

---

## 권장 순서

1. **먼저 Git Clone 시도** (방법 1)
2. 실패하면 **ZIP + Git 연결** (방법 2)
3. 그래도 안 되면 **자동 스크립트** (방법 3)

---

## 다음 단계

프로젝트를 받았다면:

```bash
# 1. 환경 설정
./scripts/setup_jetson.sh

# 2. Python 환경
conda env create -f environment.yml
conda activate litevlm

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 모델 다운로드
python scripts/download_models.py

# 5. TensorRT 변환
python scripts/convert_to_tensorrt.py --fp8

# 6. Web UI 실행
python webui.py
```

문제가 있으면 Issues에 남겨주세요!
