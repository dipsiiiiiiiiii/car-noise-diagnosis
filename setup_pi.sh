ㄷ#!/bin/bash
# 라즈베리파이 자동 설치 스크립트

set -e  # 에러 발생 시 중단

echo "=========================================="
echo "🚗 차량 소음 진단 시스템 - 라즈베리파이 설치"
echo "=========================================="
echo ""

# 1. 시스템 패키지 업데이트
echo "📦 시스템 패키지 업데이트 중..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y portaudio19-dev libatlas-base-dev
sudo apt-get install -y ffmpeg libsndfile1

# 2. 가상환경 생성
echo ""
echo "🐍 Python 가상환경 생성 중..."
if [ -d "venv" ]; then
    echo "⚠️  기존 가상환경이 있습니다. 삭제하고 재생성하시겠습니까? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf venv
    else
        echo "설치를 중단합니다."
        exit 1
    fi
fi

python3 -m venv venv
source venv/bin/activate

# 3. pip 업그레이드
echo ""
echo "📦 pip 업그레이드 중..."
pip install --upgrade pip setuptools wheel

# 4. 의존성 설치
echo ""
echo "📦 의존성 설치 중... (시간이 걸릴 수 있습니다)"
if [ -f "requirements-pi.txt" ]; then
    pip install -r requirements-pi.txt
else
    echo "⚠️  requirements-pi.txt 파일이 없습니다."
    echo "기본 requirements.txt를 사용합니다."
    pip install -r requirements.txt
fi

# 5. 디렉토리 구조 확인
echo ""
echo "📁 디렉토리 구조 확인 중..."
mkdir -p data/models
mkdir -p data/training

# 6. 모델 파일 확인
echo ""
echo "🔍 모델 파일 확인 중..."
if [ ! -f "data/models/yamnet.tflite" ]; then
    echo "⚠️  YAMNet 모델이 없습니다."
    echo "Mac에서 다음 명령어로 전송하세요:"
    echo "  scp data/models/yamnet.tflite pi@$(hostname -I | awk '{print $1}'):~/noise-diagnosis/data/models/"
fi

if [ ! -f "data/models/car_classifier_binary.pkl" ]; then
    echo "⚠️  학습된 모델이 없습니다."
    echo "Mac에서 다음 명령어로 전송하세요:"
    echo "  scp data/models/car_classifier_binary.pkl pi@$(hostname -I | awk '{print $1}'):~/noise-diagnosis/data/models/"
fi

# 7. 마이크 테스트
echo ""
echo "🎤 마이크 테스트"
echo "연결된 오디오 장치:"
arecord -l || echo "⚠️  오디오 장치를 찾을 수 없습니다. USB 마이크를 연결하세요."

# 8. 설치 완료
echo ""
echo "=========================================="
echo "✅ 설치 완료!"
echo "=========================================="
echo ""
echo "실행 방법:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "자동 실행 설정은 RASPBERRY_PI_SETUP.md 참고"
echo ""
