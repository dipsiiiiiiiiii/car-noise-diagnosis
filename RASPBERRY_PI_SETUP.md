# 라즈베리파이 배포 가이드

## 필요한 하드웨어
- 라즈베리파이 4 (최소 2GB RAM, 4GB 권장)
- USB 마이크 또는 오디오 입력 장치
- 32GB 이상 SD카드

## 설치 단계

### 1. 시스템 업데이트 및 기본 패키지 설치
```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv git
sudo apt-get install -y portaudio19-dev libatlas-base-dev
sudo apt-get install -y ffmpeg libsndfile1
```

### 2. 프로젝트 복사
**방법 A: Git 사용 (권장)**
```bash
cd ~
git clone <your-repo-url> noise-diagnosis
cd noise-diagnosis
```

**방법 B: SCP로 파일 전송 (Mac에서 실행)**
```bash
scp -r /Users/jacob/workspace/school/final-project/noise-diagnosis pi@<라즈베리파이IP>:~/
```

### 3. Python 가상환경 설정
```bash
cd ~/noise-diagnosis
python3 -m venv venv
source venv/bin/activate
```

### 4. 의존성 설치
```bash
# 기본 패키지 먼저 설치
pip install --upgrade pip setuptools wheel

# NumPy (라즈베리파이에서 시간이 걸릴 수 있음)
pip install numpy

# 오디오 처리
pip install soundfile librosa

# 머신러닝
pip install scikit-learn scipy

# MediaPipe (라즈베리파이용)
pip install mediapipe==0.10.21

# 시각화 (선택사항, 헤드리스 모드에서는 불필요)
# pip install matplotlib seaborn opencv-python
```

### 5. 마이크 설정 확인
```bash
# 마이크 연결 확인
arecord -l

# 테스트 녹음 (5초)
arecord -d 5 -f cd test.wav

# 재생 테스트
aplay test.wav
```

### 6. 모델 파일 확인
필요한 파일들이 있는지 확인:
```bash
ls data/models/yamnet.tflite
ls data/models/car_classifier_binary.pkl
```

파일이 없으면 Mac에서 전송:
```bash
# Mac에서 실행
scp data/models/* pi@<라즈베리파이IP>:~/noise-diagnosis/data/models/
```

### 7. 실행 테스트
```bash
source venv/bin/activate
python main.py
```

## 문제 해결

### 메모리 부족 에러
swap 메모리 늘리기:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048 로 변경
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 마이크가 인식 안 됨
```bash
# USB 마이크를 기본 장치로 설정
sudo nano /usr/share/alsa/alsa.conf
# defaults.pcm.card 0 을 defaults.pcm.card 1 로 변경
```

### MediaPipe 설치 실패
라즈베리파이 아키텍처에 따라 다를 수 있음:
```bash
# ARM 64비트 확인
uname -m  # aarch64 또는 armv7l

# 만약 설치 실패하면 이전 버전 시도
pip install mediapipe==0.10.9
```

## 자동 실행 설정 (선택사항)

부팅 시 자동으로 모니터링 시작:

```bash
# systemd 서비스 생성
sudo nano /etc/systemd/system/noise-diagnosis.service
```

파일 내용:
```ini
[Unit]
Description=Car Noise Diagnosis Service
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/noise-diagnosis
Environment="PATH=/home/pi/noise-diagnosis/venv/bin"
ExecStart=/home/pi/noise-diagnosis/venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

서비스 활성화:
```bash
sudo systemctl daemon-reload
sudo systemctl enable noise-diagnosis.service
sudo systemctl start noise-diagnosis.service

# 상태 확인
sudo systemctl status noise-diagnosis.service

# 로그 확인
sudo journalctl -u noise-diagnosis.service -f
```

## 성능 최적화 팁

1. **분석 간격 늘리기**: main.py에서 `analysis_interval` 을 3초에서 5초로 변경
2. **윈도우 수 줄이기**: hop_size를 0.5에서 1.0으로 변경
3. **불필요한 출력 제거**: 디버그 모드 비활성화
4. **메모리 정리**: 주기적으로 `gc.collect()` 호출

## 원격 모니터링

SSH로 접속해서 실시간 모니터링:
```bash
ssh pi@<라즈베리파이IP>
cd ~/noise-diagnosis
source venv/bin/activate
python main.py
```

## 추가 고려사항

- **전원 관리**: 차량 전원에 연결 시 안정적인 전원 공급 필요
- **오디오 품질**: 차량 소음 환경에서 마이크 위치가 중요
- **저장 공간**: SD카드 용량 주기적 확인
- **네트워크**: 필요시 Wi-Fi 또는 이더넷 설정
