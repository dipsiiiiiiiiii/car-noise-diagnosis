# YAMNet 커스텀 학습 - 새 환경 구축

## 1. 새 가상환경 생성

```bash
# Python 3.11로 새 가상환경 생성
pyenv virtualenv 3.11.13 noise-yamnet-custom

# 가상환경 활성화
pyenv activate noise-yamnet-custom

# pip 업그레이드
pip install --upgrade pip
```

## 2. 필요한 패키지 설치

```bash
# TensorFlow 및 관련 패키지
pip install tensorflow==2.17.0
pip install tensorflow-hub
pip install numpy
pip install scikit-learn
pip install librosa
pip install soundfile
pip install matplotlib
pip install seaborn
```

## 3. 학습 실행

```bash
# YAMNet 커스텀 모델 학습
python train_yamnet_transfer.py
```

## 4. 결과 파일

- `data/models/yamnet_custom_model.h5` - Keras 모델
- `data/models/yamnet_custom_metadata.pkl` - 메타데이터
