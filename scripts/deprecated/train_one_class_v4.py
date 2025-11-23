#!/usr/bin/env python3
"""
Train One-Class v4 Model with Enhanced Features
YAMNet 임베딩 + 음향 특성 (MFCC, Spectral features 등)
"""

import sys
import numpy as np
import librosa
from pathlib import Path
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def extract_enhanced_features(audio: np.ndarray, sr: int, classifier) -> np.ndarray:
    """
    YAMNet + 음향 특성 추출

    Returns:
        117차원 특성 벡터
    """
    # 1. YAMNet 임베딩 (88차원)
    yamnet_emb = classifier.extract_embedding(audio, sr)

    # 2. MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)  # 13차원
    mfcc_std = mfcc.std(axis=1)    # 13차원

    # 3. Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

    # 4. Zero Crossing Rate (금속성 소리 감지에 유용)
    zcr = librosa.feature.zero_crossing_rate(audio)

    # 5. RMS Energy
    rms = librosa.feature.rms(y=audio)

    # 결합
    features = np.concatenate([
        yamnet_emb,                      # 88차원
        mfcc_mean,                       # 13차원
        mfcc_std,                        # 13차원
        [spectral_centroid.mean()],      # 1차원
        [spectral_rolloff.mean()],       # 1차원
        [spectral_bandwidth.mean()],     # 1차원
        [zcr.mean()],                    # 1차원
        [rms.mean()]                     # 1차원
    ])  # 총 117차원

    return features


def main():
    print("="*80)
    print("🔊 One-Class v4 모델 학습 (YAMNet + 음향 특성)")
    print("="*80)

    # Paths
    knocking_dir = Path("data/training/engine_knocking_segments_v2")
    output_path = Path("data/models/car_classifier_oneclass_v4.pkl")
    yamnet_model = Path("data/models/yamnet.tflite")

    # Load YAMNet
    print("\n🤖 YAMNet 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Load data and extract enhanced features
    print(f"\n📂 데이터 로딩 및 특성 추출: {knocking_dir}")
    print("="*80)

    audio_files = sorted(knocking_dir.glob("*.wav"))
    print(f"🔊 총 {len(audio_files)}개 파일")
    print("   (YAMNet 88차원 + 음향 특성 29차원 = 117차원)\n")

    X_list = []
    for i, audio_file in enumerate(audio_files, 1):
        try:
            # Load audio
            audio, sr = AudioFileLoader.load_audio(str(audio_file))

            # Extract enhanced features
            features = extract_enhanced_features(audio, sr, classifier)
            X_list.append(features)

            if i % 30 == 0 or i == len(audio_files):
                print(f"  ✅ [{i}/{len(audio_files)}] 처리 완료 (특성: {len(features)}차원)")

        except Exception as e:
            print(f"  ⚠️  {audio_file.name}: {e}")
            continue

    X = np.array(X_list)
    print(f"\n✅ 총 {len(X)}개 샘플 로드")
    print(f"   특성 차원: {X.shape[1]}D (YAMNet 88 + 음향 29)")

    # Train
    print("\n🎯 Isolation Forest 학습 중...")
    print("="*80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)
    print("✅ 학습 완료")

    # Evaluate
    predictions = model.predict(X_scaled)
    n_inliers = np.sum(predictions == 1)
    n_outliers = np.sum(predictions == -1)

    print(f"\n📊 학습 데이터 평가:")
    print(f"   - Inliers (노킹 패턴): {n_inliers}개 ({n_inliers/len(X)*100:.1f}%)")
    print(f"   - Outliers (노이즈): {n_outliers}개 ({n_outliers/len(X)*100:.1f}%)")

    scores = model.decision_function(X_scaled)
    print(f"   - Decision score 범위: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"   - Decision score 평균: {scores.mean():.3f}")

    # Save
    print(f"\n💾 모델 저장: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'model_type': 'one_class_enhanced',
            'info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'features': 'YAMNet(88) + MFCC(26) + Spectral(3)',
                'model_type': 'isolation_forest',
                'description': 'One-Class with YAMNet + acoustic features',
            }
        }, f)

    print("✅ 저장 완료!")

    print("\n"+"="*80)
    print("✅ v4 모델 학습 완료!")
    print("="*80)
    print(f"📊 모델 정보:")
    print(f"   - 샘플: {len(X)}개")
    print(f"   - 특성: {X.shape[1]}D (향상됨!)")
    print(f"   - 저장: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
