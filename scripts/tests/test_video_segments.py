#!/usr/bin/env python3
"""
Test One-Class Model on YouTube Video
"""

import sys
import numpy as np
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def get_engine_knocking_score(classifications):
    """Get 'Engine knocking' score"""
    knocking_score = 0.0
    top_pred = ""

    for result in classifications:
        categories = result.get('categories', [])
        if categories:
            top_pred = categories[0]['category_name']

        for category in categories:
            category_name = category['category_name'].lower()
            score = category['score']

            if 'engine' in category_name and 'knock' in category_name:
                knocking_score = max(knocking_score, score)

    return knocking_score, top_pred


def extract_enhanced_features(audio: np.ndarray, sr: int, classifier) -> np.ndarray:
    """Extract enhanced features (for v4 model)"""
    import librosa

    # 1. YAMNet embedding
    yamnet_emb = classifier.extract_embedding(audio, sr)

    # 2. MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # 3. Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)

    # 5. RMS Energy
    rms = librosa.feature.rms(y=audio)

    features = np.concatenate([
        yamnet_emb,
        mfcc_mean,
        mfcc_std,
        [spectral_centroid.mean()],
        [spectral_rolloff.mean()],
        [spectral_bandwidth.mean()],
        [zcr.mean()],
        [rms.mean()]
    ])

    return features


def main():
    print("="*80)
    print("🔍 One-Class 모델 테스트")
    print("="*80)

    # Paths
    audio_path = Path("data/testing/knocking_test3.wav")
    v2_model_path = Path("data/models/car_classifier_oneclass_v2.pkl")
    v4_model_path = Path("data/models/car_classifier_oneclass_v4.pkl")
    yamnet_model = Path("data/models/yamnet.tflite")

    # Load YAMNet
    print("\n🤖 YAMNet 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Load audio
    print(f"\n📂 오디오 로딩: {audio_path.name}")
    audio, sr = AudioFileLoader.load_audio(str(audio_path))
    duration = len(audio) / sr
    print(f"   ⏱️  길이: {duration:.1f}초")

    # Load models
    print("\n📦 One-Class 모델 로드 중...")
    with open(v2_model_path, 'rb') as f:
        v2_data = pickle.load(f)
        v2_model = v2_data['model']
        v2_scaler = v2_data['scaler']
    print(f"   ✅ v2 모델 로드 완료 ({v2_data['info']['n_samples']} 샘플)")

    with open(v4_model_path, 'rb') as f:
        v4_data = pickle.load(f)
        v4_model = v4_data['model']
        v4_scaler = v4_data['scaler']
    print(f"   ✅ v4 모델 로드 완료 ({v4_data['info']['n_samples']} 샘플)")

    # Test with sliding window
    print("\n" + "="*80)
    print("🔬 슬라이딩 윈도우 분석 (1.5초 윈도우)")
    print("="*80)

    window_size = 1.5
    hop_size = 1.5  # No overlap for testing
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    results = []

    for start_sample in range(0, len(audio) - window_samples, hop_samples):
        end_sample = start_sample + window_samples
        segment = audio[start_sample:end_sample]

        # Skip if too quiet
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 0.01:
            continue

        # Get YAMNet analysis
        classifications = classifier.classify_audio(segment, sr)
        knocking_score, top_pred = get_engine_knocking_score(classifications)

        # Extract features for v2 (YAMNet only)
        embedding_v2 = classifier.extract_embedding(segment, sr)

        # Extract features for v4 (YAMNet + acoustic)
        features_v4 = extract_enhanced_features(segment, sr, classifier)

        # Predict with v2
        X_v2 = v2_scaler.transform([embedding_v2])
        pred_v2 = v2_model.predict(X_v2)[0]
        score_v2 = v2_model.decision_function(X_v2)[0]

        # Predict with v4
        X_v4 = v4_scaler.transform([features_v4])
        pred_v4 = v4_model.predict(X_v4)[0]
        score_v4 = v4_model.decision_function(X_v4)[0]

        results.append({
            'time': start_sample / sr,
            'yamnet_knocking': knocking_score,
            'top_pred': top_pred,
            'v2_pred': pred_v2,
            'v2_score': score_v2,
            'v4_pred': pred_v4,
            'v4_score': score_v4,
        })

    # Display results
    print("\n시간   | YAMNet 노킹 | v2     | v2 점수 | v4     | v4 점수 | Top 분류")
    print("-" * 80)

    v2_knocking_count = 0
    v4_knocking_count = 0
    max_yamnet_score = 0.0

    for r in results:
        max_yamnet_score = max(max_yamnet_score, r['yamnet_knocking'])

        v2_status = "🚨" if r['v2_pred'] == -1 else "✅"
        v4_status = "🚨" if r['v4_pred'] == -1 else "✅"

        if r['v2_pred'] == -1:
            v2_knocking_count += 1
        if r['v4_pred'] == -1:
            v4_knocking_count += 1

        yamnet_str = f"{r['yamnet_knocking']:.0%}" if r['yamnet_knocking'] > 0 else "없음"

        print(f"{r['time']:5.1f}s | {yamnet_str:11} | {v2_status:6} | {r['v2_score']:7.3f} | "
              f"{v4_status:6} | {r['v4_score']:7.3f} | {r['top_pred']}")

    # Summary
    print("\n" + "="*80)
    print("📊 요약:")
    print("="*80)
    print(f"YAMNet 최대 노킹 점수: {max_yamnet_score:.1%}")
    print(f"v2 노킹 감지: {v2_knocking_count}개")
    print(f"v4 노킹 감지: {v4_knocking_count}개")

    if v2_knocking_count > 0 or v4_knocking_count > 0:
        print("\n✅ 노킹 감지 성공!")
    else:
        print("\n❌ 노킹 미감지")

    print("="*80)


if __name__ == "__main__":
    main()
