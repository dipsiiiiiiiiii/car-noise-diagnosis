#!/usr/bin/env python3
"""
Test Verified Model on Video
"""

import sys
import numpy as np
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def get_engine_knocking_score(classifications):
    """Get 'Engine knocking' score from YAMNet"""
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


def main():
    print("="*80)
    print("🔍 Verified 모델 테스트")
    print("="*80)

    # Paths
    audio_path = Path("data/testing/test_verified.wav")
    verified_model_path = Path("data/models/car_classifier_oneclass_verified.pkl")
    yamnet_model = Path("data/models/yamnet.tflite")

    # Load YAMNet
    print("\n🤖 YAMNet 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 완료")

    # Load audio
    print(f"\n📂 오디오 로딩: {audio_path.name}")
    audio, sr = AudioFileLoader.load_audio(str(audio_path))
    duration = len(audio) / sr
    print(f"   길이: {duration:.1f}초")

    # Load verified model
    print(f"\n📦 Verified 모델 로드 중...")
    with open(verified_model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
    print(f"   ✅ 완료 ({data['info']['n_samples']} 샘플로 학습됨)")

    # Sliding window analysis
    print("\n" + "="*80)
    print("🔬 분석 중 (1.5초 윈도우)")
    print("="*80)

    window_size = 1.5
    hop_size = 1.5
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    results = []
    knocking_detections = 0
    total_segments = 0

    for start_sample in range(0, len(audio) - window_samples, hop_samples):
        end_sample = start_sample + window_samples
        segment = audio[start_sample:end_sample]

        # Skip if too quiet
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 0.01:
            continue

        total_segments += 1
        time_sec = start_sample / sr

        # YAMNet analysis
        classifications = classifier.classify_audio(segment, sr)
        yamnet_score, top_pred = get_engine_knocking_score(classifications)

        # Verified model prediction
        embedding = classifier.extract_embedding(segment, sr)
        X = scaler.transform([embedding])
        prediction = model.predict(X)[0]
        decision_score = model.decision_function(X)[0]

        # Isolation Forest: prediction = 1 (inlier) = matches knocking pattern
        #                   prediction = -1 (outlier) = doesn't match knocking
        is_knocking = (prediction == 1)
        if is_knocking:
            knocking_detections += 1

        results.append({
            'time': time_sec,
            'yamnet_score': yamnet_score,
            'top_pred': top_pred,
            'is_knocking': is_knocking,
            'decision_score': decision_score
        })

    # Display results
    print(f"\n시간   | YAMNet | Verified | Decision Score | Top 분류")
    print("-" * 80)

    for r in results:
        yamnet_str = f"{r['yamnet_score']:.0%}" if r['yamnet_score'] > 0 else "없음"
        status = "🚨" if r['is_knocking'] else "✅"

        print(f"{r['time']:5.1f}s | {yamnet_str:6} | {status:8} | "
              f"{r['decision_score']:7.3f}      | {r['top_pred']}")

    # Summary
    print("\n" + "="*80)
    print("📊 요약:")
    print("="*80)
    max_yamnet = max([r['yamnet_score'] for r in results]) if results else 0
    detection_rate = knocking_detections / total_segments if total_segments > 0 else 0

    print(f"총 세그먼트: {total_segments}개")
    print(f"YAMNet 최대 노킹 점수: {max_yamnet:.1%}")
    print(f"Verified 모델 노킹 감지: {knocking_detections}개 ({detection_rate:.1%})")

    if knocking_detections > 0:
        print("\n✅ 노킹 감지됨!")
    else:
        print("\n❌ 노킹 미감지")

    print("="*80)


if __name__ == "__main__":
    main()
