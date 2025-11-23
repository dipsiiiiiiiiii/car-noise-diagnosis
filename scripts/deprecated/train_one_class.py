#!/usr/bin/env python3
"""
Train One-Class Classifier for Engine Knocking Detection (Anomaly Detection)

This approach learns ONLY from knocking samples and detects anomalies.
Any sound that is NOT knocking will be classified as "normal".

Usage:
    python train_one_class.py
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader
from diagnosis.analyzer import PROBLEM_LABELS


def load_knocking_data(data_dir: Path, classifier: MediaPipeAudioClassifier) -> Tuple[np.ndarray, List[str]]:
    """Load knocking data only and extract embeddings

    Args:
        data_dir: Directory containing knocking audio segments
        classifier: MediaPipe classifier for embedding extraction

    Returns:
        X: Feature embeddings (n_samples, n_features)
        file_paths: List of file paths for reference
    """
    X_list = []
    file_paths = []

    print(f"\n📂 데이터 로딩 중: {data_dir}")
    print("=" * 60)

    # Supported audio formats
    audio_extensions = ['*.wav', '*.mp3', '*.mp4', '*.m4a', '*.flac', '*.ogg']

    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_dir.glob(ext))

    audio_files = sorted(audio_files)

    if not audio_files:
        raise ValueError(f"오디오 파일을 찾을 수 없습니다: {data_dir}")

    print(f"🔊 노킹 샘플: {len(audio_files)}개 파일")

    for i, audio_file in enumerate(audio_files, 1):
        try:
            # Load audio
            audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_file))

            if len(audio_data) == 0:
                print(f"  ⚠️  [{i}/{len(audio_files)}] 빈 파일: {audio_file.name}")
                continue

            # Extract embedding
            embedding = classifier.extract_embedding(audio_data, sample_rate)

            if embedding is None:
                print(f"  ⚠️  [{i}/{len(audio_files)}] 임베딩 추출 실패: {audio_file.name}")
                continue

            X_list.append(embedding)
            file_paths.append(str(audio_file))

            if i % 20 == 0 or i == len(audio_files):
                print(f"  ✅ [{i}/{len(audio_files)}] 처리 완료 (특성: {len(embedding)}차원)")

        except Exception as e:
            print(f"  ❌ [{i}/{len(audio_files)}] 오류: {audio_file.name} - {e}")
            continue

    if not X_list:
        raise ValueError("데이터를 로드할 수 없습니다.")

    X = np.array(X_list)

    print("\n" + "=" * 60)
    print(f"✅ 총 {len(X)}개 노킹 샘플 로드 완료")
    print(f"   특성 차원: {X.shape[1]}")

    return X, file_paths


def train_one_class_model(X: np.ndarray, model_type: str = 'isolation_forest') -> Tuple[object, StandardScaler]:
    """Train One-Class model

    Args:
        X: Feature embeddings from knocking samples only
        model_type: 'isolation_forest' or 'one_class_svm'

    Returns:
        Trained model and scaler
    """
    print(f"\n🎯 One-Class {model_type.upper()} 학습 중...")
    print("=" * 60)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == 'isolation_forest':
        # Isolation Forest
        # contamination: 예상 이상치 비율 (0.1 = 10%)
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # 노킹이 전체의 10%라고 가정
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'one_class_svm':
        # One-Class SVM
        # nu: 이상치 상한선 (0.1 = 최대 10%)
        model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fit model (learns what knocking looks like)
    model.fit(X_scaled)
    print("✅ 학습 완료")

    # Evaluate on training data
    predictions = model.predict(X_scaled)
    n_inliers = np.sum(predictions == 1)
    n_outliers = np.sum(predictions == -1)

    print(f"\n📊 학습 데이터 평가:")
    print(f"   - Inliers (노킹 패턴): {n_inliers}개 ({n_inliers/len(X)*100:.1f}%)")
    print(f"   - Outliers (노이즈): {n_outliers}개 ({n_outliers/len(X)*100:.1f}%)")

    # Decision scores (lower = more anomalous)
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_scaled)
        print(f"   - Decision score 범위: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"   - Decision score 평균: {scores.mean():.3f}")

    return model, scaler


def save_model(model, scaler, output_path: Path, model_info: dict):
    """Save trained model and scaler"""
    print(f"\n💾 모델 저장 중: {output_path}")

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'model_type': 'one_class',
            'info': model_info
        }, f)

    print("✅ 모델 저장 완료!")


def main():
    """Main training process"""
    print("=" * 80)
    print("🔊 One-Class 엔진 노킹 감지 모델 학습")
    print("   (Anomaly Detection - 노킹만 학습)")
    print("=" * 80)

    # Paths
    knocking_dir = Path("data/training/engine_knocking_segments_v2")  # 개선된 데이터
    output_path = Path("data/models/car_classifier_oneclass_v2.pkl")  # 새 모델
    yamnet_model = Path("data/models/yamnet.tflite")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check paths
    if not knocking_dir.exists():
        print(f"\n❌ 노킹 데이터를 찾을 수 없습니다: {knocking_dir}")
        print("먼저 python extract_knocking_segments.py를 실행하세요.")
        return

    if not yamnet_model.exists():
        print(f"\n❌ YAMNet 모델을 찾을 수 없습니다: {yamnet_model}")
        print("다음 명령으로 다운로드하세요:")
        print("curl -L 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite' -o data/models/yamnet.tflite")
        return

    # Initialize YAMNet classifier
    print(f"\n🤖 YAMNet 모델 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Load knocking data
    X, file_paths = load_knocking_data(knocking_dir, classifier)

    # Train model (try both and compare)
    print("\n" + "=" * 80)
    print("🎯 모델 학습")
    print("=" * 80)

    # Isolation Forest (recommended for this use case)
    model_if, scaler_if = train_one_class_model(X, model_type='isolation_forest')

    # Save model
    model_info = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'model_type': 'isolation_forest',
        'description': 'One-Class Isolation Forest for engine knocking detection'
    }

    save_model(model_if, scaler_if, output_path, model_info)

    # Summary
    print("\n" + "=" * 80)
    print("✅ 학습 완료!")
    print("=" * 80)
    print(f"📊 모델 정보:")
    print(f"   - 노킹 샘플: {len(X)}개")
    print(f"   - 특성 차원: {X.shape[1]}D")
    print(f"   - 모델 타입: Isolation Forest (One-Class)")
    print(f"   - 저장 위치: {output_path}")
    print(f"\n💡 사용 방법:")
    print(f"   1. python main.py 실행")
    print(f"   2. 시스템이 자동으로 One-Class 모델을 로드합니다")
    print(f"   3. 모든 소리 입력 가능 (음악, 말소리, 엔진 등)")
    print(f"   4. 노킹 패턴 감지 시 경고")
    print(f"\n✨ 특징:")
    print(f"   - 정상 데이터 불필요")
    print(f"   - 음악, 말소리, 정적 → 모두 '정상'으로 분류")
    print(f"   - 엔진 노킹만 '이상'으로 감지")
    print("=" * 80)


if __name__ == "__main__":
    main()
