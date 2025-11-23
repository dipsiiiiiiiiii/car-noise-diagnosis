#!/usr/bin/env python3
"""
Train custom classifier on collected car noise data

Usage:
    python train.py
    python train.py --data-dir data/training --output data/models/car_classifier.pkl
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader
from diagnosis.analyzer import PROBLEM_LABELS, LABEL_TO_NAME


def load_training_data(data_dir: Path, classifier: MediaPipeAudioClassifier) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all training data and extract embeddings

    Args:
        data_dir: Directory containing subdirectories for each class
        classifier: MediaPipe classifier for embedding extraction

    Returns:
        X: Feature embeddings (n_samples, n_features)
        y: Labels (n_samples,)
        file_paths: List of file paths for reference
    """
    X_list = []
    y_list = []
    file_paths = []

    print(f"\n📂 데이터 로딩 중: {data_dir}")
    print("=" * 60)

    # Supported audio formats
    audio_extensions = ['*.wav', '*.mp3', '*.mp4', '*.m4a', '*.flac', '*.ogg']

    for label_name, label_id in PROBLEM_LABELS.items():
        class_dir = data_dir / label_name

        if not class_dir.exists():
            print(f"⚠️  폴더 없음: {class_dir}")
            continue

        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(class_dir.glob(ext))

        if not audio_files:
            print(f"⚠️  {label_name}: 오디오 파일 없음")
            continue

        print(f"\n🔊 {label_name} ({label_id}): {len(audio_files)}개 파일")

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
                y_list.append(label_id)
                file_paths.append(str(audio_file))

                print(f"  ✅ [{i}/{len(audio_files)}] {audio_file.name} (특성: {len(embedding)}차원)")

            except Exception as e:
                print(f"  ❌ [{i}/{len(audio_files)}] 오류: {audio_file.name} - {e}")
                continue

    if not X_list:
        raise ValueError("데이터를 로드할 수 없습니다. data/training/ 폴더에 오디오 파일을 추가하세요.")

    X = np.array(X_list)
    y = np.array(y_list)

    print("\n" + "=" * 60)
    print(f"✅ 총 {len(X)}개 샘플 로드 완료")
    print(f"   특성 차원: {X.shape[1]}")
    print(f"   클래스 분포:")
    for label_id in np.unique(y):
        count = np.sum(y == label_id)
        label_name = LABEL_TO_NAME.get(label_id, f"unknown_{label_id}")
        print(f"     - {label_name}: {count}개 ({count/len(y)*100:.1f}%)")

    return X, y, file_paths


def train_classifier(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest classifier

    Args:
        X: Feature embeddings
        y: Labels

    Returns:
        Trained classifier
    """
    print("\n🎯 모델 학습 중...")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"훈련 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("\nRandom Forest 학습 중...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✅ 학습 완료 ({training_time:.1f}초)")

    # Evaluate
    print("\n📊 모델 평가:")
    print("-" * 60)

    # Training accuracy
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"훈련 정확도: {train_acc:.2%}")

    # Test accuracy
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"테스트 정확도: {test_acc:.2%}")

    # Cross-validation
    print("\n5-Fold Cross Validation 중...")
    cv_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print(f"CV 평균 정확도: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")

    # Classification report
    print("\n분류 보고서 (테스트 데이터):")
    print("-" * 60)
    target_names = [LABEL_TO_NAME.get(i, f"class_{i}") for i in sorted(np.unique(y))]
    print(classification_report(y_test, test_pred, target_names=target_names, zero_division=0))

    # Confusion matrix
    print("혼동 행렬:")
    print("-" * 60)
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    # Feature importance (top 10)
    print("\n중요 특성 (Top 10):")
    print("-" * 60)
    feature_importance = clf.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2}. Feature {idx:3}: {feature_importance[idx]:.4f}")

    return clf


def main():
    parser = argparse.ArgumentParser(description="Train car noise classifier")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/training',
        help='Directory containing training data (default: data/training)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/models/car_classifier.pkl',
        help='Output model path (default: data/models/car_classifier.pkl)'
    )
    parser.add_argument(
        '--yamnet-model',
        type=str,
        default='data/models/yamnet.tflite',
        help='YAMNet model path (default: data/models/yamnet.tflite)'
    )

    args = parser.parse_args()

    print("🚗 자동차 소음 분류기 학습")
    print("=" * 60)

    # Check paths
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    yamnet_path = Path(args.yamnet_model)

    if not data_dir.exists():
        print(f"❌ 데이터 디렉토리가 없습니다: {data_dir}")
        print(f"다음 구조로 데이터를 준비하세요:")
        print(f"  {data_dir}/")
        print(f"    ├── normal/")
        print(f"    ├── engine_problem/")
        print(f"    ├── brake_problem/")
        print(f"    └── ...")
        return 1

    if not yamnet_path.exists():
        print(f"❌ YAMNet 모델이 없습니다: {yamnet_path}")
        print("다음 명령어로 다운로드하세요:")
        print(f"curl -L 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite' -o {yamnet_path}")
        return 1

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize classifier
    print(f"\n🤖 YAMNet 모델 로딩: {yamnet_path}")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_path),
        max_results=50,
        score_threshold=0.0
    )

    # Load data
    try:
        X, y, file_paths = load_training_data(data_dir, classifier)
    except ValueError as e:
        print(f"\n❌ {e}")
        return 1

    # Check minimum samples
    if len(X) < 10:
        print(f"\n❌ 데이터가 너무 적습니다 ({len(X)}개). 최소 10개 이상 필요합니다.")
        return 1

    # Train model
    trained_model = train_classifier(X, y)

    # Save model
    print(f"\n💾 모델 저장 중: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(trained_model, f)

    print(f"✅ 모델 저장 완료!")
    print("\n사용 방법:")
    print("  python main.py")
    print("  → 저장된 모델이 자동으로 로드됩니다.")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
