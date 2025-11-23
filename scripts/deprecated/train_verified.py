#!/usr/bin/env python3
"""
Train One-Class Model with Manually Verified Knocking Data
수동 검수 완료된 노킹 데이터로 학습
"""

import sys
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def load_verified_data(data_dir: Path, classifier: MediaPipeAudioClassifier):
    """Load manually verified knocking data"""

    X_list = []
    file_paths = []

    print(f"\n📂 데이터 로딩 중: {data_dir}")
    print("="*60)

    # Find all audio files
    audio_files = sorted(data_dir.glob("*.wav"))

    if not audio_files:
        raise ValueError(f"검수된 파일을 찾을 수 없습니다: {data_dir}")

    print(f"🔊 검수 완료된 노킹 샘플: {len(audio_files)}개 파일")

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
                print(f"  ✅ [{i}/{len(audio_files)}] 처리 완료")

        except Exception as e:
            print(f"  ❌ [{i}/{len(audio_files)}] 오류: {audio_file.name} - {e}")
            continue

    if not X_list:
        raise ValueError("데이터를 로드할 수 없습니다.")

    X = np.array(X_list)

    print("\n" + "="*60)
    print(f"✅ 총 {len(X)}개 검수된 노킹 샘플 로드 완료")
    print(f"   특성 차원: {X.shape[1]}D (YAMNet 임베딩)")

    return X, file_paths


def train_model(X: np.ndarray):
    """Train One-Class Isolation Forest"""

    print(f"\n🎯 Isolation Forest 학습 중...")
    print("="*60)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # 10% contamination
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)
    print("✅ 학습 완료")

    # Evaluate on training data
    predictions = model.predict(X_scaled)
    n_inliers = np.sum(predictions == 1)
    n_outliers = np.sum(predictions == -1)

    print(f"\n📊 학습 데이터 평가:")
    print(f"   - Inliers (노킹 패턴): {n_inliers}개 ({n_inliers/len(X)*100:.1f}%)")
    print(f"   - Outliers (노이즈): {n_outliers}개 ({n_outliers/len(X)*100:.1f}%)")

    scores = model.decision_function(X_scaled)
    print(f"   - Decision score 범위: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"   - Decision score 평균: {scores.mean():.3f}")

    return model, scaler


def save_model(model, scaler, output_path: Path, n_samples: int, n_features: int):
    """Save trained model"""

    print(f"\n💾 모델 저장 중: {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'model_type': 'one_class_verified',
            'info': {
                'n_samples': n_samples,
                'n_features': n_features,
                'model_type': 'isolation_forest',
                'description': 'One-Class model trained on manually verified knocking data'
            }
        }, f)

    print("✅ 모델 저장 완료!")


def main():
    print("="*80)
    print("🔊 검수된 데이터로 One-Class 모델 학습")
    print("   (Manually Verified Knocking Data)")
    print("="*80)

    # Paths
    verified_dir = Path("data/training/manual_workflow/2_verified")
    output_path = Path("data/models/car_classifier_oneclass_verified.pkl")
    yamnet_model = Path("data/models/yamnet.tflite")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check paths
    if not verified_dir.exists():
        print(f"\n❌ 검수된 데이터를 찾을 수 없습니다: {verified_dir}")
        print("먼저 다음을 실행하세요:")
        print("  1. python extract_candidates.py  # 후보 추출")
        print("  2. python review_segments.py     # 수동 검수")
        return

    verified_files = list(verified_dir.glob("*.wav"))
    if len(verified_files) < 10:
        print(f"\n⚠️  검수된 샘플이 너무 적습니다: {len(verified_files)}개")
        print("최소 10개 이상의 샘플이 필요합니다.")
        print("python review_segments.py를 실행해서 더 많은 샘플을 검수하세요.")
        return

    if not yamnet_model.exists():
        print(f"\n❌ YAMNet 모델을 찾을 수 없습니다: {yamnet_model}")
        return

    # Initialize YAMNet
    print(f"\n🤖 YAMNet 모델 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Load verified data
    X, file_paths = load_verified_data(verified_dir, classifier)

    # Train model
    print("\n" + "="*80)
    print("🎯 모델 학습")
    print("="*80)
    model, scaler = train_model(X)

    # Save model
    save_model(model, scaler, output_path, len(X), X.shape[1])

    # Summary
    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print(f"📊 모델 정보:")
    print(f"   - 검수된 노킹 샘플: {len(X)}개")
    print(f"   - 특성 차원: {X.shape[1]}D")
    print(f"   - 모델 타입: Isolation Forest (One-Class)")
    print(f"   - 저장 위치: {output_path}")
    print(f"\n💡 사용 방법:")
    print(f"   1. main.py를 실행하면 자동으로 이 모델을 로드합니다")
    print(f"   2. 또는 test_video_segments.py에서 이 모델을 테스트하세요")
    print(f"\n✨ 특징:")
    print(f"   - 수동 검수된 고품질 노킹 데이터만 사용")
    print(f"   - 노이즈와 거짓 양성 최소화")
    print("="*80)


if __name__ == "__main__":
    main()
