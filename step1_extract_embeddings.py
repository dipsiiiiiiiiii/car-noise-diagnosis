#!/usr/bin/env python3
"""
Step 1: YAMNet Embedding 추출 (MediaPipe만 사용)
"""

import sys
import numpy as np
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def load_yamnet_embeddings(data_dirs: dict, classifier: MediaPipeAudioClassifier):
    """YAMNet embedding 추출"""
    X_list = []
    y_list = []

    print("\n" + "="*80)
    print("📂 데이터 로딩 및 YAMNet Embedding 추출")
    print("="*80)

    # Load normal samples
    print("\n[정상 샘플 로딩]")
    normal_count = 0
    for data_dir in data_dirs.get('normal', []):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            continue

        audio_files = sorted(data_dir.glob("*.wav"))
        print(f"  📁 {data_dir.name}: {len(audio_files)}개 파일")

        for i, audio_file in enumerate(audio_files, 1):
            try:
                audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_file))
                if len(audio_data) == 0:
                    continue

                embedding = classifier.extract_embedding(audio_data, sample_rate)
                if embedding is None:
                    continue

                X_list.append(embedding)
                y_list.append(0)  # Normal
                normal_count += 1

                if i % 50 == 0:
                    print(f"    [{i}/{len(audio_files)}] 처리 중...")

            except Exception as e:
                print(f"    오류: {audio_file.name} - {e}")
                continue

    print(f"  ✅ 총 {normal_count}개 정상 샘플 로드")

    # Load knocking samples
    print("\n[노킹 샘플 로딩]")
    knocking_count = 0
    for data_dir in data_dirs.get('knocking', []):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            continue

        audio_files = sorted(data_dir.glob("*.wav"))
        print(f"  📁 {data_dir.name}: {len(audio_files)}개 파일")

        for i, audio_file in enumerate(audio_files, 1):
            try:
                audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_file))
                if len(audio_data) == 0:
                    continue

                embedding = classifier.extract_embedding(audio_data, sample_rate)
                if embedding is None:
                    continue

                X_list.append(embedding)
                y_list.append(1)  # Knocking
                knocking_count += 1

                if i % 50 == 0:
                    print(f"    [{i}/{len(audio_files)}] 처리 중...")

            except Exception as e:
                print(f"    오류: {audio_file.name} - {e}")
                continue

    print(f"  ✅ 총 {knocking_count}개 노킹 샘플 로드")

    X = np.array(X_list)
    y = np.array(y_list)

    print("\n" + "="*80)
    print(f"📊 데이터셋 요약")
    print("="*80)
    print(f"   정상: {normal_count}개 (클래스 0)")
    print(f"   노킹: {knocking_count}개 (클래스 1)")
    print(f"   총합: {len(X)}개")
    print(f"   임베딩 차원: {X.shape[1]}D")

    return X, y


def main():
    print("="*80)
    print("🚗 Step 1: YAMNet Embedding 추출")
    print("="*80)

    # Data directories
    data_dirs = {
        'normal': [
            'data/training/raw/audioset/idling',
            'data/training/manual_review/normal/2_verified',
            'data/training/processed/normal',
        ],
        'knocking': [
            'data/training/manual_review/knocking/2_verified',
            'data/training/processed/knocking',
        ]
    }

    # YAMNet path
    yamnet_model = Path("data/models/yamnet.tflite")
    if not yamnet_model.exists():
        print(f"\n❌ YAMNet 모델을 찾을 수 없습니다: {yamnet_model}")
        return

    # Initialize YAMNet
    print("\n🤖 YAMNet 모델 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Extract embeddings
    X, y = load_yamnet_embeddings(data_dirs, classifier)

    # Save embeddings
    output_path = Path("data/embeddings_yamnet.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, X=X, y=y)
    print(f"\n💾 임베딩 저장: {output_path}")
    print("✅ Step 1 완료!")
    print(f"\n다음 단계: python step2_train_keras.py")


if __name__ == "__main__":
    main()
