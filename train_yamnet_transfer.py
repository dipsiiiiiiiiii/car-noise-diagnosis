#!/usr/bin/env python3
"""
YAMNet Transfer Learning
- TensorFlow Hub YAMNet 사용
- 위에 Dense Layer 추가
- 정상/노킹 분류 학습
"""

import numpy as np
from pathlib import Path
import pickle
import librosa
import soundfile as sf

# TensorFlow
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

print(f"✅ TensorFlow {tf.__version__}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


class YAMNetFeatureExtractor:
    """YAMNet 특성 추출기"""

    def __init__(self):
        print("\n🤖 YAMNet 모델 로드 중...")
        # Load YAMNet from TensorFlow Hub
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("✅ YAMNet 로드 완료")

    def extract_embedding(self, audio_path: str):
        """오디오 파일에서 YAMNet embedding 추출"""
        # Load audio
        audio_data, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        # Ensure float32
        audio_data = audio_data.astype(np.float32)

        # YAMNet expects waveform in [-1.0, 1.0]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Get YAMNet embeddings
        # YAMNet returns: (scores, embeddings, spectrogram)
        # embeddings shape: (N, 1024) where N is number of frames
        scores, embeddings, spectrogram = self.yamnet_model(audio_data)

        # Average embeddings across time
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()

        return embedding_mean


def load_data(data_dirs: dict, extractor: YAMNetFeatureExtractor):
    """데이터 로딩 및 YAMNet embedding 추출"""
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
                embedding = extractor.extract_embedding(str(audio_file))
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
                embedding = extractor.extract_embedding(str(audio_file))
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
    print(f"   YAMNet 임베딩 차원: {X.shape[1]}D")

    return X, y


def build_model(input_dim: int):
    """YAMNet embedding 위에 커스텀 분류 레이어"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Dense layers
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu', name='dense_3'),
        layers.Dropout(0.2),

        # Output
        layers.Dense(2, activation='softmax', name='output')
    ], name='YAMNet_Transfer_Learning')

    return model


def train_model(X, y):
    """모델 학습"""
    print("\n" + "="*80)
    print("🎯 YAMNet Transfer Learning")
    print("="*80)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n분할:")
    print(f"   학습: {len(X_train)}개 (정상 {np.sum(y_train==0)}, 노킹 {np.sum(y_train==1)})")
    print(f"   테스트: {len(X_test)}개 (정상 {np.sum(y_test==0)}, 노킹 {np.sum(y_test==1)})")

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, 2)
    y_test_cat = keras.utils.to_categorical(y_test, 2)

    # Build model
    print(f"\n모델 구성:")
    model = build_model(input_dim=X.shape[1])
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    print(f"\n학습 시작...")
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*80)
    print("📊 평가 결과")
    print("="*80)

    results = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"\n테스트 Loss: {results[0]:.4f}")
    print(f"테스트 Accuracy: {results[1]:.1%}")
    print(f"테스트 Precision: {results[2]:.3f}")
    print(f"테스트 Recall: {results[3]:.3f}")

    # Predictions
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred,
                                target_names=['정상', '노킹'],
                                digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print("\n혼동 행렬:")
    print("              예측")
    print("            정상   노킹")
    print(f"  정상  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"  노킹  [{cm[1,0]:4d}  {cm[1,1]:4d}]")

    # Probability distribution
    print("\n확률 분포:")
    normal_probs = y_pred_probs[y_test == 0][:, 0]
    knocking_probs = y_pred_probs[y_test == 1][:, 1]
    print(f"   정상 샘플의 정상 확률: 평균 {normal_probs.mean():.1%}")
    print(f"   노킹 샘플의 노킹 확률: 평균 {knocking_probs.mean():.1%}")

    return model, scaler, results[1], history


def save_model(model, scaler, test_acc: float, n_samples: int, n_features: int):
    """모델 저장"""
    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras model
    model_path = output_dir / "yamnet_custom_model.keras"
    model.save(model_path)
    print(f"\n💾 Keras 모델 저장: {model_path}")

    # Save metadata
    metadata_path = output_dir / "yamnet_custom_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'model_path': str(model_path),
            'model_type': 'yamnet_transfer_learning',
            'info': {
                'n_samples': n_samples,
                'n_features': n_features,
                'test_accuracy': test_acc,
                'description': 'YAMNet (TensorFlow Hub) + Transfer Learning'
            }
        }, f)
    print(f"💾 메타데이터 저장: {metadata_path}")


def main():
    print("="*80)
    print("🚗 YAMNet Transfer Learning")
    print("   (TensorFlow Hub YAMNet + Custom Dense Layers)")
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

    # Extract YAMNet embeddings
    extractor = YAMNetFeatureExtractor()
    X, y = load_data(data_dirs, extractor)

    if len(X) < 50:
        print(f"\n⚠️  샘플 수가 너무 적습니다: {len(X)}개")
        return

    # Train model
    model, scaler, test_acc, history = train_model(X, y)

    # Save
    save_model(model, scaler, test_acc, len(X), X.shape[1])

    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print(f"📊 최종 결과:")
    print(f"   - 총 샘플: {len(X)}개")
    print(f"   - YAMNet 임베딩: {X.shape[1]}차원")
    print(f"   - 테스트 정확도: {test_acc:.1%}")
    print("="*80)


if __name__ == "__main__":
    main()
