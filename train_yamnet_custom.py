#!/usr/bin/env python3
"""
YAMNet 위에 커스텀 Dense Layer 학습
- YAMNet embedding 추출 (frozen)
- 위에 Dense Layer 추가하여 정상/노킹 분류
"""

import sys
import numpy as np
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader

# TensorFlow/Keras import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"✅ TensorFlow {tf.__version__} 로드 완료")
except ImportError:
    print("❌ TensorFlow가 설치되지 않았습니다.")
    print("설치: pip install tensorflow")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_yamnet_embeddings(data_dirs: dict, classifier: MediaPipeAudioClassifier):
    """YAMNet embedding 추출"""
    X_list = []
    y_list = []
    file_paths = []

    print("\n" + "="*80)
    print("📂 데이터 로딩 및 YAMNet Embedding 추출")
    print("="*80)

    # Load normal samples (label=0)
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
                file_paths.append(str(audio_file))
                normal_count += 1

                if i % 50 == 0:
                    print(f"    [{i}/{len(audio_files)}] 처리 중...")

            except Exception as e:
                continue

    print(f"  ✅ 총 {normal_count}개 정상 샘플 로드")

    # Load knocking samples (label=1)
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
                file_paths.append(str(audio_file))
                knocking_count += 1

                if i % 50 == 0:
                    print(f"    [{i}/{len(audio_files)}] 처리 중...")

            except Exception as e:
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

    return X, y, file_paths


def build_custom_model(input_dim: int):
    """YAMNet embedding 위에 커스텀 Dense Layer 구성"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),

        # Dense layers
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout_1'),

        layers.Dense(64, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_2'),

        layers.Dense(32, activation='relu', name='dense_3'),
        layers.Dropout(0.2, name='dropout_3'),

        # Output layer (2 classes: normal, knocking)
        layers.Dense(2, activation='softmax', name='output')
    ], name='YAMNet_Custom_Classifier')

    return model


def train_model(X, y):
    """모델 학습"""
    print("\n" + "="*80)
    print("🎯 YAMNet 커스텀 모델 학습")
    print("="*80)

    # Split data
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

    # Convert labels to categorical (one-hot encoding)
    y_train_cat = keras.utils.to_categorical(y_train, 2)
    y_test_cat = keras.utils.to_categorical(y_test, 2)

    # Build model
    print(f"\n모델 구성:")
    model = build_custom_model(input_dim=X.shape[1])
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
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

    train_loss, train_acc = model.evaluate(X_train_scaled, y_train_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)

    print(f"\n학습 정확도: {train_acc:.1%}")
    print(f"테스트 정확도: {test_acc:.1%}")

    # Confusion matrix
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix

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

    return model, scaler, test_acc, history


def save_model(model, scaler, output_path: Path, n_samples: int,
               n_features: int, test_acc: float):
    """모델 저장 (Keras + Scaler)"""
    print(f"\n💾 모델 저장: {output_path}")

    # Save Keras model
    keras_model_path = output_path.parent / (output_path.stem + '_keras.h5')
    model.save(keras_model_path)
    print(f"  ✅ Keras 모델: {keras_model_path}")

    # Save scaler and metadata
    with open(output_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'keras_model_path': str(keras_model_path),
            'model_type': 'yamnet_custom_dense',
            'info': {
                'n_samples': n_samples,
                'n_features': n_features,
                'test_accuracy': test_acc,
                'description': 'YAMNet embedding + Custom Dense layers'
            }
        }, f)

    print(f"  ✅ Scaler 및 메타데이터: {output_path}")
    print("✅ 모델 저장 완료!")


def main():
    print("="*80)
    print("🚗 YAMNet 커스텀 Dense Layer 학습")
    print("   (YAMNet embedding + 커스텀 신경망)")
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

    # Output path
    output_path = Path("data/models/car_classifier_yamnet_custom.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Load data and extract embeddings
    X, y, file_paths = load_yamnet_embeddings(data_dirs, classifier)

    if len(X) < 50:
        print(f"\n⚠️  샘플 수가 너무 적습니다: {len(X)}개")
        return

    # Train
    model, scaler, test_acc, history = train_model(X, y)

    # Save
    save_model(model, scaler, output_path, len(X), X.shape[1], test_acc)

    # Summary
    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print(f"📊 최종 모델:")
    print(f"   - 총 샘플: {len(X)}개")
    print(f"   - 임베딩 차원: {X.shape[1]}D")
    print(f"   - 테스트 정확도: {test_acc:.1%}")
    print(f"   - 저장 위치: {output_path}")
    print(f"\n💡 사용 방법:")
    print(f"   main.py에서 이 모델을 로드하여 사용할 수 있습니다")
    print("="*80)


if __name__ == "__main__":
    main()
