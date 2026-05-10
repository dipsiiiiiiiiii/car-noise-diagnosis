#!/usr/bin/env python3
"""
Step 2: Keras 모델 학습 (TensorFlow만 사용)
"""

import numpy as np
from pathlib import Path
import pickle

# TensorFlow/Keras import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(f"✅ TensorFlow {tf.__version__} 로드 완료")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def build_custom_model(input_dim: int):
    """커스텀 Dense Layer 구성"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout_1'),
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_2'),
        layers.Dense(32, activation='relu', name='dense_3'),
        layers.Dropout(0.2, name='dropout_3'),
        layers.Dense(2, activation='softmax', name='output')
    ], name='YAMNet_Custom_Classifier')

    return model


def main():
    print("="*80)
    print("🚗 Step 2: Keras 모델 학습")
    print("="*80)

    # Load embeddings
    embeddings_path = Path("data/embeddings_yamnet.npz")
    if not embeddings_path.exists():
        print(f"\n❌ 임베딩 파일이 없습니다: {embeddings_path}")
        print("먼저 step1_extract_embeddings.py를 실행하세요")
        return

    print(f"\n📂 임베딩 로드 중: {embeddings_path}")
    data = np.load(embeddings_path)
    X = data['X']
    y = data['y']
    print(f"✅ 로드 완료: {len(X)}개 샘플, {X.shape[1]}차원")

    # Split data
    print("\n" + "="*80)
    print("🎯 모델 학습")
    print("="*80)

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

    # Convert to categorical
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

    # Save model
    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    keras_model_path = output_dir / "car_classifier_yamnet_custom_keras.h5"
    model.save(keras_model_path)
    print(f"\n💾 Keras 모델 저장: {keras_model_path}")

    metadata_path = output_dir / "car_classifier_yamnet_custom.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'keras_model_path': str(keras_model_path),
            'model_type': 'yamnet_custom_dense',
            'info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'test_accuracy': test_acc,
                'description': 'YAMNet embedding + Custom Dense layers'
            }
        }, f)
    print(f"💾 메타데이터 저장: {metadata_path}")

    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print(f"📊 최종 결과:")
    print(f"   - 테스트 정확도: {test_acc:.1%}")
    print(f"   - 모델: {keras_model_path}")


if __name__ == "__main__":
    main()
