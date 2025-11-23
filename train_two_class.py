#!/usr/bin/env python3
"""
Two-Class Binary Classifier for Engine Knocking Detection
정상 vs 노킹 이진 분류기 학습
"""

import sys
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def load_data(data_dirs: dict, classifier: MediaPipeAudioClassifier):
    """Load normal and knocking data

    Args:
        data_dirs: {'normal': [dir1, dir2, ...], 'knocking': [dir1, dir2, ...]}
        classifier: YAMNet classifier for embedding extraction

    Returns:
        X: Feature array
        y: Labels (0=normal, 1=knocking)
        file_paths: Source file paths
    """
    X_list = []
    y_list = []
    file_paths = []

    print("\n" + "="*80)
    print("📂 데이터 로딩")
    print("="*80)

    # Load normal samples (label=0)
    print("\n[정상 샘플 로딩]")
    normal_count = 0
    for data_dir in data_dirs.get('normal', []):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"  ⚠️  디렉토리 없음: {data_dir}")
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
                print(f"    ⚠️  오류: {audio_file.name} - {e}")
                continue

    print(f"  ✅ 총 {normal_count}개 정상 샘플 로드")

    # Load knocking samples (label=1)
    print("\n[노킹 샘플 로딩]")
    knocking_count = 0
    for data_dir in data_dirs.get('knocking', []):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"  ⚠️  디렉토리 없음: {data_dir}")
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
                print(f"    ⚠️  오류: {audio_file.name} - {e}")
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
    print(f"   특성 차원: {X.shape[1]}D (YAMNet 임베딩)")
    print(f"   비율: 정상 {normal_count/len(X)*100:.1f}% / 노킹 {knocking_count/len(X)*100:.1f}%")

    return X, y, file_paths


def train_model(X, y, model_type='random_forest'):
    """Train binary classifier

    Args:
        X: Feature array
        y: Labels
        model_type: 'random_forest' or 'svm'
    """
    print("\n" + "="*80)
    print(f"🎯 {model_type.upper()} 학습")
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

    # Train
    print(f"\n학습 중...")
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'svm':
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train_scaled, y_train)
    print("✅ 학습 완료")

    # Evaluate
    print("\n" + "="*80)
    print("📊 평가 결과")
    print("="*80)

    # Train accuracy
    train_pred = model.predict(X_train_scaled)
    train_acc = np.mean(train_pred == y_train)
    print(f"\n학습 정확도: {train_acc:.1%}")

    # Test accuracy
    y_pred = model.predict(X_test_scaled)
    test_acc = np.mean(y_pred == y_test)
    print(f"테스트 정확도: {test_acc:.1%}")

    # Classification report
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred,
                                target_names=['정상', '노킹'],
                                digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n혼동 행렬:")
    print("              예측")
    print("            정상   노킹")
    print(f"  정상  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"  노킹  [{cm[1,0]:4d}  {cm[1,1]:4d}]")

    # Probability distribution
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        print(f"\n확률 분포:")
        print(f"   노킹 확률 (노킹 샘플): 평균 {y_proba[y_test==1].mean():.3f}")
        print(f"   노킹 확률 (정상 샘플): 평균 {y_proba[y_test==0].mean():.3f}")

    return model, scaler, test_acc


def save_model(model, scaler, output_path: Path, n_samples: int,
               n_features: int, test_acc: float, model_type: str):
    """Save trained model"""
    print(f"\n💾 모델 저장: {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'model_type': 'two_class_binary',
            'info': {
                'n_samples': n_samples,
                'n_features': n_features,
                'test_accuracy': test_acc,
                'classifier_type': model_type,
                'description': 'Binary classifier for normal vs knocking'
            }
        }, f)

    print("✅ 모델 저장 완료!")


def main():
    print("="*80)
    print("🚗 Two-Class Binary Classifier 학습")
    print("   (정상 vs 노킹 이진 분류)")
    print("="*80)

    # Data directories (새로 정리된 구조)
    data_dirs = {
        'normal': [
            'data/training/raw/audioset/idling',           # AudioSet Idling 원본
            'data/training/manual_review/normal/2_verified',  # 수동 검수 완료 정상 구간
            'data/training/processed/normal',                # 증강된 정상 데이터
        ],
        'knocking': [
            'data/training/manual_review/knocking/2_verified',  # 수동 검수 완료 노킹 구간
            'data/training/processed/knocking',                 # 증강된 노킹 데이터
        ]
    }

    # Output path
    output_path = Path("data/models/car_classifier_binary.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # YAMNet path
    yamnet_model = Path("data/models/yamnet.tflite")
    if not yamnet_model.exists():
        print(f"\n❌ YAMNet 모델을 찾을 수 없습니다: {yamnet_model}")
        return

    # Check if at least one normal directory exists
    normal_exists = any(Path(d).exists() for d in data_dirs['normal'])
    if not normal_exists:
        print("\n❌ 정상 샘플 디렉토리를 찾을 수 없습니다!")
        print("\n먼저 정상 엔진 소리를 수집하세요:")
        print("  1. python download_audioset_data.py  # AudioSet에서 다운로드")
        print("  2. 또는 YouTube에서 직접 수집")
        return

    # Initialize YAMNet
    print("\n🤖 YAMNet 모델 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Load data
    X, y, file_paths = load_data(data_dirs, classifier)

    if len(X) < 50:
        print(f"\n⚠️  샘플 수가 너무 적습니다: {len(X)}개")
        print("최소 50개 이상의 샘플이 필요합니다.")
        return

    # Check class balance
    n_normal = np.sum(y == 0)
    n_knocking = np.sum(y == 1)

    if n_normal == 0 or n_knocking == 0:
        print("\n❌ 정상 또는 노킹 샘플이 없습니다!")
        print(f"   정상: {n_normal}개, 노킹: {n_knocking}개")
        return

    # Train
    model, scaler, test_acc = train_model(X, y, model_type='random_forest')

    # Save
    save_model(model, scaler, output_path, len(X), X.shape[1],
               test_acc, 'random_forest')

    # Summary
    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print(f"📊 최종 모델:")
    print(f"   - 정상 샘플: {n_normal}개")
    print(f"   - 노킹 샘플: {n_knocking}개")
    print(f"   - 테스트 정확도: {test_acc:.1%}")
    print(f"   - 저장 위치: {output_path}")
    print(f"\n💡 사용 방법:")
    print(f"   main.py를 실행하면 자동으로 이 모델을 로드합니다")
    print("="*80)


if __name__ == "__main__":
    main()
