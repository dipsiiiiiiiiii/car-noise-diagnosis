#!/usr/bin/env python3
"""
Batch Test Multiple YouTube Videos
여러 영상을 한번에 테스트하고 결과 비교
"""

import sys
import subprocess
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def download_video(url: str, output_path: Path) -> bool:
    """Download YouTube video as WAV"""
    try:
        cmd = [
            'yt-dlp', '-x', '--audio-format', 'wav',
            '-o', str(output_path.with_suffix('.%(ext)s')),
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return output_path.exists()
    except Exception as e:
        print(f"  ❌ 다운로드 실패: {e}")
        return False


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


def analyze_video(audio_path: Path, classifier, v2_model, v2_scaler) -> Dict:
    """Analyze single video"""

    # Load audio
    audio, sr = AudioFileLoader.load_audio(str(audio_path))
    duration = len(audio) / sr

    # Sliding window analysis
    window_size = 1.5
    hop_size = 1.5
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    max_yamnet_score = 0.0
    v2_knocking_count = 0
    total_segments = 0

    for start_sample in range(0, len(audio) - window_samples, hop_samples):
        end_sample = start_sample + window_samples
        segment = audio[start_sample:end_sample]

        # Skip if too quiet
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 0.01:
            continue

        total_segments += 1

        # YAMNet analysis
        classifications = classifier.classify_audio(segment, sr)
        knocking_score, _ = get_engine_knocking_score(classifications)
        max_yamnet_score = max(max_yamnet_score, knocking_score)

        # v2 model prediction
        embedding = classifier.extract_embedding(segment, sr)
        X_v2 = v2_scaler.transform([embedding])
        pred_v2 = v2_model.predict(X_v2)[0]

        if pred_v2 == -1:
            v2_knocking_count += 1

    return {
        'duration': duration,
        'total_segments': total_segments,
        'max_yamnet_score': max_yamnet_score,
        'v2_detections': v2_knocking_count,
        'v2_detection_rate': v2_knocking_count / total_segments if total_segments > 0 else 0
    }


def main():
    print("="*80)
    print("🔍 배치 영상 테스트 (9-23번)")
    print("="*80)

    # Read video URLs
    links_file = Path("youtube_links.txt")
    with open(links_file, 'r') as f:
        all_links = [line.strip() for line in f if line.strip()]

    # Get videos 9-23 (index 8-22)
    test_links = all_links[8:23]
    print(f"\n📊 총 {len(test_links)}개 영상 테스트")

    # Setup
    test_dir = Path("data/testing/batch_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    yamnet_model = Path("data/models/yamnet.tflite")
    v2_model_path = Path("data/models/car_classifier_oneclass_v2.pkl")

    # Load models
    print("\n🤖 모델 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )

    with open(v2_model_path, 'rb') as f:
        v2_data = pickle.load(f)
        v2_model = v2_data['model']
        v2_scaler = v2_data['scaler']

    print("✅ 모델 로드 완료")

    # Test each video
    results = []

    for i, url in enumerate(test_links, 9):
        print(f"\n{'='*80}")
        print(f"[{i}/23] 영상 테스트")
        print(f"{'='*80}")
        print(f"URL: {url}")

        # Download
        output_path = test_dir / f"video_{i:02d}.wav"

        if not output_path.exists():
            print("  📥 다운로드 중...")
            if not download_video(url, output_path):
                print("  ❌ 다운로드 실패 - 건너뜀")
                results.append({
                    'index': i,
                    'url': url,
                    'status': 'download_failed'
                })
                continue
        else:
            print("  ✅ 이미 다운로드됨")

        # Analyze
        print("  🔬 분석 중...")
        try:
            result = analyze_video(output_path, classifier, v2_model, v2_scaler)
            result['index'] = i
            result['url'] = url
            result['status'] = 'success'
            results.append(result)

            print(f"  📊 결과:")
            print(f"     길이: {result['duration']:.1f}초")
            print(f"     YAMNet 최대 노킹 점수: {result['max_yamnet_score']:.1%}")
            print(f"     v2 노킹 감지: {result['v2_detections']}/{result['total_segments']} "
                  f"({result['v2_detection_rate']:.1%})")

        except Exception as e:
            print(f"  ❌ 분석 실패: {e}")
            results.append({
                'index': i,
                'url': url,
                'status': 'analysis_failed',
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*80}")
    print("📊 전체 결과 요약")
    print(f"{'='*80}")
    print(f"\n{'번호':<4} {'길이':<6} {'YAMNet':<8} {'v2 감지':<10} {'상태':<15}")
    print("-"*80)

    for r in results:
        if r['status'] == 'success':
            print(f"{r['index']:<4} {r['duration']:5.1f}s "
                  f"{r['max_yamnet_score']:6.1%}   "
                  f"{r['v2_detections']:2}/{r['total_segments']:2} ({r['v2_detection_rate']:5.1%})   "
                  f"✅")
        else:
            print(f"{r['index']:<4} {'N/A':<6} {'N/A':<8} {'N/A':<10} "
                  f"❌ {r['status']}")

    # Statistics
    success_results = [r for r in results if r['status'] == 'success']

    if success_results:
        print(f"\n{'='*80}")
        print("📈 통계")
        print(f"{'='*80}")

        yamnet_detected = sum(1 for r in success_results if r['max_yamnet_score'] > 0.2)
        v2_detected = sum(1 for r in success_results if r['v2_detection_rate'] > 0.5)

        print(f"성공: {len(success_results)}/{len(results)}개")
        print(f"YAMNet 노킹 감지 (>20%): {yamnet_detected}개")
        print(f"v2 모델 노킹 감지 (>50%): {v2_detected}개")

    print("="*80)


if __name__ == "__main__":
    main()
