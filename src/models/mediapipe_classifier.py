import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
import soundfile as sf
import time


class MediaPipeAudioClassifier:
    def __init__(self, model_path: Optional[str] = None, max_results: int = 5, score_threshold: float = 0.0, use_fallback: bool = False):
        """Initialize MediaPipe Audio Classifier
        
        Args:
            model_path: Path to custom model. If None, uses default YAMNet model
            max_results: Maximum number of classification results
            score_threshold: Minimum score threshold for results
        """
        try:
            # Verify model file exists
            if model_path and not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            self.base_options = python.BaseOptions(
                model_asset_path=model_path if model_path else None
            )
            
            self.options = audio.AudioClassifierOptions(
                base_options=self.base_options,
                running_mode=audio.RunningMode.AUDIO_CLIPS,
                max_results=max_results,
                score_threshold=score_threshold
            )
            
            print("MediaPipe YAMNet 모델 로딩 중...")
            print(f"모델 파일: {model_path}")
            print(f"최대 결과: {max_results}, 임계값: {score_threshold}")
            
            self.classifier = audio.AudioClassifier.create_from_options(self.options)
            print("✅ 모델 로딩 완료")
            self.latest_result = None
            
        except Exception as e:
            print(f"❌ MediaPipe 초기화 오류: {e}")
            raise
        
    def classify_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """Classify audio data using MediaPipe official approach
        
        Args:
            audio_data: Audio data as numpy array (float32, range [-1, 1])
            sample_rate: Sample rate of audio data
            
        Returns:
            List of classification results with categories and scores
        """
        try:
            print(f"🔧 classify_audio 시작: 데이터 길이 {len(audio_data) if audio_data is not None else 'None'}")
            
            # Validate input data
            if audio_data is None or len(audio_data) == 0:
                print("❌ 빈 오디오 데이터")
                return []
            
            print(f"원본 데이터: dtype={audio_data.dtype}, shape={audio_data.shape}, range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
            
            # Convert to the format expected by MediaPipe
            if audio_data.dtype != np.float32:
                print("데이터 타입을 float32로 변환")
                audio_data = audio_data.astype(np.float32)
                
            # Ensure audio is in [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                print(f"정규화: {max_val:.3f} -> 1.0")
                audio_data = audio_data / max_val
            elif max_val == 0:
                print("❌ 무음 오디오 데이터")
                return []
            
            # Ensure minimum length and reasonable maximum
            original_len = len(audio_data)
            if len(audio_data) < 1024:
                audio_data = np.pad(audio_data, (0, 1024 - len(audio_data)))
                print(f"패딩: {original_len} -> {len(audio_data)}")
            elif len(audio_data) > 16000 * 10:  # Limit to 10 seconds max
                audio_data = audio_data[:16000 * 10]
                print(f"자르기: {original_len} -> {len(audio_data)}")
            
            # Validate sample rate
            if sample_rate != 16000:
                print(f"⚠️  샘플 레이트: {sample_rate}Hz (YAMNet 최적화: 16kHz)")
            
            # Ensure the audio data is properly formatted for MediaPipe
            # YAMNet expects mono audio
            if len(audio_data.shape) > 1:
                print(f"스테레오 -> 모노: {audio_data.shape}")
                audio_data = np.mean(audio_data, axis=1)
            
            print(f"최종 데이터: shape={audio_data.shape}, range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
            
            # Create AudioData using official MediaPipe containers
            print("AudioData 컨테이너 생성 중...")
            AudioData = mp.tasks.components.containers.AudioData
            
            print(f"create_from_array 호출: data.shape={audio_data.shape}, sr={sample_rate}")
            audio_container = AudioData.create_from_array(
                audio_data, sample_rate
            )
            print("✅ AudioData 컨테이너 생성 완료")
            
            # Run classification with error handling
            print("🤖 분류 실행 중...")
            
            # Try to prevent segfault by ensuring data is continuous and aligned
            audio_data_copy = np.ascontiguousarray(audio_data.copy())
            print(f"연속 메모리 확보: {audio_data_copy.flags}")
            
            # Create new container with copy
            audio_container_safe = AudioData.create_from_array(
                audio_data_copy, sample_rate
            )
            
            result = self.classifier.classify(audio_container_safe)
            print("✅ 분류 완료")
            print(f"분류 결과 타입: {type(result)}")
            print(f"결과 속성: {dir(result) if result else 'None'}")
            
            # Process results
            classifications = []
            if result and hasattr(result, 'classifications') and result.classifications:
                print(f"분류 개수: {len(result.classifications)}")
                for i, classification in enumerate(result.classifications):
                    print(f"분류 {i}: {len(classification.categories)}개 카테고리")
            elif result:
                print(f"❌ 분류 결과가 빈 리스트입니다. 길이: {len(result)}")
                # MediaPipe 0.10.x에서는 다른 구조일 수 있음
                if isinstance(result, list) and len(result) > 0:
                    print(f"리스트 첫 번째 요소: {type(result[0])}")
                    print(f"리스트 첫 번째 요소 속성: {dir(result[0])}")
            else:
                print("❌ 분류 결과가 None입니다")
                
            if result and hasattr(result, 'classifications') and result.classifications:
                for classification in result.classifications:
                    categories = []
                    for category in classification.categories:
                        categories.append({
                            'category_name': category.category_name,
                            'score': category.score,
                            'display_name': getattr(category, 'display_name', category.category_name)
                        })
                    classifications.append({
                        'head_index': getattr(classification, 'head_index', 0),
                        'head_name': getattr(classification, 'head_name', 'default'),
                        'categories': categories
                    })
                    
            return classifications
            
        except Exception as e:
            print(f"MediaPipe 분류 오류: {e}")
            import traceback
            traceback.print_exc()
            # Return empty result on error
            return []
    
    def get_top_predictions(self, classifications: List[Dict], top_k: int = 5) -> List[Dict]:
        """Get top k predictions from classification results"""
        all_categories = []
        
        for classification in classifications:
            for category in classification['categories']:
                all_categories.append(category)
                
        # Sort by score and return top k
        all_categories.sort(key=lambda x: x['score'], reverse=True)
        return all_categories[:top_k]
    
    def filter_vehicle_sounds(self, classifications: List[Dict]) -> List[Dict]:
        """Filter for vehicle-related sound categories"""
        vehicle_keywords = [
            'car', 'vehicle', 'engine', 'motor', 'brake', 'horn', 'tire',
            'exhaust', 'diesel', 'truck', 'motorcycle', 'scooter', 'bus',
            'traffic', 'road', 'highway', 'automotive', 'mechanical'
        ]
        
        vehicle_sounds = []
        all_categories = []
        
        for classification in classifications:
            for category in classification['categories']:
                all_categories.append(category)
                
        for category in all_categories:
            category_name = category['category_name'].lower()
            display_name = category.get('display_name', '').lower()
            
            if any(keyword in category_name or keyword in display_name 
                   for keyword in vehicle_keywords):
                vehicle_sounds.append(category)
                
        return sorted(vehicle_sounds, key=lambda x: x['score'], reverse=True)


class AudioPreprocessor:
    """Preprocessing utilities for car audio analysis"""
    
    @staticmethod
    def extract_features(audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Extract audio features for car noise analysis"""
        import librosa
        
        features = {}
        
        # Basic statistics
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
        return features
    
    @staticmethod
    def filter_background_noise(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Background noise filtering for mixed audio environments"""
        import librosa
        
        # Apply noise reduction using spectral gating
        # This helps separate mechanical sounds from human voice
        
        # Get STFT
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        # Estimate noise floor (assume first 0.5 seconds contain some noise)
        noise_samples = int(0.5 * sample_rate)
        if len(audio_data) > noise_samples:
            noise_data = audio_data[:noise_samples]
            noise_stft = librosa.stft(noise_data)
            noise_magnitude = np.abs(noise_stft)
            noise_floor = np.mean(noise_magnitude, axis=1, keepdims=True)
        else:
            noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
        
        # Spectral gating: attenuate frequencies below threshold
        gate_ratio = 2.0  # Signal must be 2x above noise floor
        mask = magnitude > (noise_floor * gate_ratio)
        
        # Apply soft masking to preserve some natural sound
        soft_mask = mask.astype(float) * 0.8 + 0.2
        filtered_stft = stft * soft_mask
        
        # Convert back to time domain
        filtered_audio = librosa.istft(filtered_stft)
        
        return filtered_audio.astype(np.float32)
    
    @staticmethod
    def detect_voice_activity(audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Detect voice activity to identify mixed audio"""
        import librosa
        
        # Voice typically has energy in 300-3400 Hz range
        # Mechanical sounds are more in 20-200 Hz and 2000+ Hz
        
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Define frequency bands
        voice_mask = (freqs >= 300) & (freqs <= 3400)
        mechanical_low_mask = (freqs >= 20) & (freqs <= 200)
        mechanical_high_mask = (freqs >= 2000) & (freqs <= 8000)
        
        voice_energy = np.sum(magnitude[voice_mask])
        mechanical_low_energy = np.sum(magnitude[mechanical_low_mask])
        mechanical_high_energy = np.sum(magnitude[mechanical_high_mask])
        total_energy = voice_energy + mechanical_low_energy + mechanical_high_energy
        
        if total_energy > 0:
            voice_ratio = voice_energy / total_energy
            mechanical_ratio = (mechanical_low_energy + mechanical_high_energy) / total_energy
        else:
            voice_ratio = 0
            mechanical_ratio = 0
        
        # Additional voice features
        # Voice has more variation in spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        centroid_variation = np.std(spectral_centroid)
        
        return {
            'voice_ratio': voice_ratio,
            'mechanical_ratio': mechanical_ratio,
            'voice_detected': voice_ratio > 0.3,  # Threshold for voice presence
            'centroid_variation': centroid_variation,
            'audio_type': 'mixed' if voice_ratio > 0.2 and mechanical_ratio > 0.2 else 
                         'voice_dominant' if voice_ratio > 0.5 else 'mechanical_dominant'
        }
    
    @staticmethod
    def detect_engine_patterns(audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Detect engine-specific patterns"""
        import librosa
        
        # Get frequency spectrum
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Focus on engine frequency ranges
        # Typical car engine: 20-200 Hz (idle), 200-2000 Hz (acceleration)
        low_freq_mask = (freqs >= 20) & (freqs <= 200)
        mid_freq_mask = (freqs >= 200) & (freqs <= 2000)
        high_freq_mask = (freqs >= 2000) & (freqs <= 8000)
        
        patterns = {
            'low_freq_energy': np.sum(magnitude[low_freq_mask]),
            'mid_freq_energy': np.sum(magnitude[mid_freq_mask]),
            'high_freq_energy': np.sum(magnitude[high_freq_mask]),
            'dominant_frequency': freqs[np.argmax(magnitude[freqs > 0])],
        }
        
        # Detect potential issues based on frequency distribution
        total_energy = patterns['low_freq_energy'] + patterns['mid_freq_energy'] + patterns['high_freq_energy']
        if total_energy > 0:
            patterns['low_freq_ratio'] = patterns['low_freq_energy'] / total_energy
            patterns['mid_freq_ratio'] = patterns['mid_freq_energy'] / total_energy
            patterns['high_freq_ratio'] = patterns['high_freq_energy'] / total_energy
        
        return patterns