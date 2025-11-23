import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import pickle
import json


class CarPartStatus(Enum):
    NORMAL = "정상"
    WARNING = "주의"
    CRITICAL = "위험"


class CarPart(Enum):
    ENGINE = "엔진"
    BRAKE = "브레이크"
    TRANSMISSION = "변속기"
    EXHAUST = "배기계통"
    SUSPENSION = "현가장치"
    BEARING = "베어링"
    BELT = "벨트"
    TIRE = "타이어"
    UNKNOWN = "알 수 없음"


# Label mapping for training data (2 classes only: normal and engine knocking)
PROBLEM_LABELS = {
    'normal': 0,
    'engine_knocking': 1
}

LABEL_TO_NAME = {v: k for k, v in PROBLEM_LABELS.items()}

LABEL_TO_PART = {
    0: (CarPart.UNKNOWN, CarPartStatus.NORMAL, "정상 작동 중"),
    1: (CarPart.ENGINE, CarPartStatus.WARNING, "엔진 노킹 감지")
}


class CarNoiseDiagnoser:
    """Data-driven car noise diagnoser

    Supports two modes:
    1. Baseline: Uses YAMNet vehicle sound detection (rule-free)
    2. Custom: Uses trained classifier on YAMNet embeddings
    """

    def __init__(self, model_path: Optional[str] = None, use_custom_model: bool = True):
        """Initialize diagnoser

        Args:
            model_path: Path to trained classifier model. If None, uses baseline only.
            use_custom_model: If True, use custom model (if available). If False, force baseline mode.
        """
        self.model_path = model_path
        self.use_custom_model = use_custom_model
        self.custom_classifier = self._load_classifier() if use_custom_model else None
        self.mode = "custom" if (self.custom_classifier and use_custom_model) else "baseline"

    def _load_classifier(self):
        """Load trained classifier or return None"""
        if self.model_path and Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Check if it's a dict with 'model' key
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.scaler = model_data.get('scaler')
                    classifier = model_data['model']
                    model_type = model_data.get('model_type', 'one_class')

                    # Check model type: 'binary' or 'two_class_binary' vs 'one_class'
                    if 'binary' in model_type.lower():
                        self.is_one_class = False
                        print(f"✅ Binary 모델 로드됨 (2-class classifier, type={model_type})")
                    else:
                        self.is_one_class = True
                        print(f"✅ One-Class 모델 로드됨 (엔진 노킹 감지 전용, type={model_type})")
                else:
                    # Regular classifier (no dict wrapper)
                    self.is_one_class = False
                    self.scaler = None
                    classifier = model_data
                    print(f"✅ 학습된 모델 로드됨")

                return classifier
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}")
                print("Baseline 모드를 사용합니다.")
        else:
            if self.use_custom_model:
                print("⚠️  학습된 모델이 없습니다. Baseline 모드 (YAMNet만 사용)")

        return None

    def switch_model(self, use_custom: bool):
        """Switch between custom model and baseline

        Args:
            use_custom: If True, switch to custom model. If False, switch to baseline.
        """
        self.use_custom_model = use_custom

        if use_custom and self.model_path:
            # Switch to custom model
            if not self.custom_classifier:
                self.custom_classifier = self._load_classifier()
            self.mode = "custom" if self.custom_classifier else "baseline"
        else:
            # Switch to baseline
            self.mode = "baseline"

        return self.mode

    def diagnose(self, audio_features: Dict, mediapipe_results: List[Dict],
                 embedding: Optional[np.ndarray] = None,
                 comparison_mode: bool = False) -> Dict:
        """Perform diagnosis using custom model or baseline

        Args:
            audio_features: Audio feature dict (for compatibility)
            mediapipe_results: YAMNet classification results
            embedding: Feature embedding for custom classifier
            comparison_mode: If True, return both baseline and custom results

        Returns:
            Diagnosis dictionary (or comparison dict if comparison_mode=True)
        """
        if comparison_mode and self.mode == "custom" and embedding is not None:
            # Run both models and return comparison
            return self._diagnose_comparison(embedding, mediapipe_results)
        elif self.mode == "custom" and embedding is not None:
            return self._custom_diagnose(embedding, mediapipe_results)
        else:
            return self._baseline_diagnose(mediapipe_results)

    def _baseline_diagnose(self, mediapipe_results: List[Dict]) -> Dict:
        """Baseline diagnosis using only YAMNet vehicle sound detection

        Simple heuristic: check if vehicle-related sounds are detected
        """
        diagnosis = {
            'mode': 'baseline',
            'overall_status': CarPartStatus.NORMAL,
            'issues': [],
            'recommendations': [],
            'confidence': 0.0,
            'detected_sounds': [],
            'prediction': None
        }

        # Filter vehicle sounds
        vehicle_sounds = self._filter_vehicle_sounds(mediapipe_results)
        diagnosis['detected_sounds'] = vehicle_sounds

        if not vehicle_sounds:
            # No vehicle sounds detected
            diagnosis['confidence'] = 0.3
            diagnosis['recommendations'].append("차량 관련 소리가 명확하게 감지되지 않았습니다.")
            return diagnosis

        # Simple baseline heuristic based on YAMNet categories
        # Look for specific problem keywords in detected sounds
        problem_keywords = {
            'squeal': (CarPart.BRAKE, CarPartStatus.CRITICAL, "브레이크 삐걱거림 감지"),
            'screech': (CarPart.BRAKE, CarPartStatus.CRITICAL, "브레이크 이상음 감지"),
            'knock': (CarPart.ENGINE, CarPartStatus.WARNING, "엔진 노킹 감지"),
            'rattle': (CarPart.UNKNOWN, CarPartStatus.WARNING, "기계적 잡음 감지"),
            'grinding': (CarPart.BEARING, CarPartStatus.WARNING, "마모음 감지"),
        }

        issues_found = []
        max_confidence = 0.0

        for sound in vehicle_sounds:
            sound_name = sound['sound_type'].lower()
            confidence = sound['confidence']
            max_confidence = max(max_confidence, confidence)

            for keyword, (part, status, description) in problem_keywords.items():
                if keyword in sound_name:
                    issues_found.append({
                        'part': part.value,
                        'status': status.value,
                        'description': f"{description} (YAMNet: {confidence:.1%})",
                        'confidence': confidence
                    })
                    break

        if issues_found:
            # Problems detected
            diagnosis['issues'] = issues_found

            # Determine overall status (CRITICAL > WARNING > NORMAL)
            if any(issue['status'] == CarPartStatus.CRITICAL.value for issue in issues_found):
                diagnosis['overall_status'] = CarPartStatus.CRITICAL
            elif any(issue['status'] == CarPartStatus.WARNING.value for issue in issues_found):
                diagnosis['overall_status'] = CarPartStatus.WARNING
            else:
                diagnosis['overall_status'] = CarPartStatus.NORMAL

            diagnosis['confidence'] = max_confidence
            diagnosis['recommendations'] = self._generate_recommendations(issues_found)
        else:
            # Vehicle sounds detected but no obvious problems
            diagnosis['overall_status'] = CarPartStatus.NORMAL
            diagnosis['confidence'] = max_confidence
            diagnosis['recommendations'].append(
                "차량 소리는 감지되었으나 명확한 문제는 발견되지 않았습니다."
            )

        return diagnosis

    def _custom_diagnose(self, embedding: np.ndarray, mediapipe_results: List[Dict]) -> Dict:
        """Custom diagnosis using trained classifier"""
        diagnosis = {
            'mode': 'custom',
            'overall_status': CarPartStatus.NORMAL,
            'issues': [],
            'recommendations': [],
            'confidence': 0.0,
            'detected_sounds': [],
            'prediction': None
        }

        try:
            # Predict using custom classifier
            embedding_2d = embedding.reshape(1, -1)

            # Check if this is a One-Class model
            if hasattr(self, 'is_one_class') and self.is_one_class:
                # One-Class Classification (Anomaly Detection)
                # Scale the embedding
                if self.scaler is not None:
                    embedding_2d = self.scaler.transform(embedding_2d)

                # Predict: +1 = inlier (knocking - learned pattern), -1 = outlier (not knocking)
                # NOTE: Since we trained ONLY on knocking samples, knocking is the "normal" pattern
                prediction = self.custom_classifier.predict(embedding_2d)[0]

                # Get anomaly score (higher = more similar to training data = more likely knocking)
                if hasattr(self.custom_classifier, 'decision_function'):
                    decision_score = self.custom_classifier.decision_function(embedding_2d)[0]

                    # Convert decision score to confidence
                    # prediction = +1 (inlier): Sound matches knocking pattern → KNOCKING DETECTED
                    # prediction = -1 (outlier): Sound doesn't match knocking → NORMAL
                    if prediction == 1:  # Inlier = Knocking pattern detected
                        # Higher score = more confident it's knocking
                        # Typical range: [0, 0.15], inliers are > 0
                        confidence = min(0.9, max(0.5, 0.5 + decision_score * 3))
                    else:  # Outlier = Not knocking (normal)
                        # Lower score = more confident it's NOT knocking
                        # Typical range: [-0.1, 0], outliers are < 0
                        confidence = min(0.9, max(0.5, 0.7 + decision_score))
                else:
                    confidence = 0.7  # Default confidence

                # Map to our label system: +1 (inlier/knocking) -> 1, -1 (outlier/normal) -> 0
                if prediction == 1:  # Inlier = matches knocking pattern
                    label_prediction = 1  # Engine knocking
                else:  # Outlier = doesn't match knocking pattern
                    label_prediction = 0  # Normal

                diagnosis['prediction'] = label_prediction
                diagnosis['confidence'] = confidence
                diagnosis['anomaly_score'] = decision_score if 'decision_score' in locals() else None

            else:
                # Regular multi-class or binary classifier
                # Scale the embedding if scaler exists
                if self.scaler is not None:
                    embedding_2d = self.scaler.transform(embedding_2d)

                # Get probability scores if available
                if hasattr(self.custom_classifier, 'predict_proba'):
                    probabilities = self.custom_classifier.predict_proba(embedding_2d)[0]

                    # Use custom threshold for knocking detection (reduce false positives)
                    KNOCKING_THRESHOLD = 0.75  # Require 75% confidence to detect knocking
                    prediction = 1 if probabilities[1] >= KNOCKING_THRESHOLD else 0
                    confidence = float(probabilities[prediction])
                else:
                    prediction = self.custom_classifier.predict(embedding_2d)[0]
                    confidence = 0.7  # Default confidence

                diagnosis['prediction'] = int(prediction)
                diagnosis['confidence'] = confidence
                diagnosis['probabilities'] = probabilities.tolist() if 'probabilities' in locals() else None

            # Map prediction to diagnosis
            label_prediction = diagnosis['prediction']
            part, status, description = LABEL_TO_PART.get(
                label_prediction,
                (CarPart.UNKNOWN, CarPartStatus.NORMAL, "분류 불가")
            )

            diagnosis['overall_status'] = status

            # Add detected sounds from YAMNet for reference
            vehicle_sounds = self._filter_vehicle_sounds(mediapipe_results)
            diagnosis['detected_sounds'] = vehicle_sounds

            # Create issue if not normal
            if label_prediction != 0:  # 0 is normal
                model_type = "One-Class" if (hasattr(self, 'is_one_class') and self.is_one_class) else "Custom"
                diagnosis['issues'].append({
                    'part': part.value,
                    'status': status.value,
                    'description': f"{description} ({model_type} Model: {confidence:.1%})",
                    'confidence': confidence
                })

            # Generate recommendations
            diagnosis['recommendations'] = self._generate_recommendations(diagnosis['issues'])

            if label_prediction == 0:
                diagnosis['recommendations'].append("정상 작동 중입니다. 문제가 감지되지 않았습니다.")

        except Exception as e:
            print(f"Custom model 예측 오류: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to baseline
            return self._baseline_diagnose(mediapipe_results)

        return diagnosis

    def _diagnose_comparison(self, embedding: np.ndarray, mediapipe_results: List[Dict]) -> Dict:
        """Run both baseline and custom models for comparison

        Returns:
            Comparison dictionary with both results
        """
        baseline_result = self._baseline_diagnose(mediapipe_results)
        custom_result = self._custom_diagnose(embedding, mediapipe_results)

        # Calculate improvement metrics
        confidence_diff = custom_result['confidence'] - baseline_result['confidence']

        # Check if predictions agree
        baseline_pred = None
        if baseline_result['issues']:
            baseline_pred = baseline_result['issues'][0]['part']

        custom_pred = None
        if custom_result.get('prediction') is not None:
            custom_pred = LABEL_TO_PART[custom_result['prediction']][0].value

        agreement = (baseline_pred == custom_pred) if (baseline_pred and custom_pred) else False

        return {
            'mode': 'comparison',
            'baseline': baseline_result,
            'custom': custom_result,
            'comparison_metrics': {
                'confidence_improvement': confidence_diff,
                'predictions_agree': agreement,
                'baseline_confidence': baseline_result['confidence'],
                'custom_confidence': custom_result['confidence']
            },
            'success': True
        }

    def _filter_vehicle_sounds(self, mediapipe_results: List[Dict]) -> List[Dict]:
        """Filter and categorize vehicle-related sounds"""
        vehicle_keywords = {
            CarPart.ENGINE: ['engine', 'motor', 'idle', 'rev', 'diesel'],
            CarPart.BRAKE: ['brake', 'squeal', 'screech'],
            CarPart.TIRE: ['tire', 'road', 'friction', 'skid'],
            CarPart.EXHAUST: ['exhaust', 'muffler'],
            CarPart.TRANSMISSION: ['gear', 'transmission'],
        }

        categorized_sounds = []

        for result in mediapipe_results:
            for category in result.get('categories', []):
                category_name = category['category_name'].lower()

                for part, keywords in vehicle_keywords.items():
                    if any(keyword in category_name for keyword in keywords):
                        categorized_sounds.append({
                            'part': part.value,
                            'sound_type': category['category_name'],
                            'confidence': category['score'],
                            'display_name': category.get('display_name', category['category_name'])
                        })
                        break

        return categorized_sounds

    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate maintenance recommendations based on detected issues"""
        recommendations = []

        if not issues:
            return recommendations

        critical_issues = [issue for issue in issues if issue['status'] == CarPartStatus.CRITICAL.value]
        warning_issues = [issue for issue in issues if issue['status'] == CarPartStatus.WARNING.value]

        if critical_issues:
            recommendations.append("🚨 즉시 정비소 방문이 필요합니다!")
            for issue in critical_issues:
                recommendations.append(f"- {issue['description']}")

        if warning_issues:
            recommendations.append("⚠️  가까운 시일 내 점검을 받으시기 바랍니다:")
            for issue in warning_issues:
                recommendations.append(f"- {issue['description']}")

        # Part-specific advice
        parts_mentioned = set(issue['part'] for issue in issues)
        if CarPart.ENGINE.value in parts_mentioned:
            recommendations.append("엔진 오일 교환 주기를 확인해보세요.")
        if CarPart.BRAKE.value in parts_mentioned:
            recommendations.append("브레이크 패드 두께를 점검해보세요.")
        if CarPart.TIRE.value in parts_mentioned:
            recommendations.append("타이어 공기압과 마모도를 확인해보세요.")

        return recommendations
