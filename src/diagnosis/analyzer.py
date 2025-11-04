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


# Label mapping for training data
PROBLEM_LABELS = {
    'normal': 0,
    'engine_problem': 1,
    'brake_problem': 2,
    'bearing_problem': 3,
    'belt_problem': 4,
    'tire_problem': 5,
    'transmission_problem': 6
}

LABEL_TO_NAME = {v: k for k, v in PROBLEM_LABELS.items()}

LABEL_TO_PART = {
    0: (CarPart.UNKNOWN, CarPartStatus.NORMAL, "정상 작동 중"),
    1: (CarPart.ENGINE, CarPartStatus.WARNING, "엔진 이상 감지"),
    2: (CarPart.BRAKE, CarPartStatus.CRITICAL, "브레이크 문제 감지"),
    3: (CarPart.BEARING, CarPartStatus.WARNING, "베어링 마모 감지"),
    4: (CarPart.BELT, CarPartStatus.WARNING, "벨트 이상 감지"),
    5: (CarPart.TIRE, CarPartStatus.WARNING, "타이어 문제 감지"),
    6: (CarPart.TRANSMISSION, CarPartStatus.WARNING, "변속기 이상 감지"),
}


class CarNoiseDiagnoser:
    """Data-driven car noise diagnoser

    Supports two modes:
    1. Baseline: Uses YAMNet vehicle sound detection (rule-free)
    2. Custom: Uses trained classifier on YAMNet embeddings
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize diagnoser

        Args:
            model_path: Path to trained classifier model. If None, uses baseline only.
        """
        self.model_path = model_path
        self.custom_classifier = self._load_classifier()
        self.mode = "custom" if self.custom_classifier else "baseline"

    def _load_classifier(self):
        """Load trained classifier or return None"""
        if self.model_path and Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    classifier = pickle.load(f)
                print(f"✅ 학습된 모델 로드됨: {self.model_path}")
                return classifier
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}")
                print("Baseline 모드를 사용합니다.")
        else:
            print("⚠️  학습된 모델이 없습니다. Baseline 모드 (YAMNet만 사용)")

        return None

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
            prediction = self.custom_classifier.predict(embedding_2d)[0]

            # Get probability scores if available
            if hasattr(self.custom_classifier, 'predict_proba'):
                probabilities = self.custom_classifier.predict_proba(embedding_2d)[0]
                confidence = float(probabilities[prediction])
            else:
                confidence = 0.7  # Default confidence

            diagnosis['prediction'] = int(prediction)
            diagnosis['confidence'] = confidence

            # Map prediction to diagnosis
            part, status, description = LABEL_TO_PART.get(
                prediction,
                (CarPart.UNKNOWN, CarPartStatus.NORMAL, "분류 불가")
            )

            diagnosis['overall_status'] = status

            # Add detected sounds from YAMNet for reference
            vehicle_sounds = self._filter_vehicle_sounds(mediapipe_results)
            diagnosis['detected_sounds'] = vehicle_sounds

            # Create issue if not normal
            if prediction != 0:  # 0 is normal
                diagnosis['issues'].append({
                    'part': part.value,
                    'status': status.value,
                    'description': f"{description} (Custom Model: {confidence:.1%})",
                    'confidence': confidence
                })

            # Generate recommendations
            diagnosis['recommendations'] = self._generate_recommendations(diagnosis['issues'])

            if prediction == 0:
                diagnosis['recommendations'].append("정상 작동 중입니다. 문제가 감지되지 않았습니다.")

        except Exception as e:
            print(f"Custom model 예측 오류: {e}")
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
