import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


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


class DiagnosisRule:
    def __init__(self, part: CarPart, condition: str, threshold: float, 
                 status: CarPartStatus, description: str):
        self.part = part
        self.condition = condition
        self.threshold = threshold
        self.status = status
        self.description = description


class CarNoiseDiagnoser:
    def __init__(self):
        self.diagnosis_rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[DiagnosisRule]:
        """Initialize diagnosis rules based on audio characteristics"""
        rules = [
            # Engine related issues
            DiagnosisRule(
                CarPart.ENGINE, "low_freq_ratio > 0.7", 0.7,
                CarPartStatus.WARNING, 
                "엔진 저주파 노이즈 증가 - 엔진 마운트나 내부 부품 점검 필요"
            ),
            DiagnosisRule(
                CarPart.ENGINE, "dominant_frequency < 50", 50,
                CarPartStatus.CRITICAL,
                "비정상적으로 낮은 주파수 - 엔진 심각한 문제 가능성"
            ),
            
            # High frequency issues (bearings, belts)
            DiagnosisRule(
                CarPart.BEARING, "high_freq_ratio > 0.4", 0.4,
                CarPartStatus.WARNING,
                "고주파 노이즈 증가 - 베어링 마모 가능성"
            ),
            DiagnosisRule(
                CarPart.BELT, "spectral_centroid_mean > 3000", 3000,
                CarPartStatus.WARNING,
                "높은 스펙트럴 중심 - 벨트 슬립이나 마모 가능성"
            ),
            
            # Brake related
            DiagnosisRule(
                CarPart.BRAKE, "dominant_frequency > 1000 and rms > 0.1", 1000,
                CarPartStatus.CRITICAL,
                "브레이크 관련 고주파 소음 - 즉시 점검 필요"
            ),
            
            # General noise level
            DiagnosisRule(
                CarPart.UNKNOWN, "rms > 0.3", 0.3,
                CarPartStatus.WARNING,
                "전체적인 소음 레벨 높음 - 종합 점검 권장"
            )
        ]
        return rules
    
    def diagnose(self, audio_features: Dict, mediapipe_results: List[Dict]) -> Dict:
        """Perform comprehensive diagnosis based on audio features and ML results"""
        diagnosis = {
            'overall_status': CarPartStatus.NORMAL,
            'issues': [],
            'recommendations': [],
            'confidence': 0.0,
            'detected_sounds': [],
            'part_analysis': {}
        }
        
        # Analyze MediaPipe results
        vehicle_sounds = self._filter_vehicle_sounds(mediapipe_results)
        diagnosis['detected_sounds'] = vehicle_sounds
        
        # Apply rule-based diagnosis
        triggered_rules = self._apply_rules(audio_features)
        
        # Combine results
        all_issues = []
        for rule in triggered_rules:
            issue = {
                'part': rule.part.value,
                'status': rule.status.value,
                'description': rule.description,
                'confidence': self._calculate_confidence(rule, audio_features)
            }
            all_issues.append(issue)
            
        # Sort by severity and confidence
        all_issues.sort(key=lambda x: (
            2 if x['status'] == CarPartStatus.CRITICAL.value else 1,
            -x['confidence']
        ))
        
        diagnosis['issues'] = all_issues
        
        # Determine overall status
        if any(issue['status'] == CarPartStatus.CRITICAL.value for issue in all_issues):
            diagnosis['overall_status'] = CarPartStatus.CRITICAL
        elif any(issue['status'] == CarPartStatus.WARNING.value for issue in all_issues):
            diagnosis['overall_status'] = CarPartStatus.WARNING
            
        # Generate recommendations
        diagnosis['recommendations'] = self._generate_recommendations(all_issues)
        diagnosis['confidence'] = self._calculate_overall_confidence(all_issues, vehicle_sounds)
        
        return diagnosis
    
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
    
    def _apply_rules(self, features: Dict) -> List[DiagnosisRule]:
        """Apply diagnosis rules to audio features"""
        triggered_rules = []
        
        for rule in self.diagnosis_rules:
            if self._evaluate_condition(rule.condition, features):
                triggered_rules.append(rule)
                
        return triggered_rules
    
    def _evaluate_condition(self, condition: str, features: Dict) -> bool:
        """Evaluate a rule condition against audio features"""
        try:
            # Create a safe evaluation context
            safe_dict = {
                'low_freq_ratio': features.get('low_freq_ratio', 0),
                'mid_freq_ratio': features.get('mid_freq_ratio', 0),
                'high_freq_ratio': features.get('high_freq_ratio', 0),
                'dominant_frequency': features.get('dominant_frequency', 0),
                'rms': features.get('rms', 0),
                'spectral_centroid_mean': features.get('spectral_centroid_mean', 0),
                'spectral_rolloff_mean': features.get('spectral_rolloff_mean', 0),
                'zero_crossing_rate': features.get('zero_crossing_rate', 0),
            }
            
            return eval(condition, {"__builtins__": {}}, safe_dict)
        except:
            return False
    
    def _calculate_confidence(self, rule: DiagnosisRule, features: Dict) -> float:
        """Calculate confidence score for a triggered rule"""
        base_confidence = 0.7
        
        # Adjust based on how much the threshold is exceeded
        if rule.condition in features:
            feature_value = features.get(rule.condition.split()[0], 0)
            if feature_value > rule.threshold:
                excess_ratio = (feature_value - rule.threshold) / rule.threshold
                confidence_boost = min(0.3, excess_ratio * 0.1)
                return min(1.0, base_confidence + confidence_boost)
                
        return base_confidence
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate maintenance recommendations based on detected issues"""
        recommendations = []
        
        if not issues:
            recommendations.append("현재 특별한 문제가 감지되지 않았습니다. 정기적인 점검을 계속 받으시기 바랍니다.")
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
                
        # General maintenance advice
        parts_mentioned = set(issue['part'] for issue in issues)
        if CarPart.ENGINE.value in parts_mentioned:
            recommendations.append("엔진 오일 교환 주기를 확인해보세요.")
        if CarPart.BRAKE.value in parts_mentioned:
            recommendations.append("브레이크 패드 두께를 점검해보세요.")
        if CarPart.TIRE.value in parts_mentioned:
            recommendations.append("타이어 공기압과 마모도를 확인해보세요.")
            
        return recommendations
    
    def _calculate_overall_confidence(self, issues: List[Dict], detected_sounds: List[Dict]) -> float:
        """Calculate overall diagnosis confidence"""
        if not issues and not detected_sounds:
            return 0.5  # Neutral confidence when nothing detected
            
        # Base confidence from issues
        if issues:
            issue_confidence = np.mean([issue['confidence'] for issue in issues])
        else:
            issue_confidence = 0.0
            
        # Boost confidence if vehicle sounds were clearly detected
        if detected_sounds:
            sound_confidence = np.mean([sound['confidence'] for sound in detected_sounds])
            detection_boost = min(0.3, sound_confidence)
        else:
            detection_boost = 0.0
            
        overall_confidence = min(1.0, issue_confidence + detection_boost)
        return overall_confidence