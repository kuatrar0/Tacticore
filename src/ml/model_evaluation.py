#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for CS2 Kill Analysis

This module provides advanced evaluation metrics specifically designed for 
CS2 kill classification models, going beyond simple accuracy to provide 
insights into model performance across different scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Tuple, Any, Optional
import pickle
from pathlib import Path
import json


class CS2ModelEvaluator:
    """
    Comprehensive evaluation suite for CS2 kill classification models.
    """
    
    def __init__(self, model, label_encoder, feature_names: List[str]):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained ML model
            label_encoder: Label encoder for class names
            feature_names: List of feature names used in training
        """
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.class_names = label_encoder.classes_
        
    def evaluate_comprehensive(self, X_test: np.ndarray, y_test: np.ndarray, 
                             test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels (encoded)
            test_data: Optional DataFrame with additional context data
            
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Basic metrics
        results['basic_metrics'] = self._calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        
        # Per-class metrics
        results['per_class_metrics'] = self._calculate_per_class_metrics(y_test, y_pred, y_pred_proba)
        
        # Confusion matrix analysis
        results['confusion_matrix'] = self._analyze_confusion_matrix(y_test, y_pred)
        
        # Feature importance
        results['feature_importance'] = self._analyze_feature_importance()
        
        # Cross-validation
        results['cross_validation'] = self._perform_cross_validation(X_test, y_test)
        
        # CS2-specific metrics
        if test_data is not None:
            results['cs2_specific'] = self._calculate_cs2_specific_metrics(
                y_test, y_pred, y_pred_proba, test_data
            )
        
        # Model confidence analysis
        results['confidence_analysis'] = self._analyze_prediction_confidence(y_pred_proba)
        
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add AUC for multi-class (one-vs-rest)
        try:
            if len(np.unique(y_true)) > 2:
                metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                metrics['auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
            else:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            metrics['auc_ovr'] = 0.0
            
        return metrics
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each class."""
        per_class = {}
        
        # Get classification report as dict
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if class_name in report:
                per_class[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support'],
                    'class_frequency': np.sum(y_true == i) / len(y_true)
                }
        
        return per_class
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix and common misclassifications."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Find most common misclassifications
        misclassifications = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm_normalized[i, j]) * 100
                    })
        
        # Sort by count
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'class_names': self.class_names.tolist(),
            'top_misclassifications': misclassifications[:10],
            'total_misclassifications': sum(cm[i, j] for i in range(len(cm)) for j in range(len(cm)) if i != j)
        }
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance."""
        importance_data = {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models (LightGBM, RandomForest)
                importances = self.model.feature_importances_
                
                feature_importance = []
                for i, importance in enumerate(importances):
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                    feature_importance.append({
                        'feature': feature_name,
                        'importance': float(importance),
                        'rank': 0  # Will be filled after sorting
                    })
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                
                # Add ranks
                for rank, feat in enumerate(feature_importance):
                    feat['rank'] = rank + 1
                
                importance_data = {
                    'available': True,
                    'features': feature_importance,
                    'top_features': feature_importance[:5],
                    'method': 'built_in_importance'
                }
            else:
                importance_data = {
                    'available': False,
                    'message': 'Model does not provide feature importance',
                    'features': [],
                    'method': 'none'
                }
                
        except Exception as e:
            importance_data = {
                'available': False,
                'error': str(e),
                'features': [],
                'method': 'error'
            }
        
        return importance_data
    
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation analysis."""
        try:
            # Stratified K-Fold to maintain class distribution
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Calculate CV scores for different metrics
            cv_results = {
                'accuracy': cross_val_score(self.model, X, y, cv=skf, scoring='accuracy'),
                'precision_macro': cross_val_score(self.model, X, y, cv=skf, scoring='precision_macro'),
                'recall_macro': cross_val_score(self.model, X, y, cv=skf, scoring='recall_macro'),
                'f1_macro': cross_val_score(self.model, X, y, cv=skf, scoring='f1_macro'),
            }
            
            # Calculate statistics
            cv_stats = {}
            for metric, scores in cv_results.items():
                cv_stats[metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'scores': scores.tolist()
                }
            
            return {
                'available': True,
                'folds': cv_folds,
                'metrics': cv_stats,
                'overall_stability': float(np.mean([cv_stats[m]['std'] for m in cv_stats]))
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def _calculate_cs2_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_pred_proba: np.ndarray, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate CS2-specific evaluation metrics."""
        cs2_metrics = {}
        
        # Distance-based performance
        if 'distance_xy' in test_data.columns:
            distance_ranges = [
                ('close', 0, 500),
                ('medium', 500, 1000), 
                ('long', 1000, float('inf'))
            ]
            
            distance_performance = {}
            for range_name, min_dist, max_dist in distance_ranges:
                mask = (test_data['distance_xy'] >= min_dist) & (test_data['distance_xy'] < max_dist)
                if mask.sum() > 0:
                    range_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                    distance_performance[range_name] = {
                        'accuracy': float(range_accuracy),
                        'sample_count': int(mask.sum()),
                        'percentage_of_total': float(mask.sum() / len(test_data) * 100)
                    }
            
            cs2_metrics['distance_performance'] = distance_performance
        
        # Weapon-based performance
        if 'weapon' in test_data.columns:
            weapon_performance = {}
            top_weapons = test_data['weapon'].value_counts().head(5)
            
            for weapon in top_weapons.index:
                mask = test_data['weapon'] == weapon
                if mask.sum() > 5:  # Only evaluate weapons with enough samples
                    weapon_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                    weapon_performance[weapon] = {
                        'accuracy': float(weapon_accuracy),
                        'sample_count': int(mask.sum())
                    }
            
            cs2_metrics['weapon_performance'] = weapon_performance
        
        # Round phase performance
        if 'time_in_round_s' in test_data.columns:
            phase_ranges = [
                ('early', 0, 30),
                ('mid', 30, 90),
                ('late', 90, float('inf'))
            ]
            
            phase_performance = {}
            for phase_name, min_time, max_time in phase_ranges:
                mask = (test_data['time_in_round_s'] >= min_time) & (test_data['time_in_round_s'] < max_time)
                if mask.sum() > 0:
                    phase_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                    phase_performance[phase_name] = {
                        'accuracy': float(phase_accuracy),
                        'sample_count': int(mask.sum())
                    }
            
            cs2_metrics['round_phase_performance'] = phase_performance
        
        # Headshot accuracy
        if 'headshot' in test_data.columns:
            headshot_mask = test_data['headshot'] == True
            bodyshot_mask = test_data['headshot'] == False
            
            cs2_metrics['headshot_performance'] = {
                'headshot_accuracy': float(accuracy_score(y_true[headshot_mask], y_pred[headshot_mask])) if headshot_mask.sum() > 0 else 0.0,
                'bodyshot_accuracy': float(accuracy_score(y_true[bodyshot_mask], y_pred[bodyshot_mask])) if bodyshot_mask.sum() > 0 else 0.0,
                'headshot_samples': int(headshot_mask.sum()),
                'bodyshot_samples': int(bodyshot_mask.sum())
            }
        
        return cs2_metrics
    
    def _analyze_prediction_confidence(self, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence patterns."""
        max_probas = np.max(y_pred_proba, axis=1)
        
        confidence_ranges = [
            ('very_low', 0.0, 0.4),
            ('low', 0.4, 0.6),
            ('medium', 0.6, 0.8),
            ('high', 0.8, 0.95),
            ('very_high', 0.95, 1.0)
        ]
        
        confidence_distribution = {}
        for range_name, min_conf, max_conf in confidence_ranges:
            mask = (max_probas >= min_conf) & (max_probas < max_conf)
            confidence_distribution[range_name] = {
                'count': int(mask.sum()),
                'percentage': float(mask.sum() / len(max_probas) * 100),
                'avg_confidence': float(np.mean(max_probas[mask])) if mask.sum() > 0 else 0.0
            }
        
        return {
            'overall_confidence': {
                'mean': float(np.mean(max_probas)),
                'std': float(np.std(max_probas)),
                'min': float(np.min(max_probas)),
                'max': float(np.max(max_probas))
            },
            'confidence_distribution': confidence_distribution,
            'low_confidence_threshold': 0.6,
            'low_confidence_predictions': int(np.sum(max_probas < 0.6)),
            'low_confidence_percentage': float(np.sum(max_probas < 0.6) / len(max_probas) * 100)
        }
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 output_dir: Path = None) -> str:
        """Generate a comprehensive evaluation report."""
        if output_dir is None:
            output_dir = Path("results/evaluation")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_lines = []
        report_lines.append("# CS2 Kill Classification Model Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Basic metrics
        basic = results['basic_metrics']
        report_lines.append("## Overall Performance Metrics")
        report_lines.append(f"- **Accuracy**: {basic['accuracy']:.3f} ({basic['accuracy']*100:.1f}%)")
        report_lines.append(f"- **Precision (Macro)**: {basic['precision_macro']:.3f}")
        report_lines.append(f"- **Recall (Macro)**: {basic['recall_macro']:.3f}")
        report_lines.append(f"- **F1-Score (Macro)**: {basic['f1_macro']:.3f}")
        report_lines.append(f"- **AUC (One-vs-Rest)**: {basic.get('auc_ovr', 'N/A'):.3f}" if isinstance(basic.get('auc_ovr'), float) else "- **AUC**: N/A")
        report_lines.append("")
        
        # Performance interpretation
        report_lines.append("## Performance Interpretation")
        accuracy = basic['accuracy']
        if accuracy >= 0.85:
            performance_level = "Excellent"
            interpretation = "Model shows excellent performance for CS2 kill classification."
        elif accuracy >= 0.75:
            performance_level = "Good"
            interpretation = "Model shows good performance with room for improvement."
        elif accuracy >= 0.65:
            performance_level = "Fair"
            interpretation = "Model shows fair performance. Consider feature engineering or more data."
        else:
            performance_level = "Poor"
            interpretation = "Model performance is poor. Significant improvements needed."
        
        report_lines.append(f"**Performance Level**: {performance_level}")
        report_lines.append(f"{interpretation}")
        report_lines.append("")
        
        # Per-class performance
        report_lines.append("## Per-Class Performance")
        per_class = results['per_class_metrics']
        
        for class_name, metrics in per_class.items():
            report_lines.append(f"### {class_name}")
            report_lines.append(f"- Precision: {metrics['precision']:.3f}")
            report_lines.append(f"- Recall: {metrics['recall']:.3f}")
            report_lines.append(f"- F1-Score: {metrics['f1_score']:.3f}")
            report_lines.append(f"- Support: {metrics['support']} samples ({metrics['class_frequency']*100:.1f}%)")
            report_lines.append("")
        
        # Feature importance
        if results['feature_importance']['available']:
            report_lines.append("## Feature Importance")
            features = results['feature_importance']['top_features']
            for feat in features:
                report_lines.append(f"- **{feat['feature']}**: {feat['importance']:.3f}")
            report_lines.append("")
        
        # Cross-validation
        if results['cross_validation']['available']:
            cv = results['cross_validation']
            report_lines.append("## Cross-Validation Results")
            report_lines.append(f"- **CV Accuracy**: {cv['metrics']['accuracy']['mean']:.3f} ± {cv['metrics']['accuracy']['std']:.3f}")
            report_lines.append(f"- **CV F1-Score**: {cv['metrics']['f1_macro']['mean']:.3f} ± {cv['metrics']['f1_macro']['std']:.3f}")
            report_lines.append(f"- **Model Stability**: {cv['overall_stability']:.3f} (lower is better)")
            report_lines.append("")
        
        # CS2-specific metrics
        if 'cs2_specific' in results:
            cs2 = results['cs2_specific']
            report_lines.append("## CS2-Specific Performance")
            
            if 'distance_performance' in cs2:
                report_lines.append("### Performance by Distance")
                for range_name, perf in cs2['distance_performance'].items():
                    report_lines.append(f"- **{range_name.title()} Range**: {perf['accuracy']:.3f} ({perf['sample_count']} samples)")
            
            if 'weapon_performance' in cs2:
                report_lines.append("### Performance by Weapon")
                for weapon, perf in cs2['weapon_performance'].items():
                    report_lines.append(f"- **{weapon}**: {perf['accuracy']:.3f} ({perf['sample_count']} samples)")
            
            if 'round_phase_performance' in cs2:
                report_lines.append("### Performance by Round Phase")
                for phase, perf in cs2['round_phase_performance'].items():
                    report_lines.append(f"- **{phase.title()} Game**: {perf['accuracy']:.3f} ({perf['sample_count']} samples)")
            
            report_lines.append("")
        
        # Confidence analysis
        conf = results['confidence_analysis']
        report_lines.append("## Prediction Confidence Analysis")
        report_lines.append(f"- **Average Confidence**: {conf['overall_confidence']['mean']:.3f}")
        report_lines.append(f"- **Low Confidence Predictions**: {conf['low_confidence_predictions']} ({conf['low_confidence_percentage']:.1f}%)")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        recommendations = self._generate_recommendations(results)
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = output_dir / "evaluation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save detailed results as JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Evaluation report saved to: {report_file}")
        print(f"Detailed results saved to: {results_file}")
        
        return report_content
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []
        
        basic = results['basic_metrics']
        
        # Overall performance recommendations
        if basic['accuracy'] < 0.7:
            recommendations.append("Consider collecting more training data or improving feature engineering")
        
        if basic['precision_macro'] < 0.7:
            recommendations.append("High false positive rate - consider adjusting classification thresholds")
        
        if basic['recall_macro'] < 0.7:
            recommendations.append("High false negative rate - model may be too conservative")
        
        # Class imbalance recommendations
        per_class = results['per_class_metrics']
        class_supports = [metrics['support'] for metrics in per_class.values()]
        max_support = max(class_supports)
        min_support = min(class_supports)
        
        if max_support / min_support > 5:
            recommendations.append("Significant class imbalance detected - consider balancing techniques (SMOTE, class weights)")
        
        # Feature importance recommendations
        if results['feature_importance']['available']:
            top_features = results['feature_importance']['top_features']
            if len(top_features) > 0 and top_features[0]['importance'] > 0.5:
                recommendations.append(f"Model heavily relies on '{top_features[0]['feature']}' - consider feature diversification")
        
        # Confidence recommendations
        conf = results['confidence_analysis']
        if conf['low_confidence_percentage'] > 30:
            recommendations.append("High percentage of low-confidence predictions - consider ensemble methods or threshold tuning")
        
        # Cross-validation recommendations
        if results['cross_validation']['available']:
            cv_std = results['cross_validation']['overall_stability']
            if cv_std > 0.05:
                recommendations.append("Model shows instability across folds - consider regularization or more data")
        
        # CS2-specific recommendations
        if 'cs2_specific' in results:
            cs2 = results['cs2_specific']
            
            if 'distance_performance' in cs2:
                dist_perfs = [perf['accuracy'] for perf in cs2['distance_performance'].values()]
                if max(dist_perfs) - min(dist_perfs) > 0.2:
                    recommendations.append("Performance varies significantly by distance - consider distance-specific features")
            
            if 'weapon_performance' in cs2:
                weapon_perfs = [perf['accuracy'] for perf in cs2['weapon_performance'].values()]
                if len(weapon_perfs) > 1 and max(weapon_perfs) - min(weapon_perfs) > 0.3:
                    recommendations.append("Performance varies by weapon type - consider weapon-specific models or features")
        
        if not recommendations:
            recommendations.append("Model performance looks good overall - consider A/B testing in production")
        
        return recommendations
    
    def plot_evaluation_results(self, results: Dict[str, Any], output_dir: Path = None):
        """Generate visualization plots for evaluation results."""
        if output_dir is None:
            output_dir = Path("results/evaluation")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(results['confusion_matrix'], output_dir)
        
        # 2. Per-class performance
        self._plot_per_class_performance(results['per_class_metrics'], output_dir)
        
        # 3. Feature importance
        if results['feature_importance']['available']:
            self._plot_feature_importance(results['feature_importance'], output_dir)
        
        # 4. Confidence distribution
        self._plot_confidence_distribution(results['confidence_analysis'], output_dir)
        
        # 5. CS2-specific plots
        if 'cs2_specific' in results:
            self._plot_cs2_specific(results['cs2_specific'], output_dir)
        
        print(f"Evaluation plots saved to: {output_dir}")
    
    def _plot_confusion_matrix(self, cm_data: Dict, output_dir: Path):
        """Plot confusion matrix."""
        cm = np.array(cm_data['confusion_matrix'])
        cm_norm = np.array(cm_data['confusion_matrix_normalized'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_performance(self, per_class_data: Dict, output_dir: Path):
        """Plot per-class performance metrics."""
        classes = list(per_class_data.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [per_class_data[cls][metric] for cls in classes]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, importance_data: Dict, output_dir: Path):
        """Plot feature importance."""
        features = importance_data['features'][:10]  # Top 10
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        feature_names = [f['feature'] for f in features]
        importances = [f['importance'] for f in features]
        
        y_pos = np.arange(len(feature_names))
        
        bars = ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 10 Feature Importances')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importances[i]:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, confidence_data: Dict, output_dir: Path):
        """Plot prediction confidence distribution."""
        dist_data = confidence_data['confidence_distribution']
        
        ranges = list(dist_data.keys())
        counts = [dist_data[r]['count'] for r in ranges]
        percentages = [dist_data[r]['percentage'] for r in ranges]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count distribution
        ax1.bar(ranges, counts)
        ax1.set_title('Prediction Confidence Distribution (Counts)')
        ax1.set_xlabel('Confidence Range')
        ax1.set_ylabel('Number of Predictions')
        ax1.tick_params(axis='x', rotation=45)
        
        # Percentage distribution
        ax2.bar(ranges, percentages)
        ax2.set_title('Prediction Confidence Distribution (%)')
        ax2.set_xlabel('Confidence Range')
        ax2.set_ylabel('Percentage of Predictions')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cs2_specific(self, cs2_data: Dict, output_dir: Path):
        """Plot CS2-specific performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Distance performance
        if 'distance_performance' in cs2_data:
            dist_data = cs2_data['distance_performance']
            ranges = list(dist_data.keys())
            accuracies = [dist_data[r]['accuracy'] for r in ranges]
            
            axes[plot_idx].bar(ranges, accuracies)
            axes[plot_idx].set_title('Performance by Distance Range')
            axes[plot_idx].set_ylabel('Accuracy')
            plot_idx += 1
        
        # Weapon performance
        if 'weapon_performance' in cs2_data:
            weapon_data = cs2_data['weapon_performance']
            weapons = list(weapon_data.keys())[:8]  # Top 8 weapons
            accuracies = [weapon_data[w]['accuracy'] for w in weapons]
            
            axes[plot_idx].bar(weapons, accuracies)
            axes[plot_idx].set_title('Performance by Weapon')
            axes[plot_idx].set_ylabel('Accuracy')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1
        
        # Round phase performance
        if 'round_phase_performance' in cs2_data:
            phase_data = cs2_data['round_phase_performance']
            phases = list(phase_data.keys())
            accuracies = [phase_data[p]['accuracy'] for p in phases]
            
            axes[plot_idx].bar(phases, accuracies)
            axes[plot_idx].set_title('Performance by Round Phase')
            axes[plot_idx].set_ylabel('Accuracy')
            plot_idx += 1
        
        # Headshot performance
        if 'headshot_performance' in cs2_data:
            hs_data = cs2_data['headshot_performance']
            categories = ['Headshots', 'Bodyshots']
            accuracies = [hs_data['headshot_accuracy'], hs_data['bodyshot_accuracy']]
            
            axes[plot_idx].bar(categories, accuracies)
            axes[plot_idx].set_title('Performance: Headshots vs Bodyshots')
            axes[plot_idx].set_ylabel('Accuracy')
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cs2_specific_performance.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_model_and_evaluate(model_dir: Path, test_data_path: Path, 
                          output_dir: Path = None) -> Dict[str, Any]:
    """
    Load a trained model and perform comprehensive evaluation.
    
    Args:
        model_dir: Directory containing model files
        test_data_path: Path to test dataset (CSV or parquet)
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    # Load model files
    model_path = model_dir / "kill_analyzer_model.pkl"
    encoder_path = model_dir / "label_encoder.pkl"
    features_path = model_dir / "available_features.pkl"
    
    if not all([model_path.exists(), encoder_path.exists()]):
        raise FileNotFoundError("Model files not found. Train a model first.")
    
    # Load model components
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load feature names (fallback to default if not available)
    if features_path.exists():
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
    else:
        feature_names = ['distance_xy', 'time_in_round_s', 'headshot', 
                        'victim_was_aware', 'had_sound_cue', 'utility_count', 'approach_align_deg']
    
    # Load test data
    if test_data_path.suffix == '.csv':
        test_df = pd.read_csv(test_data_path)
    else:
        test_df = pd.read_parquet(test_data_path)
    
    print(f"Loaded test data: {len(test_df)} samples")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Features: {feature_names}")
    
    # Prepare test features
    X_test = []
    y_test = []
    
    for _, row in test_df.iterrows():
        # Extract features in the same order as training
        features = []
        for feature_name in feature_names:
            if feature_name in row:
                features.append(float(row[feature_name]))
            else:
                features.append(0.0)  # Default value
        
        X_test.append(features)
        
        # Get label
        if 'attacker_labels' in row and row['attacker_labels']:
            if isinstance(row['attacker_labels'], str):
                # Handle string representation of list
                import ast
                try:
                    labels = ast.literal_eval(row['attacker_labels'])
                    label = labels[0] if labels else 'other'
                except:
                    label = row['attacker_labels']
            else:
                label = row['attacker_labels'][0] if row['attacker_labels'] else 'other'
        else:
            label = 'other'
        
        y_test.append(label)
    
    X_test = np.array(X_test)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Create evaluator
    evaluator = CS2ModelEvaluator(model, label_encoder, feature_names)
    
    # Perform evaluation
    print("Running comprehensive evaluation...")
    results = evaluator.evaluate_comprehensive(X_test, y_test_encoded, test_df)
    
    # Generate report and plots
    if output_dir is None:
        output_dir = Path("results/evaluation")
    
    report_content = evaluator.generate_evaluation_report(results, output_dir)
    evaluator.plot_evaluation_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    basic = results['basic_metrics']
    print(f"Accuracy: {basic['accuracy']:.3f} ({basic['accuracy']*100:.1f}%)")
    print(f"F1-Score (Macro): {basic['f1_macro']:.3f}")
    print(f"Precision (Macro): {basic['precision_macro']:.3f}")
    print(f"Recall (Macro): {basic['recall_macro']:.3f}")
    
    if results['cross_validation']['available']:
        cv_acc = results['cross_validation']['metrics']['accuracy']['mean']
        cv_std = results['cross_validation']['metrics']['accuracy']['std']
        print(f"Cross-Validation Accuracy: {cv_acc:.3f} ± {cv_std:.3f}")
    
    conf_analysis = results['confidence_analysis']
    print(f"Average Confidence: {conf_analysis['overall_confidence']['mean']:.3f}")
    print(f"Low Confidence Predictions: {conf_analysis['low_confidence_percentage']:.1f}%")
    
    print(f"\nFull report: {output_dir}/evaluation_report.md")
    print(f"Plots: {output_dir}/*.png")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CS2 kill classification model")
    parser.add_argument("--model-dir", type=Path, default="src/backend/models",
                       help="Directory containing model files")
    parser.add_argument("--test-data", type=Path, required=True,
                       help="Path to test dataset (CSV or parquet)")
    parser.add_argument("--output-dir", type=Path, default="results/evaluation",
                       help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    try:
        results = load_model_and_evaluate(args.model_dir, args.test_data, args.output_dir)
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        exit(1)
