#!/usr/bin/env python3
"""
Filtered Dual Perspective Kill Analyzer
Excludes redundant labels and low-frequency labels
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

class FilteredDualKillAnalyzer:
    """
    Filtered analyzer that excludes redundant and low-frequency labels.
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize the filtered dual analyzer.
        
        Args:
            models_dir: Directory containing model files
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        
        self.models_dir = models_dir
        self.attacker_models = None
        self.victim_models = None
        self.attacker_mlb = None
        self.victim_mlb = None
        self.feature_names = None
        
        self.load_models()
    
    def load_models(self):
        """Load trained filtered models and encoders."""
        try:
            with open(self.models_dir / "filtered_attacker_models.pkl", 'rb') as f:
                self.attacker_models = pickle.load(f)
            
            with open(self.models_dir / "filtered_victim_models.pkl", 'rb') as f:
                self.victim_models = pickle.load(f)
            
            with open(self.models_dir / "filtered_attacker_binarizer.pkl", 'rb') as f:
                self.attacker_mlb = pickle.load(f)
            
            with open(self.models_dir / "filtered_victim_binarizer.pkl", 'rb') as f:
                self.victim_mlb = pickle.load(f)
            
            with open(self.models_dir / "filtered_available_features.pkl", 'rb') as f:
                self.feature_names = pickle.load(f)
            
            print(f"SUCCESS: Filtered dual analyzer loaded successfully")
            print(f"   Attacker labels: {len(self.attacker_mlb.classes_)} (filtered)")
            print(f"   Victim labels: {len(self.victim_mlb.classes_)} (filtered)")
            print(f"   Features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"ERROR: Failed to load filtered dual analyzer: {e}")
            self.attacker_models = None
            self.victim_models = None
    
    def prepare_features(self, kill_data: Dict) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            kill_data: Dictionary containing kill information
            
        Returns:
            numpy array of features
        """
        if self.feature_names is None:
            return None
        
        features = []
        for feature_name in self.feature_names:
            if feature_name in kill_data:
                value = kill_data[feature_name]
                try:
                    features.append(float(value) if value is not None else 0.0)
                except (ValueError, TypeError):
                    features.append(0.0)
            else:
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def analyze_kill(self, kill_data: Dict, threshold: float = 0.5) -> Dict:
        """
        Analyze a kill and return attacker strengths and victim errors.
        
        Args:
            kill_data: Dictionary containing kill information
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary with attacker_strengths and victim_errors
        """
        if not self.is_loaded():
            return {"attacker_strengths": {}, "victim_errors": {}}
        
        features = self.prepare_features(kill_data)
        if features is None:
            return {"attacker_strengths": {}, "victim_errors": {}}
        
        attacker_strengths = {}
        for label, model in self.attacker_models.items():
            prob = model.predict(features)[0]
            if prob >= threshold:
                attacker_strengths[label] = prob
        
        victim_errors = {}
        for label, model in self.victim_models.items():
            prob = model.predict(features)[0]
            if prob >= threshold:
                victim_errors[label] = prob
        
        return {
            "attacker_strengths": attacker_strengths,
            "victim_errors": victim_errors
        }
    
    def get_analysis_summary(self, analysis: Dict) -> str:
        """
        Get a summary of the analysis.
        
        Args:
            analysis: Analysis result from analyze_kill
            
        Returns:
            String summary
        """
        attacker = analysis.get("attacker_strengths", {})
        victim = analysis.get("victim_errors", {})
        
        summary_parts = []
        
        if attacker:
            summary_parts.append("Attacker Strengths:")
            for label, prob in attacker.items():
                summary_parts.append(f"   - {label}: {prob:.1%}")
        else:
            summary_parts.append("No significant attacker strengths detected")
        
        if victim:
            summary_parts.append("\nVictim Errors:")
            for label, prob in victim.items():
                summary_parts.append(f"   - {label}: {prob:.1%}")
        else:
            summary_parts.append("\nNo significant victim errors detected")
        
        return "\n".join(summary_parts)
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return (self.attacker_models is not None and 
                self.victim_models is not None and
                self.attacker_mlb is not None and
                self.victim_mlb is not None and
                self.feature_names is not None)

# Create global instance
filtered_dual_analyzer = FilteredDualKillAnalyzer()
