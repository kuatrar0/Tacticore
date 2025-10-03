#!/usr/bin/env python3
"""
Refined Dual Perspective Kill Analyzer
Excludes redundant labels and low-frequency labels
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

class RefinedDualKillAnalyzer:
    """
    Refined analyzer that excludes redundant and low-frequency labels.
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize the refined dual analyzer.
        
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
        """Load trained refined models and encoders."""
        try:
            # Load refined attacker models
            with open(self.models_dir / "refined_attacker_models.pkl", 'rb') as f:
                self.attacker_models = pickle.load(f)
            
            # Load refined victim models
            with open(self.models_dir / "refined_victim_models.pkl", 'rb') as f:
                self.victim_models = pickle.load(f)
            
            # Load refined binarizers
            with open(self.models_dir / "refined_attacker_binarizer.pkl", 'rb') as f:
                self.attacker_mlb = pickle.load(f)
            
            with open(self.models_dir / "refined_victim_binarizer.pkl", 'rb') as f:
                self.victim_mlb = pickle.load(f)
            
            # Load feature names
            with open(self.models_dir / "refined_available_features.pkl", 'rb') as f:
                self.feature_names = pickle.load(f)
            
            print(f"SUCCESS: Refined dual analyzer loaded successfully")
            print(f"   Attacker labels: {len(self.attacker_mlb.classes_)} (refined)")
            print(f"   Victim labels: {len(self.victim_mlb.classes_)} (refined)")
            print(f"   Features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"ERROR: Failed to load refined dual analyzer: {e}")
            self.attacker_models = None
            self.victim_models = None
    
    def prepare_features(self, kill_data: Dict) -> np.ndarray:
        """
        Prepare features from kill data.
        
        Args:
            kill_data: Dictionary with kill information
            
        Returns:
            Feature array
        """
        features = []
        
        for feature_name in self.feature_names:
            if feature_name in kill_data:
                value = kill_data[feature_name]
                if pd.isna(value):
                    features.append(0.0)
                else:
                    features.append(float(value))
            else:
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def predict_attacker_strengths(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict attacker strengths (refined labels only).
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary of label -> probability
        """
        if self.attacker_models is None:
            return {}
        
        predictions = {}
        
        for label in self.attacker_mlb.classes_:
            if label in self.attacker_models:
                model = self.attacker_models[label]
                prob = model.predict(features, num_iteration=model.best_iteration)[0]
                predictions[label] = float(prob)
        
        return predictions
    
    def predict_victim_errors(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict victim errors (refined labels only).
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary of label -> probability
        """
        if self.victim_models is None:
            return {}
        
        predictions = {}
        
        for label in self.victim_mlb.classes_:
            if label in self.victim_models:
                model = self.victim_models[label]
                prob = model.predict(features, num_iteration=model.best_iteration)[0]
                predictions[label] = float(prob)
        
        return predictions
    
    def analyze_kill(self, kill_data: Dict, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze a kill from both perspectives (refined labels only).
        
        Args:
            kill_data: Dictionary with kill information
            threshold: Probability threshold for predictions
            
        Returns:
            Analysis results
        """
        if self.attacker_models is None or self.victim_models is None:
            return {"error": "Models not loaded"}
        
        # Prepare features
        features = self.prepare_features(kill_data)
        
        # Get predictions
        attacker_probs = self.predict_attacker_strengths(features)
        victim_probs = self.predict_victim_errors(features)
        
        # Filter by threshold
        attacker_strengths = {
            label: prob for label, prob in attacker_probs.items() 
            if prob >= threshold
        }
        victim_errors = {
            label: prob for label, prob in victim_probs.items() 
            if prob >= threshold
        }
        
        # Sort by probability
        attacker_strengths = dict(sorted(attacker_strengths.items(), key=lambda x: x[1], reverse=True))
        victim_errors = dict(sorted(victim_errors.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "attacker_strengths": attacker_strengths,
            "victim_errors": victim_errors,
            "attacker_all_probs": attacker_probs,
            "victim_all_probs": victim_probs,
            "threshold": threshold
        }
    
    def get_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of the analysis.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Summary string
        """
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        attacker = analysis["attacker_strengths"]
        victim = analysis["victim_errors"]
        
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
                self.victim_mlb is not None)

# Global refined analyzer instance
refined_dual_analyzer = RefinedDualKillAnalyzer()
