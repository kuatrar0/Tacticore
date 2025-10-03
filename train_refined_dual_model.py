#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined Dual Perspective Model Training
Excludes redundant labels and low-frequency labels
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, jaccard_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def load_and_prepare_refined_data(csv_path: str):
    """Load and prepare CSV data excluding redundant and low-frequency labels."""
    print(f"Loading CSV data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Filter out empty labels
    df_clean = df[
        (df['attacker_label'].notna() & (df['attacker_label'] != '')) &
        (df['victim_label'].notna() & (df['victim_label'] != ''))
    ].copy()
    print(f"After filtering empty labels: {len(df_clean)} samples")
    
    # Parse multi-label strings
    def parse_labels(label_string):
        """Parse comma-separated labels into list."""
        if pd.isna(label_string) or label_string == '':
            return []
        return [label.strip() for label in str(label_string).split(',')]
    
    # Parse both attacker and victim labels
    attacker_labels = df_clean['attacker_label'].apply(parse_labels)
    victim_labels = df_clean['victim_label'].apply(parse_labels)
    
    # Define labels to exclude
    redundant_attacker_labels = {'is_alive', 'visible'}
    redundant_victim_labels = {'is_alive'}
    
    # Low-frequency victim labels to exclude (less than 5 samples)
    low_frequency_victim_labels = {
        'bad_support', 'overexposed', 'good_position', 'isolated', 'predictable',
        'no_communication', 'bad_site_hold', 'bad_rotation', 'trapped',
        'no_utility', 'force_buy', 'no_utility_usage', 'wide_peek', 'no_sound_awareness'
    }
    
    # Filter out redundant and low-frequency labels
    def filter_labels(labels, redundant_set, low_freq_set=None):
        """Remove redundant and low-frequency labels from the list."""
        filtered = [label for label in labels if label not in redundant_set]
        if low_freq_set:
            filtered = [label for label in filtered if label not in low_freq_set]
        return filtered
    
    attacker_labels_filtered = attacker_labels.apply(lambda x: filter_labels(x, redundant_attacker_labels))
    victim_labels_filtered = victim_labels.apply(lambda x: filter_labels(x, redundant_victim_labels, low_frequency_victim_labels))
    
    # Get all unique labels for each perspective (excluding redundant and low-frequency)
    attacker_unique = set()
    victim_unique = set()
    
    for labels in attacker_labels_filtered:
        attacker_unique.update(labels)
    for labels in victim_labels_filtered:
        victim_unique.update(labels)
    
    attacker_unique = sorted(list(attacker_unique))
    victim_unique = sorted(list(victim_unique))
    
    print(f"Found {len(attacker_unique)} unique attacker labels (excluding redundant): {attacker_unique}")
    print(f"Found {len(victim_unique)} unique victim labels (excluding redundant and low-frequency): {victim_unique}")
    
    # Show excluded labels
    print(f"\nExcluded redundant labels: {redundant_attacker_labels | redundant_victim_labels}")
    print(f"Excluded low-frequency victim labels: {low_frequency_victim_labels}")
    
    # Define feature columns
    feature_columns = [
        'distance_xy', 'time_in_round_s', 'headshot', 'approach_align_deg',
        'victim_was_aware', 'had_sound_cue', 'utility_count',
        'attacker_health', 'victim_health', 'flash_near', 'smoke_near', 
        'molotov_near', 'he_near'
    ]
    
    # Filter to available features
    available_features = [col for col in feature_columns if col in df_clean.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Prepare features
    X = df_clean[available_features].copy()
    
    # Handle missing values and convert to numeric
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        elif X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    X = X.astype(float)
    
    # Create multi-label binary matrices
    attacker_mlb = MultiLabelBinarizer()
    victim_mlb = MultiLabelBinarizer()
    
    attacker_y = attacker_mlb.fit_transform(attacker_labels_filtered)
    victim_y = victim_mlb.fit_transform(victim_labels_filtered)
    
    print(f"Attacker matrix shape: {attacker_y.shape}")
    print(f"Victim matrix shape: {victim_y.shape}")
    print(f"Average attacker labels per sample: {attacker_y.sum(axis=1).mean():.2f}")
    print(f"Average victim labels per sample: {victim_y.sum(axis=1).mean():.2f}")
    
    return X, attacker_y, victim_y, available_features, attacker_mlb, victim_mlb

def train_refined_dual_models(X, attacker_y, victim_y, feature_names, attacker_mlb, victim_mlb):
    """Train refined dual perspective models."""
    print("Training refined dual perspective models...")
    
    # Split data
    X_train, X_test, attacker_y_train, attacker_y_test, victim_y_train, victim_y_test = train_test_split(
        X, attacker_y, victim_y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Configure LightGBM
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Train attacker models
    print(f"\nTraining {attacker_y.shape[1]} attacker models...")
    attacker_models = {}
    attacker_predictions = {}
    
    for i, label in enumerate(attacker_mlb.classes_):
        print(f"  Training attacker model for: {label}")
        
        train_data = lgb.Dataset(X_train, label=attacker_y_train[:, i], feature_name=feature_names)
        test_data = lgb.Dataset(X_test, label=attacker_y_test[:, i], feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        attacker_models[label] = model
        attacker_predictions[label] = (model.predict(X_test, num_iteration=model.best_iteration) > 0.5).astype(int)
    
    # Train victim models
    print(f"\nTraining {victim_y.shape[1]} victim models...")
    victim_models = {}
    victim_predictions = {}
    
    for i, label in enumerate(victim_mlb.classes_):
        print(f"  Training victim model for: {label}")
        
        train_data = lgb.Dataset(X_train, label=victim_y_train[:, i], feature_name=feature_names)
        test_data = lgb.Dataset(X_test, label=victim_y_test[:, i], feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        victim_models[label] = model
        victim_predictions[label] = (model.predict(X_test, num_iteration=model.best_iteration) > 0.5).astype(int)
    
    # Calculate metrics
    attacker_pred_combined = np.column_stack([attacker_predictions[label] for label in attacker_mlb.classes_])
    victim_pred_combined = np.column_stack([victim_predictions[label] for label in victim_mlb.classes_])
    
    attacker_hamming = hamming_loss(attacker_y_test, attacker_pred_combined)
    attacker_jaccard = jaccard_score(attacker_y_test, attacker_pred_combined, average='macro', zero_division=0)
    
    victim_hamming = hamming_loss(victim_y_test, victim_pred_combined)
    victim_jaccard = jaccard_score(victim_y_test, victim_pred_combined, average='macro', zero_division=0)
    
    print(f"\nRefined Dual Perspective Performance:")
    print(f"Attacker - Hamming Loss: {attacker_hamming:.3f}, Jaccard Score: {attacker_jaccard:.3f}")
    print(f"Victim - Hamming Loss: {victim_hamming:.3f}, Jaccard Score: {victim_jaccard:.3f}")
    
    return attacker_models, victim_models, attacker_mlb, victim_mlb, attacker_hamming, victim_hamming

def save_refined_model(attacker_models, victim_models, attacker_mlb, victim_mlb, feature_names, output_dir: Path):
    """Save refined dual perspective model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    attacker_models_path = output_dir / "refined_attacker_models.pkl"
    victim_models_path = output_dir / "refined_victim_models.pkl"
    attacker_mlb_path = output_dir / "refined_attacker_binarizer.pkl"
    victim_mlb_path = output_dir / "refined_victim_binarizer.pkl"
    features_path = output_dir / "refined_available_features.pkl"
    
    with open(attacker_models_path, 'wb') as f:
        pickle.dump(attacker_models, f)
    with open(victim_models_path, 'wb') as f:
        pickle.dump(victim_models, f)
    with open(attacker_mlb_path, 'wb') as f:
        pickle.dump(attacker_mlb, f)
    with open(victim_mlb_path, 'wb') as f:
        pickle.dump(victim_mlb, f)
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"\nRefined dual perspective model saved to:")
    print(f"  Attacker Models: {attacker_models_path}")
    print(f"  Victim Models: {victim_models_path}")
    print(f"  Attacker Binarizer: {attacker_mlb_path}")
    print(f"  Victim Binarizer: {victim_mlb_path}")
    print(f"  Features: {features_path}")

def main():
    """Main function."""
    print("=" * 60)
    print("CS2 Refined Dual Perspective Model Training")
    print("Excluding redundant and low-frequency labels")
    print("=" * 60)
    
    # Paths
    csv_path = "features_labeled_context (2).csv"
    output_dir = Path("src/backend/models")
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return False
    
    try:
        # Load and prepare data
        X, attacker_y, victim_y, feature_names, attacker_mlb, victim_mlb = load_and_prepare_refined_data(csv_path)
        
        # Train models
        attacker_models, victim_models, attacker_mlb, victim_mlb, attacker_hamming, victim_hamming = train_refined_dual_models(
            X, attacker_y, victim_y, feature_names, attacker_mlb, victim_mlb
        )
        
        # Save model
        save_refined_model(attacker_models, victim_models, attacker_mlb, victim_mlb, feature_names, output_dir)
        
        print(f"\nSUCCESS: Refined dual perspective training completed!")
        print(f"Attacker Hamming Loss: {attacker_hamming:.3f}")
        print(f"Victim Hamming Loss: {victim_hamming:.3f}")
        print(f"Model saved to: {output_dir}")
        
        # Show label counts
        print(f"\nRefined model labels:")
        print(f"  Attacker labels: {len(attacker_mlb.classes_)}")
        print(f"  Victim labels: {len(victim_mlb.classes_)}")
        print(f"  Total models: {len(attacker_mlb.classes_) + len(victim_mlb.classes_)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
