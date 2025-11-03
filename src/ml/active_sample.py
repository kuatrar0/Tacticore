#!/usr/bin/env python3
"""
Active Learning Sampling - Identify uncertain samples for labeling.

This script trains a quick LightGBM model on existing labeled data and
identifies the most uncertain unlabeled samples for human labeling.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features_data(features_path: Path) -> pd.DataFrame:
    """
    Load features data and prepare for active learning.
    
    Args:
        features_path: Path to features CSV file
        
    Returns:
        DataFrame with features and labels
    """
    logger.info(f"Loading features data from {features_path}")
    
    try:
        df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Failed to load features data: {e}")
        raise


def prepare_training_data(df: pd.DataFrame, target_column: str = 'attacker_label') -> tuple:
    """
    Prepare data for training by separating labeled and unlabeled samples.
    
    Args:
        df: DataFrame with features and labels
        target_column: Column name for the target variable
        
    Returns:
        Tuple of (labeled_features, labeled_targets, unlabeled_features, feature_names)
    """
    logger.info(f"Preparing training data for target: {target_column}")
    
    labeled_mask = df[target_column] != ''
    labeled_df = df[labeled_mask].copy()
    unlabeled_df = df[~labeled_mask].copy()
    
    logger.info(f"Labeled samples: {len(labeled_df)}")
    logger.info(f"Unlabeled samples: {len(unlabeled_df)}")
    
    if len(labeled_df) == 0:
        raise ValueError("No labeled samples found for training")
    
    feature_columns = [
        'distance_xy', 'time_in_round_s', 'approach_align_deg',
        'attacker_health', 'victim_health', 'headshot',
        'flash_near', 'smoke_near', 'molotov_near', 'he_near'
    ]
    
    categorical_features = ['side', 'place']
    for col in categorical_features:
        if col in df.columns:
            feature_columns.append(col)
    
    available_features = [col for col in feature_columns if col in df.columns]
    
    X_labeled = labeled_df[available_features].copy()
    y_labeled = labeled_df[target_column].copy()
    
    X_unlabeled = unlabeled_df[available_features].copy() if len(unlabeled_df) > 0 else pd.DataFrame()
    
    for col in categorical_features:
        if col in available_features:
            le = LabelEncoder()
            if len(X_labeled) > 0:
                X_labeled[col] = le.fit_transform(X_labeled[col].astype(str))
            if len(X_unlabeled) > 0:
                X_unlabeled[col] = le.transform(X_unlabeled[col].astype(str))
    
    X_labeled = X_labeled.fillna(0)
    if len(X_unlabeled) > 0:
        X_unlabeled = X_unlabeled.fillna(0)
    
    target_encoder = LabelEncoder()
    y_labeled_encoded = target_encoder.fit_transform(y_labeled)
    
    logger.info(f"Prepared {len(available_features)} features for training")
    logger.info(f"Target classes: {list(target_encoder.classes_)}")
    
    return X_labeled, y_labeled_encoded, X_unlabeled, available_features, target_encoder


def train_uncertainty_model(X_labeled: pd.DataFrame, y_labeled: np.ndarray,
                          feature_names: List[str]) -> lgb.LGBMClassifier:
    """
    Train a LightGBM model for uncertainty estimation.
    
    Args:
        X_labeled: Features for labeled samples
        y_labeled: Targets for labeled samples
        feature_names: List of feature names
        
    Returns:
        Trained LightGBM classifier
    """
    logger.info("Training uncertainty estimation model...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )
    
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_labeled)),
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    logger.info(f"Model training completed. Best iteration: {model.best_iteration}")
    
    return model


def calculate_uncertainty(model: lgb.Booster, X_unlabeled: pd.DataFrame) -> np.ndarray:
    """
    Calculate uncertainty scores for unlabeled samples.
    
    Args:
        model: Trained LightGBM model
        X_unlabeled: Features for unlabeled samples
        
    Returns:
        Array of uncertainty scores
    """
    if len(X_unlabeled) == 0:
        return np.array([])
    
    logger.info("Calculating uncertainty scores...")
    
    predictions = model.predict(X_unlabeled, num_iteration=model.best_iteration)
    
    if predictions.shape[1] == 2:
        probs = predictions[:, 1]
        uncertainty = np.abs(probs - 0.5)
    else:
        epsilon = 1e-10
        probs = np.clip(predictions, epsilon, 1 - epsilon)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        uncertainty = entropy / np.log(probs.shape[1])
    
    logger.info(f"Calculated uncertainty scores for {len(uncertainty)} samples")
    logger.info(f"Uncertainty range: {uncertainty.min():.3f} - {uncertainty.max():.3f}")
    
    return uncertainty


def select_samples_for_labeling(df: pd.DataFrame, uncertainty_scores: np.ndarray,
                               n_samples: int = 50) -> pd.DataFrame:
    """
    Select samples with highest uncertainty for labeling.
    
    Args:
        df: Original DataFrame
        uncertainty_scores: Uncertainty scores for unlabeled samples
        n_samples: Number of samples to select
        
    Returns:
        DataFrame with samples to label
    """
    if len(uncertainty_scores) == 0:
        logger.warning("No unlabeled samples available for selection")
        return pd.DataFrame()
    
    unlabeled_mask = df['attacker_label'] == ''
    unlabeled_df = df[unlabeled_mask].copy()
    
    unlabeled_df['uncertainty'] = uncertainty_scores
    unlabeled_df = unlabeled_df.sort_values('uncertainty', ascending=False)
    
    n_samples = min(n_samples, len(unlabeled_df))
    selected_df = unlabeled_df.head(n_samples).copy()
    
    logger.info(f"Selected {n_samples} samples for labeling")
    logger.info(f"Uncertainty range of selected samples: {selected_df['uncertainty'].min():.3f} - {selected_df['uncertainty'].max():.3f}")
    
    return selected_df


def main():
    """Main function for active learning sampling."""
    parser = argparse.ArgumentParser(
        description="Active learning sampling for CS2 kill labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python active_sample.py --features results/features_labeled_context.csv --output results/to_label_next.csv
  python active_sample.py --features results/features_labeled_context.csv --target victim_label --samples 100 --output results/to_label_next.csv
        """
    )
    
    parser.add_argument(
        '--features',
        type=Path,
        required=True,
        help='Path to features CSV file'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='attacker_label',
        choices=['attacker_label', 'victim_label'],
        help='Target column for active learning (default: attacker_label)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Number of samples to select for labeling (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/to_label_next.csv'),
        help='Output path for samples to label (default: results/to_label_next.csv)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    df = load_features_data(args.features)
    
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in data")
        sys.exit(1)
    
    labeled_count = len(df[df[args.target] != ''])
    if labeled_count == 0:
        logger.error(f"No labeled samples found for target '{args.target}'")
        sys.exit(1)
    
    logger.info(f"Found {labeled_count} labeled samples for target '{args.target}'")
    
    try:
        X_labeled, y_labeled, X_unlabeled, feature_names, target_encoder = prepare_training_data(
            df, args.target
        )
    except ValueError as e:
        logger.error(f"Data preparation failed: {e}")
        sys.exit(1)
    
    model = train_uncertainty_model(X_labeled, y_labeled, feature_names)
    uncertainty_scores = calculate_uncertainty(model, X_unlabeled)
    selected_df = select_samples_for_labeling(df, uncertainty_scores, args.samples)
    
    if len(selected_df) == 0:
        logger.warning("No samples selected for labeling")
        return
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    output_columns = [
        'kill_tick', 'attacker_name', 'victim_name', 'side', 'place',
        'distance_xy', 'time_in_round_s', 'approach_align_deg',
        'attacker_health', 'victim_health', 'headshot',
        'uncertainty'
    ]
    
    available_columns = [col for col in output_columns if col in selected_df.columns]
    output_df = selected_df[available_columns].copy()
    
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(output_df)} samples to {args.output}")
    
    logger.info("Active learning sampling completed!")
    logger.info(f"Target variable: {args.target}")
    logger.info(f"Labeled samples used for training: {len(X_labeled)}")
    logger.info(f"Unlabeled samples available: {len(X_unlabeled)}")
    logger.info(f"Samples selected for labeling: {len(selected_df)}")
    
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.3f}")


if __name__ == "__main__":
    main()
