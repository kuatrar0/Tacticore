#!/usr/bin/env python3
"""
LightGBM Training Script - Train classification models on labeled CS2 data.

This script loads labeled features and trains LightGBM models for both
attacker and victim classification tasks.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(features_path: Path) -> pd.DataFrame:
    """
    Load labeled features data for training.
    
    Args:
        features_path: Path to features CSV file
        
    Returns:
        DataFrame with features and labels
    """
    logger.info(f"Loading training data from {features_path}")
    
    try:
        df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise


def prepare_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], LabelEncoder]:
    """
    Prepare features and target for training.
    
    Args:
        df: DataFrame with features and labels
        target_column: Column name for the target variable
        
    Returns:
        Tuple of (features, targets, feature_names, target_encoder)
    """
    logger.info(f"Preparing features for target: {target_column}")
    
    # Filter to labeled samples only
    labeled_mask = df[target_column] != ''
    labeled_df = df[labeled_mask].copy()
    
    if len(labeled_df) == 0:
        raise ValueError(f"No labeled samples found for target '{target_column}'")
    
    logger.info(f"Using {len(labeled_df)} labeled samples for training")
    
    # Define feature columns
    feature_columns = [
        'distance_xy', 'time_in_round_s', 'approach_align_deg',
        'attacker_health', 'victim_health', 'headshot',
        'flash_near', 'smoke_near', 'molotov_near', 'he_near'
    ]
    
    # Add categorical features if they exist
    categorical_features = ['side', 'place']
    for col in categorical_features:
        if col in df.columns:
            feature_columns.append(col)
    
    # Filter to available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Prepare features
    X = labeled_df[available_features].copy()
    y = labeled_df[target_column].copy()
    
    # Handle categorical variables
    for col in categorical_features:
        if col in available_features:
            # Convert to numeric for LightGBM
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Fill missing values
    X = X.fillna(0)
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    logger.info(f"Prepared {len(available_features)} features for training")
    logger.info(f"Target classes: {list(target_encoder.classes_)}")
    logger.info(f"Class distribution: {dict(zip(target_encoder.classes_, np.bincount(y_encoded)))}")
    
    return X, y_encoded, available_features, target_encoder


def train_lightgbm_model(X: pd.DataFrame, y: np.ndarray, feature_names: List[str],
                        target_name: str, random_state: int = 42) -> Tuple[lgb.LGBMClassifier, Dict]:
    """
    Train a LightGBM classification model.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        target_name: Name of the target variable
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    logger.info(f"Training LightGBM model for {target_name}...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Configure LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': random_state
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'best_iteration': model.best_iteration
    }
    
    logger.info(f"Model training completed. Best iteration: {model.best_iteration}")
    logger.info(f"Test accuracy: {accuracy:.3f}")
    logger.info(f"Test F1-score: {f1:.3f}")
    
    return model, metrics


def create_feature_importance_plot(model: lgb.Booster, feature_names: List[str], 
                                 target_name: str, output_dir: Path) -> None:
    """
    Create and save feature importance plot.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        target_name: Name of the target variable
        output_dir: Directory to save the plot
    """
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'Feature Importance - {target_name}')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"feature_importance_{target_name.lower().replace('_', '')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance plot to {plot_path}")


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                               target_encoder: LabelEncoder, target_name: str,
                               output_dir: Path) -> None:
    """
    Create and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_encoder: Label encoder for target variable
        target_name: Name of the target variable
        output_dir: Directory to save the plot
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    plt.title(f'Confusion Matrix - {target_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"confusion_matrix_{target_name.lower().replace('_', '')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix plot to {plot_path}")


def save_model(model: lgb.Booster, target_encoder: LabelEncoder, feature_names: List[str],
              target_name: str, metrics: Dict, output_dir: Path) -> None:
    """
    Save trained model and metadata.
    
    Args:
        model: Trained LightGBM model
        target_encoder: Label encoder for target variable
        feature_names: List of feature names
        target_name: Name of the target variable
        metrics: Training metrics
        output_dir: Directory to save the model
    """
    # Save model
    model_path = output_dir / f"tacticore_model_{target_name.lower().replace('_', '')}.pkl"
    
    model_data = {
        'model': model,
        'target_encoder': target_encoder,
        'feature_names': feature_names,
        'target_name': target_name,
        'metrics': metrics
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Saved model to {model_path}")


def main():
    """Main function to train LightGBM models."""
    parser = argparse.ArgumentParser(
        description="Train LightGBM models on labeled CS2 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training_lightgbm.py --features results/features_labeled_context.csv
  python training_lightgbm.py --features results/features_labeled_context.csv --target victim_label --output results/models/
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
        help='Target column to train on (default: attacker_label)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results'),
        help='Output directory for models and plots (default: results)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_training_data(args.features)
    
    # Check if target column exists
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in data")
        sys.exit(1)
    
    # Prepare features
    try:
        X, y, feature_names, target_encoder = prepare_features(df, args.target)
    except ValueError as e:
        logger.error(f"Feature preparation failed: {e}")
        sys.exit(1)
    
    # Train model
    model, metrics = train_lightgbm_model(X, y, feature_names, args.target, args.random_state)
    
    # Create plots
    create_feature_importance_plot(model, feature_names, args.target, args.output)
    
    # Create confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_classes = np.argmax(y_pred, axis=1)
    create_confusion_matrix_plot(y_test, y_pred_classes, target_encoder, args.target, args.output)
    
    # Save model
    save_model(model, target_encoder, feature_names, args.target, metrics, args.output)
    
    # Print detailed classification report
    logger.info("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=target_encoder.classes_))
    
    # Print summary
    logger.info("Model training completed!")
    logger.info(f"Target variable: {args.target}")
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Features used: {len(feature_names)}")
    logger.info(f"Model saved to: {args.output}")
    logger.info(f"Test accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"Test F1-score: {metrics['f1_score']:.3f}")


if __name__ == "__main__":
    main()
