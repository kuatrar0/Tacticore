#!/usr/bin/env python3
"""
Feature Engineering Script - Build ML-ready features from labeled kill data.

This script takes parsed demo data and labeled kills to create a comprehensive
feature set for machine learning training.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from features.schemas import (
    KillSchema, TickSchema, GrenadeSchema, LabeledFeatureSchema,
    validate_dataframe_schema, get_schema_warnings, get_feature_columns
)
from streamlit_app.transforms import (
    find_nearest_tick, calculate_time_in_round, calculate_distance_2d,
    calculate_approach_alignment, find_nearby_utility
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(kills_path: Path, ticks_path: Path, 
              grenades_path: Optional[Path] = None,
              labels_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all required data files.
    
    Args:
        kills_path: Path to kills.parquet
        ticks_path: Path to ticks.parquet
        grenades_path: Optional path to grenades.parquet
        labels_path: Optional path to labeled CSV
        
    Returns:
        Dictionary with loaded DataFrames
    """
    data = {}
    
    logger.info(f"Loading kills data from {kills_path}")
    try:
        data['kills'] = pd.read_parquet(kills_path)
        logger.info(f"Loaded {len(data['kills'])} kills")
    except Exception as e:
        logger.error(f"Failed to load kills data: {e}")
        raise
    
    logger.info(f"Loading ticks data from {ticks_path}")
    try:
        data['ticks'] = pd.read_parquet(ticks_path)
        logger.info(f"Loaded {len(data['ticks'])} ticks")
    except Exception as e:
        logger.error(f"Failed to load ticks data: {e}")
        raise
    
    if grenades_path and grenades_path.exists():
        logger.info(f"Loading grenades data from {grenades_path}")
        try:
            data['grenades'] = pd.read_parquet(grenades_path)
            logger.info(f"Loaded {len(data['grenades'])} grenade events")
        except Exception as e:
            logger.warning(f"Failed to load grenades data: {e}")
            data['grenades'] = pd.DataFrame()
    else:
        data['grenades'] = pd.DataFrame()
    
    if labels_path and labels_path.exists():
        logger.info(f"Loading labels data from {labels_path}")
        try:
            data['labels'] = pd.read_csv(labels_path)
            logger.info(f"Loaded {len(data['labels'])} labeled kills")
        except Exception as e:
            logger.warning(f"Failed to load labels data: {e}")
            data['labels'] = pd.DataFrame()
    else:
        data['labels'] = pd.DataFrame()
    
    return data


def validate_data(data: Dict[str, pd.DataFrame]) -> None:
    """
    Validate loaded data against schemas.
    
    Args:
        data: Dictionary with loaded DataFrames
    """
    logger.info("Validating data schemas...")
    
    kills_validation = validate_dataframe_schema(data['kills'], KillSchema)
    if not kills_validation['valid']:
        logger.error("Kills data validation failed")
        for warning in get_schema_warnings(kills_validation):
            logger.error(warning)
        raise ValueError("Invalid kills data")
    
    ticks_validation = validate_dataframe_schema(data['ticks'], TickSchema)
    if not ticks_validation['valid']:
        logger.error("Ticks data validation failed")
        for warning in get_schema_warnings(ticks_validation):
            logger.error(warning)
        raise ValueError("Invalid ticks data")
    
    if not data['grenades'].empty:
        grenades_validation = validate_dataframe_schema(data['grenades'], GrenadeSchema)
        for warning in get_schema_warnings(grenades_validation):
            logger.warning(warning)
    
    logger.info("Data validation completed")


def build_kill_context(kills_df: pd.DataFrame, ticks_df: pd.DataFrame, 
                      grenades_df: pd.DataFrame, tickrate: int = 64) -> pd.DataFrame:
    """
    Build comprehensive context for each kill event.
    
    Args:
        kills_df: DataFrame with kill events
        ticks_df: DataFrame with tick data
        grenades_df: DataFrame with grenade events
        tickrate: Game tickrate
        
    Returns:
        DataFrame with kill context features
    """
    logger.info("Building kill context features...")
    
    context_features = []
    
    for idx, kill_row in kills_df.iterrows():
        kill_tick = kill_row['tick']
        attacker_name = kill_row['attacker_name']
        victim_name = kill_row['victim_name']
        
        attacker_tick = find_nearest_tick(kill_tick, ticks_df, attacker_name)
        victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
        
        context = {
            'kill_tick': kill_tick,
            'attacker_name': attacker_name,
            'victim_name': victim_name,
            'side': kill_row.get('side', 'Unknown'),
            'place': kill_row.get('place', 'Unknown'),
            'headshot': kill_row.get('headshot', False)
        }
        
        if attacker_tick is not None:
            context.update({
                'attacker_x': attacker_tick.get('x', 0),
                'attacker_y': attacker_tick.get('y', 0),
                'attacker_z': attacker_tick.get('z', 0),
                'attacker_health': attacker_tick.get('health', 100),
                'attacker_vel_x': attacker_tick.get('vel_x', 0),
                'attacker_vel_y': attacker_tick.get('vel_y', 0)
            })
        else:
            context.update({
                'attacker_x': 0, 'attacker_y': 0, 'attacker_z': 0,
                'attacker_health': 100, 'attacker_vel_x': 0, 'attacker_vel_y': 0
            })
        
        if victim_tick is not None:
            context.update({
                'victim_x': victim_tick.get('x', 0),
                'victim_y': victim_tick.get('y', 0),
                'victim_z': victim_tick.get('z', 0),
                'victim_health': victim_tick.get('health', 100)
            })
        else:
            context.update({
                'victim_x': 0, 'victim_y': 0, 'victim_z': 0, 'victim_health': 100
            })
        
        context['time_in_round_s'] = calculate_time_in_round(kill_tick, pd.DataFrame(), tickrate)
        
        context['distance_xy'] = calculate_distance_2d(
            context['attacker_x'], context['attacker_y'],
            context['victim_x'], context['victim_y']
        )
        
        context['approach_align_deg'] = calculate_approach_alignment(
            context['attacker_x'], context['attacker_y'],
            context['victim_x'], context['victim_y'],
            context['attacker_vel_x'], context['attacker_vel_y']
        )
        
        utility_flags = find_nearby_utility(
            kill_tick, context['victim_x'], context['victim_y'], grenades_df
        )
        context.update(utility_flags)
        
        context_features.append(context)
    
    context_df = pd.DataFrame(context_features)
    logger.info(f"Built context for {len(context_df)} kills")
    
    return context_df


def merge_labels(context_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge context features with labels.
    
    Args:
        context_df: DataFrame with kill context
        labels_df: DataFrame with labels
        
    Returns:
        DataFrame with merged features and labels
    """
    if labels_df.empty:
        logger.warning("No labels provided, creating empty label columns")
        context_df['attacker_label'] = ''
        context_df['victim_label'] = ''
        return context_df
    
    logger.info("Merging context with labels...")
    
    merge_keys = ['kill_tick', 'attacker_name', 'victim_name']
    
    context_keys = [key for key in merge_keys if key in context_df.columns]
    label_keys = [key for key in merge_keys if key in labels_df.columns]
    
    if not context_keys or not label_keys:
        logger.warning("No common merge keys found, using index merge")
        merged_df = context_df.copy()
        merged_df['attacker_label'] = ''
        merged_df['victim_label'] = ''
        return merged_df
    
    common_keys = list(set(context_keys) & set(label_keys))
    
    if not common_keys:
        logger.warning("No common merge keys found, using index merge")
        merged_df = context_df.copy()
        merged_df['attacker_label'] = ''
        merged_df['victim_label'] = ''
        return merged_df
    
    merged_df = context_df.merge(
        labels_df[common_keys + ['attacker_label', 'victim_label']],
        on=common_keys,
        how='left'
    )
    
    merged_df['attacker_label'] = merged_df['attacker_label'].fillna('')
    merged_df['victim_label'] = merged_df['victim_label'].fillna('')
    
    logger.info(f"Merged {len(merged_df)} kills with labels")
    
    return merged_df


def engineer_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features for ML training.
    
    Args:
        features_df: DataFrame with basic features
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering additional features...")
    
    df = features_df.copy()
    
    df['is_headshot'] = df['headshot'].astype(int)
    df['is_attacker_healthy'] = (df['attacker_health'] > 50).astype(int)
    df['is_victim_healthy'] = (df['victim_health'] > 50).astype(int)
    
    df['distance_category'] = pd.cut(
        df['distance_xy'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['very_close', 'close', 'medium', 'far']
    )
    
    df['time_category'] = pd.cut(
        df['time_in_round_s'],
        bins=[0, 30, 60, 90, float('inf')],
        labels=['early', 'mid_early', 'mid_late', 'late']
    )
    
    df['alignment_category'] = pd.cut(
        df['approach_align_deg'],
        bins=[0, 30, 60, 90, 180],
        labels=['excellent', 'good', 'fair', 'poor']
    )
    
    utility_cols = ['flash_near', 'smoke_near', 'molotov_near', 'he_near']
    df['utility_count'] = df[utility_cols].sum(axis=1)
    
    if 'side' in df.columns:
        df['is_terrorist'] = (df['side'] == 'T').astype(int)
        df['is_counter_terrorist'] = (df['side'] == 'CT').astype(int)
    
    logger.info("Feature engineering completed")
    
    return df


def prepare_ml_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for ML training (handle categorical variables, etc.).
    
    Args:
        features_df: DataFrame with all features
        
    Returns:
        DataFrame ready for ML training
    """
    logger.info("Preparing features for ML training...")
    
    df = features_df.copy()
    
    categorical_cols = ['side', 'place', 'distance_category', 'time_category', 'alignment_category']
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    bool_cols = ['headshot', 'flash_near', 'smoke_near', 'molotov_near', 'he_near']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    non_feature_cols = ['kill_tick', 'attacker_name', 'victim_name', 'attacker_label', 'victim_label']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    logger.info(f"Prepared {len(feature_cols)} features for ML training")
    
    return df


def main():
    """Main function to build features."""
    parser = argparse.ArgumentParser(
        description="Build ML-ready features from CS2 demo data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_features.py --kills dataset/demo/kills.parquet --ticks dataset/demo/ticks.parquet --output results/features.csv
  python build_features.py --kills dataset/demo/kills.parquet --ticks dataset/demo/ticks.parquet --grenades dataset/demo/grenades.parquet --labels results/labeled.csv --output results/features.csv
        """
    )
    
    parser.add_argument(
        '--kills',
        type=Path,
        required=True,
        help='Path to kills.parquet file'
    )
    
    parser.add_argument(
        '--ticks',
        type=Path,
        required=True,
        help='Path to ticks.parquet file'
    )
    
    parser.add_argument(
        '--grenades',
        type=Path,
        help='Optional path to grenades.parquet file'
    )
    
    parser.add_argument(
        '--labels',
        type=Path,
        help='Optional path to labeled CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/features_labeled_context.csv'),
        help='Output path for features CSV (default: results/features_labeled_context.csv)'
    )
    
    parser.add_argument(
        '--tickrate',
        type=int,
        default=64,
        help='Game tickrate (default: 64)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    data = load_data(args.kills, args.ticks, args.grenades, args.labels)
    validate_data(data)
    
    context_df = build_kill_context(
        data['kills'], data['ticks'], data['grenades'], args.tickrate
    )
    
    features_df = merge_labels(context_df, data['labels'])
    features_df = engineer_features(features_df)
    ml_features_df = prepare_ml_features(features_df)
    
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_df.to_csv(args.output, index=False)
    logger.info(f"Saved features to {args.output}")
    
    ml_output = args.output.parent / f"ml_{args.output.name}"
    ml_features_df.to_csv(ml_output, index=False)
    logger.info(f"Saved ML features to {ml_output}")
    
    logger.info("Feature engineering completed!")
    logger.info(f"Total kills processed: {len(features_df)}")
    logger.info(f"Features with attacker labels: {len(features_df[features_df['attacker_label'] != ''])}")
    logger.info(f"Features with victim labels: {len(features_df[features_df['victim_label'] != ''])}")
    logger.info(f"ML features created: {len(ml_features_df.columns)}")


if __name__ == "__main__":
    main()
