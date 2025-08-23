#!/usr/bin/env python3
"""
Pre-labeling Rules - Apply heuristic rules to generate initial labels.

This script applies domain knowledge and heuristics to automatically
generate initial labels for kill events, reducing manual labeling effort.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from streamlit_app.transforms import (
    find_nearest_tick, calculate_distance_2d, calculate_approach_alignment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_attacker_rules(kill_row: pd.Series, ticks_df: pd.DataFrame) -> str:
    """
    Apply heuristic rules to determine attacker label.
    
    Args:
        kill_row: Row from kills DataFrame
        ticks_df: DataFrame with tick data
        
    Returns:
        Attacker label string
    """
    kill_tick = kill_row['tick']
    attacker_name = kill_row['attacker_name']
    victim_name = kill_row['victim_name']
    
    # Find nearest tick data
    attacker_tick = find_nearest_tick(kill_tick, ticks_df, attacker_name)
    victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
    
    if attacker_tick is None or victim_tick is None:
        return ''
    
    # Extract features
    attacker_health = attacker_tick.get('health', 100)
    victim_health = victim_tick.get('health', 100)
    headshot = kill_row.get('headshot', False)
    
    # Calculate distance and alignment
    distance = calculate_distance_2d(
        attacker_tick.get('x', 0), attacker_tick.get('y', 0),
        victim_tick.get('x', 0), victim_tick.get('y', 0)
    )
    
    alignment = calculate_approach_alignment(
        attacker_tick.get('x', 0), attacker_tick.get('y', 0),
        victim_tick.get('x', 0), victim_tick.get('y', 0),
        attacker_tick.get('vel_x', 0), attacker_tick.get('vel_y', 0)
    )
    
    # Rule 1: Precise headshot at short distance
    if headshot and distance < 1000 and alignment < 30:
        return 'precise'
    
    # Rule 2: Good positioning - attacker healthy, victim low health
    if attacker_health > 80 and victim_health < 50:
        return 'good_positioning'
    
    # Rule 3: Bad positioning - attacker low health, victim healthy
    if attacker_health < 30 and victim_health > 80:
        return 'bad_positioning'
    
    # Rule 4: Good decision - attacker at advantage
    if attacker_health > victim_health and distance < 1500:
        return 'good_decision'
    
    # Rule 5: Bad decision - attacker at disadvantage
    if attacker_health < victim_health and distance > 2000:
        return 'bad_decision'
    
    # Rule 6: Imprecise - poor alignment or long distance
    if alignment > 90 or distance > 2500:
        return 'imprecise'
    
    return ''


def apply_victim_rules(kill_row: pd.Series, ticks_df: pd.DataFrame) -> str:
    """
    Apply heuristic rules to determine victim label.
    
    Args:
        kill_row: Row from kills DataFrame
        ticks_df: DataFrame with tick data
        
    Returns:
        Victim label string
    """
    kill_tick = kill_row['tick']
    attacker_name = kill_row['attacker_name']
    victim_name = kill_row['victim_name']
    
    # Find nearest tick data
    attacker_tick = find_nearest_tick(kill_tick, ticks_df, attacker_name)
    victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
    
    if attacker_tick is None or victim_tick is None:
        return ''
    
    # Extract features
    attacker_health = attacker_tick.get('health', 100)
    victim_health = victim_tick.get('health', 100)
    headshot = kill_row.get('headshot', False)
    
    # Calculate distance
    distance = calculate_distance_2d(
        attacker_tick.get('x', 0), attacker_tick.get('y', 0),
        victim_tick.get('x', 0), victim_tick.get('y', 0)
    )
    
    # Rule 1: Exposed - victim healthy but killed by headshot at distance
    if victim_health > 80 and headshot and distance > 1500:
        return 'exposed'
    
    # Rule 2: No cover - victim killed quickly at close range
    if distance < 500 and victim_health > 50:
        return 'no_cover'
    
    # Rule 3: Good position - victim had advantage but still died
    if victim_health > attacker_health and distance > 1000:
        return 'good_position'
    
    # Rule 4: Mistake - victim low health but attacker also low
    if victim_health < 30 and attacker_health < 50:
        return 'mistake'
    
    # Rule 5: Exposed - victim at full health killed at long range
    if victim_health > 90 and distance > 2000:
        return 'exposed'
    
    return ''


def apply_advanced_rules(kill_row: pd.Series, ticks_df: pd.DataFrame, 
                        grenades_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Apply more advanced rules using additional context.
    
    Args:
        kill_row: Row from kills DataFrame
        ticks_df: DataFrame with tick data
        grenades_df: DataFrame with grenade data
        
    Returns:
        Tuple of (attacker_label, victim_label)
    """
    kill_tick = kill_row['tick']
    attacker_name = kill_row['attacker_name']
    victim_name = kill_row['victim_name']
    
    # Find nearest tick data
    attacker_tick = find_nearest_tick(kill_tick, ticks_df, attacker_name)
    victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
    
    if attacker_tick is None or victim_tick is None:
        return '', ''
    
    attacker_label = ''
    victim_label = ''
    
    # Check for nearby utility
    if not grenades_df.empty:
        # Filter grenades around kill time (±2 seconds)
        time_window = 128  # 2 seconds at 64 tickrate
        nearby_grenades = grenades_df[
            (grenades_df['tick'] >= kill_tick - time_window) &
            (grenades_df['tick'] <= kill_tick + time_window)
        ]
        
        if not nearby_grenades.empty:
            # Calculate distances to grenades
            victim_x = victim_tick.get('x', 0)
            victim_y = victim_tick.get('y', 0)
            
            nearby_grenades['distance'] = calculate_distance_2d(
                victim_x, victim_y,
                nearby_grenades['x'], nearby_grenades['y']
            )
            
            # Check for flash kills
            flash_grenades = nearby_grenades[
                (nearby_grenades['grenade_type'].str.contains('flash', case=False, na=False)) &
                (nearby_grenades['distance'] < 300)
            ]
            
            if not flash_grenades.empty:
                attacker_label = 'good_decision'  # Using utility effectively
                victim_label = 'exposed'  # Victim was flashed
    
    # Check for weapon-specific rules
    weapon = kill_row.get('weapon', '').lower()
    
    if 'awp' in weapon or 'scout' in weapon:
        # Sniper rifle kills
        distance = calculate_distance_2d(
            attacker_tick.get('x', 0), attacker_tick.get('y', 0),
            victim_tick.get('x', 0), victim_tick.get('y', 0)
        )
        
        if distance > 2000:
            attacker_label = 'precise'
            victim_label = 'exposed'
    
    elif 'pistol' in weapon or 'deagle' in weapon:
        # Pistol kills
        if kill_row.get('headshot', False):
            attacker_label = 'precise'
    
    return attacker_label, victim_label


def generate_prelabels(kills_df: pd.DataFrame, ticks_df: pd.DataFrame,
                      grenades_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate pre-labels for all kills using heuristic rules.
    
    Args:
        kills_df: DataFrame with kill events
        ticks_df: DataFrame with tick data
        grenades_df: Optional DataFrame with grenade data
        
    Returns:
        DataFrame with pre-labels
    """
    logger.info("Generating pre-labels using heuristic rules...")
    
    if grenades_df is None:
        grenades_df = pd.DataFrame()
    
    prelabels = []
    
    for idx, kill_row in kills_df.iterrows():
        # Apply basic rules
        attacker_label = apply_attacker_rules(kill_row, ticks_df)
        victim_label = apply_victim_rules(kill_row, ticks_df)
        
        # Apply advanced rules if basic rules didn't find anything
        if not attacker_label or not victim_label:
            adv_attacker, adv_victim = apply_advanced_rules(kill_row, ticks_df, grenades_df)
            if not attacker_label:
                attacker_label = adv_attacker
            if not victim_label:
                victim_label = adv_victim
        
        prelabels.append({
            'kill_index': idx,
            'kill_tick': kill_row.get('tick', 0),
            'attacker_name': kill_row.get('attacker_name', ''),
            'victim_name': kill_row.get('victim_name', ''),
            'pre_att': attacker_label,
            'pre_vic': victim_label
        })
    
    prelabels_df = pd.DataFrame(prelabels)
    
    # Count pre-labels
    attacker_count = len(prelabels_df[prelabels_df['pre_att'] != ''])
    victim_count = len(prelabels_df[prelabels_df['pre_vic'] != ''])
    
    logger.info(f"Generated {attacker_count} attacker pre-labels")
    logger.info(f"Generated {victim_count} victim pre-labels")
    
    return prelabels_df


def main():
    """Main function to generate pre-labels."""
    parser = argparse.ArgumentParser(
        description="Generate pre-labels using heuristic rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prelabel_rules.py --kills dataset/demo/kills.parquet --ticks dataset/demo/ticks.parquet --output results/prelabels.csv
  python prelabel_rules.py --kills dataset/demo/kills.parquet --ticks dataset/demo/ticks.parquet --grenades dataset/demo/grenades.parquet --output results/prelabels.csv
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
        '--output',
        type=Path,
        default=Path('results/prelabels.csv'),
        help='Output path for pre-labels CSV (default: results/prelabels.csv)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load data
    logger.info(f"Loading kills data from {args.kills}")
    kills_df = pd.read_parquet(args.kills)
    logger.info(f"Loaded {len(kills_df)} kills")
    
    logger.info(f"Loading ticks data from {args.ticks}")
    ticks_df = pd.read_parquet(args.ticks)
    logger.info(f"Loaded {len(ticks_df)} ticks")
    
    grenades_df = pd.DataFrame()
    if args.grenades and args.grenades.exists():
        logger.info(f"Loading grenades data from {args.grenades}")
        grenades_df = pd.read_parquet(args.grenades)
        logger.info(f"Loaded {len(grenades_df)} grenade events")
    
    # Generate pre-labels
    prelabels_df = generate_prelabels(kills_df, ticks_df, grenades_df)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prelabels_df.to_csv(args.output, index=False)
    logger.info(f"Saved pre-labels to {args.output}")
    
    # Print summary
    logger.info("Pre-labeling completed!")
    logger.info(f"Total kills processed: {len(prelabels_df)}")
    logger.info(f"Kills with attacker pre-labels: {len(prelabels_df[prelabels_df['pre_att'] != ''])}")
    logger.info(f"Kills with victim pre-labels: {len(prelabels_df[prelabels_df['pre_vic'] != ''])}")
    
    # Show label distribution
    if len(prelabels_df[prelabels_df['pre_att'] != '']) > 0:
        logger.info("Attacker pre-label distribution:")
        for label, count in prelabels_df['pre_att'].value_counts().items():
            if label != '':
                logger.info(f"  {label}: {count}")
    
    if len(prelabels_df[prelabels_df['pre_vic'] != '']) > 0:
        logger.info("Victim pre-label distribution:")
        for label, count in prelabels_df['pre_vic'].value_counts().items():
            if label != '':
                logger.info(f"  {label}: {count}")


if __name__ == "__main__":
    main()
