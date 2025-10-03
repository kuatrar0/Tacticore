#!/usr/bin/env python3
"""
CS2 Demo Parser - Convert .dem files to structured parquet datasets.

This script uses AWPy to parse Counter-Strike 2 demo files and extract
various game events and player data into parquet format for analysis.
Enhanced version with more detailed context for ML training.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parser.utils import (
    to_pandas_maybe, safe_save_parquet, safe_save_json, 
    get_awpy_attributes, validate_demo_file, get_demo_stem,
    update_dataset_index
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_enhanced_data(demo) -> Dict[str, pd.DataFrame]:
    """
    Extract enhanced data from demo with additional context.
    
    Args:
        demo: AWPy Demo object
        
    Returns:
        Dictionary of DataFrames with enhanced data
    """
    enhanced_data = {}
    
    # Extract basic tables
    basic_tables = [
        'ticks', 'kills', 'damages', 'shots', 'grenades', 
        'smokes', 'infernos', 'bomb', 'rounds'
    ]
    
    for table_name in basic_tables:
        if hasattr(demo, table_name):
            table_data = getattr(demo, table_name)
            df = to_pandas_maybe(table_data)
            if df is not None and not df.empty:
                enhanced_data[table_name] = df
    
    # Extract additional context data
    enhanced_data.update(extract_sound_events(demo))
    enhanced_data.update(extract_player_states(demo))
    enhanced_data.update(extract_round_context(demo))
    
    return enhanced_data


def extract_sound_events(demo) -> Dict[str, pd.DataFrame]:
    """
    Extract sound-related events for audio context analysis.
    
    Args:
        demo: AWPy Demo object
        
    Returns:
        Dictionary with sound events DataFrames
    """
    sound_data = {}
    
    # Try to extract footsteps, weapon sounds, etc.
    # Note: AWPy may not expose all sound events directly
    # We'll work with what's available and enhance with derived features
    
    if hasattr(demo, 'damages'):
        damages_df = to_pandas_maybe(demo.damages)
        if damages_df is not None and not damages_df.empty:
            # Add sound context to damages
            damages_df['has_sound_cue'] = True
            damages_df['sound_type'] = 'damage'
            sound_data['damage_sounds'] = damages_df
    
    if hasattr(demo, 'shots'):
        shots_df = to_pandas_maybe(demo.shots)
        if shots_df is not None and not shots_df.empty:
            # Add sound context to shots
            shots_df['has_sound_cue'] = True
            shots_df['sound_type'] = 'shot'
            sound_data['shot_sounds'] = shots_df
    
    return sound_data


def extract_player_states(demo) -> Dict[str, pd.DataFrame]:
    """
    Extract detailed player state information.
    
    Args:
        demo: AWPy Demo object
        
    Returns:
        Dictionary with player state DataFrames
    """
    player_data = {}
    
    if hasattr(demo, 'ticks'):
        ticks_df = to_pandas_maybe(demo.ticks)
        if ticks_df is not None and not ticks_df.empty:
            # Enhance ticks with derived features
            enhanced_ticks = enhance_ticks_data(ticks_df)
            player_data['enhanced_ticks'] = enhanced_ticks
    
    return player_data


def enhance_ticks_data(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance tick data with derived features for better context.
    
    Args:
        ticks_df: Original ticks DataFrame
        
    Returns:
        Enhanced ticks DataFrame
    """
    if ticks_df.empty:
        return ticks_df
    
    enhanced = ticks_df.copy()
    
    # Add derived features
    enhanced['is_alive'] = enhanced.get('health', 100) > 0
    enhanced['is_ducking'] = enhanced.get('ducking', False)
    enhanced['is_scoped'] = enhanced.get('scoped', False)
    
    # Calculate movement speed
    if 'vel_x' in enhanced.columns and 'vel_y' in enhanced.columns:
        enhanced['movement_speed'] = np.sqrt(
            enhanced['vel_x']**2 + enhanced['vel_y']**2
        )
        enhanced['is_moving'] = enhanced['movement_speed'] > 10
    else:
        enhanced['movement_speed'] = 0
        enhanced['is_moving'] = False
    
    # Add weapon context
    enhanced['has_primary'] = enhanced.get('primary_weapon', '') != ''
    enhanced['has_secondary'] = enhanced.get('secondary_weapon', '') != ''
    enhanced['has_utility'] = enhanced.get('utility_weapon', '') != ''
    
    return enhanced


def extract_round_context(demo) -> Dict[str, pd.DataFrame]:
    """
    Extract round-level context information.
    
    Args:
        demo: AWPy Demo object
        
    Returns:
        Dictionary with round context DataFrames
    """
    round_data = {}
    
    if hasattr(demo, 'rounds'):
        rounds_df = to_pandas_maybe(demo.rounds)
        if rounds_df is not None and not rounds_df.empty:
            # Enhance round data with additional context
            enhanced_rounds = enhance_rounds_data(rounds_df)
            round_data['enhanced_rounds'] = enhanced_rounds
    
    return round_data


def enhance_rounds_data(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance round data with additional context.
    
    Args:
        rounds_df: Original rounds DataFrame
        
    Returns:
        Enhanced rounds DataFrame
    """
    if rounds_df.empty:
        return rounds_df
    
    enhanced = rounds_df.copy()
    
    # Map AWPy column names to expected names
    column_mapping = {
        'start': 'start_tick',
        'end': 'end_tick', 
        'official_end': 'official_end_tick',
        'round_num': 'round',
        'winner': 'winner',
        'reason': 'round_winner_reason',
        'bomb_plant': 'bomb_plant_tick',
        'bomb_site': 'bomb_site'
    }
    
    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in enhanced.columns:
            enhanced = enhanced.rename(columns={old_name: new_name})
    
    # Add round phase information
    enhanced['round_phase'] = 'unknown'
    
    # Add time-based features
    if 'start_tick' in enhanced.columns and 'end_tick' in enhanced.columns:
        enhanced['round_duration_ticks'] = enhanced['end_tick'] - enhanced['start_tick']
        enhanced['round_duration_seconds'] = enhanced['round_duration_ticks'] / 64  # Assuming 64 tickrate
    
    return enhanced


def parse_single_demo(demo_path: Path, output_dir: Path, partition_rounds: bool = False) -> Dict[str, int]:
    """
    Parse a single demo file and save data to parquet files.
    
    Args:
        demo_path: Path to the demo file
        output_dir: Directory to save parsed data
        partition_rounds: Whether to partition data by rounds
        
    Returns:
        Dictionary of saved table names and row counts
    """
    try:
        # Import AWPy here to handle import errors gracefully
        from awpy import Demo
        
        logger.info(f"Parsing demo: {demo_path}")
        
        # Create demo object and parse (AWPy 2.0+ API)
        demo = Demo(demo_path)
        demo.parse()
        
        # Get demo name and create output directory
        demo_name = get_demo_stem(demo_path)
        demo_output_dir = output_dir / demo_name
        demo_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved data
        saved_data = {}
        
        # Extract enhanced data
        enhanced_data = extract_enhanced_data(demo)
        
        # Define tables to extract (in order of importance)
        tables_to_extract = [
            ('enhanced_ticks', 'ticks.parquet'),
            ('kills', 'kills.parquet'),
            ('damages', 'damages.parquet'),
            ('shots', 'shots.parquet'),
            ('grenades', 'grenades.parquet'),
            ('smokes', 'smokes.parquet'),
            ('infernos', 'infernos.parquet'),
            ('bomb', 'bomb.parquet'),
            ('enhanced_rounds', 'rounds.parquet'),
            ('damage_sounds', 'damage_sounds.parquet'),
            ('shot_sounds', 'shot_sounds.parquet'),
        ]
        
        # Extract each table
        for table_name, filename in tables_to_extract:
            if table_name in enhanced_data:
                df = enhanced_data[table_name]
                
                if df is not None and not df.empty:
                    filepath = demo_output_dir / filename
                    if safe_save_parquet(df, filepath, table_name):
                        saved_data[table_name] = len(df)
                        
                        # Optionally partition by rounds
                        if partition_rounds and 'round' in df.columns:
                            partition_by_rounds(df, demo_output_dir, table_name)
                else:
                    logger.info(f"No data for {table_name}")
                    saved_data[table_name] = 0
            else:
                logger.warning(f"Table {table_name} not available in this AWPy version")
                saved_data[table_name] = None
        
        # Debug: Check what's in the demo header
        demo_header = getattr(demo, 'header', {})
        print(f"DEBUG: Demo header keys: {list(demo_header.keys()) if demo_header else 'No header'}")
        print(f"DEBUG: Demo header content: {demo_header}")
        
        # Try different possible map name fields
        map_name = 'unknown'
        if demo_header:
            for key in ['map_name', 'mapName', 'map', 'Map', 'mapname']:
                if key in demo_header:
                    map_name = demo_header[key]
                    print(f"DEBUG: Found map name '{map_name}' in header field '{key}'")
                    break
        
        # Save metadata
        meta_data = {
            'demo_file': str(demo_path),
            'map': map_name,
            'saved': saved_data,
            'awpy_attributes': get_awpy_attributes(demo)
        }
        
        meta_filepath = demo_output_dir / 'meta.json'
        safe_save_json(meta_data, meta_filepath)
        
        logger.info(f"Successfully parsed {demo_path}")
        
        # Add map name to saved_data for easy access
        saved_data['map'] = meta_data['map']
        
        return saved_data
        
    except ImportError:
        logger.error("AWPy not installed. Please install with: pip install awpy")
        return {}
    except Exception as e:
        logger.error(f"Failed to parse {demo_path}: {e}")
        return {}


def partition_by_rounds(df: pd.DataFrame, output_dir: Path, table_name: str) -> None:
    """
    Partition DataFrame by rounds and save separate files.
    
    Args:
        df: DataFrame to partition
        output_dir: Output directory
        table_name: Name of the table
    """
    try:
        if 'round' not in df.columns:
            return
            
        rounds_dir = output_dir / f"{table_name}_by_round"
        rounds_dir.mkdir(exist_ok=True)
        
        for round_num, round_df in df.groupby('round'):
            if not round_df.empty:
                round_filepath = rounds_dir / f"round_{round_num}.parquet"
                safe_save_parquet(round_df, round_filepath, f"{table_name}_round_{round_num}")
                
    except Exception as e:
        logger.warning(f"Failed to partition {table_name} by rounds: {e}")


def find_demo_files(input_path: Path) -> List[Path]:
    """
    Find all .dem files in the input path.
    
    Args:
        input_path: Path to file or directory
        
    Returns:
        List of demo file paths
    """
    demo_files = []
    
    if input_path.is_file():
        if validate_demo_file(input_path):
            demo_files.append(input_path)
    elif input_path.is_dir():
        demo_files = list(input_path.glob("*.dem"))
        demo_files.extend(list(input_path.glob("**/*.dem")))  # Recursive search
    else:
        logger.error(f"Input path does not exist: {input_path}")
    
    return demo_files


def main():
    """Main function to parse demo files."""
    parser = argparse.ArgumentParser(
        description="Parse CS2 demo files to parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_dem_to_parquet.py -i "C:\\demos\\match.dem" -o dataset
  python parse_dem_to_parquet.py -i "C:\\demos\\" -o dataset --partition-rounds
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Input demo file or directory containing .dem files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('dataset'),
        help='Output directory for parsed data (default: dataset)'
    )
    
    parser.add_argument(
        '--partition-rounds',
        action='store_true',
        help='Partition data by rounds (creates additional round-specific files)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find demo files
    demo_files = find_demo_files(args.input)
    
    if not demo_files:
        logger.error("No valid demo files found")
        sys.exit(1)
    
    logger.info(f"Found {len(demo_files)} demo file(s) to process")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process each demo file
    total_saved = {}
    for demo_file in tqdm(demo_files, desc="Parsing demos"):
        saved_data = parse_single_demo(demo_file, args.output, args.partition_rounds)
        
        # Update total counts
        for table, count in saved_data.items():
            if count is not None:
                total_saved[table] = total_saved.get(table, 0) + count
    
    # Update global dataset index
    index_file = args.output / 'index.csv'
    for demo_file in demo_files:
        demo_name = get_demo_stem(demo_file)
        demo_saved = parse_single_demo(demo_file, args.output, args.partition_rounds)
        update_dataset_index(index_file, demo_name, demo_saved)
    
    # Print summary
    logger.info("Parsing complete!")
    logger.info(f"Output directory: {args.output}")
    logger.info("Total rows saved:")
    for table, count in total_saved.items():
        logger.info(f"  {table}: {count:,} rows")


if __name__ == "__main__":
    main()
