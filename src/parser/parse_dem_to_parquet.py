#!/usr/bin/env python3
"""
CS2 Demo Parser - Convert .dem files to structured parquet datasets.

This script uses AWPy to parse Counter-Strike 2 demo files and extract
various game events and player data into parquet format for analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

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
        
        # Create demo object and parse
        demo = Demo(str(demo_path))
        demo.parse()
        
        # Get demo name and create output directory
        demo_name = get_demo_stem(demo_path)
        demo_output_dir = output_dir / demo_name
        demo_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved data
        saved_data = {}
        
        # Define tables to extract (in order of importance)
        tables_to_extract = [
            ('ticks', 'ticks.parquet'),
            ('kills', 'kills.parquet'),
            ('damages', 'damages.parquet'),
            ('shots', 'shots.parquet'),
            ('grenades', 'grenades.parquet'),
            ('smokes', 'smokes.parquet'),
            ('infernos', 'infernos.parquet'),
            ('bomb', 'bomb.parquet'),
            ('rounds', 'rounds.parquet'),
        ]
        
        # Extract each table
        for table_name, filename in tables_to_extract:
            if hasattr(demo, table_name):
                table_data = getattr(demo, table_name)
                df = to_pandas_maybe(table_data)
                
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
        
        # Save metadata
        meta_data = {
            'demo_file': str(demo_path),
            'map': getattr(demo, 'header', {}).get('map_name', 'unknown'),
            'saved': saved_data,
            'awpy_attributes': get_awpy_attributes(demo)
        }
        
        meta_filepath = demo_output_dir / 'meta.json'
        safe_save_json(meta_data, meta_filepath)
        
        logger.info(f"Successfully parsed {demo_path}")
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
