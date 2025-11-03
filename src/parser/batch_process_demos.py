#!/usr/bin/env python3
"""
Batch Demo Processor - Efficiently process multiple CS2 demo files for ML training.

This script processes multiple demo files in parallel and generates
enhanced datasets optimized for machine learning training.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parser.parse_dem_to_parquet import parse_single_demo, find_demo_files
from parser.utils import update_dataset_index, get_demo_stem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_demo_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function for processing a single demo file.
    
    Args:
        args: Tuple of (demo_path, output_dir, partition_rounds)
        
    Returns:
        Dictionary with processing results
    """
    demo_path, output_dir, partition_rounds = args
    
    try:
        result = parse_single_demo(demo_path, output_dir, partition_rounds)
        return {
            'demo_path': str(demo_path),
            'demo_name': get_demo_stem(demo_path),
            'success': True,
            'saved_data': result,
            'error': None
        }
    except Exception as e:
        return {
            'demo_path': str(demo_path),
            'demo_name': get_demo_stem(demo_path),
            'success': False,
            'saved_data': {},
            'error': str(e)
        }


def batch_process_demos(input_path: Path, output_dir: Path, 
                       max_workers: int = 4, partition_rounds: bool = False,
                       create_ml_dataset: bool = True) -> Dict[str, Any]:
    """
    Process multiple demo files in parallel.
    
    Args:
        input_path: Path to directory containing demo files
        output_dir: Output directory for processed data
        max_workers: Maximum number of parallel workers
        partition_rounds: Whether to partition data by rounds
        create_ml_dataset: Whether to create ML-ready dataset
        
    Returns:
        Dictionary with processing results
    """
    demo_files = find_demo_files(input_path)
    
    if not demo_files:
        logger.error("No demo files found")
        return {'success': False, 'error': 'No demo files found'}
    
    logger.info(f"Found {len(demo_files)} demo files to process")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    successful_demos = []
    failed_demos = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_demo = {
            executor.submit(process_demo_worker, (demo_file, output_dir, partition_rounds)): demo_file
            for demo_file in demo_files
        }
        
        with tqdm(total=len(demo_files), desc="Processing demos") as pbar:
            for future in as_completed(future_to_demo):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    successful_demos.append(result)
                    logger.info(f"Successfully processed: {result['demo_name']}")
                else:
                    failed_demos.append(result)
                    logger.error(f"Failed to process {result['demo_name']}: {result['error']}")
                
                pbar.update(1)
    
    index_file = output_dir / 'index.csv'
    for result in successful_demos:
        update_dataset_index(index_file, result['demo_name'], result['saved_data'])
    
    ml_dataset_path = None
    if create_ml_dataset and successful_demos:
        ml_dataset_path = create_ml_ready_dataset(output_dir, successful_demos)
    summary = {
        'total_demos': len(demo_files),
        'successful_demos': len(successful_demos),
        'failed_demos': len(failed_demos),
        'success_rate': len(successful_demos) / len(demo_files) * 100,
        'ml_dataset_path': ml_dataset_path,
        'failed_demos': [d['demo_name'] for d in failed_demos],
        'total_rows': sum(sum(d['saved_data'].values()) for d in successful_demos if d['saved_data'])
    }
    
    return summary


def create_ml_ready_dataset(output_dir: Path, successful_demos: List[Dict]) -> Path:
    """
    Create a ML-ready dataset from processed demo data.
    
    Args:
        output_dir: Output directory containing processed data
        successful_demos: List of successfully processed demos
        
    Returns:
        Path to the ML-ready dataset
    """
    logger.info("Creating ML-ready dataset...")
    
    all_kills = []
    all_ticks = []
    all_damages = []
    all_shots = []
    all_grenades = []
    all_rounds = []
    
    for demo_result in successful_demos:
        demo_name = demo_result['demo_name']
        demo_dir = output_dir / demo_name
        
        kills_file = demo_dir / 'kills.parquet'
        if kills_file.exists():
            kills_df = pd.read_parquet(kills_file)
            kills_df['demo_name'] = demo_name
            all_kills.append(kills_df)
        
        ticks_file = demo_dir / 'ticks.parquet'
        if ticks_file.exists():
            ticks_df = pd.read_parquet(ticks_file)
            ticks_df['demo_name'] = demo_name
            all_ticks.append(ticks_df)
        
        damages_file = demo_dir / 'damages.parquet'
        if damages_file.exists():
            damages_df = pd.read_parquet(damages_file)
            damages_df['demo_name'] = demo_name
            all_damages.append(damages_df)
        
        shots_file = demo_dir / 'shots.parquet'
        if shots_file.exists():
            shots_df = pd.read_parquet(shots_file)
            shots_df['demo_name'] = demo_name
            all_shots.append(shots_df)
        
        grenades_file = demo_dir / 'grenades.parquet'
        if grenades_file.exists():
            grenades_df = pd.read_parquet(grenades_file)
            grenades_df['demo_name'] = demo_name
            all_grenades.append(grenades_df)
        
        rounds_file = demo_dir / 'rounds.parquet'
        if rounds_file.exists():
            rounds_df = pd.read_parquet(rounds_file)
            rounds_df['demo_name'] = demo_name
            all_rounds.append(rounds_df)
    
    ml_dataset_dir = output_dir / 'ml_dataset'
    ml_dataset_dir.mkdir(exist_ok=True)
    
    if all_kills:
        combined_kills = pd.concat(all_kills, ignore_index=True)
        combined_kills.to_parquet(ml_dataset_dir / 'all_kills.parquet', index=False)
        logger.info(f"Saved {len(combined_kills)} kills to ML dataset")
    
    if all_ticks:
        combined_ticks = pd.concat(all_ticks, ignore_index=True)
        combined_ticks.to_parquet(ml_dataset_dir / 'all_ticks.parquet', index=False)
        logger.info(f"Saved {len(combined_ticks)} ticks to ML dataset")
    
    if all_damages:
        combined_damages = pd.concat(all_damages, ignore_index=True)
        combined_damages.to_parquet(ml_dataset_dir / 'all_damages.parquet', index=False)
        logger.info(f"Saved {len(combined_damages)} damage events to ML dataset")
    
    if all_shots:
        combined_shots = pd.concat(all_shots, ignore_index=True)
        combined_shots.to_parquet(ml_dataset_dir / 'all_shots.parquet', index=False)
        logger.info(f"Saved {len(combined_shots)} shots to ML dataset")
    
    if all_grenades:
        combined_grenades = pd.concat(all_grenades, ignore_index=True)
        combined_grenades.to_parquet(ml_dataset_dir / 'all_grenades.parquet', index=False)
        logger.info(f"Saved {len(combined_grenades)} grenades to ML dataset")
    
    if all_rounds:
        combined_rounds = pd.concat(all_rounds, ignore_index=True)
        combined_rounds.to_parquet(ml_dataset_dir / 'all_rounds.parquet', index=False)
        logger.info(f"Saved {len(combined_rounds)} rounds to ML dataset")
    
    metadata = {
        'total_demos': len(successful_demos),
        'demo_names': [d['demo_name'] for d in successful_demos],
        'total_kills': len(combined_kills) if all_kills else 0,
        'total_ticks': len(combined_ticks) if all_ticks else 0,
        'total_damages': len(combined_damages) if all_damages else 0,
        'total_shots': len(combined_shots) if all_shots else 0,
        'total_grenades': len(combined_grenades) if all_grenades else 0,
        'total_rounds': len(combined_rounds) if all_rounds else 0,
        'created_at': pd.Timestamp.now().isoformat()
    }
    
    with open(ml_dataset_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"ML dataset created at: {ml_dataset_dir}")
    return ml_dataset_dir


def main():
    """Main function to batch process demo files."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple CS2 demo files for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_demos.py -i "C:\\demos\\" -o dataset --max-workers 8
  python batch_process_demos.py -i "C:\\demos\\" -o dataset --partition-rounds --no-ml-dataset
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Input directory containing .dem files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('dataset'),
        help='Output directory for processed data (default: dataset)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--partition-rounds',
        action='store_true',
        help='Partition data by rounds (creates additional round-specific files)'
    )
    
    parser.add_argument(
        '--no-ml-dataset',
        action='store_true',
        help='Skip creation of ML-ready dataset'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting batch processing of demos in: {args.input}")
    
    summary = batch_process_demos(
        input_path=args.input,
        output_dir=args.output,
        max_workers=args.max_workers,
        partition_rounds=args.partition_rounds,
        create_ml_dataset=not args.no_ml_dataset
    )
    
    logger.info("Batch processing complete!")
    logger.info(f"Total demos: {summary['total_demos']}")
    logger.info(f"Successful: {summary['successful_demos']}")
    logger.info(f"Failed: {summary['failed_demos']}")
    logger.info(f"Success rate: {summary['success_rate']:.1f}%")
    logger.info(f"Total rows processed: {summary['total_rows']:,}")
    
    if summary['ml_dataset_path']:
        logger.info(f"ML dataset created at: {summary['ml_dataset_path']}")
    
    if summary['failed_demos']:
        logger.warning(f"Failed demos: {', '.join(summary['failed_demos'])}")
    
    logger.info(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
