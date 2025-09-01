#!/usr/bin/env python3
"""
Combine multiple parsed demo datasets into a single dataset.

This script takes multiple parsed demo directories and combines them into
a single dataset with all kills, ticks, rounds, etc. from all games.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_parsed_demos(input_dir: Path) -> List[Path]:
    """
    Find all parsed demo directories.
    
    Args:
        input_dir: Directory containing parsed demo folders
        
    Returns:
        List of parsed demo directory paths
    """
    demo_dirs = []
    
    if input_dir.is_dir():
        # Look for directories that contain parquet files
        for item in input_dir.iterdir():
            if item.is_dir():
                # Check if this directory contains parquet files
                parquet_files = list(item.glob("*.parquet"))
                if parquet_files:
                    demo_dirs.append(item)
                    logger.info(f"Found parsed demo: {item.name}")
    
    return demo_dirs


def combine_parquet_files(demo_dirs: List[Path], output_dir: Path) -> Dict[str, int]:
    """
    Combine parquet files from multiple demos into a single dataset.
    
    Args:
        demo_dirs: List of parsed demo directories
        output_dir: Output directory for combined dataset
        
    Returns:
        Dictionary with file counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track combined data
    combined_data = {}
    file_counts = {}
    
    # Process each demo directory
    for demo_dir in tqdm(demo_dirs, desc="Combining datasets"):
        demo_name = demo_dir.name
        logger.info(f"Processing {demo_name}...")
        
        # Find all parquet files in this demo
        parquet_files = list(demo_dir.glob("*.parquet"))
        
        for parquet_file in parquet_files:
            file_type = parquet_file.stem  # e.g., "kills", "ticks", "rounds"
            
            try:
                # Read the parquet file
                df = pd.read_parquet(parquet_file)
                
                # Add demo identifier
                df['demo_name'] = demo_name
                df['demo_path'] = str(demo_dir)
                
                # Initialize combined data for this file type if not exists
                if file_type not in combined_data:
                    combined_data[file_type] = []
                
                # Add to combined data
                combined_data[file_type].append(df)
                
                logger.info(f"  Added {len(df)} rows from {file_type}.parquet")
                
            except Exception as e:
                logger.error(f"Error reading {parquet_file}: {e}")
    
    # Combine and save each file type
    for file_type, dataframes in combined_data.items():
        if dataframes:
            try:
                # Combine all dataframes for this file type
                combined_df = pd.concat(dataframes, ignore_index=True)
                
                # Save combined file
                output_file = output_dir / f"{file_type}.parquet"
                combined_df.to_parquet(output_file, index=False)
                
                file_counts[file_type] = len(combined_df)
                logger.info(f"Saved combined {file_type}.parquet: {len(combined_df):,} rows")
                
            except Exception as e:
                logger.error(f"Error combining {file_type}: {e}")
    
    return file_counts


def create_combined_metadata(demo_dirs: List[Path], output_dir: Path, file_counts: Dict[str, int]):
    """
    Create metadata for the combined dataset.
    
    Args:
        demo_dirs: List of parsed demo directories
        output_dir: Output directory
        file_counts: Dictionary with file counts
    """
    metadata = {
        "combined_dataset": True,
        "source_demos": [demo.name for demo in demo_dirs],
        "demo_count": len(demo_dirs),
        "file_counts": file_counts,
        "total_kills": file_counts.get("kills", 0),
        "total_rounds": file_counts.get("rounds", 0),
        "combined_at": pd.Timestamp.now().isoformat()
    }
    
    # Save metadata
    metadata_file = output_dir / "combined_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata: {metadata_file}")


def main():
    """Main function to combine datasets."""
    parser = argparse.ArgumentParser(
        description="Combine multiple parsed demo datasets into a single dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combine_datasets.py -i "C:/demos/parsed" -o "C:/demos/combined"
  python combine_datasets.py -i "C:/demos/parsed" -o "dataset/combined"
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Input directory containing parsed demo folders'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('combined_dataset'),
        help='Output directory for combined dataset (default: combined_dataset)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find parsed demo directories
    demo_dirs = find_parsed_demos(args.input)
    
    if not demo_dirs:
        logger.error("No parsed demo directories found")
        sys.exit(1)
    
    logger.info(f"Found {len(demo_dirs)} parsed demo directories")
    
    # Combine datasets
    file_counts = combine_parquet_files(demo_dirs, args.output)
    
    # Create metadata
    create_combined_metadata(demo_dirs, args.output, file_counts)
    
    # Print summary
    logger.info("Dataset combination complete!")
    logger.info(f"Output directory: {args.output}")
    logger.info("Combined files:")
    for file_type, count in file_counts.items():
        logger.info(f"  {file_type}.parquet: {count:,} rows")


if __name__ == "__main__":
    main()
