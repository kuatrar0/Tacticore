"""
Utility functions for demo parsing and data handling.
"""

import logging
import json
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_pandas_maybe(obj: Any) -> Optional[pd.DataFrame]:
    """
    Normalize various data types to pandas DataFrame.
    
    Args:
        obj: Object to convert (Polars DataFrame, Pandas DataFrame, dict, or None)
        
    Returns:
        Pandas DataFrame or None if conversion fails
    """
    if obj is None:
        return None
    
    try:
        if isinstance(obj, pl.DataFrame):
            return obj.to_pandas()
        elif isinstance(obj, pd.DataFrame):
            return obj.copy()
        elif isinstance(obj, dict):
            return pd.DataFrame(obj)
        else:
            logger.warning(f"Unknown data type: {type(obj)}")
            return None
    except Exception as e:
        logger.warning(f"Failed to convert to pandas: {e}")
        return None


def safe_save_parquet(df: Optional[pd.DataFrame], filepath: Path, table_name: str) -> bool:
    """
    Safely save DataFrame to parquet file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file
        table_name: Name of the table for logging
        
    Returns:
        True if saved successfully, False otherwise
    """
    if df is None or df.empty:
        logger.info(f"No data to save for {table_name}")
        return False
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {table_name} to {filepath}: {e}")
        return False


def safe_save_json(data: Dict, filepath: Path) -> bool:
    """
    Safely save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False


def get_awpy_attributes(demo_obj: Any) -> Dict[str, Any]:
    """
    Get available attributes from AWPy demo object, handling version differences.
    
    Args:
        demo_obj: AWPy Demo object
        
    Returns:
        Dictionary of available attributes and their row counts
    """
    available_attrs = {}
    
    attr_names = [
        'ticks', 'kills', 'damages', 'shots', 'grenades', 
        'smokes', 'infernos', 'bomb', 'rounds', 'header'
    ]
    
    for attr_name in attr_names:
        if hasattr(demo_obj, attr_name):
            attr_value = getattr(demo_obj, attr_name)
            if attr_value is not None:
                df = to_pandas_maybe(attr_value)
                if df is not None and not df.empty:
                    available_attrs[attr_name] = len(df)
                else:
                    available_attrs[attr_name] = 0
            else:
                available_attrs[attr_name] = 0
        else:
            available_attrs[attr_name] = None
    
    return available_attrs


def validate_demo_file(filepath: Path) -> bool:
    """
    Validate that a demo file exists and has correct extension.
    
    Args:
        filepath: Path to demo file
        
    Returns:
        True if valid, False otherwise
    """
    if not filepath.exists():
        logger.error(f"Demo file not found: {filepath}")
        return False
    
    if filepath.suffix.lower() != '.dem':
        logger.error(f"File must have .dem extension: {filepath}")
        return False
    
    return True


def get_demo_stem(filepath: Path) -> str:
    """
    Get demo name without extension and path.
    
    Args:
        filepath: Path to demo file
        
    Returns:
        Demo name stem
    """
    return filepath.stem


def update_dataset_index(index_file: Path, demo_name: str, saved_data: Dict[str, int]) -> None:
    """
    Update the global dataset index with demo information.
    
    Args:
        index_file: Path to index.csv file
        demo_name: Name of the demo
        saved_data: Dictionary of saved table names and row counts
    """
    try:
        index_entry = {
            'demo_name': demo_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            **{f'{table}_rows': count for table, count in saved_data.items()}
        }
        
        if index_file.exists():
            index_df = pd.read_csv(index_file)
            index_df = index_df[index_df['demo_name'] != demo_name]
        else:
            index_df = pd.DataFrame()
        
        new_df = pd.DataFrame([index_entry])
        index_df = pd.concat([index_df, new_df], ignore_index=True)
        
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(index_file, index=False)
        logger.info(f"Updated dataset index: {index_file}")
        
    except Exception as e:
        logger.error(f"Failed to update dataset index: {e}")


def calculate_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate 2D Euclidean distance between two points.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
        
    Returns:
        Distance in units
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate angle between two 2D points in degrees.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
        
    Returns:
        Angle in degrees (0-360)
    """
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return (angle_deg + 360) % 360
