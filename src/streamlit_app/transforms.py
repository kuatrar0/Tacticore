"""
Transform functions for the Streamlit labeling app.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path


def world_to_map_coords(x: float, y: float, map_data: Dict, x_adjust: float = 50, y_adjust: float = -30) -> Tuple[float, float]:
    """
    Convert CS2 world coordinates to map image coordinates.
    
    Args:
        x, y: World coordinates
        map_data: Dictionary with pos_x, pos_y, scale
        x_adjust: Fine-tuning adjustment for X axis
        y_adjust: Fine-tuning adjustment for Y axis
        
    Returns:
        Tuple of (image_x, image_y) coordinates
    """
    pos_x = map_data.get('pos_x', 0)
    pos_y = map_data.get('pos_y', 0)
    scale = map_data.get('scale', 1.0)
    
    # Apply transformation
    # Center the coordinates and scale them
    image_x = (x - pos_x) / scale
    image_y = (pos_y - y) / scale  # Note: Y is inverted for image coordinates
    
    # Add offset to center the map in the 1024x1024 image
    # Fine-tuned offsets for better positioning
    image_x += 512  # Center horizontally
    image_y += 512  # Center vertically
    
    # Apply fine-tuning adjustments
    image_x += x_adjust
    image_y += y_adjust
    
    # Clamp to reasonable bounds for a 1024x1024 image
    image_x = max(0, min(image_x, 1023))
    image_y = max(0, min(image_y, 1023))
    
    return image_x, image_y


def world_to_map_coords_advanced(x: float, y: float, map_data: Dict, x_adjust: float = 50, y_adjust: float = -30) -> Tuple[float, float]:
    """
    Advanced CS2 world to map coordinate conversion with area-specific adjustments.
    
    Args:
        x, y: World coordinates
        map_data: Dictionary with pos_x, pos_y, scale
        x_adjust: Fine-tuning adjustment for X axis
        y_adjust: Fine-tuning adjustment for Y axis
        
    Returns:
        Tuple of (image_x, image_y) coordinates
    """
    pos_x = map_data.get('pos_x', 0)
    pos_y = map_data.get('pos_y', 0)
    scale = map_data.get('scale', 1.0)
    
    # Base transformation
    image_x = (x - pos_x) / scale
    image_y = (pos_y - y) / scale
    
    # Add centering offset
    image_x += 512
    image_y += 512
    
    # Apply area-specific adjustments based on coordinate ranges
    # This helps with maps that have different coordinate systems in different areas
    
    # For de_mirage specifically, apply different adjustments based on area
    if abs(x) > 2000 or abs(y) > 2000:
        # Far areas might need different scaling
        image_x *= 0.95  # Slight scale adjustment for far areas
        image_y *= 0.95
    
    # Apply fine-tuning adjustments
    image_x += x_adjust
    image_y += y_adjust
    
    # Clamp to image bounds
    image_x = max(0, min(image_x, 1023))
    image_y = max(0, min(image_y, 1023))
    
    return image_x, image_y


def world_to_map_coords_fixed(x: float, y: float, map_data: Dict, x_adjust: float = 25, y_adjust: float = 0) -> Tuple[float, float]:
    """
    Fixed CS2 world to map coordinate conversion that maps to the actual playable map area.
    
    For de_mirage.png:
    - Full image: 1024x1024
    - Actual map area: 990x832 at offset (0, 97)
    
    Args:
        x, y: World coordinates
        map_data: Dictionary with pos_x, pos_y, scale
        x_adjust: Fine-tuning adjustment for X axis
        y_adjust: Fine-tuning adjustment for Y axis
        
    Returns:
        Tuple of (image_x, image_y) coordinates
    """
    pos_x = map_data.get('pos_x', 0)
    pos_y = map_data.get('pos_y', 0)
    scale = map_data.get('scale', 1.0)
    
    # Map dimensions for de_mirage (actual playable area)
    MAP_OFFSET_X = 0
    MAP_OFFSET_Y = 97
    MAP_WIDTH = 990
    MAP_HEIGHT = 832
    
    # Base transformation to map coordinate space
    map_x = (x - pos_x) / scale
    map_y = (pos_y - y) / scale  # Y is inverted
    
    # Center within the actual map area (not the full image)
    map_x += MAP_WIDTH / 2   # Center in 990px width
    map_y += MAP_HEIGHT / 2  # Center in 832px height
    
    # Apply fine-tuning adjustments
    map_x += x_adjust
    map_y += y_adjust
    
    # Convert from map space to full image space
    image_x = MAP_OFFSET_X + map_x
    image_y = MAP_OFFSET_Y + map_y
    
    # Clamp to the actual map area bounds
    image_x = max(MAP_OFFSET_X, min(image_x, MAP_OFFSET_X + MAP_WIDTH - 1))
    image_y = max(MAP_OFFSET_Y, min(image_y, MAP_OFFSET_Y + MAP_HEIGHT - 1))
    
    return image_x, image_y


def debug_coordinate_transformation(x: float, y: float, map_data: Dict, x_adjust: float = 50, y_adjust: float = -30) -> Dict:
    """
    Debug coordinate transformation to understand what's happening.
    
    Args:
        x, y: World coordinates
        map_data: Dictionary with pos_x, pos_y, scale
        x_adjust, y_adjust: Fine-tuning adjustments
        
    Returns:
        Dictionary with debug information
    """
    pos_x = map_data.get('pos_x', 0)
    pos_y = map_data.get('pos_y', 0)
    scale = map_data.get('scale', 1.0)
    
    # Step-by-step transformation
    step1_x = x - pos_x
    step1_y = pos_y - y
    
    step2_x = step1_x / scale
    step2_y = step1_y / scale
    
    step3_x = step2_x + 512
    step3_y = step2_y + 512
    
    step4_x = step3_x + x_adjust
    step4_y = step3_y + y_adjust
    
    return {
        'original': (x, y),
        'step1_centered': (step1_x, step1_y),
        'step2_scaled': (step2_x, step2_y),
        'step3_centered': (step3_x, step3_y),
        'step4_adjusted': (step4_x, step4_y),
        'final': (max(0, min(step4_x, 1023)), max(0, min(step4_y, 1023))),
        'parameters': {
            'pos_x': pos_x,
            'pos_y': pos_y,
            'scale': scale,
            'x_adjust': x_adjust,
            'y_adjust': y_adjust
        }
    }


def load_map_data(map_data_path: str, map_name: str = "de_mirage") -> Dict:
    """
    Load map coordinate data from JSON file.
    
    Args:
        map_data_path: Path to map-data.json file
        map_name: Name of the map to load (default: de_mirage)
        
    Returns:
        Dictionary with map coordinate data
    """
    try:
        with open(map_data_path, 'r') as f:
            data = json.load(f)
        
        # If data contains multiple maps, select the specific one
        if isinstance(data, dict) and map_name in data:
            return data[map_name]
        elif isinstance(data, dict) and len(data) == 1:
            # Assume first key is the map name
            map_name = list(data.keys())[0]
            return data[map_name]
        
        return data
    except Exception as e:
        print(f"Error loading map data: {e}")
        return {}


def suggest_transformation_params(ticks_df: pd.DataFrame) -> Dict:
    """
    Suggest transformation parameters based on actual coordinate ranges.
    
    Args:
        ticks_df: DataFrame with tick data containing X, Y coordinates
        
    Returns:
        Dictionary with suggested pos_x, pos_y, scale values
    """
    if ticks_df.empty:
        return {'pos_x': -2000, 'pos_y': 2000, 'scale': 8.0}
    
    # Get coordinate ranges - use the actual column names from AWPy
    if 'X' not in ticks_df.columns or 'Y' not in ticks_df.columns:
        return {'pos_x': -2000, 'pos_y': 2000, 'scale': 8.0}
    
    x_coords = ticks_df['X'].dropna()
    y_coords = ticks_df['Y'].dropna()
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return {'pos_x': -2000, 'pos_y': 2000, 'scale': 8.0}
    
    # Calculate ranges
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Calculate center and range
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # For CS2 maps, we want to map the world coordinates to a 1024x1024 image
    # The transformation should center the map and scale it appropriately
    # CS2 world coordinates are typically centered around (0,0) with ranges in the thousands
    
    # Calculate scale to fit the larger dimension into ~900 pixels (leaving some margin)
    # Since we add 512 offset to center, we want the range to fit within ~900 pixels
    max_range = max(x_range, y_range)
    suggested_scale = max_range / 900
    
    # The pos_x and pos_y should be the center of the coordinate system
    # This will be subtracted from world coordinates to center them around 0
    suggested_pos_x = x_center
    suggested_pos_y = y_center
    
    print(f"Coordinate analysis:")
    print(f"  X range: {x_min:.1f} to {x_max:.1f} (center: {x_center:.1f})")
    print(f"  Y range: {y_min:.1f} to {y_max:.1f} (center: {y_center:.1f})")
    print(f"  Max range: {max_range:.1f}")
    print(f"  Suggested: pos_x={suggested_pos_x:.1f}, pos_y={suggested_pos_y:.1f}, scale={suggested_scale:.1f}")
    
    return {
        'pos_x': suggested_pos_x,
        'pos_y': suggested_pos_y,
        'scale': suggested_scale
    }


def find_nearest_tick(kill_tick: int, ticks_df: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
    """
    Find the nearest tick data for a player around a kill event.
    
    Args:
        kill_tick: Tick when the kill occurred
        ticks_df: DataFrame with tick data
        player_name: Name of the player to find
        
    Returns:
        Series with nearest tick data or None if not found
    """
    if ticks_df.empty:
        return None
    
    # Try different possible column names for player identification
    player_col = None
    for col_name in ['player_name', 'name', 'player', 'attacker_name', 'victim_name']:
        if col_name in ticks_df.columns:
            player_col = col_name
            break
    
    if player_col is None:
        # If no player column found, return None
        return None
    
    # Filter ticks for this player
    player_ticks = ticks_df[ticks_df[player_col] == player_name].copy()
    
    if player_ticks.empty:
        return None
    
    # Find nearest tick
    player_ticks['tick_diff'] = abs(player_ticks['tick'] - kill_tick)
    nearest_idx = player_ticks['tick_diff'].idxmin()
    
    return player_ticks.loc[nearest_idx]


def calculate_time_in_round(kill_tick: int, rounds_df: pd.DataFrame, tickrate: int = 64) -> float:
    """
    Calculate time in round for a kill event.
    
    Args:
        kill_tick: Tick when the kill occurred
        rounds_df: DataFrame with round data
        tickrate: Game tickrate (default 64)
        
    Returns:
        Time in seconds since round start
    """
    if rounds_df.empty:
        # If no round data, estimate based on tick
        return kill_tick / tickrate
    
    # Find the round this kill belongs to
    for _, round_data in rounds_df.iterrows():
        start_tick = round_data.get('start_tick', 0)
        end_tick = round_data.get('end_tick', float('inf'))
        
        if start_tick <= kill_tick <= end_tick:
            return (kill_tick - start_tick) / tickrate
    
    # Fallback to tick-based calculation
    return kill_tick / tickrate


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


def calculate_approach_alignment(attacker_x: float, attacker_y: float, 
                               victim_x: float, victim_y: float,
                               attacker_vel_x: float, attacker_vel_y: float) -> float:
    """
    Calculate alignment of attacker movement toward victim.
    
    Args:
        attacker_x, attacker_y: Attacker position
        victim_x, victim_y: Victim position
        attacker_vel_x, attacker_vel_y: Attacker velocity
        
    Returns:
        Alignment angle in degrees (0-180, where 0 is perfect alignment)
    """
    # Vector from attacker to victim
    to_victim_x = victim_x - attacker_x
    to_victim_y = victim_y - attacker_y
    
    # Normalize vectors
    to_victim_mag = np.sqrt(to_victim_x**2 + to_victim_y**2)
    vel_mag = np.sqrt(attacker_vel_x**2 + attacker_vel_y**2)
    
    if to_victim_mag == 0 or vel_mag == 0:
        return 90.0  # Default to perpendicular
    
    to_victim_x /= to_victim_mag
    to_victim_y /= to_victim_mag
    attacker_vel_x /= vel_mag
    attacker_vel_y /= vel_mag
    
    # Calculate dot product
    dot_product = to_victim_x * attacker_vel_x + to_victim_y * attacker_vel_y
    
    # Clamp to valid range
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Convert to angle
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def find_nearby_utility(kill_tick: int, kill_x: float, kill_y: float, 
                       grenades_df: pd.DataFrame, radius: float = 500.0) -> Dict[str, bool]:
    """
    Find nearby utility at the time of kill.
    
    Args:
        kill_tick: Tick when the kill occurred
        kill_x, kill_y: Position of the kill
        grenades_df: DataFrame with grenade data
        radius: Search radius in units
        
    Returns:
        Dictionary with utility flags
    """
    if grenades_df.empty:
        return {
            'flash_near': False,
            'smoke_near': False,
            'molotov_near': False,
            'he_near': False
        }
    
    # Filter grenades around the kill time (±2 seconds)
    time_window = 128  # 2 seconds at 64 tickrate
    nearby_grenades = grenades_df[
        (grenades_df['tick'] >= kill_tick - time_window) &
        (grenades_df['tick'] <= kill_tick + time_window)
    ].copy()
    
    if nearby_grenades.empty:
        return {
            'flash_near': False,
            'smoke_near': False,
            'molotov_near': False,
            'he_near': False
        }
    
    # Calculate distances - AWPy uses uppercase X, Y for coordinates
    nearby_grenades['distance'] = calculate_distance_2d(
        kill_x, kill_y,
        nearby_grenades.get('X', nearby_grenades.get('x', 0)), 
        nearby_grenades.get('Y', nearby_grenades.get('y', 0))
    )
    
    # Check for nearby utility
    nearby = nearby_grenades[nearby_grenades['distance'] <= radius]
    
    utility_flags = {
        'flash_near': False,
        'smoke_near': False,
        'molotov_near': False,
        'he_near': False
    }
    
    for _, grenade in nearby.iterrows():
        grenade_type = str(grenade.get('grenade_type', '')).lower()
        
        if 'flash' in grenade_type:
            utility_flags['flash_near'] = True
        elif 'smoke' in grenade_type:
            utility_flags['smoke_near'] = True
        elif 'molotov' in grenade_type or 'incendiary' in grenade_type:
            utility_flags['molotov_near'] = True
        elif 'he' in grenade_type or 'grenade' in grenade_type:
            utility_flags['he_near'] = True
    
    return utility_flags


def get_kill_context(kill_row: pd.Series, ticks_df: pd.DataFrame, 
                     rounds_df: pd.DataFrame, grenades_df: pd.DataFrame,
                     map_data: Dict, tickrate: int = 64, x_adjust: float = 25, y_adjust: float = 0, 
                     use_advanced: bool = True) -> Dict:
    """
    Get comprehensive context for a kill event.
    
    Args:
        kill_row: Row from kills DataFrame
        ticks_df: DataFrame with tick data
        rounds_df: DataFrame with round data
        grenades_df: DataFrame with grenade data
        map_data: Map coordinate data
        tickrate: Game tickrate
        
    Returns:
        Dictionary with kill context
    """
    context = {}
    
    # Basic kill info
    context['kill_tick'] = kill_row.get('tick', 0)
    context['attacker_name'] = kill_row.get('attacker_name', 'Unknown')
    context['victim_name'] = kill_row.get('victim_name', 'Unknown')
    # Use correct column names from AWPy data
    context['side'] = kill_row.get('attacker_side', kill_row.get('side', 'Unknown'))
    context['place'] = kill_row.get('attacker_place', kill_row.get('place', 'Unknown'))
    context['headshot'] = kill_row.get('headshot', False)
    
    # Find nearest tick data for attacker and victim
    attacker_tick = find_nearest_tick(context['kill_tick'], ticks_df, context['attacker_name'])
    victim_tick = find_nearest_tick(context['kill_tick'], ticks_df, context['victim_name'])
    
    if attacker_tick is not None:
        # AWPy uses uppercase X, Y, Z for coordinates
        context['attacker_x'] = attacker_tick.get('X', attacker_tick.get('x', 0))
        context['attacker_y'] = attacker_tick.get('Y', attacker_tick.get('y', 0))
        context['attacker_z'] = attacker_tick.get('Z', attacker_tick.get('z', 0))
        context['attacker_health'] = attacker_tick.get('health', 100)
        # Velocity columns might not exist in AWPy data
        context['attacker_vel_x'] = attacker_tick.get('vel_x', attacker_tick.get('velocity_x', 0))
        context['attacker_vel_y'] = attacker_tick.get('vel_y', attacker_tick.get('velocity_y', 0))
    else:
        context.update({
            'attacker_x': 0, 'attacker_y': 0, 'attacker_z': 0,
            'attacker_health': 100, 'attacker_vel_x': 0, 'attacker_vel_y': 0
        })
    
    if victim_tick is not None:
        # AWPy uses uppercase X, Y, Z for coordinates
        context['victim_x'] = victim_tick.get('X', victim_tick.get('x', 0))
        context['victim_y'] = victim_tick.get('Y', victim_tick.get('y', 0))
        context['victim_z'] = victim_tick.get('Z', victim_tick.get('z', 0))
        context['victim_health'] = victim_tick.get('health', 100)
    else:
        context.update({
            'victim_x': 0, 'victim_y': 0, 'victim_z': 0, 'victim_health': 100
        })
    
    # Calculate derived features
    context['time_in_round_s'] = calculate_time_in_round(
        context['kill_tick'], rounds_df, tickrate
    )
    
    context['distance_xy'] = calculate_distance_2d(
        context['attacker_x'], context['attacker_y'],
        context['victim_x'], context['victim_y']
    )
    
    context['approach_align_deg'] = calculate_approach_alignment(
        context['attacker_x'], context['attacker_y'],
        context['victim_x'], context['victim_y'],
        context['attacker_vel_x'], context['attacker_vel_y']
    )
    
    # Find nearby utility
    utility_flags = find_nearby_utility(
        context['kill_tick'], context['victim_x'], context['victim_y'],
        grenades_df
    )
    context.update(utility_flags)
    
    # Convert to map coordinates
    if map_data:
        # Choose transformation method
        if use_advanced:
            context['attacker_image_x'], context['attacker_image_y'] = world_to_map_coords_fixed(
                context['attacker_x'], context['attacker_y'], map_data, x_adjust, y_adjust
            )
            context['victim_image_x'], context['victim_image_y'] = world_to_map_coords_fixed(
                context['victim_x'], context['victim_y'], map_data, x_adjust, y_adjust
            )
        else:
            context['attacker_image_x'], context['attacker_image_y'] = world_to_map_coords(
                context['attacker_x'], context['attacker_y'], map_data, x_adjust, y_adjust
            )
            context['victim_image_x'], context['victim_image_y'] = world_to_map_coords(
                context['victim_x'], context['victim_y'], map_data, x_adjust, y_adjust
            )
        
        # Add debug information for troubleshooting
        context['debug_attacker'] = debug_coordinate_transformation(
            context['attacker_x'], context['attacker_y'], map_data, x_adjust, y_adjust
        )
        context['debug_victim'] = debug_coordinate_transformation(
            context['victim_x'], context['victim_y'], map_data, x_adjust, y_adjust
        )
    else:
        context.update({
            'attacker_image_x': 0, 'attacker_image_y': 0,
            'victim_image_x': 0, 'victim_image_y': 0
        })
    
    return context
