"""
Transform functions for the Streamlit labeling app.
Enhanced version with more detailed context for ML training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import json
from pathlib import Path
from collections import Counter


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
    if not rounds_df.empty:
        # Find the round this kill belongs to
        for _, round_data in rounds_df.iterrows():
            start_tick = round_data.get('start_tick', 0)
            end_tick = round_data.get('end_tick', float('inf'))
            
            if start_tick <= kill_tick <= end_tick:
                time_in_round = (kill_tick - start_tick) / tickrate
                # Ensure time is reasonable (CS2 rounds are ~115 seconds max)
                return max(0.0, min(time_in_round, 115.0))
    
    # If no round data or kill not found in rounds, use better estimation
    # CS2 rounds are typically 115 seconds = 7360 ticks at 64 tickrate
    round_duration_ticks = 7360
    
    # Find which round this kill belongs to
    round_number = kill_tick // round_duration_ticks
    estimated_round_start = round_number * round_duration_ticks
    
    time_in_round = (kill_tick - estimated_round_start) / tickrate
    
    # Ensure time is reasonable
    return max(0.0, min(time_in_round, 115.0))


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
                               attacker_vel_x: float, attacker_vel_y: float) -> Optional[float]:
    """
    Calculate alignment of attacker movement toward victim.
    
    Args:
        attacker_x, attacker_y: Attacker position
        victim_x, victim_y: Victim position
        attacker_vel_x, attacker_vel_y: Attacker velocity
        
    Returns:
        Alignment angle in degrees (0-180, where 0 is perfect alignment) or None if not moving
    """
    # Vector from attacker to victim
    to_victim_x = victim_x - attacker_x
    to_victim_y = victim_y - attacker_y
    
    # Calculate magnitudes
    to_victim_mag = np.sqrt(to_victim_x**2 + to_victim_y**2)
    vel_mag = np.sqrt(attacker_vel_x**2 + attacker_vel_y**2)
    
    # If attacker is not moving or vectors are invalid, return None
    if to_victim_mag < 1.0 or vel_mag < 10:  # Higher movement threshold for more accurate detection
        return None
    
    # Normalize vectors
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
    
    # Try to find the tick column with different possible names
    tick_col = None
    for col_name in ['tick', 'tick_id', 'time', 'timestamp']:
        if col_name in grenades_df.columns:
            tick_col = col_name
            break
    
    if tick_col is None:
        # If no tick column found, return default values
        return {
            'flash_near': False,
            'smoke_near': False,
            'molotov_near': False,
            'he_near': False
        }
    
    # Filter grenades around the kill time (±2 seconds)
    time_window = 128  # 2 seconds at 64 tickrate
    nearby_grenades = grenades_df[
        (grenades_df[tick_col] >= kill_tick - time_window) &
        (grenades_df[tick_col] <= kill_tick + time_window)
    ].copy()
    
    if nearby_grenades.empty:
        return {
            'flash_near': False,
            'smoke_near': False,
            'molotov_near': False,
            'he_near': False
        }
    
    # Calculate distances - AWPy uses uppercase X, Y for coordinates
    # Get X and Y columns (try both uppercase and lowercase)
    x_col = 'X' if 'X' in nearby_grenades.columns else 'x'
    y_col = 'Y' if 'Y' in nearby_grenades.columns else 'y'
    
    nearby_grenades['distance'] = calculate_distance_2d(
        kill_x, kill_y,
        nearby_grenades[x_col], 
        nearby_grenades[y_col]
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
    
    # Weapon information
    context['attacker_weapon'] = kill_row.get('weapon', kill_row.get('attacker_weapon', 'Unknown'))
    
    # Try to get victim weapon from tick data (what they had before death)
    victim_weapon = 'Unknown'
    victim_weapon_tick = None  # Initialize victim_weapon_tick variable
    
    # Find victim tick data for weapon
    if not ticks_df.empty and context['victim_name'] != 'Unknown':
        # Try different possible column names for player identification
        player_col = None
        for col_name in ['player_name', 'name', 'player', 'attacker_name', 'victim_name']:
            if col_name in ticks_df.columns:
                player_col = col_name
                break
        
        if player_col is not None:
            victim_weapon_ticks = ticks_df[
                (ticks_df['tick'] == context['kill_tick']) &
                (ticks_df[player_col] == context['victim_name'])
            ]
            if not victim_weapon_ticks.empty:
                victim_weapon_tick = victim_weapon_ticks.iloc[0]
    
    if victim_weapon_tick is not None:
        # Check various possible weapon column names
        for weapon_col in ['weapon', 'active_weapon', 'current_weapon', 'equipped_weapon']:
            if weapon_col in victim_weapon_tick and pd.notna(victim_weapon_tick[weapon_col]):
                victim_weapon = str(victim_weapon_tick[weapon_col])
                break
    
    # Fallback to kill data if available
    if victim_weapon == 'Unknown':
        victim_weapon = kill_row.get('victim_weapon', 'Unknown')
    
    context['victim_weapon'] = victim_weapon
    
    # Find nearest tick data for attacker and victim
    attacker_tick = find_nearest_tick(context['kill_tick'], ticks_df, context['attacker_name'])
    victim_tick = find_nearest_tick(context['kill_tick'], ticks_df, context['victim_name'])
    
    if attacker_tick is not None:
        # AWPy uses uppercase X, Y, Z for coordinates
        context['attacker_x'] = attacker_tick.get('X', attacker_tick.get('x', 0))
        context['attacker_y'] = attacker_tick.get('Y', attacker_tick.get('y', 0))
        context['attacker_z'] = attacker_tick.get('Z', attacker_tick.get('z', 0))
        context['attacker_health'] = attacker_tick.get('health', 100)
        
        # Try multiple velocity column names that might exist in AWPy data
        vel_x = 0
        vel_y = 0
        for vel_col_x in ['vel_x', 'velocity_x', 'velX', 'velocityX']:
            if vel_col_x in attacker_tick and pd.notna(attacker_tick[vel_col_x]):
                vel_x = float(attacker_tick[vel_col_x])
                break
        
        for vel_col_y in ['vel_y', 'velocity_y', 'velY', 'velocityY']:
            if vel_col_y in attacker_tick and pd.notna(attacker_tick[vel_col_y]):
                vel_y = float(attacker_tick[vel_col_y])
                break
        
        context['attacker_vel_x'] = vel_x
        context['attacker_vel_y'] = vel_y
        
        # Extract attacker view angle
        attacker_view_angle = attacker_tick.get('view_x', attacker_tick.get('viewX', 0))
        context['attacker_view_angle'] = (attacker_view_angle + 360) % 360
    else:
        context.update({
            'attacker_x': 0, 'attacker_y': 0, 'attacker_z': 0,
            'attacker_health': 100, 'attacker_vel_x': 0, 'attacker_vel_y': 0,
            'attacker_view_angle': 0
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


def detect_victim_awareness(kill_tick: int, victim_name: str, attacker_name: str, 
                            ticks_df: pd.DataFrame, time_window_seconds: float = 3.0) -> Dict[str, Any]:
    """
    Detect if victim was aware of the attacker before death.
    
    Logic:
    - Backstab: Killer was NOT in victim's field of view at moment of death
    - Awareness: Victim was aware of killer in the seconds BEFORE the kill
    
    Args:
        kill_tick: Tick when the kill occurred
        victim_name: Name of the victim
        attacker_name: Name of the attacker
        ticks_df: DataFrame with tick data
        time_window_seconds: Time window to check for awareness before kill
        
    Returns:
        Dictionary with awareness information
    """
    time_window_ticks = int(time_window_seconds * 64)
    
    awareness = {
        'victim_was_aware': False,
        'victim_was_watching': False,
        'victim_was_backstabbed': False,
        'time_since_last_sight': None,
        'awareness_confidence': 0.0,
        'angle_to_attacker': None,
        'victim_view_angle': None,
        'angle_difference': None,
        'awareness_detected_at_tick': None
    }
    
    # Get victim and attacker positions at kill time
    victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
    attacker_tick = find_nearest_tick(kill_tick, ticks_df, attacker_name)
    
    if victim_tick is not None and attacker_tick is not None:
        # Get positions at kill time
        victim_x = victim_tick.get('X', 0)
        victim_y = victim_tick.get('Y', 0)
        attacker_x = attacker_tick.get('X', 0)
        attacker_y = attacker_tick.get('Y', 0)
        
        # Calculate angle from victim to attacker at kill time
        to_attacker_x = attacker_x - victim_x
        to_attacker_y = attacker_y - victim_y
        
        # Calculate the angle where the attacker is relative to victim (0-360 degrees)
        angle_to_attacker = np.degrees(np.arctan2(to_attacker_y, to_attacker_x))
        angle_to_attacker = (angle_to_attacker + 360) % 360
        
        # Get victim's view angle at kill time
        victim_view_angle = victim_tick.get('view_x', victim_tick.get('viewX', 0))
        victim_view_angle = (victim_view_angle + 360) % 360
        
        # Calculate angle difference at kill time
        angle_diff = abs(victim_view_angle - angle_to_attacker)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # Store angles for debugging
        awareness['angle_to_attacker'] = angle_to_attacker
        awareness['victim_view_angle'] = victim_view_angle
        awareness['angle_difference'] = angle_diff
        
        # DETERMINE BACKSTAB: If attacker was NOT in victim's field of view at death
        # Field of view is typically ~180 degrees (90 degrees to each side)
        if angle_diff > 90:
            awareness['victim_was_backstabbed'] = True
            awareness['awareness_confidence'] = 0.9
        else:
            awareness['victim_was_backstabbed'] = False
            awareness['awareness_confidence'] = 0.8
        
        # DETERMINE AWARENESS: Check if victim was aware of attacker BEFORE the kill
        # Look at victim's ticks in the time window before death
        victim_ticks_before = ticks_df[
            (ticks_df.get('name', ticks_df.get('player_name', '')) == victim_name) &
            (ticks_df['tick'] >= kill_tick - time_window_ticks) &
            (ticks_df['tick'] < kill_tick)
        ].copy()
        
        if not victim_ticks_before.empty:
            # Check each tick to see if victim was looking towards attacker
            for _, tick_data in victim_ticks_before.iterrows():
                tick_view_angle = tick_data.get('view_x', tick_data.get('viewX', 0))
                tick_view_angle = (tick_view_angle + 360) % 360
                
                # Calculate angle difference at this tick
                tick_angle_diff = abs(tick_view_angle - angle_to_attacker)
                tick_angle_diff = min(tick_angle_diff, 360 - tick_angle_diff)
                
                # If victim was looking within 90 degrees of attacker, they were aware
                if tick_angle_diff < 90:
                    awareness['victim_was_aware'] = True
                    awareness['victim_was_watching'] = True
                    awareness['awareness_detected_at_tick'] = tick_data['tick']
                    awareness['time_since_last_sight'] = (kill_tick - tick_data['tick']) / 64
                    awareness['awareness_confidence'] = 1.0 - (tick_angle_diff / 90.0)
                    break
        
        # If no awareness detected in the time window, victim was not aware
        if not awareness['victim_was_aware']:
            awareness['awareness_confidence'] = 0.7  # High confidence they weren't aware
    
    return awareness


def detect_sound_cues(kill_tick: int, victim_name: str, attacker_name: str, ticks_df: pd.DataFrame, 
                       damages_df: pd.DataFrame, shots_df: pd.DataFrame, 
                       time_window_seconds: float = 3.0) -> Dict[str, Any]:
    """
    Detect if victim had sound cues about the attacker before death.
    
    Args:
        kill_tick: Tick when the kill occurred
        victim_name: Name of the victim
        ticks_df: DataFrame with tick data
        damages_df: DataFrame with damage events
        shots_df: DataFrame with shot events
        time_window_seconds: Time window to check for sound cues
        
    Returns:
        Dictionary with sound cue information
    """
    time_window_ticks = int(time_window_seconds * 64)  # Assuming 64 tickrate
    
    sound_cues = {
        'had_sound_cue': False,
        'sound_cue_types': [],
        'sound_cue_count': 0,
        'last_sound_tick': None,
        'time_since_last_sound': None,
        'attacker_visible': False,
        'attacker_distance_when_heard': None
    }
    
    # Check for damage events (victim taking damage)
    if not damages_df.empty and 'victim_name' in damages_df.columns and 'tick' in damages_df.columns:
        victim_damages = damages_df[
            (damages_df['victim_name'] == victim_name) &
            (damages_df['tick'] >= kill_tick - time_window_ticks) &
            (damages_df['tick'] < kill_tick)
        ]
        
        if not victim_damages.empty:
            sound_cues['had_sound_cue'] = True
            sound_cues['sound_cue_types'].append('damage_taken')
            sound_cues['sound_cue_count'] += len(victim_damages)
            
            # Get last sound cue
            last_damage = victim_damages.iloc[-1]
            sound_cues['last_sound_tick'] = last_damage['tick']
            sound_cues['time_since_last_sound'] = (kill_tick - last_damage['tick']) / 64
    
    # Check for nearby shots
    if not shots_df.empty and 'tick' in shots_df.columns:
        # Get victim position at kill time
        victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
        if victim_tick is not None:
            victim_x = victim_tick.get('X', 0)
            victim_y = victim_tick.get('Y', 0)
            
            # Find shots near victim in time window
            nearby_shots = shots_df[
                (shots_df['tick'] >= kill_tick - time_window_ticks) &
                (shots_df['tick'] < kill_tick)
            ].copy()
            
            if not nearby_shots.empty:
                # Calculate distances to shots
                nearby_shots['distance'] = calculate_distance_2d(
                    victim_x, victim_y,
                    nearby_shots.get('X', 0), nearby_shots.get('Y', 0)
                )
                
                # Consider shots within 1000 units as audible
                audible_shots = nearby_shots[nearby_shots['distance'] <= 1000]
                
                if not audible_shots.empty:
                    sound_cues['had_sound_cue'] = True
                    sound_cues['sound_cue_types'].append('nearby_shots')
                    sound_cues['sound_cue_count'] += len(audible_shots)
                    
                    # Get closest shot
                    closest_shot = audible_shots.loc[audible_shots['distance'].idxmin()]
                    sound_cues['attacker_distance_when_heard'] = closest_shot['distance']
    
    # Check for attacker visibility (line of sight)
    attacker_tick = find_nearest_tick(kill_tick, ticks_df, attacker_name)
    victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
    if attacker_tick is not None and victim_tick is not None:
        # Simple line of sight check (can be enhanced with map geometry)
        attacker_x = attacker_tick.get('X', 0)
        attacker_y = attacker_tick.get('Y', 0)
        victim_x = victim_tick.get('X', 0)
        victim_y = victim_tick.get('Y', 0)
        
        distance = calculate_distance_2d(attacker_x, attacker_y, victim_x, victim_y)
        # Assume visibility if distance is reasonable (can be enhanced)
        sound_cues['attacker_visible'] = distance <= 2000
    
    return sound_cues


def extract_team_names(rounds_df: pd.DataFrame, ticks_df: pd.DataFrame, game_id: str = None) -> Tuple[str, str]:
    """
    Extract team names from round data or ticks data.
    
    Args:
        rounds_df: DataFrame with round data
        ticks_df: DataFrame with tick data
        game_id: Optional game ID to filter data for specific game
        
    Returns:
        Tuple of (team1_name, team2_name)
    """
    team1_name = 'Team 1'
    team2_name = 'Team 2'
    
    # Filter data by game_id if provided
    if game_id is not None and 'game_id' in rounds_df.columns:
        game_rounds = rounds_df[rounds_df['game_id'] == game_id]
    else:
        game_rounds = rounds_df
    
    if game_id is not None and 'game_id' in ticks_df.columns:
        game_ticks = ticks_df[ticks_df['game_id'] == game_id]
    else:
        game_ticks = ticks_df
    
    # Try to get team names from rounds data first
    if not game_rounds.empty:
        for _, round_data in game_rounds.iterrows():
            # Check for various team name column patterns
            team1_cols = ['team1_name', 'team1', 'team_1', 'team_1_name', 't_team', 't_team_name']
            team2_cols = ['team2_name', 'team2', 'team_2', 'team_2_name', 'ct_team', 'ct_team_name']
            
            for col in team1_cols:
                if col in round_data and pd.notna(round_data[col]) and str(round_data[col]).strip():
                    team1_name = str(round_data[col]).strip()
                    break
            
            for col in team2_cols:
                if col in round_data and pd.notna(round_data[col]) and str(round_data[col]).strip():
                    team2_name = str(round_data[col]).strip()
                    break
            
            if team1_name != 'Team 1' and team2_name != 'Team 2':
                break
    
    # If no team names found in rounds, try to extract from ticks data
    if (team1_name == 'Team 1' or team2_name == 'Team 2') and not game_ticks.empty:
        # Look for team-related columns
        team_columns = [col for col in game_ticks.columns if 'team' in col.lower()]
        if team_columns:
            for team_col in team_columns:
                unique_teams = game_ticks[team_col].dropna().unique()
                if len(unique_teams) >= 2:
                    # Filter out numeric IDs and very long strings
                    valid_teams = []
                    for team in unique_teams:
                        team_str = str(team).strip()
                        # Skip if it's a numeric ID or too long
                        if (not team_str.isdigit() and 
                            len(team_str) < 50 and 
                            not team_str.startswith('7656119')):  # Skip Steam IDs
                            valid_teams.append(team_str)
                    
                    if len(valid_teams) >= 2:
                        team1_name = valid_teams[0]
                        team2_name = valid_teams[1]
                        break
    
    # Enhanced team name extraction from player names
    if (team1_name == 'Team 1' or team2_name == 'Team 2') and not game_ticks.empty:
        # Look for player name columns
        player_col = None
        for col_name in ['player_name', 'name', 'player', 'attacker_name', 'victim_name']:
            if col_name in game_ticks.columns:
                player_col = col_name
                break
        
        if player_col:
            # Get unique player names and try to identify teams
            unique_players = game_ticks[player_col].dropna().unique()
            if len(unique_players) >= 10:  # Should have at least 10 players
                # Filter out Steam IDs and very long names
                valid_players = []
                for player in unique_players:
                    player_str = str(player).strip()
                    if (not player_str.isdigit() and 
                        len(player_str) < 50 and 
                        not player_str.startswith('7656119')):
                        valid_players.append(player_str)
                
                if len(valid_players) >= 10:
                    # Enhanced team name extraction
                    team_names = extract_teams_from_player_names(valid_players)
                    if len(team_names) >= 2:
                        team1_name = team_names[0]
                        team2_name = team_names[1]
    
    # Final fallback: use more descriptive names if we still have default values
    if team1_name == 'Team 1':
        # Try to generate a descriptive name based on player patterns
        if not game_ticks.empty:
            player_col = None
            for col_name in ['player_name', 'name', 'player', 'attacker_name', 'victim_name']:
                if col_name in game_ticks.columns:
                    player_col = col_name
                    break
            
            if player_col:
                unique_players = game_ticks[player_col].dropna().unique()
                if len(unique_players) >= 10:
                    # Try to find a pattern in player names
                    player_words = []
                    for player in unique_players[:5]:  # Look at first 5 players
                        words = str(player).split()
                        player_words.extend(words)
                    
                    # Find most common word that could be a team identifier
                    word_counts = Counter(player_words)
                    common_words = [word for word, count in word_counts.most_common(10) 
                                  if len(word) > 2 and not word.isdigit() and word.lower() not in ['the', 'and', 'for', 'with']]
                    
                    if common_words:
                        team1_name = f"Team {common_words[0]}"
                    else:
                        team1_name = 'Terrorists'
                else:
                    team1_name = 'Terrorists'
            else:
                team1_name = 'Terrorists'
        else:
            team1_name = 'Terrorists'
    
    if team2_name == 'Team 2':
        # Generate a different team name for the second team
        if team1_name != 'Terrorists':
            # If we found a pattern for team1, try to find a different pattern for team2
            if not game_ticks.empty:
                player_col = None
                for col_name in ['player_name', 'name', 'player', 'attacker_name', 'victim_name']:
                    if col_name in game_ticks.columns:
                        player_col = col_name
                        break
                
                if player_col:
                    unique_players = game_ticks[player_col].dropna().unique()
                    if len(unique_players) >= 10:
                        # Look for different patterns in remaining players
                        player_words = []
                        for player in unique_players[5:10]:  # Look at next 5 players
                            words = str(player).split()
                            player_words.extend(words)
                        
                        word_counts = Counter(player_words)
                        common_words = [word for word, count in word_counts.most_common(10) 
                                      if len(word) > 2 and not word.isdigit() and word.lower() not in ['the', 'and', 'for', 'with']]
                        
                        if common_words and common_words[0] not in team1_name:
                            team2_name = f"Team {common_words[0]}"
                        else:
                            team2_name = 'Counter-Terrorists'
                    else:
                        team2_name = 'Counter-Terrorists'
                else:
                    team2_name = 'Counter-Terrorists'
            else:
                team2_name = 'Counter-Terrorists'
        else:
            team2_name = 'Counter-Terrorists'
    
    # Debug: Print what we found
    print(f"DEBUG: Extracted team names - {team1_name} vs {team2_name}")
    if not rounds_df.empty:
        print(f"DEBUG: Rounds columns: {list(rounds_df.columns)}")
        if 'winner' in rounds_df.columns:
            print(f"DEBUG: Winner values: {rounds_df['winner'].dropna().unique()}")
    if not ticks_df.empty:
        print(f"DEBUG: Ticks columns: {list(ticks_df.columns)}")
        team_cols = [col for col in ticks_df.columns if 'team' in col.lower()]
        if team_cols:
            for col in team_cols:
                print(f"DEBUG: {col} values: {ticks_df[col].dropna().unique()[:5]}")
    
    return team1_name, team2_name


def extract_teams_from_player_names(player_names: List[str]) -> List[str]:
    """
    Extract team names from player names using pattern analysis.
    
    Args:
        player_names: List of player names
        
    Returns:
        List of detected team names
    """
    # Common team name patterns and keywords
    team_keywords = [
        'team', 'clan', 'org', 'esports', 'gaming', 'gaming', 'pro', 'esports',
        'liquid', 'navi', 'faze', 'astralis', 'vitality', 'g2', 'cloud9', 'c9',
        'fnatic', 'nip', 'mouz', 'heroic', 'ence', 'big', 'spirit', 'outsiders',
        'imperial', 'pain', 'furia', 'mibr', 'complexity', 'col', 'eg', 'evil',
        'geniuses', 'tsm', '100t', 'thieves', 'optic', 'og', 'vitality', 'vit',
        'gambit', 'virtus', 'pro', 'vp', 'sk', 'skgaming', 'lg', 'luminosity',
        'monte', 'apeks', 'eternal', 'fire', 'flames', 'flame', 'fire', 'flame',
        'natus', 'vincere', 'natus vincere', 'team liquid', 'team vitality',
        'team spirit', 'team g2', 'team fnatic', 'team nip', 'team mouz',
        'team heroic', 'team ence', 'team big', 'team outsiders', 'team imperial',
        'team pain', 'team furia', 'team mibr', 'team complexity', 'team eg',
        'team tsm', 'team optic', 'team og', 'team gambit', 'team vp', 'team sk',
        'team lg', 'team luminosity', 'team monte', 'team apeks', 'team eternal'
    ]
    
    # Analyze player names for team patterns
    potential_teams = []
    
    for player in player_names:
        player_lower = player.lower()
        
        # Check for team prefixes (e.g., "Team Liquid | s1mple")
        if '|' in player:
            parts = player.split('|')
            if len(parts) >= 2:
                team_part = parts[0].strip()
                if len(team_part) > 2 and not team_part.isdigit():
                    potential_teams.append(team_part)
        
        # Check for team keywords in player names
        for keyword in team_keywords:
            if keyword in player_lower:
                # Extract potential team name
                words = player.split()
                for i, word in enumerate(words):
                    if keyword in word.lower():
                        # Try to get the full team name
                        if i > 0:
                            # Check if previous word might be part of team name
                            potential_team = f"{words[i-1]} {word}"
                            if len(potential_team) > 3:
                                potential_teams.append(potential_team)
                        else:
                            # Check if next word might be part of team name
                            if i + 1 < len(words):
                                potential_team = f"{word} {words[i+1]}"
                                if len(potential_team) > 3:
                                    potential_teams.append(potential_team)
                            else:
                                potential_teams.append(word)
                        break
        
        # Check for common team name patterns (e.g., "Liquid.s1mple", "NAVI-electronic")
        if '.' in player:
            parts = player.split('.')
            if len(parts) >= 2:
                team_part = parts[0].strip()
                if len(team_part) > 2 and not team_part.isdigit():
                    potential_teams.append(team_part)
        
        if '-' in player:
            parts = player.split('-')
            if len(parts) >= 2:
                team_part = parts[0].strip()
                if len(team_part) > 2 and not team_part.isdigit():
                    potential_teams.append(team_part)
        
        # Check for brackets pattern (e.g., "[Team] Player", "Player [Team]")
        if '[' in player and ']' in player:
            start = player.find('[')
            end = player.find(']')
            if start < end:
                team_part = player[start+1:end].strip()
                if len(team_part) > 2 and not team_part.isdigit():
                    potential_teams.append(team_part)
        
        # Check for parentheses pattern (e.g., "(Team) Player", "Player (Team)")
        if '(' in player and ')' in player:
            start = player.find('(')
            end = player.find(')')
            if start < end:
                team_part = player[start+1:end].strip()
                if len(team_part) > 2 and not team_part.isdigit():
                    potential_teams.append(team_part)
        
        # Check for underscore pattern (e.g., "Team_Player")
        if '_' in player:
            parts = player.split('_')
            if len(parts) >= 2:
                team_part = parts[0].strip()
                if len(team_part) > 2 and not team_part.isdigit():
                    potential_teams.append(team_part)
    
    # Filter and rank potential teams
    team_counts = {}
    for team in potential_teams:
        # Clean up team name
        team_clean = team.strip()
        if len(team_clean) > 2 and not team_clean.isdigit():
            team_counts[team_clean] = team_counts.get(team_clean, 0) + 1
    
    # Sort by frequency and return top teams
    sorted_teams = sorted(team_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract unique team names (avoid duplicates)
    unique_teams = []
    for team, count in sorted_teams:
        # Check if this team name is not too similar to already added teams
        is_unique = True
        for existing_team in unique_teams:
            if (team.lower() in existing_team.lower() or 
                existing_team.lower() in team.lower() or
                team.lower() == existing_team.lower()):
                is_unique = False
                break
        
        if is_unique and len(unique_teams) < 2:
            unique_teams.append(team)
    
    # If we found teams, return them
    if len(unique_teams) >= 2:
        return unique_teams[:2]
    
    # If we only found one team, try to infer the second team
    if len(unique_teams) == 1:
        # Look for players that don't match the first team pattern
        other_players = []
        first_team = unique_teams[0].lower()
        
        for player in player_names:
            player_lower = player.lower()
            # Check if this player doesn't belong to the first team
            if (first_team not in player_lower and 
                not any(keyword in player_lower for keyword in first_team.split())):
                other_players.append(player)
        
        # Try to extract team name from other players
        if other_players:
            other_team_names = extract_teams_from_player_names(other_players)
            if other_team_names:
                return [unique_teams[0], other_team_names[0]]
    
    return unique_teams


def analyze_round_context(kill_tick: int, rounds_df: pd.DataFrame, 
                         bomb_df: pd.DataFrame, ticks_df: pd.DataFrame = None, tickrate: int = 64) -> Dict[str, Any]:
    print(f"DEBUG: analyze_round_context called with kill_tick={kill_tick}, ticks_df empty={ticks_df.empty if ticks_df is not None else 'None'}")
    """
    Analyze round context at the time of kill.
    
    Args:
        kill_tick: Tick when the kill occurred
        rounds_df: DataFrame with round data
        bomb_df: DataFrame with bomb events
        ticks_df: DataFrame with tick data (for team name extraction)
        tickrate: Game tickrate
        
    Returns:
        Dictionary with round context information
    """
    # Check if this is a combined dataset with game_id column
    has_game_id = 'game_id' in rounds_df.columns
    
    # Find the game_id for this kill if we have game_id column
    kill_game_id = None
    if has_game_id and not ticks_df.empty:
        # Find the game_id for this kill by looking at ticks around this time
        kill_tick_row = ticks_df[ticks_df['tick'] == kill_tick]
        if not kill_tick_row.empty:
            kill_game_id = kill_tick_row.iloc[0].get('game_id')
    
    # Extract team names first
    team1_name, team2_name = extract_team_names(rounds_df, ticks_df if ticks_df is not None else pd.DataFrame(), kill_game_id)
    
    round_context = {
        'round_number': None,
        'time_in_round_s': 0,
        'round_phase': 'unknown',
        'bomb_planted': False,
        'time_since_bomb_plant': None,
        'time_until_bomb_explode': None,
        'players_alive_t': 0,
        'players_alive_ct': 0,
        'round_win_probability': 0.5,
        'match_score_t': 0,
        'match_score_ct': 0,
        'team1_name': team1_name,
        'team2_name': team2_name,
        'game_id': kill_game_id
    }
    
    # Calculate alive players at the time of kill
    print(f"DEBUG: Starting player count calculation for kill tick {kill_tick}")
    if not ticks_df.empty:
        print(f"DEBUG: Ticks DataFrame has {len(ticks_df)} rows")
        # Filter ticks around the kill time (within 1 second)
        time_window_ticks = 64  # 1 second at 64 tickrate
        nearby_ticks = ticks_df[
            (ticks_df['tick'] >= kill_tick - time_window_ticks) & 
            (ticks_df['tick'] <= kill_tick + time_window_ticks)
        ]
        print(f"DEBUG: Found {len(nearby_ticks)} nearby ticks")
        
        if not nearby_ticks.empty:
            # Get the closest tick to the kill
            closest_tick = nearby_ticks.iloc[nearby_ticks['tick'].sub(kill_tick).abs().argsort()[:1]]
            print(f"DEBUG: Closest tick has {len(closest_tick)} rows")
            
            if not closest_tick.empty:
                # Debug: Check what columns are available
                print(f"DEBUG: Available columns in ticks: {list(closest_tick.columns)}")
                
                # Try different possible column names for alive status
                is_alive_col = None
                for col in ['is_alive', 'alive', 'isAlive', 'IsAlive']:
                    if col in closest_tick.columns:
                        is_alive_col = col
                        break
                
                # Try different possible column names for side
                side_col = None
                for col in ['side', 'Side', 'team', 'Team', 'team_side', 'teamSide']:
                    if col in closest_tick.columns:
                        side_col = col
                        break
                
                if is_alive_col and side_col:
                    # Count alive players by side
                    alive_players = closest_tick[closest_tick[is_alive_col] == True]
                    
                    if not alive_players.empty:
                        # Count by side (T vs CT)
                        t_alive = alive_players[alive_players[side_col].str.upper() == 'T']
                        ct_alive = alive_players[alive_players[side_col].str.upper() == 'CT']
                        
                        round_context['players_alive_t'] = len(t_alive)
                        round_context['players_alive_ct'] = len(ct_alive)
                        
                        print(f"DEBUG: Found {len(t_alive)} T players alive, {len(ct_alive)} CT players alive")
                        # Add debug info to the context so it shows up in the response
                        round_context['debug_player_count'] = f"Found {len(t_alive)} T, {len(ct_alive)} CT players alive"
                else:
                    print(f"DEBUG: Could not find is_alive column ({is_alive_col}) or side column ({side_col})")
                    # Fallback: assume 5 players per side (default CS2 team size)
                    round_context['players_alive_t'] = 5
                    round_context['players_alive_ct'] = 5
                    round_context['debug_player_count'] = f"Fallback: Could not find columns (is_alive: {is_alive_col}, side: {side_col})"
    
    # Find current round and calculate match score with proper team tracking
    if not rounds_df.empty:
        current_round = None
        team1_score = 0
        team2_score = 0
        
        # Filter rounds to only include rounds from the same game if we have game_id
        if has_game_id and kill_game_id is not None:
            game_rounds = rounds_df[rounds_df['game_id'] == kill_game_id].copy()
        else:
            game_rounds = rounds_df.copy()
        
        # Track which team is on which side for each round
        # In CS2, teams switch sides after round 12 (or when a team reaches 13 wins)
        team1_side_history = {}  # Maps round number to side (T or CT)
        team2_side_history = {}
        
        for _, round_data in game_rounds.iterrows():
            round_num = round_data.get('round', 0)
            start_tick = round_data.get('start_tick', 0)
            end_tick = round_data.get('end_tick', float('inf'))
            
            # Determine which team is on which side for this round
            # For simplicity, assume team1 starts T side, team2 starts CT side
            # After round 12, they switch
            if round_num <= 12:
                team1_side_history[round_num] = 'T'
                team2_side_history[round_num] = 'CT'
            else:
                team1_side_history[round_num] = 'CT'
                team2_side_history[round_num] = 'T'
            
            # Calculate scores from completed rounds (rounds that ended before this kill)
            if end_tick < kill_tick:
                # Try multiple possible winner column names
                winner_cols = ['winner', 'round_winner', 'winning_team', 'winning_side', 'winner_side']
                round_winner = None
                
                for col in winner_cols:
                    if col in round_data and pd.notna(round_data[col]):
                        round_winner = round_data[col]
                        break
                
                if round_winner is not None:
                    winner_str = str(round_winner).lower().strip()
                    
                    # Determine which team won based on the side that won
                    if (winner_str == 't' or winner_str == 'terrorist' or 
                        'terrorist' in winner_str or winner_str == '2' or 
                        winner_str == 'team1' or winner_str == 'team_1'):
                        # T side won - determine which team was T side this round
                        if team1_side_history.get(round_num) == 'T':
                            team1_score += 1
                        else:
                            team2_score += 1
                    elif (winner_str == 'ct' or winner_str == 'counter' or 
                          'counter' in winner_str or winner_str == '3' or
                          winner_str == 'team2' or winner_str == 'team_2'):
                        # CT side won - determine which team was CT side this round
                        if team1_side_history.get(round_num) == 'CT':
                            team1_score += 1
                        else:
                            team2_score += 1
            
            # Find current round
            if start_tick <= kill_tick <= end_tick:
                current_round = round_data
                round_context['round_number'] = round_num
                round_context['time_in_round_s'] = (kill_tick - start_tick) / tickrate
                
                # Determine round phase based on time
                time_in_round = round_context['time_in_round_s']
                if time_in_round < 30:
                    round_context['round_phase'] = 'early'
                elif time_in_round < 90:
                    round_context['round_phase'] = 'mid'
                else:
                    round_context['round_phase'] = 'late'
        
        # Update match score with actual team scores
        round_context['match_score_t'] = team1_score
        round_context['match_score_ct'] = team2_score
        
        # Debug: Print score calculation
        print(f"DEBUG: Match score calculation - {team1_name}: {team1_score}, {team2_name}: {team2_score}")
        print(f"DEBUG: Kill tick: {kill_tick}")
        if not rounds_df.empty:
            print(f"DEBUG: Rounds data shape: {rounds_df.shape}")
            for _, round_data in rounds_df.iterrows():
                start_tick = round_data.get('start_tick', 0)
                end_tick = round_data.get('end_tick', float('inf'))
                winner = None
                for col in ['winner', 'round_winner', 'winning_team', 'winning_side', 'winner_side']:
                    if col in round_data and pd.notna(round_data[col]):
                        winner = round_data[col]
                        break
                print(f"DEBUG: Round {round_data.get('round', 'N/A')} - Start: {start_tick}, End: {end_tick}, Winner: {winner}, Completed: {end_tick < kill_tick}")
    
    # Check bomb status
    if not bomb_df.empty:
        round_bomb_events = bomb_df[
            (bomb_df['tick'] <= kill_tick) &
            (bomb_df['tick'] >= kill_tick - 30000)  # Check last 30 seconds
        ]
        
        if not round_bomb_events.empty:
            # Check if bomb is planted
            if 'bomb_action' in round_bomb_events.columns:
                planted_events = round_bomb_events[
                    round_bomb_events['bomb_action'].str.contains('plant', case=False, na=False)
                ]
            else:
                # If no bomb_action column, assume no bomb planted
                planted_events = pd.DataFrame()
            
            if not planted_events.empty:
                round_context['bomb_planted'] = True
                last_plant = planted_events.iloc[-1]
                time_since_plant = (kill_tick - last_plant['tick']) / tickrate
                round_context['time_since_bomb_plant'] = time_since_plant
                
                # Calculate time until bomb explosion (40 seconds after plant)
                if time_since_plant < 40:
                    round_context['time_until_bomb_explode'] = 40 - time_since_plant
    
    return round_context


def analyze_utility_context(kill_tick: int, kill_x: float, kill_y: float,
                           grenades_df: pd.DataFrame, smokes_df: pd.DataFrame,
                           infernos_df: pd.DataFrame, time_window_seconds: float = 5.0) -> Dict[str, Any]:
    """
    Analyze utility context around the kill.
    
    Args:
        kill_tick: Tick when the kill occurred
        kill_x, kill_y: Position of the kill
        grenades_df: DataFrame with grenade data
        smokes_df: DataFrame with smoke data
        infernos_df: DataFrame with molotov/incendiary data
        time_window_seconds: Time window to check for utility
        
    Returns:
        Dictionary with utility context information
    """
    time_window_ticks = int(time_window_seconds * 64)
    
    utility_context = {
        'flash_active': False,
        'smoke_active': False,
        'molotov_active': False,
        'he_active': False,
        'utility_count': 0,
        'closest_utility_distance': None,
        'utility_thrower': None,
        'time_since_utility': None,
        'utility_affecting_kill': False
    }
    
    # Check grenades
    if not grenades_df.empty and 'tick' in grenades_df.columns:
        nearby_grenades = grenades_df[
            (grenades_df['tick'] >= kill_tick - time_window_ticks) &
            (grenades_df['tick'] <= kill_tick + time_window_ticks)
        ].copy()
        
        if not nearby_grenades.empty:
            # Calculate distances
            # Get X and Y columns (try both uppercase and lowercase)
            x_col = 'X' if 'X' in nearby_grenades.columns else 'x'
            y_col = 'Y' if 'Y' in nearby_grenades.columns else 'y'
            
            nearby_grenades['distance'] = calculate_distance_2d(
                kill_x, kill_y,
                nearby_grenades[x_col], nearby_grenades[y_col]
            )
            
            # Check for active utility within 500 units
            active_utility = nearby_grenades[nearby_grenades['distance'] <= 500]
            
            if not active_utility.empty:
                utility_context['utility_count'] = len(active_utility)
                utility_context['closest_utility_distance'] = active_utility['distance'].min()
                
                # Get closest utility
                closest_utility = active_utility.loc[active_utility['distance'].idxmin()]
                utility_context['utility_thrower'] = closest_utility.get('thrower', 'Unknown')
                utility_context['time_since_utility'] = (kill_tick - closest_utility['tick']) / 64
                
                # Check utility types
                for _, utility in active_utility.iterrows():
                    grenade_type = str(utility.get('grenade_type', '')).lower()
                    
                    if 'flash' in grenade_type:
                        utility_context['flash_active'] = True
                    elif 'smoke' in grenade_type:
                        utility_context['smoke_active'] = True
                    elif 'molotov' in grenade_type or 'incendiary' in grenade_type:
                        utility_context['molotov_active'] = True
                    elif 'he' in grenade_type:
                        utility_context['he_active'] = True
                
                # Determine if utility affected the kill
                if utility_context['flash_active'] or utility_context['smoke_active']:
                    utility_context['utility_affecting_kill'] = True
    
    # Check smokes
    if not smokes_df.empty and 'tick' in smokes_df.columns:
        nearby_smokes = smokes_df[
            (smokes_df['tick'] >= kill_tick - time_window_ticks) &
            (smokes_df['tick'] <= kill_tick + time_window_ticks)
        ].copy()
        
        if not nearby_smokes.empty:
            # Get X and Y columns (try both uppercase and lowercase)
            x_col = 'X' if 'X' in nearby_smokes.columns else 'x'
            y_col = 'Y' if 'Y' in nearby_smokes.columns else 'y'
            
            nearby_smokes['distance'] = calculate_distance_2d(
                kill_x, kill_y,
                nearby_smokes[x_col], nearby_smokes[y_col]
            )
            
            if (nearby_smokes['distance'] <= 500).any():
                utility_context['smoke_active'] = True
                utility_context['utility_affecting_kill'] = True
    
    # Check infernos (molotovs)
    if not infernos_df.empty and 'tick' in infernos_df.columns:
        nearby_infernos = infernos_df[
            (infernos_df['tick'] >= kill_tick - time_window_ticks) &
            (infernos_df['tick'] <= kill_tick + time_window_ticks)
        ].copy()
        
        if not nearby_infernos.empty:
            # Get X and Y columns (try both uppercase and lowercase)
            x_col = 'X' if 'X' in nearby_infernos.columns else 'x'
            y_col = 'Y' if 'Y' in nearby_infernos.columns else 'y'
            
            nearby_infernos['distance'] = calculate_distance_2d(
                kill_x, kill_y,
                nearby_infernos[x_col], nearby_infernos[y_col]
            )
            
            if (nearby_infernos['distance'] <= 500).any():
                utility_context['molotov_active'] = True
                utility_context['utility_affecting_kill'] = True
    
    return utility_context


def get_enhanced_kill_context(kill_row: pd.Series, ticks_df: pd.DataFrame, 
                             rounds_df: pd.DataFrame, grenades_df: pd.DataFrame,
                             damages_df: pd.DataFrame, shots_df: pd.DataFrame,
                             smokes_df: pd.DataFrame, infernos_df: pd.DataFrame,
                             bomb_df: pd.DataFrame, map_data: Dict, 
                             tickrate: int = 64, x_adjust: float = 25, y_adjust: float = 0, 
                             use_advanced: bool = True) -> Dict:
    """
    Get comprehensive enhanced context for a kill event.
    
    Args:
        kill_row: Row from kills DataFrame
        ticks_df: DataFrame with tick data
        rounds_df: DataFrame with round data
        grenades_df: DataFrame with grenade data
        damages_df: DataFrame with damage events
        shots_df: DataFrame with shot events
        smokes_df: DataFrame with smoke events
        infernos_df: DataFrame with molotov events
        bomb_df: DataFrame with bomb events
        map_data: Map coordinate data
        tickrate: Game tickrate
        x_adjust: Fine-tuning adjustment for X axis
        y_adjust: Fine-tuning adjustment for Y axis
        use_advanced: Whether to use advanced coordinate transformation
        
    Returns:
        Dictionary with enhanced kill context
    """
    # Get basic context first
    context = get_kill_context(kill_row, ticks_df, rounds_df, grenades_df, 
                              map_data, tickrate, x_adjust, y_adjust, use_advanced)
    
    # Add enhanced context
    kill_tick = context['kill_tick']
    victim_name = context['victim_name']
    attacker_name = context['attacker_name']
    
    # Sound cue analysis
    sound_cues = detect_sound_cues(kill_tick, victim_name, attacker_name, ticks_df, damages_df, shots_df)
    context.update(sound_cues)
    
    # Victim awareness analysis
    awareness = detect_victim_awareness(kill_tick, victim_name, attacker_name, ticks_df)
    context.update(awareness)
    
    # Round context analysis
    round_context = analyze_round_context(kill_tick, rounds_df, bomb_df, ticks_df, tickrate)
    
    context.update(round_context)
    
    # Utility context analysis
    utility_context = analyze_utility_context(
        kill_tick, context['victim_x'], context['victim_y'],
        grenades_df, smokes_df, infernos_df
    )
    context.update(utility_context)
    
    # Add player state analysis
    attacker_tick = find_nearest_tick(kill_tick, ticks_df, attacker_name)
    victim_tick = find_nearest_tick(kill_tick, ticks_df, victim_name)
    
    if attacker_tick is not None:
        context.update({
            'attacker_is_alive': attacker_tick.get('is_alive', True),
            'attacker_is_ducking': attacker_tick.get('is_ducking', False),
            'attacker_is_scoped': attacker_tick.get('is_scoped', False),
            'attacker_is_moving': attacker_tick.get('is_moving', False),
            'attacker_movement_speed': attacker_tick.get('movement_speed', 0),
            'attacker_has_primary': attacker_tick.get('has_primary', False),
            'attacker_has_secondary': attacker_tick.get('has_secondary', False),
            'attacker_has_utility': attacker_tick.get('has_utility', False),
        })
    
    if victim_tick is not None:
        context.update({
            'victim_is_alive': victim_tick.get('is_alive', True),
            'victim_is_ducking': victim_tick.get('is_ducking', False),
            'victim_is_scoped': victim_tick.get('is_scoped', False),
            'victim_is_moving': victim_tick.get('is_moving', False),
            'victim_movement_speed': victim_tick.get('movement_speed', 0),
            'victim_has_primary': victim_tick.get('has_primary', False),
            'victim_has_secondary': victim_tick.get('has_secondary', False),
            'victim_has_utility': victim_tick.get('has_utility', False),
        })
    
    # Add tactical analysis
    context.update({
        'kill_advantage': context.get('attacker_health', 100) - context.get('victim_health', 100),
        'distance_category': 'close' if context['distance_xy'] < 500 else 'medium' if context['distance_xy'] < 1000 else 'far',
        'is_eco_kill': context.get('attacker_health', 100) < 50 or context.get('victim_health', 100) < 50,
        'is_trade_kill': False,  # Can be enhanced with kill chain analysis
        'is_clutch_situation': False,  # Can be enhanced with player count analysis
    })
    
    return context
