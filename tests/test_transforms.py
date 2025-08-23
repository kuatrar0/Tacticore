"""
Unit tests for transform functions.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from streamlit_app.transforms import (
    world_to_map_coords, load_map_data, find_nearest_tick,
    calculate_time_in_round, calculate_distance_2d,
    calculate_approach_alignment, find_nearby_utility
)


class TestTransforms(unittest.TestCase):
    """Test cases for transform functions."""
    
    def setUp(self):
        """Set up test data."""
        # Sample map data
        self.map_data = {
            'pos_x': 1000,
            'pos_y': 2000,
            'scale': 2.0
        }
        
        # Sample ticks data
        self.ticks_df = pd.DataFrame({
            'tick': [100, 101, 102, 103, 104],
            'player_name': ['player1', 'player1', 'player2', 'player2', 'player1'],
            'x': [100, 110, 200, 210, 120],
            'y': [150, 160, 250, 260, 170],
            'z': [50, 55, 60, 65, 70],
            'health': [100, 95, 80, 75, 90]
        })
        
        # Sample grenades data
        self.grenades_df = pd.DataFrame({
            'tick': [100, 101, 102],
            'grenade_type': ['flashbang', 'smoke', 'molotov'],
            'x': [150, 160, 170],
            'y': [200, 210, 220],
            'z': [30, 35, 40]
        })
    
    def test_world_to_map_coords(self):
        """Test world to map coordinate conversion."""
        # Test basic conversion
        x, y = world_to_map_coords(1500, 1800, self.map_data)
        expected_x = (1500 - 1000) / 2.0  # 250
        expected_y = (2000 - 1800) / 2.0  # 100
        self.assertAlmostEqual(x, expected_x)
        self.assertAlmostEqual(y, expected_y)
        
        # Test with zero coordinates
        x, y = world_to_map_coords(0, 0, self.map_data)
        expected_x = (0 - 1000) / 2.0  # -500
        expected_y = (2000 - 0) / 2.0   # 1000
        self.assertAlmostEqual(x, expected_x)
        self.assertAlmostEqual(y, expected_y)
        
        # Test with missing map data
        x, y = world_to_map_coords(100, 100, {})
        self.assertAlmostEqual(x, 100)  # Default pos_x = 0
        self.assertAlmostEqual(y, -100)  # Default pos_y = 0, scale = 1
    
    def test_calculate_distance_2d(self):
        """Test 2D distance calculation."""
        # Test basic distance
        distance = calculate_distance_2d(0, 0, 3, 4)
        self.assertAlmostEqual(distance, 5.0)  # 3-4-5 triangle
        
        # Test zero distance
        distance = calculate_distance_2d(10, 20, 10, 20)
        self.assertAlmostEqual(distance, 0.0)
        
        # Test negative coordinates
        distance = calculate_distance_2d(-1, -1, 1, 1)
        self.assertAlmostEqual(distance, 2.8284271247461903)  # sqrt(8)
    
    def test_calculate_approach_alignment(self):
        """Test approach alignment calculation."""
        # Test perfect alignment (attacker moving directly toward victim)
        alignment = calculate_approach_alignment(0, 0, 10, 0, 1, 0)
        self.assertAlmostEqual(alignment, 0.0)
        
        # Test perpendicular movement
        alignment = calculate_approach_alignment(0, 0, 10, 0, 0, 1)
        self.assertAlmostEqual(alignment, 90.0)
        
        # Test opposite movement
        alignment = calculate_approach_alignment(0, 0, 10, 0, -1, 0)
        self.assertAlmostEqual(alignment, 180.0)
        
        # Test zero velocity
        alignment = calculate_approach_alignment(0, 0, 10, 0, 0, 0)
        self.assertAlmostEqual(alignment, 90.0)  # Default to perpendicular
    
    def test_find_nearest_tick(self):
        """Test nearest tick finding."""
        # Test finding nearest tick
        nearest = find_nearest_tick(102, self.ticks_df, 'player1')
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest['tick'], 102)
        self.assertEqual(nearest['player_name'], 'player1')
        
        # Test with exact match
        nearest = find_nearest_tick(100, self.ticks_df, 'player1')
        self.assertEqual(nearest['tick'], 100)
        
        # Test with no data for player
        nearest = find_nearest_tick(100, self.ticks_df, 'nonexistent')
        self.assertIsNone(nearest)
        
        # Test with empty DataFrame
        nearest = find_nearest_tick(100, pd.DataFrame(), 'player1')
        self.assertIsNone(nearest)
    
    def test_calculate_time_in_round(self):
        """Test time in round calculation."""
        # Test with no round data (fallback to tick-based)
        time = calculate_time_in_round(128, pd.DataFrame(), 64)
        self.assertAlmostEqual(time, 2.0)  # 128 / 64 = 2 seconds
        
        # Test with round data
        rounds_df = pd.DataFrame({
            'start_tick': [0, 1000],
            'end_tick': [999, 1999]
        })
        time = calculate_time_in_round(1100, rounds_df, 64)
        self.assertAlmostEqual(time, 1.5625)  # (1100 - 1000) / 64
    
    def test_find_nearby_utility(self):
        """Test nearby utility detection."""
        # Test with nearby grenades
        utility = find_nearby_utility(101, 160, 210, self.grenades_df, radius=50)
        self.assertTrue(utility['smoke_near'])
        self.assertFalse(utility['flash_near'])
        
        # Test with no nearby grenades
        utility = find_nearby_utility(101, 1000, 1000, self.grenades_df, radius=50)
        self.assertFalse(utility['flash_near'])
        self.assertFalse(utility['smoke_near'])
        self.assertFalse(utility['molotov_near'])
        self.assertFalse(utility['he_near'])
        
        # Test with empty grenades DataFrame
        utility = find_nearby_utility(101, 160, 210, pd.DataFrame(), radius=50)
        self.assertFalse(utility['flash_near'])
        self.assertFalse(utility['smoke_near'])
        self.assertFalse(utility['molotov_near'])
        self.assertFalse(utility['he_near'])
    
    def test_load_map_data(self):
        """Test map data loading."""
        # Test with flat structure
        data = {'pos_x': 100, 'pos_y': 200, 'scale': 1.5}
        loaded = load_map_data_from_dict(data)
        self.assertEqual(loaded, data)
        
        # Test with nested structure
        nested_data = {'de_mirage': {'pos_x': 100, 'pos_y': 200, 'scale': 1.5}}
        loaded = load_map_data_from_dict(nested_data)
        self.assertEqual(loaded, nested_data['de_mirage'])


def load_map_data_from_dict(data):
    """Helper function to simulate load_map_data for testing."""
    # Handle nested structure (e.g., {"de_mirage": {"pos_x": 100, ...}})
    if isinstance(data, dict) and len(data) == 1:
        # Assume first key is the map name
        map_name = list(data.keys())[0]
        return data[map_name]
    
    return data


if __name__ == '__main__':
    unittest.main()
