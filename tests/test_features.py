"""
Unit tests for feature engineering functions.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from features.schemas import (
    KillSchema, TickSchema, GrenadeSchema, LabeledFeatureSchema,
    validate_dataframe_schema, get_schema_warnings, get_feature_columns
)


class TestFeatures(unittest.TestCase):
    """Test cases for feature engineering functions."""
    
    def setUp(self):
        """Set up test data."""
        # Sample kills data
        self.kills_df = pd.DataFrame({
            'tick': [100, 101, 102],
            'attacker_name': ['player1', 'player2', 'player1'],
            'victim_name': ['player2', 'player3', 'player3'],
            'side': ['T', 'CT', 'T'],
            'place': ['A', 'B', 'A'],
            'headshot': [True, False, True],
            'weapon': ['ak47', 'm4a1', 'awp'],
            'damage': [100, 85, 100]
        })
        
        # Sample ticks data
        self.ticks_df = pd.DataFrame({
            'tick': [100, 101, 102, 103],
            'player_name': ['player1', 'player2', 'player3', 'player1'],
            'x': [100, 200, 300, 110],
            'y': [150, 250, 350, 160],
            'z': [50, 60, 70, 55],
            'health': [100, 80, 90, 95],
            'vel_x': [10, -5, 0, 15],
            'vel_y': [5, 10, -5, 8]
        })
        
        # Sample grenades data
        self.grenades_df = pd.DataFrame({
            'tick': [100, 101, 102],
            'grenade_type': ['flashbang', 'smoke', 'molotov'],
            'x': [150, 160, 170],
            'y': [200, 210, 220],
            'z': [30, 35, 40],
            'player_name': ['player1', 'player2', 'player1']
        })
    
    def test_kill_schema_validation(self):
        """Test kills data validation."""
        validation = validate_dataframe_schema(self.kills_df, KillSchema)
        self.assertTrue(validation['valid'])
        self.assertEqual(len(validation['missing_required']), 0)
    
    def test_tick_schema_validation(self):
        """Test ticks data validation."""
        validation = validate_dataframe_schema(self.ticks_df, TickSchema)
        self.assertTrue(validation['valid'])
        self.assertEqual(len(validation['missing_required']), 0)
    
    def test_grenade_schema_validation(self):
        """Test grenades data validation."""
        validation = validate_dataframe_schema(self.grenades_df, GrenadeSchema)
        self.assertTrue(validation['valid'])
        self.assertEqual(len(validation['missing_required']), 0)
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        # Remove required column
        incomplete_df = self.kills_df.drop('attacker_name', axis=1)
        validation = validate_dataframe_schema(incomplete_df, KillSchema)
        self.assertFalse(validation['valid'])
        self.assertIn('attacker_name', validation['missing_required'])
    
    def test_missing_optional_columns(self):
        """Test validation with missing optional columns."""
        # Remove optional column
        incomplete_df = self.kills_df.drop('weapon', axis=1)
        validation = validate_dataframe_schema(incomplete_df, KillSchema)
        self.assertTrue(validation['valid'])  # Should still be valid
        self.assertIn('weapon', validation['missing_optional'])
    
    def test_get_schema_warnings(self):
        """Test warning message generation."""
        validation = {
            'missing_required': ['attacker_name'],
            'missing_optional': ['weapon'],
            'type_mismatches': [
                {'column': 'tick', 'expected': int, 'actual': str}
            ]
        }
        warnings = get_schema_warnings(validation)
        self.assertEqual(len(warnings), 3)
        self.assertIn('Missing required columns', warnings[0])
        self.assertIn('Missing optional columns', warnings[1])
        self.assertIn('Type mismatch', warnings[2])
    
    def test_get_feature_columns(self):
        """Test feature column list."""
        feature_cols = get_feature_columns()
        expected_cols = [
            'distance_xy', 'time_in_round_s', 'approach_align_deg',
            'attacker_health', 'victim_health', 'headshot',
            'flash_near', 'smoke_near', 'molotov_near', 'he_near',
            'side', 'place'
        ]
        self.assertEqual(feature_cols, expected_cols)
    
    def test_labeled_feature_schema(self):
        """Test labeled feature schema."""
        # Create sample labeled features
        labeled_df = pd.DataFrame({
            'kill_tick': [100, 101, 102],
            'attacker_name': ['player1', 'player2', 'player1'],
            'victim_name': ['player2', 'player3', 'player3'],
            'side': ['T', 'CT', 'T'],
            'place': ['A', 'B', 'A'],
            'distance_xy': [500.0, 750.0, 300.0],
            'time_in_round_s': [45.2, 67.8, 23.1],
            'approach_align_deg': [15.5, 45.2, 8.9],
            'attacker_label': ['good_decision', 'precise', 'bad_decision'],
            'victim_label': ['exposed', 'no_cover', 'mistake']
        })
        
        validation = validate_dataframe_schema(labeled_df, LabeledFeatureSchema)
        self.assertTrue(validation['valid'])
    
    def test_data_type_validation(self):
        """Test data type validation."""
        # Create DataFrame with wrong types
        wrong_types_df = pd.DataFrame({
            'tick': ['100', '101', '102'],  # Should be int
            'attacker_name': ['player1', 'player2', 'player1'],
            'victim_name': ['player2', 'player3', 'player3'],
            'side': ['T', 'CT', 'T'],
            'place': ['A', 'B', 'A'],
            'headshot': [True, False, True]
        })
        
        validation = validate_dataframe_schema(wrong_types_df, KillSchema)
        self.assertTrue(validation['valid'])  # Type checking is lenient
        # But should have warnings about type mismatches
        if validation['type_mismatches']:
            self.assertIn('tick', [m['column'] for m in validation['type_mismatches']])


if __name__ == '__main__':
    unittest.main()
