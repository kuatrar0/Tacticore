"""
Schema definitions for feature engineering and validation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class KillSchema:
    """Schema for kill event data."""
    required_columns = [
        'tick', 'attacker_name', 'victim_name', 'side', 'place', 'headshot'
    ]
    
    optional_columns = [
        'weapon', 'damage', 'armor', 'helmet'
    ]
    
    expected_types = {
        'tick': int,
        'attacker_name': str,
        'victim_name': str,
        'side': str,
        'place': str,
        'headshot': bool,
        'weapon': str,
        'damage': (int, float),
        'armor': (int, float),
        'helmet': bool
    }


@dataclass
class TickSchema:
    """Schema for tick data."""
    required_columns = [
        'tick', 'player_name', 'x', 'y', 'z', 'health'
    ]
    
    optional_columns = [
        'vel_x', 'vel_y', 'vel_z', 'side', 'alive', 'armor', 'helmet'
    ]
    
    expected_types = {
        'tick': int,
        'player_name': str,
        'x': (int, float),
        'y': (int, float),
        'z': (int, float),
        'health': (int, float),
        'vel_x': (int, float),
        'vel_y': (int, float),
        'vel_z': (int, float),
        'side': str,
        'alive': bool,
        'armor': (int, float),
        'helmet': bool
    }


@dataclass
class GrenadeSchema:
    """Schema for grenade data."""
    required_columns = [
        'tick', 'grenade_type', 'x', 'y', 'z'
    ]
    
    optional_columns = [
        'player_name', 'side', 'thrown', 'detonated'
    ]
    
    expected_types = {
        'tick': int,
        'grenade_type': str,
        'x': (int, float),
        'y': (int, float),
        'z': (int, float),
        'player_name': str,
        'side': str,
        'thrown': bool,
        'detonated': bool
    }


@dataclass
class LabeledFeatureSchema:
    """Schema for final labeled features."""
    required_columns = [
        'kill_tick', 'attacker_name', 'victim_name', 'side', 'place',
        'distance_xy', 'time_in_round_s', 'approach_align_deg',
        'attacker_label', 'victim_label'
    ]
    
    optional_columns = [
        'attacker_health', 'victim_health', 'headshot',
        'flash_near', 'smoke_near', 'molotov_near', 'he_near',
        'attacker_x', 'attacker_y', 'attacker_z',
        'victim_x', 'victim_y', 'victim_z'
    ]
    
    expected_types = {
        'kill_tick': int,
        'attacker_name': str,
        'victim_name': str,
        'side': str,
        'place': str,
        'distance_xy': (int, float),
        'time_in_round_s': (int, float),
        'approach_align_deg': (int, float),
        'attacker_label': str,
        'victim_label': str,
        'attacker_health': (int, float),
        'victim_health': (int, float),
        'headshot': bool,
        'flash_near': bool,
        'smoke_near': bool,
        'molotov_near': bool,
        'he_near': bool,
        'attacker_x': (int, float),
        'attacker_y': (int, float),
        'attacker_z': (int, float),
        'victim_x': (int, float),
        'victim_y': (int, float),
        'victim_z': (int, float)
    }


def validate_dataframe_schema(df, schema) -> Dict[str, Any]:
    """
    Validate DataFrame against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Schema class to validate against
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'valid': True,
        'missing_required': [],
        'missing_optional': [],
        'type_mismatches': [],
        'warnings': []
    }
    
    for col in schema.required_columns:
        if col not in df.columns:
            validation_result['missing_required'].append(col)
            validation_result['valid'] = False
    
    for col in schema.optional_columns:
        if col not in df.columns:
            validation_result['missing_optional'].append(col)
    
    for col, expected_type in schema.expected_types.items():
        if col in df.columns:
            if not df[col].empty:
                sample_value = df[col].iloc[0]
                if not isinstance(sample_value, expected_type):
                    validation_result['type_mismatches'].append({
                        'column': col,
                        'expected': expected_type,
                        'actual': type(sample_value)
                    })
                    validation_result['warnings'].append(
                        f"Column '{col}' has unexpected type: {type(sample_value)}"
                    )
    
    return validation_result


def get_schema_warnings(validation_result: Dict[str, Any]) -> List[str]:
    """
    Get warning messages from validation result.
    
    Args:
        validation_result: Result from validate_dataframe_schema
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    if validation_result['missing_required']:
        warnings.append(f"Missing required columns: {validation_result['missing_required']}")
    
    if validation_result['missing_optional']:
        warnings.append(f"Missing optional columns: {validation_result['missing_optional']}")
    
    if validation_result['type_mismatches']:
        for mismatch in validation_result['type_mismatches']:
            warnings.append(
                f"Type mismatch in '{mismatch['column']}': "
                f"expected {mismatch['expected']}, got {mismatch['actual']}"
            )
    
    return warnings


def get_feature_columns() -> List[str]:
    """
    Get list of feature columns for ML training.
    
    Returns:
        List of feature column names
    """
    return [
        'distance_xy',
        'time_in_round_s', 
        'approach_align_deg',
        'attacker_health',
        'victim_health',
        'headshot',
        'flash_near',
        'smoke_near',
        'molotov_near',
        'he_near',
        # Categorical features (will be one-hot encoded)
        'side',
        'place'
    ]


def get_label_columns() -> List[str]:
    """
    Get list of label columns.
    
    Returns:
        List of label column names
    """
    return ['attacker_label', 'victim_label']


def get_attacker_labels() -> List[str]:
    """
    Get list of valid attacker labels.
    
    Returns:
        List of attacker label values
    """
    return [
        'good_decision',
        'bad_decision', 
        'precise',
        'imprecise',
        'good_positioning',
        'bad_positioning',
        'other'
    ]


def get_victim_labels() -> List[str]:
    """
    Get list of valid victim labels.
    
    Returns:
        List of victim label values
    """
    return [
        'exposed',
        'no_cover',
        'good_position',
        'mistake',
        'other'
    ]
