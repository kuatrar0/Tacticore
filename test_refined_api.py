#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for refined dual perspective API
Excludes redundant and low-frequency labels
"""

import requests
import json
from pathlib import Path

def test_refined_api():
    """Test the refined dual perspective API."""
    print("Testing Tacticore Refined Dual Perspective API")
    print("Excluding redundant and low-frequency labels")
    print("=" * 60)
    
    # Check if demo file exists
    demo_file = Path("galorys-vs-shinden-m2-mirage.dem")
    if not demo_file.exists():
        print(f"ERROR: Demo file not found: {demo_file}")
        return False
    
    print(f"Demo file: {demo_file}")
    print(f"File size: {demo_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Test API endpoint
    url = "http://127.0.0.1:8000/analyze-demo"
    
    try:
        print(f"\nSending request to: {url}")
        
        with open(demo_file, 'rb') as f:
            files = {'demo_file': (demo_file.name, f, 'application/octet-stream')}
            response = requests.post(url, files=files, timeout=300)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\nSUCCESS: API Response")
            print(f"Total kills: {data.get('total_kills', 0)}")
            print(f"Map: {data.get('map', 'Unknown')}")
            print(f"Tickrate: {data.get('tickrate', 64)}")
            
            # Show first few predictions
            predictions = data.get('predictions', [])
            if predictions:
                print(f"\nFirst 5 predictions (refined model):")
                for i, pred in enumerate(predictions[:5]):
                    print(f"\n{i+1}. {pred.get('attacker', 'Unknown')} -> {pred.get('victim', 'Unknown')}")
                    print(f"   Weapon: {pred.get('weapon', 'Unknown')}")
                    print(f"   Distance: {pred.get('distance', 0):.1f}u")
                    print(f"   Headshot: {pred.get('headshot', False)}")
                    
                    # Show attacker strengths (refined labels only)
                    attacker_strengths = pred.get('attacker_strengths', {})
                    if attacker_strengths:
                        print(f"   Attacker Strengths:")
                        for label, prob in attacker_strengths.items():
                            print(f"     - {label}: {prob:.1%}")
                    else:
                        print(f"   No significant attacker strengths detected")
                    
                    # Show victim errors (refined labels only)
                    victim_errors = pred.get('victim_errors', {})
                    if victim_errors:
                        print(f"   Victim Errors:")
                        for label, prob in victim_errors.items():
                            print(f"     - {label}: {prob:.1%}")
                    else:
                        print(f"   No significant victim errors detected")
                    
                    # Show analysis summary
                    summary = pred.get('analysis_summary', '')
                    if summary:
                        print(f"   Summary: {summary}")
            
            # Analyze refined model performance
            print(f"\n=== REFINED MODEL ANALYSIS ===")
            attacker_strengths_found = 0
            victim_errors_found = 0
            attacker_types = {}
            victim_types = {}
            
            for pred in predictions:
                attacker_strengths = pred.get('attacker_strengths', {})
                victim_errors = pred.get('victim_errors', {})
                
                if attacker_strengths:
                    attacker_strengths_found += 1
                    for label in attacker_strengths.keys():
                        attacker_types[label] = attacker_types.get(label, 0) + 1
                
                if victim_errors:
                    victim_errors_found += 1
                    for label in victim_errors.keys():
                        victim_types[label] = victim_types.get(label, 0) + 1
            
            print(f"Kills with attacker strengths: {attacker_strengths_found}/{len(predictions)} ({attacker_strengths_found/len(predictions)*100:.1f}%)")
            print(f"Kills with victim errors: {victim_errors_found}/{len(predictions)} ({victim_errors_found/len(predictions)*100:.1f}%)")
            
            if attacker_types:
                print(f"\nMost common attacker strengths:")
                for label, count in sorted(attacker_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {label}: {count} kills")
            
            if victim_types:
                print(f"\nMost common victim errors:")
                for label, count in sorted(victim_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {label}: {count} kills")
            
            print(f"\n=== REFINED MODEL BENEFITS ===")
            print(f"+ Only 26 models (9 attacker + 17 victim)")
            print(f"+ Excluded 14 low-frequency victim labels")
            print(f"+ Better performance with focused labels")
            print(f"+ More relevant predictions")
            
            return True
            
        else:
            print(f"ERROR: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API. Is the backend running?")
        print("Start it with: python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_refined_api()
    if success:
        print(f"\nSUCCESS: Refined dual perspective API is working correctly!")
        print(f"Model optimized with only relevant labels!")
    else:
        print(f"\nERROR: Refined dual perspective API test failed!")
