#!/usr/bin/env python3
"""
FastAPI Backend for Tacticore CS2 Kill Analysis

A simple monolithic backend that loads a trained ML model and provides
endpoints for analyzing .dem files and returning kill predictions.
"""

import os
import sys
import tempfile
import shutil
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from streamlit_app.transforms import get_enhanced_kill_context
from parser.parse_dem_to_parquet import parse_single_demo
from .filtered_dual_analyzer import filtered_dual_analyzer

app = FastAPI(
    title="Tacticore CS2 Kill Analyzer",
    description="FastAPI backend for analyzing CS2 demo files and providing kill predictions",
    version="1.0.0"
)

class KillAnalyzer:
    """Manages model loading, saving, and prediction."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.available_features = None
        self.models_dir = Path(__file__).parent / "models"
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            model_path = self.models_dir / "kill_analyzer_model.pkl"
            encoder_path = self.models_dir / "label_encoder.pkl"
            features_path = self.models_dir / "available_features.pkl"
            
            if model_path.exists() and encoder_path.exists() and features_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                with open(features_path, 'rb') as f:
                    self.available_features = pickle.load(f)
                print(f"SUCCESS: Model loaded successfully from {self.models_dir}")
            else:
                print(f"WARNING: Model files not found in {self.models_dir}")
                print(f"   Expected: {model_path}, {encoder_path}, {features_path}")
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
    
    def predict_kill(self, kill_dict: Dict, map_data: Dict, tickrate: int = 64) -> Dict:
        """Predict labels for a single kill."""
        if not self.model or not self.label_encoder:
            return {"error": "No trained model available"}
        
        try:
            features = []
            features.append(float(kill_dict.get('distance_xy', 0)))
            features.append(float(kill_dict.get('time_in_round_s', 0)))
            features.append(1 if kill_dict.get('headshot', False) else 0)
            features.append(1 if kill_dict.get('victim_was_aware', False) else 0)
            features.append(1 if kill_dict.get('had_sound_cue', False) else 0)
            features.append(float(kill_dict.get('utility_count', 0)))
            features.append(float(kill_dict.get('approach_align_deg', 0) or 0))
            
            # Make prediction
            prediction_proba = self.model.predict_proba([features])[0]
            predicted_class = self.model.predict([features])[0]
            
            # Get class names
            class_names = self.label_encoder.classes_
            predicted_label = class_names[predicted_class]
            
            # Calculate confidence
            confidence = max(prediction_proba)
            
            # Get top 3 predictions
            top_indices = np.argsort(prediction_proba)[::-1][:3]
            top_predictions = [
                {
                    "label": class_names[i],
                    "confidence": float(prediction_proba[i])
                }
                for i in top_indices
            ]
            
            return {
                "predicted_label": predicted_label,
                "confidence": float(confidence),
                "top_predictions": top_predictions,
                "all_probabilities": {
                    class_names[i]: float(prediction_proba[i])
                    for i in range(len(class_names))
                }
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    import numpy as np
    
    if obj is None:
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj

# Initialize analyzer
analyzer = KillAnalyzer()

def process_demo_file(demo_path: str) -> Dict:
    """
    Process a demo file by converting it to parquet and loading the data.
    
    Args:
        demo_path: Path to the .dem file
        
    Returns:
        Dictionary containing all loaded DataFrames
    """
    try:
        demo_path = Path(demo_path)
        
        # Create temporary directory for parquet files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Parse demo to parquet
            print(f"Parsing demo: {demo_path}")
            saved_data = parse_single_demo(demo_path, temp_path)
            
            if not saved_data:
                raise Exception("Failed to parse demo file")
            
            # Load parquet files
            data = {}
            
            # Load kills
            kills_file = temp_path / demo_path.stem / "kills.parquet"
            if kills_file.exists():
                data['kills_df'] = pd.read_parquet(kills_file)
                print(f"Loaded {len(data['kills_df'])} kills")
            
            # Load ticks
            ticks_file = temp_path / demo_path.stem / "ticks.parquet"
            if ticks_file.exists():
                data['ticks_df'] = pd.read_parquet(ticks_file)
                print(f"Loaded {len(data['ticks_df'])} ticks")
            
            # Load rounds
            rounds_file = temp_path / demo_path.stem / "rounds.parquet"
            if rounds_file.exists():
                data['rounds_df'] = pd.read_parquet(rounds_file)
                print(f"Loaded {len(data['rounds_df'])} rounds")
            
            # Load grenades
            grenades_file = temp_path / demo_path.stem / "grenades.parquet"
            if grenades_file.exists():
                data['grenades_df'] = pd.read_parquet(grenades_file)
                print(f"Loaded {len(data['grenades_df'])} grenade events")
            else:
                data['grenades_df'] = pd.DataFrame()
            
            # Load damages
            damages_file = temp_path / demo_path.stem / "damages.parquet"
            if damages_file.exists():
                data['damages_df'] = pd.read_parquet(damages_file)
                print(f"Loaded {len(data['damages_df'])} damage events")
            else:
                data['damages_df'] = pd.DataFrame()
            
            # Load shots
            shots_file = temp_path / demo_path.stem / "shots.parquet"
            if shots_file.exists():
                data['shots_df'] = pd.read_parquet(shots_file)
                print(f"Loaded {len(data['shots_df'])} shot events")
            else:
                data['shots_df'] = pd.DataFrame()
            
            # Load smokes
            smokes_file = temp_path / demo_path.stem / "smokes.parquet"
            if smokes_file.exists():
                data['smokes_df'] = pd.read_parquet(smokes_file)
                print(f"Loaded {len(data['smokes_df'])} smoke events")
            else:
                data['smokes_df'] = pd.DataFrame()
            
            # Load infernos
            infernos_file = temp_path / demo_path.stem / "infernos.parquet"
            if infernos_file.exists():
                data['infernos_df'] = pd.read_parquet(infernos_file)
                print(f"Loaded {len(data['infernos_df'])} inferno events")
            else:
                data['infernos_df'] = pd.DataFrame()
            
            # Load bomb events
            bomb_file = temp_path / demo_path.stem / "bomb.parquet"
            if bomb_file.exists():
                data['bomb_df'] = pd.read_parquet(bomb_file)
                print(f"Loaded {len(data['bomb_df'])} bomb events")
            else:
                data['bomb_df'] = pd.DataFrame()
            
            # Load map data
            map_name = saved_data.get('map', 'de_mirage')
            maps_dir = Path(__file__).parent.parent.parent / "maps"
            map_data_path = maps_dir / "map-data.json"
            
            if map_data_path.exists():
                import json
                with open(map_data_path, 'r') as f:
                    all_maps = json.load(f)
                data['map_data'] = all_maps.get(map_name, {})
            else:
                data['map_data'] = {}
            
            # Get tickrate (default to 64)
            data['tickrate'] = 64
            
            data['map'] = saved_data.get('map', 'unknown')
            return data
            
    except Exception as e:
        raise Exception(f"Failed to process demo file: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Tacticore CS2 Kill Analyzer",
        "model_loaded": analyzer.model is not None
    }

@app.post("/train")
async def train_model_endpoint():
    """
    Placeholder for training endpoint.
    Currently expects labeled_kills.csv to be available.
    """
    return {
        "message": "Training endpoint - use Streamlit app to train model first",
        "instructions": "1. Use the Streamlit app in ML Training Mode to train a model",
        "instructions2": "2. The model will be automatically saved to backend/models/",
        "instructions3": "3. Restart this backend to load the trained model"
    }

@app.post("/analyze-demo")
async def analyze_demo(demo_file: UploadFile = File(...)):
    """
    Analyze a demo file and return kill predictions.
    
    Args:
        demo_file: Uploaded .dem file
        
    Returns:
        JSON with kill analysis and predictions
    """
    if not filtered_dual_analyzer.is_loaded():
        return JSONResponse(
            status_code=400,
            content={"error": "No trained filtered dual model available. Please train a model first using the CSV data."}
        )
    
    if not demo_file.filename.lower().endswith('.dem'):
        return JSONResponse(
            status_code=400,
            content={"error": "File must be a .dem file"}
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dem") as tmp_file:
            shutil.copyfileobj(demo_file.file, tmp_file)
            tmp_path = tmp_file.name
        
        result = process_demo_file(tmp_path)
        
        # Extract data
        kills_df = result.get('kills_df')
        ticks_df = result.get('ticks_df')
        rounds_df = result.get('rounds_df')
        grenades_df = result.get('grenades_df', pd.DataFrame())
        damages_df = result.get('damages_df', pd.DataFrame())
        shots_df = result.get('shots_df', pd.DataFrame())
        smokes_df = result.get('smokes_df', pd.DataFrame())
        infernos_df = result.get('infernos_df', pd.DataFrame())
        bomb_df = result.get('bomb_df', pd.DataFrame())
        map_data = result.get('map_data', {})
        tickrate = result.get('tickrate', 64)
        
        if kills_df is None or ticks_df is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract kills or ticks data from demo"}
            )
        
        print(f"Processing {len(kills_df)} kills from demo...")
        
        # Analyze each kill
        predictions = []
        for idx, kill in kills_df.iterrows():
            try:
                kill_dict = kill.to_dict()
                
                kill_context = get_enhanced_kill_context(
                    kill, ticks_df, rounds_df, grenades_df, damages_df, shots_df,
                    smokes_df, infernos_df, bomb_df, map_data, tickrate
                )
                kill_context = convert_numpy_types(kill_context)
                
                enhanced_kill_dict = kill_dict.copy()
                enhanced_kill_dict.update(kill_context)
                dual_analysis = filtered_dual_analyzer.analyze_kill(enhanced_kill_dict, threshold=0.5)
                kill_result = {
                    "kill_id": str(f"{kill.get('tick', '')}_{kill.get('attacker_name', '')}_{kill.get('victim_name', '')}"),
                    "attacker": str(kill.get('attacker_name', 'Unknown')),
                    "victim": str(kill.get('victim_name', 'Unknown')),
                    "place": str(kill.get('place', 'Unknown')),
                    "round": int(kill_context.get('round_number', kill_context.get('round', 0))),
                    "weapon": str(kill.get('weapon', 'Unknown')),
                    "headshot": bool(kill.get('headshot', False)),
                    "distance": float(kill_context.get('distance_xy', 0.0)),
                    "time_in_round": float(kill_context.get('time_in_round_s', 0.0)),
                    "context": kill_context,
                    "attacker_strengths": dual_analysis.get("attacker_strengths", {}),
                    "victim_errors": dual_analysis.get("victim_errors", {}),
                    "analysis_summary": filtered_dual_analyzer.get_analysis_summary(dual_analysis)
                }
                predictions.append(kill_result)
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(kills_df)} kills...")
                    
            except Exception as e:
                print(f"Error processing kill {idx}: {e}")
                continue
        
        os.unlink(tmp_path)
        
        actual_map_name = result.get('map', 'Unknown')
        response_data = {
            "status": "success",
            "total_kills": len(predictions),
            "map": str(actual_map_name),
            "tickrate": int(tickrate),
            "predictions": predictions
        }
        
        response_data = convert_numpy_types(response_data)
        
        return response_data
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if not analyzer.model:
        return {"error": "No model loaded"}
    
    return {
        "model_type": type(analyzer.model).__name__,
        "n_classes": len(analyzer.label_encoder.classes_) if analyzer.label_encoder else 0,
        "classes": analyzer.label_encoder.classes_.tolist() if analyzer.label_encoder else [],
        "n_features": len(analyzer.available_features) if analyzer.available_features else 0,
        "features": list(analyzer.available_features) if analyzer.available_features else []
    }

if __name__ == "__main__":
    print("🚀 Starting Tacticore CS2 Kill Analyzer Backend...")
    print(f"📁 Models directory: {analyzer.models_dir}")
    print(f"🤖 Model loaded: {analyzer.model is not None}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
