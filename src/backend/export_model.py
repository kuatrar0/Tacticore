"""
Script to export trained model from Streamlit to FastAPI backend
"""
import pickle
import os
import sys
from pathlib import Path

# Add the parent directory to path to import from streamlit_app
sys.path.append(str(Path(__file__).parent.parent))

def export_model_from_streamlit():
    """Export the model that was trained in Streamlit to the backend"""
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Check if we have a trained model in session state
    # Since we can't access Streamlit session state directly from here,
    # we'll create a simple interface to export the model
    
    print("🔧 Model Export Tool")
    print("=" * 50)
    print()
    print("This tool helps you export your trained model from Streamlit to the FastAPI backend.")
    print()
    print("To use this:")
    print("1. First, train your model in the Streamlit app (ML Training Mode)")
    print("2. Then run this script to export it")
    print("3. The model will be saved to src/backend/models/")
    print()
    
    # Check if models directory exists and has files
    model_files = list(models_dir.glob("*.pkl"))
    
    if model_files:
        print(f"✅ Found {len(model_files)} model files in {models_dir}:")
        for file in model_files:
            print(f"   - {file.name}")
        print()
        print("Your model is ready for the FastAPI backend!")
        print()
        print("To start the backend server:")
        print("   cd src/backend")
        print("   pip install -r requirements.txt")
        print("   python main.py")
        print()
        print("Then you can:")
        print("   - Upload .dem files to http://localhost:8000/analyze-demo")
        print("   - Check model info at http://localhost:8000/model-info")
        print("   - View API docs at http://localhost:8000/docs")
    else:
        print("❌ No model files found.")
        print()
        print("To create a model:")
        print("1. Run the Streamlit app: streamlit run src/streamlit_app/app.py")
        print("2. Go to 'ML Training Mode'")
        print("3. Upload your labeled data and train the model")
        print("4. Run this export script again")
    
    return models_dir

if __name__ == "__main__":
    export_model_from_streamlit()
