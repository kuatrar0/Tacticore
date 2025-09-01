# CS2 Kill Analyzer Backend

A FastAPI backend that analyzes CS2 demo files and provides kill predictions using a trained machine learning model.

## 🎯 Overview

This backend service:
- Loads a pre-trained ML model for CS2 kill analysis
- Processes `.dem` files and extracts kill data
- Provides predictions for each kill with confidence scores
- Returns comprehensive JSON analysis results

## 📋 Prerequisites

### Required Software
- **Python 3.8+**
- **Git** (to clone the repository)
- **CS2 Demo Files** (`.dem` format)

### Python Dependencies
All dependencies are listed in `requirements.txt` and will be installed automatically.

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Tacticore

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install backend dependencies
pip install -r src/backend/requirements.txt

# Install other project dependencies (if needed)
pip install -r requirements.txt
```

### 3. Train the Model (First Time Only)

Before using the backend, you need to train a model using the Streamlit app:

```bash
# Start the Streamlit app
cd src/streamlit_app
streamlit run app.py
```

**In the Streamlit app:**
1. Go to **"ML Training Mode"**
2. Upload your labeled `.parquet` files or CSV data
3. Click **"Train Model"**
4. The model will be automatically saved to `src/backend/models/`

### 4. Start the Backend

```bash
# Navigate to backend directory
cd src/backend

# Start the FastAPI server
python main.py
```

You should see:
```
✅ Model loaded successfully from C:\path\to\src\backend\models
🚀 Starting Tacticore CS2 Kill Analyzer Backend...
📁 Models directory: C:\path\to\src\backend\models
🤖 Model loaded: True
INFO: Uvicorn running on http://0.0.0.0:8000
```

## 📡 API Endpoints

### Health Check
```bash
GET http://localhost:8000/
```
Returns service status and model loading status.

### Model Information
```bash
GET http://localhost:8000/model-info
```
Returns information about the loaded model (features, classes, etc.).

### Analyze Demo File
```bash
POST http://localhost:8000/analyze-demo
Content-Type: multipart/form-data

Body: demo_file=<your-demo-file.dem>
```

## 🔧 Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/

# Get model info
curl http://localhost:8000/model-info

# Analyze a demo file
curl -X POST http://localhost:8000/analyze-demo \
  -F "demo_file=@path/to/your/demo.dem"
```

### Using Postman

1. **Method**: `POST`
2. **URL**: `http://localhost:8000/analyze-demo`
3. **Body**: `form-data`
4. **Key**: `demo_file` (type: File)
5. **Value**: Select your `.dem` file

### Using Python Requests

```python
import requests

# Analyze demo file
with open('path/to/demo.dem', 'rb') as f:
    files = {'demo_file': f}
    response = requests.post('http://localhost:8000/analyze-demo', files=files)
    
    if response.status_code == 200:
        results = response.json()
        print(f"Analyzed {results['total_kills']} kills")
        print(f"Map: {results['map']}")
        
        for kill in results['predictions']:
            print(f"Kill {kill['kill_id']}: {kill['prediction']['predicted_label']}")
    else:
        print(f"Error: {response.json()}")
```

## 📊 Response Format

The `/analyze-demo` endpoint returns:

```json
{
  "status": "success",
  "total_kills": 143,
  "map": "de_mirage",
  "tickrate": 64,
  "predictions": [
    {
      "kill_id": "181009_Aleksib_karrigan",
      "attacker": "Aleksib",
      "victim": "karrigan", 
      "place": "B Site",
      "round": 17,
      "weapon": "m4a1_silencer",
      "headshot": true,
      "distance": 553.21,
      "time_in_round": 147.84,
      "context": {
        "round_number": 17,
        "distance_xy": 553.21,
        "time_in_round_s": 147.84,
        "victim_was_aware": false,
        "had_sound_cue": false,
        "utility_count": 0,
        "approach_align_deg": 0
      },
      "prediction": {
        "predicted_label": "good_shot",
        "confidence": 0.85,
        "top_predictions": [
          {"label": "good_shot", "confidence": 0.85},
          {"label": "lucky_shot", "confidence": 0.12},
          {"label": "other", "confidence": 0.03}
        ],
        "all_probabilities": {
          "good_shot": 0.85,
          "lucky_shot": 0.12,
          "other": 0.03
        }
      }
    }
  ]
}
```

## ⚙️ Configuration

### Model Files
The backend expects these files in `src/backend/models/`:
- `kill_analyzer_model.pkl` - Trained ML model
- `label_encoder.pkl` - Label encoder for predictions
- `available_features.pkl` - Feature list (legacy, not used)

### Environment Variables
- `PORT` - Server port (default: 8000)
- `HOST` - Server host (default: 0.0.0.0)

## 🐛 Troubleshooting

### Common Issues

**1. "No trained model available"**
- Make sure you've trained a model using the Streamlit app
- Check that model files exist in `src/backend/models/`

**2. "Feature mismatch" errors**
- The model expects 7 features in specific order
- This should be automatically handled by the backend

**3. "Analysis failed" errors**
- Ensure your `.dem` file is valid
- Check that all required dependencies are installed

**4. Port already in use**
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <PID> /F
```

### Debug Information

The backend provides detailed debug output:
- Model loading status
- Feature extraction details
- Processing progress
- Error details

## 📁 Project Structure

```
src/backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # Trained model files
│   ├── kill_analyzer_model.pkl
│   ├── label_encoder.pkl
│   └── available_features.pkl
└── export_model.py     # Model export utility
```

## 🔄 Development

### Adding New Features
1. Modify `main.py` to add new endpoints
2. Update the `KillAnalyzer` class for new functionality
3. Test with sample data
4. Update this README

### Model Retraining
1. Use the Streamlit app to retrain with new data
2. The model will be automatically saved
3. Restart the backend to load the new model

## 📝 License

[Your License Here]

## 🤝 Contributing

[Your Contributing Guidelines Here]
