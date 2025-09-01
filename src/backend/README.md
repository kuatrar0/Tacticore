# CS2 Kill Analysis FastAPI Backend

This FastAPI backend provides a production-ready service for analyzing CS2 demo files and providing kill predictions using a trained machine learning model.

## 🚀 Quick Start

### 1. Train Your Model (Streamlit)

First, train your model using the Streamlit app:

```bash
cd /c/Users/nicol/Documents/Tacticore
streamlit run src/streamlit_app/app.py
```

1. Go to "ML Training Mode"
2. Upload your labeled data and train the model
3. The model will automatically be saved to `src/backend/models/`

### 2. Start the FastAPI Backend

```bash
cd src/backend
pip install -r requirements.txt
python main.py
```

The server will start at `http://localhost:8000`

### 3. Use the API

#### Health Check
```bash
curl http://localhost:8000/
```

#### Upload and Analyze Demo
```bash
curl -X POST "http://localhost:8000/analyze-demo" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "demo_file=@your_demo.dem"
```

#### Get Model Info
```bash
curl http://localhost:8000/model-info
```

## 📋 API Endpoints

### `GET /`
Health check endpoint that shows if the model is loaded.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "available_features": ["distance_xy", "time_in_round_s", ...]
}
```

### `POST /analyze-demo`
Upload a `.dem` file and get kill predictions.

**Request:** Multipart form with `demo_file`

**Response:**
```json
{
  "status": "success",
  "total_kills": 25,
  "map": "de_dust2",
  "tickrate": 128,
  "predictions": [
    {
      "kill_id": "12345_Player1_Player2",
      "attacker": "Player1",
      "victim": "Player2",
      "place": "A Site",
      "round": 1,
      "weapon": "AK-47",
      "headshot": true,
      "distance": 150.5,
      "time_in_round": 45.2,
      "context": {
        "round_score": "15-10",
        "bomb_status": "planted",
        "team_sides": "T vs CT"
      },
      "prediction": {
        "predicted_label": "Good positioning",
        "confidence": 0.85,
        "uncertainty": 0.12,
        "probabilities": [0.1, 0.85, 0.05],
        "available_labels": ["Bad positioning", "Good positioning", "Lucky shot"]
      }
    }
  ]
}
```

### `GET /model-info`
Get information about the loaded model.

**Response:**
```json
{
  "model_type": "LGBMClassifier",
  "available_features": ["distance_xy", "time_in_round_s", ...],
  "available_labels": ["Good positioning", "Bad positioning", ...],
  "model_path": "models/kill_analyzer_model.pkl"
}
```

## 🏗️ Architecture

### Model Persistence
- Models are saved to disk in `models/` directory
- Automatically loaded when the server starts
- Persistent across server restarts

### File Structure
```
src/backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── export_model.py      # Model export utility
├── models/              # Saved models (auto-created)
│   ├── kill_analyzer_model.pkl
│   ├── label_encoder.pkl
│   └── available_features.pkl
└── README.md           # This file
```

## 🔧 Development

### Adding New Features
1. Modify the model training in Streamlit
2. The model will automatically be saved to the backend
3. Restart the FastAPI server to load the new model

### Customizing Predictions
Edit `predict_kill()` method in `main.py` to modify how predictions are made.

### Error Handling
The API includes comprehensive error handling:
- Missing model files
- Invalid demo files
- Processing errors
- Feature extraction failures

## 🚀 Production Deployment

### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📊 Integration Examples

### Python Client
```python
import requests

# Upload demo file
with open('demo.dem', 'rb') as f:
    files = {'demo_file': f}
    response = requests.post('http://localhost:8000/analyze-demo', files=files)

predictions = response.json()
for kill in predictions['predictions']:
    print(f"{kill['attacker']} → {kill['victim']}: {kill['prediction']['predicted_label']}")
```

### JavaScript/Node.js Client
```javascript
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('demo_file', fs.createReadStream('demo.dem'));

fetch('http://localhost:8000/analyze-demo', {
  method: 'POST',
  body: form
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🎯 Your End Goal

This backend achieves exactly what you wanted:

1. **User uploads .dem file** → API processes it
2. **Model analyzes kills** → Provides predictions and context
3. **Returns JSON** → Ready for AI feedback processing
4. **Persistent model** → Trained once, used many times
5. **Simple architecture** → Single service, no microservices

The JSON output includes all the information you need:
- Kill details (attacker, victim, place, weapon)
- Game context (round, score, bomb status)
- ML predictions (labels, confidence, uncertainty)
- Ready for your AI feedback system
