#!/bin/bash

# Start FastAPI in the background
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the background
streamlit run src/streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 &

# Wait for both processes
wait
