#!/bin/bash

# Exit on any error
set -e

# Check if we should run a simple command instead of starting services
if [ "$1" = "python" ] || [ "$1" = "--version" ] || [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ "$1" = "streamlit" ] || [ "$1" = "uvicorn" ]; then
    echo "🔧 Running command: $@"
    exec "$@"
fi

echo "🚀 Starting Tacticore Application..."

# Function to start FastAPI backend
start_backend() {
    echo "📡 Starting FastAPI backend on port 8000..."
    cd /app
    python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
}

# Function to start Streamlit frontend
start_frontend() {
    echo "🎨 Starting Streamlit frontend on port 8501..."
    cd /app
    streamlit run src/streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"
}

# Function to wait for services
wait_for_services() {
    echo "⏳ Waiting for services to start..."
    
    # Wait for backend
    echo "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/ >/dev/null 2>&1; then
            echo "✅ Backend is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Backend failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Wait for frontend
    echo "Waiting for frontend to be ready..."
    for i in {1..30}; do
        if curl -f http://localhost:8501 >/dev/null 2>&1; then
            echo "✅ Frontend is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Frontend failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
}

# Function to handle shutdown
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Check if we should run in backend-only mode
if [ "$1" = "backend-only" ]; then
    echo "🔧 Running in backend-only mode..."
    start_backend
    wait $BACKEND_PID
else
    # Start both services
    start_backend
    sleep 2  # Give backend a moment to start
    start_frontend
    
    # Wait for services to be ready
    wait_for_services
    
    echo ""
    echo "🎉 Tacticore is ready!"
    echo "📊 Frontend: http://localhost:8501"
    echo "🔌 Backend API: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo ""
    echo "Press Ctrl+C to stop the application"
    
    # Wait for either process to exit
    wait $BACKEND_PID $FRONTEND_PID
fi
