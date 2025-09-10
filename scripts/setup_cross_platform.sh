#!/bin/bash

# Cross-platform setup script for Tacticore
# This script helps set up the project on Linux/Mac

set -e

echo "🎯 Tacticore Cross-Platform Setup"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed. Please install Python 3.8+ first."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD --version
print_success "Python is installed"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Create virtual environment
print_status "Creating virtual environment..."
if [ -d ".venv" ]; then
    print_status "Virtual environment already exists"
else
    $PYTHON_CMD -m venv .venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
print_success "Pip upgraded"

# Install requirements
print_status "Installing requirements..."
if pip install -r requirements.txt; then
    print_success "Requirements installed"
else
    print_warning "Standard installation failed, trying with --no-cache-dir..."
    if pip install --no-cache-dir -r requirements.txt; then
        print_success "Requirements installed with --no-cache-dir"
    else
        print_error "Failed to install requirements"
        exit 1
    fi
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p dataset src/backend/models results maps
print_success "Directories created"

# Check if demo files exist
print_status "Checking for demo files..."
if ls dataset/*.parquet 1> /dev/null 2>&1; then
    print_success "Found existing parsed demo files"
else
    print_status "No parsed demo files found. You can add .dem files to parse them."
fi

# Test Streamlit installation
print_status "Testing Streamlit installation..."
if streamlit --version &> /dev/null; then
    print_success "Streamlit is working"
else
    print_warning "Streamlit test failed, but installation might still work"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Add your .dem files to the dataset folder"
echo "2. Run: streamlit run src/streamlit_app/app.py"
echo "3. Open http://localhost:8501 in your browser"
echo ""
echo "For Docker setup (recommended for sharing):"
echo "1. Install Docker from https://docker.com"
echo "2. Run: docker-compose up --build"
echo ""
