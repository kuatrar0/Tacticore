@echo off
REM Cross-platform setup script for Tacticore
REM This script helps set up the project on any Windows machine

echo 🎯 Tacticore Cross-Platform Setup
echo =================================

REM Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed. Please install Python 3.8+ from https://python.org
    echo [INFO] Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo [SUCCESS] Python is installed

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python version: %PYTHON_VERSION%

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist .venv (
    echo [INFO] Virtual environment already exists
) else (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
echo [SUCCESS] Pip upgraded

REM Install requirements
echo [INFO] Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements
    echo [INFO] Trying with --no-cache-dir...
    pip install --no-cache-dir -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Still failed to install requirements
        pause
        exit /b 1
    )
)
echo [SUCCESS] Requirements installed

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist dataset mkdir dataset
if not exist src\backend\models mkdir src\backend\models
if not exist results mkdir results
if not exist maps mkdir maps
echo [SUCCESS] Directories created

REM Check if demo files exist
echo [INFO] Checking for demo files...
if exist "dataset\*.parquet" (
    echo [SUCCESS] Found existing parsed demo files
) else (
    echo [INFO] No parsed demo files found. You can add .dem files to parse them.
)

REM Test Streamlit installation
echo [INFO] Testing Streamlit installation...
streamlit --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Streamlit is working
) else (
    echo [WARNING] Streamlit test failed, but installation might still work
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Add your .dem files to the dataset folder
echo 2. Run: streamlit run src/streamlit_app/app.py
echo 3. Open http://localhost:8501 in your browser
echo.
echo For Docker setup (recommended for sharing):
echo 1. Install Docker Desktop from https://docker.com
echo 2. Run: docker-compose up --build
echo.
pause
