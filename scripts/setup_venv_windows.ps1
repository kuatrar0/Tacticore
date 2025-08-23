# Tacticore Windows Environment Setup Script
# This script creates a virtual environment and installs all dependencies

Write-Host "Setting up Tacticore environment..." -ForegroundColor Green

# Check if Python 3.11+ is available
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = [version]$matches[1]
        if ($version -lt [version]"3.11") {
            Write-Host "Error: Python 3.11 or higher is required. Found version $($matches[1])" -ForegroundColor Red
            exit 1
        }
        Write-Host "Found Python $($matches[1]) - OK" -ForegroundColor Green
    } else {
        Write-Host "Error: Python not found or version could not be determined" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error: Python not found. Please install Python 3.11 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists. Removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nSetup complete! Here's how to use Tacticore:" -ForegroundColor Green
Write-Host "`n1. Activate the environment:" -ForegroundColor Cyan
Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor White

Write-Host "`n2. Parse a demo file:" -ForegroundColor Cyan
Write-Host "   python src/parser/parse_dem_to_parquet.py -i `"C:\path\to\demo.dem`" -o dataset" -ForegroundColor White

Write-Host "`n3. Launch the labeling app:" -ForegroundColor Cyan
Write-Host "   streamlit run src/streamlit_app/app.py" -ForegroundColor White

Write-Host "`n4. Build features and train model:" -ForegroundColor Cyan
Write-Host "   python src/features/build_features.py --kills dataset/demo_name/kills.parquet --ticks dataset/demo_name/ticks.parquet --labels results/features_labeled_context.csv" -ForegroundColor White
Write-Host "   python src/ml/training_lightgbm.py" -ForegroundColor White

Write-Host "`nFor more information, see README.md" -ForegroundColor Yellow
