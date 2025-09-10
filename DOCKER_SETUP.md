# 🐳 Docker Setup for Tacticore

This guide will help you run Tacticore in a Docker container, ensuring consistent behavior across different environments.

## 📋 Prerequisites

### Windows Users
1. **Install Docker Desktop for Windows**
   - Download from: https://www.docker.com/products/docker-desktop/
   - Enable WSL 2 integration (recommended)
   - Ensure virtualization is enabled in BIOS

2. **Install WSL 2 (Recommended)**
   ```powershell
   wsl --install
   ```

### Linux/Mac Users
1. **Install Docker**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install docker.io docker-compose
   
   # macOS
   brew install docker docker-compose
   ```

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone and navigate to the project**
   ```bash
   git clone <your-repo-url>
   cd Tacticore
   ```

2. **Build and start the application**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - **Frontend (Streamlit)**: http://localhost:8501
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

### Option 2: Using Docker directly

1. **Build the image**
   ```bash
   docker build -t tacticore .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 -p 8000:8000 \
     -v $(pwd)/dataset:/app/dataset \
     -v $(pwd)/src/backend/models:/app/src/backend/models \
     -v $(pwd)/results:/app/results \
     -v $(pwd)/maps:/app/maps \
     tacticore
   ```

## 🔧 Configuration

### Environment Variables

You can customize the setup using environment variables:

```bash
# In docker-compose.yml or docker run command
environment:
  - STREAMLIT_SERVER_PORT=8501
  - FASTAPI_PORT=8000
  - PYTHONPATH=/app
```

### Volume Mounts

The following directories are mounted for data persistence:

- `./dataset` → `/app/dataset` (Demo files and parsed data)
- `./src/backend/models` → `/app/src/backend/models` (Trained ML models)
- `./results` → `/app/results` (Analysis results)
- `./maps` → `/app/maps` (Map data and images)

## 🛠️ Development Mode

For development with live code changes:

```bash
# Run with volume mounts for live reloading
docker-compose -f docker-compose.dev.yml up
```

## 📊 Monitoring and Logs

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f tacticore
```

### Check container status
```bash
docker-compose ps
```

### Access container shell
```bash
docker-compose exec tacticore bash
```

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using the ports
   netstat -tulpn | grep :8501
   netstat -tulpn | grep :8000
   
   # Kill the process or change ports in docker-compose.yml
   ```

2. **Permission issues (Linux/Mac)**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER dataset/ src/backend/models/ results/
   ```

3. **Windows WSL2 issues**
   - Ensure WSL 2 is enabled in Docker Desktop settings
   - Restart Docker Desktop
   - Try running in WSL 2 terminal instead of PowerShell

4. **Memory issues**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   # Recommended: 4GB+ RAM for Docker
   ```

### Health Checks

The application includes health checks:

```bash
# Check backend health
curl http://localhost:8000/

# Check frontend health
curl http://localhost:8501
```

## 🏗️ Building for Production

### Multi-stage build (Optional)

For smaller production images, you can use a multi-stage build:

```dockerfile
# Add to Dockerfile
FROM python:3.11-slim as builder
# ... build dependencies ...

FROM python:3.11-slim as production
# ... copy from builder ...
```

### Security considerations

1. **Run as non-root user**
   ```dockerfile
   RUN adduser --disabled-password --gecos '' appuser
   USER appuser
   ```

2. **Use specific versions**
   - Pin all dependency versions
   - Use `--no-cache` for builds

## 📝 Notes

- The Docker setup includes both Streamlit frontend and FastAPI backend
- All data is persisted through volume mounts
- The application automatically starts both services
- Health checks ensure services are ready before use

## 🆘 Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Verify Docker is running: `docker --version`
3. Check available resources: `docker system df`
4. Restart Docker Desktop (Windows) or Docker daemon (Linux/Mac)
