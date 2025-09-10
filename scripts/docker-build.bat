@echo off
REM Docker build script for Tacticore (Windows)
REM This script helps build and run the Docker container

echo 🐳 Tacticore Docker Build Script
echo =================================

REM Check if Docker is installed and running
echo [INFO] Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

echo [SUCCESS] Docker is installed and running

REM Check if Docker Compose is available
echo [INFO] Checking Docker Compose...
docker-compose --version >nul 2>&1
if %errorlevel% equ 0 (
    set COMPOSE_CMD=docker-compose
    goto :compose_found
)

docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    set COMPOSE_CMD=docker compose
    goto :compose_found
)

echo [ERROR] Docker Compose is not available. Please install Docker Compose.
exit /b 1

:compose_found
echo [SUCCESS] Docker Compose is available

REM Main script logic
if "%1"=="build" goto :build
if "%1"=="run" goto :run
if "%1"=="dev" goto :dev
if "%1"=="stop" goto :stop
if "%1"=="logs" goto :logs
if "%1"=="cleanup" goto :cleanup
goto :help

:build
echo [INFO] Building Docker image...
docker build -t tacticore:latest .
if %errorlevel% equ 0 (
    echo [SUCCESS] Docker image built successfully
) else (
    echo [ERROR] Failed to build Docker image
    exit /b 1
)
goto :end

:run
echo [INFO] Starting Tacticore with Docker Compose...
%COMPOSE_CMD% up --build
goto :end

:dev
echo [INFO] Starting Tacticore in development mode...
%COMPOSE_CMD% -f docker-compose.dev.yml up --build
goto :end

:stop
echo [INFO] Stopping Tacticore containers...
%COMPOSE_CMD% down 2>nul
%COMPOSE_CMD% -f docker-compose.dev.yml down 2>nul
docker stop tacticore-app 2>nul
docker rm tacticore-app 2>nul
echo [SUCCESS] Containers stopped
goto :end

:logs
echo [INFO] Showing Tacticore logs...
%COMPOSE_CMD% logs -f
goto :end

:cleanup
echo [INFO] Cleaning up Docker resources...
docker container prune -f
docker image prune -f
docker volume prune -f
echo [SUCCESS] Cleanup completed
goto :end

:help
echo Usage: %0 {build^|run^|dev^|stop^|logs^|cleanup^|help}
echo.
echo Commands:
echo   build      - Build the Docker image
echo   run        - Run with Docker Compose (production mode)
echo   dev        - Run in development mode with live reloading
echo   stop       - Stop all running containers
echo   logs       - Show application logs
echo   cleanup    - Clean up Docker resources
echo   help       - Show this help message
echo.
echo Examples:
echo   %0 build
echo   %0 run
echo   %0 dev
echo   %0 stop

:end
