#!/bin/bash

# Docker build script for Tacticore
# This script helps build and run the Docker container

set -e

echo "🐳 Tacticore Docker Build Script"
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

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_success "Docker Compose is available"
}

# Build the Docker image
build_image() {
    print_status "Building Docker image..."
    
    if docker build -t tacticore:latest .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Run with Docker Compose
run_with_compose() {
    local mode=${1:-"production"}
    
    print_status "Starting Tacticore with Docker Compose (mode: $mode)..."
    
    if [ "$mode" = "dev" ]; then
        $COMPOSE_CMD -f docker-compose.dev.yml up --build
    else
        $COMPOSE_CMD up --build
    fi
}

# Run with Docker directly
run_direct() {
    print_status "Starting Tacticore with Docker directly..."
    
    # Create necessary directories
    mkdir -p dataset src/backend/models results maps
    
    docker run -d \
        --name tacticore-app \
        -p 8501:8501 \
        -p 8000:8000 \
        -v "$(pwd)/dataset:/app/dataset" \
        -v "$(pwd)/src/backend/models:/app/src/backend/models" \
        -v "$(pwd)/results:/app/results" \
        -v "$(pwd)/maps:/app/maps" \
        tacticore:latest
    
    print_success "Tacticore started in background"
    print_status "Frontend: http://localhost:8501"
    print_status "Backend: http://localhost:8000"
}

# Stop running containers
stop_containers() {
    print_status "Stopping Tacticore containers..."
    
    # Stop docker-compose containers
    $COMPOSE_CMD down 2>/dev/null || true
    $COMPOSE_CMD -f docker-compose.dev.yml down 2>/dev/null || true
    
    # Stop direct docker containers
    docker stop tacticore-app 2>/dev/null || true
    docker rm tacticore-app 2>/dev/null || true
    
    print_success "Containers stopped"
}

# Clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    print_success "Cleanup completed"
}

# Show logs
show_logs() {
    print_status "Showing Tacticore logs..."
    
    if docker ps -q -f name=tacticore-app | grep -q .; then
        docker logs -f tacticore-app
    else
        $COMPOSE_CMD logs -f
    fi
}

# Main script logic
main() {
    case "${1:-help}" in
        "build")
            check_docker
            build_image
            ;;
        "run")
            check_docker
            check_docker_compose
            run_with_compose "${2:-production}"
            ;;
        "run-direct")
            check_docker
            build_image
            run_direct
            ;;
        "dev")
            check_docker
            check_docker_compose
            run_with_compose "dev"
            ;;
        "stop")
            check_docker_compose
            stop_containers
            ;;
        "logs")
            check_docker_compose
            show_logs
            ;;
        "cleanup")
            check_docker
            cleanup
            ;;
        "help"|*)
            echo "Usage: $0 {build|run|run-direct|dev|stop|logs|cleanup|help}"
            echo ""
            echo "Commands:"
            echo "  build      - Build the Docker image"
            echo "  run        - Run with Docker Compose (production mode)"
            echo "  run-direct - Run with Docker directly"
            echo "  dev        - Run in development mode with live reloading"
            echo "  stop       - Stop all running containers"
            echo "  logs       - Show application logs"
            echo "  cleanup    - Clean up Docker resources"
            echo "  help       - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 build"
            echo "  $0 run"
            echo "  $0 dev"
            echo "  $0 stop"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
