# Use Python 3.11 slim image for better performance and smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Go
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Go (required by AWPy)
RUN wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz \
    && rm go1.21.5.linux-amd64.tar.gz

# Add Go to PATH
ENV PATH="/usr/local/go/bin:${PATH}"

# Copy requirements first for better Docker layer caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the entire project

# Create necessary directories
RUN mkdir -p src/backend/models || true
RUN mkdir -p dataset
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=8000

# Expose ports
EXPOSE 8501 8000

# Make the start script executable
RUN chmod +x start_services.sh

# Default command - start both services
CMD ["./start_services.sh"]
