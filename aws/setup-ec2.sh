#!/bin/bash
# ==============================================
# Tacticore - EC2 Setup Script para Amazon Linux 2023
# ==============================================

set -e

echo "🚀 Configurando instancia para Tacticore..."

# Instalar Git y Docker
echo "📦 Instalando Git y Docker..."
sudo yum install -y git docker

# Iniciar Docker
echo "🐳 Iniciando Docker..."
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Instalar Docker Compose V2 y Buildx
echo "📦 Instalando Docker Compose V2 y Buildx..."
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins

# Docker Compose V2
curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
  -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

# Docker Buildx
curl -SL https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-amd64 \
  -o $DOCKER_CONFIG/cli-plugins/docker-buildx
chmod +x $DOCKER_CONFIG/cli-plugins/docker-buildx

echo ""
echo "=========================================="
echo "✅ Setup completado!"
echo ""
echo "Versiones instaladas:"
docker --version
$DOCKER_CONFIG/cli-plugins/docker-compose version
$DOCKER_CONFIG/cli-plugins/docker-buildx version
echo ""
echo "=========================================="
echo ""
echo "⚠️  IMPORTANTE: Cerrá y volvé a abrir la sesión SSH"
echo "   para que los permisos de Docker se apliquen."
echo ""
echo "   exit"
echo "   ssh -i tu-clave.pem ec2-user@<IP>"
echo ""
echo "Luego ejecutá:"
echo "   cd Tacticore"
echo "   docker compose up -d --build"
echo "=========================================="
