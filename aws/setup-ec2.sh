#!/bin/bash
# ==============================================
# Tacticore - EC2 Setup Script
# Ejecutar en una instancia Amazon Linux 2023 o Ubuntu
# ==============================================

set -e

echo "🚀 Configurando instancia para Tacticore..."

# Detectar sistema operativo
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

echo "📦 Instalando Docker..."

if [ "$OS" = "amzn" ] || [ "$OS" = "amazon" ]; then
    # Amazon Linux
    sudo yum update -y
    sudo yum install -y docker git
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ec2-user
elif [ "$OS" = "ubuntu" ]; then
    # Ubuntu
    sudo apt-get update
    sudo apt-get install -y docker.io git
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ubuntu
else
    echo "⚠️  Sistema operativo no reconocido. Instalando Docker manualmente..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
fi

echo "📦 Instalando Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "✅ Docker instalado correctamente"
docker --version
docker-compose --version

echo ""
echo "=========================================="
echo "✅ Setup completado!"
echo ""
echo "⚠️  IMPORTANTE: Cerrá y volvé a abrir la sesión SSH"
echo "   para que los permisos de Docker se apliquen."
echo ""
echo "Luego ejecutá:"
echo "   cd Tacticore"
echo "   docker-compose up -d --build"
echo "=========================================="

