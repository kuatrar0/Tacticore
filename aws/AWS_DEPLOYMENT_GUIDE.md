# 🚀 Guía de Despliegue en AWS EC2 - Tacticore

Guía completa para desplegar Tacticore en una instancia EC2 para demo.

## Requisitos Previos

- Cuenta de AWS con acceso a EC2
- Par de claves SSH (.pem) para conectarte a la instancia

---

## Paso 1: Crear la Instancia EC2

### Desde la Consola de AWS:

1. Ir a **EC2 → Launch Instance**

2. **Configuración:**
   | Campo | Valor |
   |-------|-------|
   | Name | `tacticore-demo` |
   | AMI | Amazon Linux 2023 |
   | Instance type | `t3.large` (8 GB RAM) - mínimo recomendado |
   | Key pair | Seleccionar o crear una |
   | Storage | 30 GB gp3 |

3. **Security Group - Reglas de entrada:**
   | Tipo | Puerto | Origen |
   |------|--------|--------|
   | SSH | 22 | Tu IP |
   | Custom TCP | 8000 | 0.0.0.0/0 (API) |
   | Custom TCP | 8501 | 0.0.0.0/0 (Frontend) |

4. Click en **Launch Instance**

---

## Paso 2: Conectarse a la Instancia

### Configurar permisos del archivo .pem

```bash
# El archivo .pem debe tener permisos restrictivos
chmod 400 tu-clave.pem
```

### Conectarse por SSH

```bash
ssh -i "tu-clave.pem" ec2-user@<IP-PUBLICA>
```

---

## Paso 3: Instalar Dependencias

### Instalar Git

```bash
sudo yum install -y git
```

### Clonar el Repositorio

```bash
git clone https://github.com/kuatrar0/Tacticore.git
cd Tacticore
git checkout feature/upload-aws
```

---

## Paso 4: Instalar Docker y Docker Compose

Amazon Linux 2023 requiere instalación manual de Docker Compose V2 y Buildx.

### Instalar Docker

```bash
# Instalar Docker desde repositorio de Amazon
sudo yum install -y docker

# Iniciar y habilitar Docker
sudo systemctl start docker
sudo systemctl enable docker

# Agregar usuario al grupo docker
sudo usermod -aG docker ec2-user
```

### Instalar Docker Compose V2 y Buildx

```bash
# Crear directorio para plugins de Docker
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins

# Descargar Docker Compose V2
curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
  -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

# Descargar Docker Buildx
curl -SL https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-amd64 \
  -o $DOCKER_CONFIG/cli-plugins/docker-buildx
chmod +x $DOCKER_CONFIG/cli-plugins/docker-buildx
```

### Reconectar SSH

```bash
# Cerrar sesión para aplicar permisos de grupo docker
exit

# Reconectar
ssh -i "tu-clave.pem" ec2-user@<IP-PUBLICA>
```

### Verificar instalación

```bash
docker --version
docker compose version
docker buildx version
```

---

## Paso 5: Levantar la Aplicación

```bash
cd Tacticore
docker compose up -d --build
```

> **Nota:** Con Docker Compose V2 se usa `docker compose` (sin guión).

El build inicial toma **5-10 minutos** (descarga de dependencias y compilación de Go).

### Verificar que está corriendo

```bash
docker compose ps
docker compose logs -f
```

---

## Paso 6: Acceder a la Aplicación

| Servicio | URL |
|----------|-----|
| **Frontend (Streamlit)** | `http://<IP-PUBLICA>:8501` |
| **API (FastAPI)** | `http://<IP-PUBLICA>:8000` |
| **API Docs** | `http://<IP-PUBLICA>:8000/docs` |

### Probar la API

```bash
# Verificar que la API responde
curl http://localhost:8000/

# Ver información del modelo
curl http://localhost:8000/model-info

# Analizar un archivo demo
curl -X POST "http://localhost:8000/analyze-demo" \
  -F "demo_file=@archivo.dem"
```

---

## Comandos Útiles

```bash
# Ver logs en tiempo real
docker compose logs -f

# Reiniciar servicios
docker compose restart

# Detener todo
docker compose down

# Ver uso de recursos
docker stats

# Ver estado de contenedores
docker compose ps
```

---

## Troubleshooting

### Error: "Permissions for .pem are too open"

```bash
chmod 400 tu-clave.pem
```

### Error: "git: command not found"

```bash
sudo yum install -y git
```

### Error: "compose build requires buildx 0.17 or later"

Docker de Amazon Linux no incluye Buildx. Instalarlo manualmente:

```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-amd64 \
  -o $DOCKER_CONFIG/cli-plugins/docker-buildx
chmod +x $DOCKER_CONFIG/cli-plugins/docker-buildx
```

### Error: "permission denied" al usar docker

Reconectar SSH después de agregar usuario al grupo docker:

```bash
exit
ssh -i "tu-clave.pem" ec2-user@<IP-PUBLICA>
```

### Error de memoria durante el build

```bash
# Verificar memoria disponible
free -h

# Si hay menos de 4 GB, crear swap
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Puerto no accesible desde internet

1. Verificar Security Group en AWS Console
2. Verificar que el servicio esté corriendo: `docker compose ps`
3. Verificar logs: `docker compose logs`

### El contenedor se reinicia constantemente

```bash
# Ver logs para identificar el error
docker compose logs --tail=100

# Ver estado detallado
docker compose ps -a
```

### El análisis de demos es muy lento

El tiempo de análisis depende de:
- **Tamaño del archivo demo** (~350 MB = ~5 min)
- **Velocidad de upload** desde tu máquina a AWS
- **Tipo de instancia** (más CPU = más rápido)

Para archivos grandes, considerar subir directamente al servidor:

```bash
# Desde tu máquina local
scp -i "tu-clave.pem" archivo.dem ec2-user@<IP>:~/Tacticore/

# Luego en el servidor
curl -X POST "http://localhost:8000/analyze-demo" -F "demo_file=@archivo.dem"
```

---

## Limpieza (Después de la Demo)

```bash
# En la instancia
docker compose down
docker system prune -a

# Desde AWS Console: Terminate la instancia EC2
```

---

## Costos Estimados

| Instancia | RAM | Costo/hora | Costo/día |
|-----------|-----|------------|-----------|
| t3.large | 8 GB | ~$0.08 | ~$2 |
| t3.xlarge | 16 GB | ~$0.16 | ~$4 |

**💡 Tip:** Apagar la instancia cuando no se use para ahorrar costos.

---

## Resumen de Comandos Rápidos

```bash
# Setup completo (copiar y pegar)
sudo yum install -y git docker
sudo systemctl start docker && sudo systemctl enable docker
sudo usermod -aG docker ec2-user

DOCKER_CONFIG=$HOME/.docker
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
curl -SL https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-amd64 -o $DOCKER_CONFIG/cli-plugins/docker-buildx
chmod +x $DOCKER_CONFIG/cli-plugins/*

# Reconectar SSH, luego:
cd Tacticore
docker compose up -d --build
```
