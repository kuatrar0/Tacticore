# 🚀 Guía de Despliegue en AWS EC2 - Tacticore

Guía rápida para desplegar Tacticore en una instancia EC2 para demo.

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
   | AMI | Amazon Linux 2023 (o Ubuntu 22.04) |
   | Instance type | `t3.large` (8 GB RAM) |
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

```bash
# Obtener la IP pública de la instancia desde la consola de AWS
ssh -i "tu-clave.pem" ec2-user@<IP-PUBLICA>

# Si usas Ubuntu:
ssh -i "tu-clave.pem" ubuntu@<IP-PUBLICA>
```

---

## Paso 3: Clonar el Repositorio

```bash
git clone <URL-DE-TU-REPO> Tacticore
cd Tacticore
```

**Alternativa - Subir archivos manualmente:**
```bash
# Desde tu máquina local:
scp -i "tu-clave.pem" -r /path/to/Tacticore ec2-user@<IP-PUBLICA>:~/
```

---

## Paso 4: Ejecutar Setup

```bash
chmod +x aws/setup-ec2.sh
./aws/setup-ec2.sh
```

**⚠️ Después del setup, cerrar y reconectar SSH:**
```bash
exit
ssh -i "tu-clave.pem" ec2-user@<IP-PUBLICA>
```

---

## Paso 5: Levantar la Aplicación

```bash
cd Tacticore
docker-compose up -d --build
```

El build inicial toma **5-10 minutos** (descarga de dependencias y Go).

### Verificar que está corriendo:
```bash
docker-compose ps
docker-compose logs -f
```

---

## Paso 6: Acceder a la Aplicación

| Servicio | URL |
|----------|-----|
| **Frontend (Streamlit)** | `http://<IP-PUBLICA>:8501` |
| **API (FastAPI)** | `http://<IP-PUBLICA>:8000` |
| **API Docs** | `http://<IP-PUBLICA>:8000/docs` |

---

## Comandos Útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Reiniciar servicios
docker-compose restart

# Detener todo
docker-compose down

# Ver uso de recursos
docker stats
```

---

## Troubleshooting

### Error de memoria durante el build
```bash
# Verificar memoria disponible
free -h

# Si no hay suficiente, crear swap
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Puerto no accesible
- Verificar Security Group en AWS Console
- Verificar que el servicio esté corriendo: `docker-compose ps`

### El contenedor se reinicia constantemente
```bash
# Ver logs para identificar el error
docker-compose logs --tail=100
```

---

## Limpieza (Después de la Demo)

```bash
# En la instancia
docker-compose down
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

