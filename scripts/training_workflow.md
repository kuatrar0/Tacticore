# 🎯 Guía de Entrenamiento Manual - Tacticore ML

## 📋 Flujo de Entrenamiento Manual

Esta guía te ayudará a continuar entrenando tu modelo de Machine Learning de manera manual y controlada.

## 🚀 **Flujo Recomendado**

### **Opción 1: Streamlit App (Recomendado)**
```bash
# 1. Ejecutar la aplicación
streamlit run src/streamlit_app/app.py

# 2. En la app:
#    - Seleccionar "ML Training Mode"
#    - Subir tu archivo de labels existente
#    - Usar "Active Learning" para ver los kills más importantes
#    - Etiquetar manualmente uno por uno o en modo "Batch"
```

### **Opción 2: Docker (Si prefieres contenedores)**
```bash
# 1. Iniciar la aplicación
docker-compose up --build

# 2. Abrir en el navegador
# http://localhost:8501

# 3. Usar la interfaz web para etiquetar manualmente
```

## 📁 **Estructura de Archivos Recomendada**

```
Tacticore/
├── dataset/
│   └── imperial-vs-shinden-m1-mirage/
│       ├── kills.parquet
│       ├── ticks.parquet
│       ├── grenades.parquet
│       └── ...
├── results/
│   ├── labeled_data.csv              # Tus labels existentes
│   ├── features_labeled_context.csv # Features generadas
│   ├── active_learning_samples.csv  # Muestras para etiquetar
│   └── models/                       # Modelos entrenados
│       ├── tacticore_model_attackerlabel.pkl
│       └── tacticore_model_victimlabel.pkl
└── scripts/
    └── training_workflow.md
```

## 🔄 **Proceso de Entrenamiento Manual**

### **Paso 1: Preparar Datos**
1. **Asegúrate de tener tus archivos de datos**:
   - `dataset/imperial-vs-shinden-m1-mirage/kills.parquet`
   - `dataset/imperial-vs-shinden-m1-mirage/ticks.parquet`
   - `dataset/imperial-vs-shinden-m1-mirage/grenades.parquet` (opcional)

2. **Si tienes labels existentes**:
   - Guárdalos en `results/labeled_data.csv`

### **Paso 2: Iniciar la Aplicación**

**Opción A: Streamlit Directo**
```bash
streamlit run src/streamlit_app/app.py
```

**Opción B: Docker**
```bash
docker-compose up --build
# Luego ir a http://localhost:8501
```

### **Paso 3: Configurar en la App**
1. **Seleccionar modo**: "ML Training Mode"
2. **Subir archivos**:
   - `kills.parquet`
   - `ticks.parquet`
   - `grenades.parquet` (opcional)
3. **Subir labels existentes**: Si tienes un archivo CSV con labels
4. **Configurar mapa**: Las rutas ya están corregidas

### **Paso 4: Etiquetar Manualmente**
1. **Usar Active Learning**: La app te mostrará los kills más importantes
2. **Etiquetar uno por uno**: Revisar cada kill y aplicar labels manualmente
3. **Modo Batch**: Etiquetar múltiples kills de una vez (pero manualmente)
4. **Auto-retraining**: El modelo se reentrena automáticamente cuando agregas labels

## 🎯 **Mejores Prácticas**

### **1. Frecuencia de Entrenamiento**
- **Reentrenar cada 10-20 nuevos labels**
- **Usar Active Learning para priorizar kills importantes**
- **Revisar métricas del modelo regularmente**

### **2. Calidad de Labels**
- **Ser consistente con los criterios de etiquetado**
- **Etiquetar múltiples aspectos cuando sea relevante**
- **Revisar labels anteriores ocasionalmente**

### **3. Organización de Datos**
- **Mantener backup de labels importantes**
- **Versionar modelos entrenados**
- **Documentar cambios en criterios de etiquetado**

## 📊 **Monitoreo del Progreso**

### **Métricas a Revisar en la App**
1. **Accuracy del modelo** (objetivo: >80%)
2. **Número de labels por clase** (balancear clases)
3. **Uncertainty scores** (priorizar kills inciertos)

### **Verificar Progreso**
- **En la app**: Ver las métricas en tiempo real
- **Exportar datos**: Descargar tu dataset etiquetado
- **Revisar modelos**: Los modelos se guardan automáticamente

## 🚨 **Solución de Problemas Comunes**

### **Error: "Map data file not found"**
✅ **SOLUCIONADO** - Las rutas han sido corregidas

### **Error: "No labeled samples found"**
- Verificar que tienes labels en la app
- Usar "Import Labeled Data" para cargar labels existentes

### **Modelo con baja accuracy**
1. **Revisar balance de clases** en la app
2. **Aumentar número de labels**
3. **Verificar calidad de etiquetado**

## 🎯 **Workflow Diario Recomendado**

### **Sesión de 30 minutos:**
1. **5 min**: Iniciar app y cargar datos
2. **20 min**: Etiquetar usando Active Learning
3. **5 min**: Revisar métricas en la app

### **Sesión de 1 hora:**
1. **10 min**: Revisar métricas y ajustar filtros
2. **40 min**: Etiquetar en modo intensivo
3. **10 min**: Exportar datos y revisar resultados

## 📈 **Objetivos de Entrenamiento**

### **Fase 1: Base (0-100 labels)**
- **Objetivo**: 60% accuracy
- **Enfoque**: Etiquetar casos obvios y diversos
- **Frecuencia**: Reentrenar cada 20 labels

### **Fase 2: Mejora (100-500 labels)**
- **Objetivo**: 75% accuracy
- **Enfoque**: Usar Active Learning intensivamente
- **Frecuencia**: Reentrenar cada 50 labels

### **Fase 3: Refinamiento (500+ labels)**
- **Objetivo**: 85%+ accuracy
- **Enfoque**: Casos edge y corrección de errores
- **Frecuencia**: Reentrenar cada 100 labels

## 🎯 **Conclusión**

Este flujo te permitirá:
- ✅ **Etiquetar manualmente con control total**
- ✅ **Priorizar los kills más importantes para el modelo**
- ✅ **Mantener un proceso organizado y escalable**
- ✅ **Monitorear el progreso del modelo en tiempo real**

**Recomendación**: Usa la **aplicación Streamlit** para un control completo y manual del proceso de etiquetado.

