# 🎯 API Unificada de Modelos de IA

Aplicación Flask que integra tres modelos de IA independientes (Lechuga, Moneda, Tomate) en una sola API con endpoints específicos para cada modelo.

---

## 🤝 Agregar Nuevo Modelo

1. Crear carpeta `nuevo_modelo_ai/`
2. Copiar estructura (inference.py, model.py, etc.)
3. Crear `routes/nuevo_modelo_routes.py`
4. Registrar blueprint en `app.py`:
   ```python
   from routes.nuevo_modelo_routes import nuevo_modelo_bp
   app.register_blueprint(nuevo_modelo_bp, url_prefix='/nuevo_modelo')
   ```

---

## 📝 Versión

- **API Version:** 1.0
- **Flask:** 3.0.0
- **Ultralytics:** 8.0.0+
- **Python:** 3.8+


## 🏗️ Arquitectura


### 1 aplicación unificada
```
app.py
  ├── /lechuga/predict   (LechugaInference)
  ├── /moneda/predict    (MonedaInference)
  └── /tomate/predict    (TomatoInference)
  
  :5000
```

---

## 📁 Estructura de Archivos

```
.
├── app.py              # 🔴 ARCHIVO PRINCIPAL
├── MANUAL_INSTALACION.md      # Guía paso a paso
├── REQUIREMENTS.txt           # Dependencias
│
├── routes/                     # 🔴 BLUEPRINTS DE MODELOS
│   ├── __init__.py
│   ├── lechuga_routes.py      # Endpoints de lechuga
│   ├── moneda_routes.py       # Endpoints de moneda
│   └── tomate_routes.py       # Endpoints de tomate
│
├── lechuga_ai/                
├── moneda_ai/                 
├── tomate_ai/                 
│
└── templates/                
    ├── index.html
    └── 404.html
```

---

## 🚀 Inicio Rápido

### 1️⃣ Crear Entorno Virtual
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# o
source venv/bin/activate      # Linux/Mac
```

### 2️⃣ Instalar Dependencias
```bash
pip install -r REQUIREMENTS.txt
```

### 3️⃣ Ejecutar Aplicación
```bash
python app.py
```

### 4️⃣ Probar Endpoints
```bash
# Información general
curl http://localhost:5000/

# Predicción de lechuga
curl -X POST http://localhost:5000/lechuga/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_STRING"}'
```

---

## 📡 Endpoints Disponibles

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Información general de la API |
| `POST` | `/lechuga/predict` | Predicción de etapa lechuga |
| `POST` | `/moneda/predict` | Predicción de monedas |
| `POST` | `/tomate/predict` | Predicción etapa tomate |

---

## 🎨 Estructura de Requests

### Request (POST a cualquier /*/predict)
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
}
```

### Response (exitosa)
```json
Toca cuadrar bien
```

### Response (error)
```json
{
  "success": false,
  "message": "No se recibió imagen en formato JSON"
}
```

---

## 🔍 ¿Cómo Funciona?

### 1. Blueprints de Flask
Cada modelo es un "blueprint" (sub-aplicación):
- Los blueprints se registran con prefijos únicos
- Permiten mantener código modular y escalable
- Cada blueprint importa su respectiva clase Inference

### 2. Clases de Inference
Cada carpeta de modelo contiene:
- `inference.py` → Carga modelo y realiza predicción
- `model.py` → Arquitectura del modelo
- `utils/` → Funciones de preprocesamiento
- `models/` → Pesos entrenados (.pt)

### 3. Rutas Dinámicas
```python
# En app.py
app.register_blueprint(lechuga_bp, url_prefix='/lechuga')
# Convierte:
#   /lechuga/predict  → lechuga_routes.py:predict_lechuga()
```

## 📚 Documentación Adicional

- **[MANUAL_INSTALACION.md](MANUAL_INSTALACION.md)** - Guía completa paso a paso

---

## 🔐 Configuración para Producción

1. Cambiar `DEBUG = False` en `app.py`
2. Generar `SECRET_KEY` fuerte
3. Usar gunicorn en lugar de Flask:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
4. Usar reverse proxy (nginx)
5. HTTPS/SSL

---


## 📊 Notas de Rendimiento

- **Modelos en memoria:** Se cargan una sola vez al iniciar
- **Concurrent requests:** Usar Gunicorn con workers
- **Timeout:** Aumentar en producción para modelos lentos

---


