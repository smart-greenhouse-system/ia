# 🎯 API Unificada de Modelos de IA

Aplicación Flask que integra tres modelos de IA independientes (Lechuga, Moneda, Tomate) en una sola API con endpoints específicos para cada modelo.

---

## 🏗️ Arquitectura

### Antes (3 microservicios separados)
```
lechuga_ai/        moneda_ai/        tomate_ai/
  app.py              app.py             app.py
  :5001               :5002              :5003
```

### Después (1 aplicación unificada)
```
unified_app.py
  ├── /lechuga/predict   (LechugaInference)
  ├── /moneda/predict    (MonedaInference)
  └── /tomate/predict    (TomatoInference)
  
  :5000
```

---

## 📁 Estructura de Archivos

```
.
├── unified_app.py              # 🔴 ARCHIVO PRINCIPAL
├── config.py                   # Configuración centralizada
├── diagnose.py                 # Script de diagnóstico
├── EJEMPLOS_USO.py            # Ejemplos en Python/JS
├── MANUAL_INSTALACION.md      # Guía paso a paso
├── REQUIREMENTS.txt           # Dependencias
│
├── routes/                     # 🔴 BLUEPRINTS DE MODELOS
│   ├── __init__.py
│   ├── lechuga_routes.py      # Endpoints de lechuga
│   ├── moneda_routes.py       # Endpoints de moneda
│   └── tomate_routes.py       # Endpoints de tomate
│
├── lechuga_ai/                # Código original (intacto)
├── moneda_ai/                 # Código original (intacto)
├── tomate_ai/                 # Código original (intacto)
│
└── templates/                 # HTML compartido
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
python unified_app.py
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
| `GET` | `/lechuga/` | Página inicio lechuga |
| `POST` | `/lechuga/predict` | Predicción de etapa lechuga |
| `GET` | `/moneda/` | Página inicio moneda |
| `POST` | `/moneda/predict` | Predicción de monedas |
| `GET` | `/tomate/` | Página inicio tomate |
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
{
  "success": true,
  "etapa": "Harvest Stage",
  "apto_cosecha": true,
  "confianza": 0.95
}
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
# En unified_app.py
app.register_blueprint(lechuga_bp, url_prefix='/lechuga')
# Convierte:
#   /lechuga/predict  → lechuga_routes.py:predict_lechuga()
```

---

## ✅ Sin Modificaciones a Lógica Original

**Importante:** La lógica de cada modelo se mantiene intacta:
- Los archivos `inference.py` no fueron modificados
- Los archivos `model.py` no fueron modificados
- Solo se creó un "wrapper" (envoltorio) en los blueprints

```python
# Antes (lechuga_ai/app.py):
model = LechugaInference()
result = model.predict_base64(image_base64)

# Después (routes/lechuga_routes.py):
modelo_lechuga = LechugaInference()  # 🟢 MISMO CÓDIGO
result = modelo_lechuga.predict_base64(image_base64)  # 🟢 MISMA FUNCIÓN
```

---

## 🛠️ Comando Paso a Paso (Resumen)

### Windows PowerShell
```powershell
# 1. Crear entorno
python -m venv venv

# 2. Activar entorno
.\venv\Scripts\Activate.ps1

# 3. Instalar paquetes
pip install -r REQUIREMENTS.txt

# 4. Ejecutar
python unified_app.py

# 5. Verificar en navegador
Start-Process http://localhost:5000
```

### Linux / Mac
```bash
# 1. Crear entorno
python3 -m venv venv

# 2. Activar entorno
source venv/bin/activate

# 3. Instalar paquetes
pip install -r REQUIREMENTS.txt

# 4. Ejecutar
python unified_app.py

# 5. Verificar
open http://localhost:5000
```

---

## 🧪 Diagnóstico Automático

```bash
python diagnose.py
```

Verifica:
- ✓ Versión de Python
- ✓ Paquetes instalados
- ✓ Estructura de carpetas
- ✓ Archivos de modelos
- ✓ Importaciones funcionales

---

## 📚 Documentación Adicional

- **[MANUAL_INSTALACION.md](MANUAL_INSTALACION.md)** - Guía completa paso a paso
- **[EJEMPLOS_USO.py](EJEMPLOS_USO.py)** - Ejemplos en Python, JavaScript y curl
- **[config.py](config.py)** - Configuración centralizada

---

## 🔐 Configuración para Producción

1. Cambiar `DEBUG = False` en `unified_app.py`
2. Generar `SECRET_KEY` fuerte
3. Usar gunicorn en lugar de Flask:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 unified_app:app
   ```
4. Usar reverse proxy (nginx)
5. HTTPS/SSL

---

## 🤝 Agregar Nuevo Modelo

1. Crear carpeta `nuevo_modelo_ai/`
2. Copiar estructura (inference.py, model.py, etc.)
3. Crear `routes/nuevo_modelo_routes.py`
4. Registrar blueprint en `unified_app.py`:
   ```python
   from routes.nuevo_modelo_routes import nuevo_modelo_bp
   app.register_blueprint(nuevo_modelo_bp, url_prefix='/nuevo_modelo')
   ```

---

## 📞 Soporte

| Problema | Solución |
|----------|----------|
| Port 5000 ya en uso | Cambiar puerto en `unified_app.py` |
| Módulo no encontrado | `pip install -r REQUIREMENTS.txt` |
| Modelos no cargan | Verificar rutas en `config.py` |
| ImportError | Ejecutar `python diagnose.py` |

---

## 📊 Notas de Rendimiento

- **Modelos en memoria:** Se cargan una sola vez al iniciar
- **Concurrent requests:** Usar Gunicorn con workers
- **Timeout:** Aumentar en producción para modelos lentos

---

## 📝 Versión

- **API Version:** 1.0
- **Flask:** 3.0.0
- **Ultralytics:** 8.0.0+
- **Python:** 3.8+

---

**¡API lista para usar! 🚀**
