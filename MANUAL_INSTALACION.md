# 🚀 Manual de Instalación y Uso - API Unificada

## 📋 Estructura Nuevo del Proyecto

```
.
├── unified_app.py           # APP PRINCIPAL
├── config.py                # Configuración centralizada
├── REQUIREMENTS.txt         # Dependencias
├── routes/                  # Blueprint para cada modelo
│   ├── __init__.py
│   ├── lechuga_routes.py
│   ├── moneda_routes.py
│   └── tomate_routes.py
├── lechuga_ai/              # Código original intacto
├── moneda_ai/               # Código original intacto
├── tomate_ai/               # Código original intacto
└── templates/               # HTML compartido
```

---

## ⚙️ PASO 1: Crear Entorno Virtual

### En Windows (PowerShell):
```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\Activate.ps1
```

### En Windows (CMD):
```cmd
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate.bat
```

### En Linux/Mac:
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

---

## 📦 PASO 2: Instalar Dependencias

Asegúrate de estar en el directorio raíz del proyecto y con el entorno virtual activado.

```bash
# Instalar todos los paquetes del REQUIREMENTS.txt
pip install -r REQUIREMENTS.txt
```

**Esto instalará:**
- Flask (framework web)
- ultralytics (modelos YOLO)
- opencv-python (procesamiento de imágenes)
- torch (pytorch para IA)
- numpy (cálculos numéricos)
- roboflow (gestión de datasets)
- python-dotenv (variables de entorno)

> ⏱️ Esto toma ~5-15 minutos dependiendo de tu conexión

---

## 🚀 PASO 3: Ejecutar la Aplicación

Desde el directorio raíz con el entorno virtual activado:

```bash
# Ejecutar el servidor
python unified_app.py
```

**Salida esperada:**
```
 * Serving Flask app 'create_app'
 * Environment: production
 * WARNING: This is a development server. Do not use it in production.
 * Running on http://127.0.0.1:5000
```

---

## 🔗 PASO 4: Probar los Endpoints

### 4.1 - Información General (GET)
```bash
curl http://localhost:5000/
```

**Respuesta:**
```json
{
  "mensaje": "API de Modelos de IA Unificada",
  "version": "1.0",
  "modelos_disponibles": [...]
}
```

---

### 4.2 - Predicción Lechuga (POST)
```bash
curl -X POST http://localhost:5000/lechuga/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_AQUI"}'
```

**Respuesta:**
```json
{
  "success": true,
  "etapa": "Harvest Stage",
  "apto_cosecha": true,
  "recomendacion": "Lechuga lista para cosechar.",
  "confianza": 0.95
}
```

---

### 4.3 - Predicción Moneda (POST)
```bash
curl -X POST http://localhost:5000/moneda/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_AQUI"}'
```

**Respuesta:**
```json
{
  "success": true,
  "total_monedas": 3,
  "monedas": [
    {"clase": "Moneda_1000", "confianza": 0.92, "tiene_mascara": false},
    {"clase": "Moneda_500", "confianza": 0.88, "tiene_mascara": true}
  ]
}
```

---

### 4.4 - Predicción Tomate (POST)
```bash
curl -X POST http://localhost:5000/tomate/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_AQUI"}'
```

**Respuesta:**
```json
{
  "success": true,
  "etapa": "Fruit Maturation",
  "dias_para_cosecha": 12,
  "confianza": 0.91
}
```

---

## 🛑 Detener la Aplicación

```bash
# En la terminal donde corre el servidor:
# Presiona: Ctrl + C
```

---

## 🔧 Configuración Avanzada

### Cambiar Puerto
Edita `unified_app.py`:
```python
app.run(debug=True, host="0.0.0.0", port=8000)  # Cambiar 5000 por 8000
```

### Cambiar Modo Debug
Para producción, edita `unified_app.py`:
```python
app.run(debug=False, host="0.0.0.0", port=5000)
```

### Variables de Entorno
Crea archivo `.env` en raíz:
```
FLASK_DEBUG=True
FLASK_ENV=development
SECRET_KEY=tu-clave-secreta-aqui
```

---

## 🐛 Solución de Problemas

### Error: "No module named 'flask'"
```bash
pip install flask==3.0.0
```

### Error: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Error: "Port 5000 already in use"
```bash
# Cambiar puerto en unified_app.py o ejecutar en otro puerto:
python unified_app.py  # Cambia el puerto en el código
```

### Los modelos no cargan
- Verifica que los archivos `.pt` existen en cada carpeta
- Comprueba rutas en `config.py`

---

## 📊 Estructura de Respuestas

### Predicción Exitosa
```json
{
  "success": true,
  "...": "otros campos específicos del modelo"
}
```

### Predicción Fallida
```json
{
  "success": false,
  "message": "Descripción del error"
}
```

---

## 💡 Notas Importantes

✅ **Los modelos originales NO fueron modificados**
✅ **Cada modelo corre independientemente**
✅ **Puedes agregar más modelos fácilmente**
✅ **La lógica de predicción se mantiene intacta**

---

## 🎯 Próximos Pasos

1. Probar cada endpoint con tus imágenes
2. Integrar con frontend
3. Agregar autenticación si es necesario
4. Desplegar en servidor (Heroku, AWS, etc.)
