# 🎯 API Unificada de Modelos de IA

Aplicación Flask que integra modelos de Inteligencia Artificial para detección de cultivos, identificación de estados de crecimiento y predicción de comportamientos en base a variables.

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

```


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


