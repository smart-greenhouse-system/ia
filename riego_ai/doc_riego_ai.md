# Documentación del Modelo de Riego - riego_ai

## Descripción

Modelo de regresión lineal simple que predice el tiempo en segundos que tardará un tanque de agua en alcanzar el nivel mínimo durante un proceso de riego con caudal constante.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/riego/` | Interfaz web interactiva para probar el modelo |
| `POST` | `/riego/predict` | API de predicción |

## Entrada (JSON)

### Modo manual

| Campo | Tipo | Unidad | Rango | Descripción |
|---|---|---|---|---|
| `cantidad` | float | litros | ≥ 0 | Nivel actual del tanque |
| `threshold_minimo` | float | litros | ≥ 0 | Nivel mínimo (umbral de alerta) |
| `caudal_bomba_lps` | float | L/s | > 0 | Caudal de la bomba (opcional, default: 0.075) |

### Modo desde base de datos

| Campo | Tipo | Descripción |
|---|---|---|
| `desde_bd` | boolean | `true` para leer datos desde MongoDB `inventory` |

### Ejemplos

```json
// Manual
{ "cantidad": 85, "threshold_minimo": 20, "caudal_bomba_lps": 0.075 }

// Desde BD
{ "desde_bd": true }
```

## Salida (JSON)

### Respuesta exitosa

```json
{
  "success": true,
  "nivel_actual": 85.0,
  "nivel_minimo": 20.0,
  "volumen_a_vaciar": 65.0,
  "caudal_bomba": 0.075,
  "tiempo_segundos": 866.7,
  "tiempo_formateado": "14 minutos y 27 segundos",
  "estado": "NORMAL"
}
```

### Tanque en nivel mínimo

```json
{
  "success": true,
  "nivel_actual": 15.0,
  "nivel_minimo": 20.0,
  "volumen_a_vaciar": 0,
  "caudal_bomba": 0.075,
  "tiempo_segundos": 0,
  "tiempo_formateado": "0 segundos",
  "estado": "TANQUE_EN_NIVEL_MINIMO"
}
```

### Error

```json
{ "success": false, "message": "cantidad (15.0) es menor que threshold_minimo (20.0). El tanque ya esta en nivel minimo." }
```

## Fórmula del Modelo

```
tiempo_segundos = coef × ((cantidad - threshold_minimo) / caudal_bomba_lps) + intercept
```

Idealmente: `coef ≈ 1.0`, `intercept ≈ 0.0`

## Cómo Probar

### Opción 1: Interfaz web

1. Inicia el servidor: `python app.py`
2. Abre en el navegador: [http://localhost:5000/riego/](http://localhost:5000/riego/)
3. Ingresa valores manualmente o haz clic en "Predecir desde BD"

### Opción 2: API con curl

```bash
# Predicción manual
curl -X POST http://localhost:5000/riego/predict \
  -H "Content-Type: application/json" \
  -d '{"cantidad":85,"threshold_minimo":20,"caudal_bomba_lps":0.075}'

# Predicción desde MongoDB
curl -X POST http://localhost:5000/riego/predict \
  -H "Content-Type: application/json" \
  -d '{"desde_bd":true}'
```

## Entrenamiento

```bash
python riego_ai/train.py
```

Genera 200 muestras sintéticas basadas en la fórmula física con ruido gaussiano (~1%), entrena el modelo de regresión lineal y guarda el `.pkl` en `models/`.

## Estructura de MongoDB

Colección: `inventory`

Documento esperado:
```json
{
  "nombre": "agua",
  "cantidad": 120,
  "unidad": "litros",
  "threshold_minimo": 20,
  "caudal_bomba_lps": 0.075
}
```

## Dependencias Adicionales

- `pymongo>=4.6.0`
- `dnspython>=2.4.0`
