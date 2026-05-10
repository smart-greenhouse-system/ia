# Documentación del Modelo Predictivo - predictivo_ai

## Descripción

Modelo de clasificación híbrido (reglas determinísticas + árbol de decisión) que evalúa variables ambientales de sensores IoT y determina si las condiciones de cultivo son normales o requieren una acción correctiva.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/predictivo/` | Interfaz web interactiva para probar el modelo |
| `POST` | `/predictivo/predict` | API de predicción (JSON) |

## Entrada (JSON)

| Campo | Tipo | Unidad | Rango | Descripción |
|---|---|---|---|---|
| `cultivo` | string | - | `"lechuga"` / `"tomate"` | Tipo de cultivo |
| `etapa` | string | - | Ver tabla abajo | Etapa fenológica |
| `temperature` | float | °C | 0 - 45 | Temperatura |
| `humidity` | float | % | 0 - 100 | Humedad relativa |
| `soil_moisture` | float | % | 0 - 100 | Humedad del suelo |
| `sunlight` | float | µmol/m²/s | 0 - 1200 | Iluminación PAR |
| `todas_las_variables` | boolean | - | `true` / `false` | Opcional. `true` retorna todas las variables fuera de rango en un array |

### Ejemplo de entrada con todas las variables

```json
{
  "cultivo": "tomate",
  "etapa": "Floracion",
  "temperature": 12.5,
  "humidity": 30,
  "soil_moisture": 20,
  "sunlight": 50,
  "todas_las_variables": true
}
```

## Rangos Óptimos por Cultivo y Etapa

### Lechuga

| Etapa | Temp (°C) | HR (%) | Suelo (%) | Luz (µmol/m²/s) |
|---|---|---|---|---|
| Germinacion | 18 - 22 | 80 - 95 | 60 - 80 | 50 - 100 |
| Plantula | 15 - 20 | 70 - 85 | 50 - 70 | 100 - 200 |
| CrecimientoVegetativo | 14 - 18 | 60 - 75 | 40 - 60 | 200 - 400 |
| FormacionCabeza | 12 - 16 | 55 - 70 | 35 - 55 | 300 - 500 |
| Cosecha | 10 - 15 | 50 - 65 | 30 - 45 | 200 - 400 |

Notas:
- Temperaturas > 28°C provocan espigado prematuro (amargor)
- Humedad > 85% + poca ventilación = riesgo de botrytis
- Luz < 150 µmol causa crecimiento etiolado (alargado)

### Tomate

| Etapa | Temp (°C) | HR (%) | Suelo (%) | Luz (µmol/m²/s) |
|---|---|---|---|---|
| Germinacion | 22 - 28 | 80 - 90 | 70 - 85 | 50 - 100 |
| Plantula | 20 - 25 | 70 - 80 | 60 - 75 | 150 - 250 |
| CrecimientoVegetativo | 18 - 24 | 65 - 75 | 55 - 70 | 300 - 500 |
| Floracion | 20 - 25 | 60 - 70 | 60 - 75 | 400 - 600 |
| CuajadoFrutos | 18 - 24 | 55 - 65 | 65 - 80 | 500 - 700 |
| Maduracion | 16 - 22 | 50 - 60 | 50 - 65 | 400 - 600 |

Notas:
- Temperatura nocturna < 12°C en Floracion → aborto floral
- Humedad > 80% + temperatura < 25°C → condiciones para mildiu
- Humedad < 50% → mala polinización y frutos rajados

## Etapas Válidas

**Lechuga:** `Germinacion`, `Plantula`, `CrecimientoVegetativo`, `FormacionCabeza`, `Cosecha`

**Tomate:** `Germinacion`, `Plantula`, `CrecimientoVegetativo`, `Floracion`, `CuajadoFrutos`, `Maduracion`

## Salida (JSON)

### Condición normal
```json
{ "success": true, "status": "OK", "cultivo": "lechuga", "etapa": "Germinacion" }
```

### Alerta (con `todas_las_variables: true`)

```json
{
  "success": true,
  "status": "alerta",
  "accion": "ACTIVAR_CALEFACCION",
  "mensaje": "TOMATE en Floracion: Temperatura 12.5 por debajo del optimo (20-25). Accion: Aumentar temperatura.",
  "nivel_urgencia": "CRITICO",
  "cultivo": "tomate",
  "etapa": "Floracion",
  "variable": "temperatura",
  "valor_actual": 12.5,
  "valor_optimo_min": 20,
  "valor_optimo_max": 25,
  "todas_las_variables": [
    {
      "variable": "temperatura",
      "valor_actual": 12.5,
      "valor_optimo_min": 20,
      "valor_optimo_max": 25,
      "accion": "ACTIVAR_CALEFACCION",
      "nivel_urgencia": "CRITICO"
    },
    {
      "variable": "humedad",
      "valor_actual": 30,
      "valor_optimo_min": 60,
      "valor_optimo_max": 70,
      "accion": "ACTIVAR_NEBULIZACION",
      "nivel_urgencia": "CRITICO"
    },
    {
      "variable": "iluminacion",
      "valor_actual": 50,
      "valor_optimo_min": 400,
      "valor_optimo_max": 600,
      "accion": "ACTIVAR_LED_SUPLEMENTAL",
      "nivel_urgencia": "CRITICO"
    }
  ]
}
```

### Error
```json
{ "success": false, "message": "Cultivo 'pepino' no valido. Cultivos validos: lechuga, tomate" }
```

## Acciones de Alerta Posibles

| Código | Descripción |
|---|---|
| `SIN_ALERTA` | Condiciones normales |
| `ACTIVAR_VENTILACION` | Reducir temperatura o humedad |
| `ACTIVAR_CALEFACCION` | Aumentar temperatura |
| `ACTIVAR_RIEGO` | Aumentar humedad del suelo |
| `ACTIVAR_NEBULIZACION` | Aumentar humedad ambiental |
| `ACTIVAR_DESHUMIDIFICADOR` | Reducir humedad ambiental |
| `ACTIVAR_LED_SUPLEMENTAL` | Aumentar iluminación |
| `REDUCIR_LUZ` | Disminuir iluminación (sombreo) |
| `ACTIVAR_RIEGO_Y_VENTILACION` | Combinación riego + ventilación |
| `ACTIVAR_CALEFACCION_Y_NEBULIZACION` | Combinación calor + humedad |

## Niveles de Urgencia

| Nivel | Descripción |
|---|---|
| `BAJO` | Monitorear, acción en 24-48h |
| `MEDIO` | Acción en 6-12h |
| `ALTO` | Acción en 1-2h |
| `CRITICO` | Acción inmediata (<30 min) |

## Cómo Probar

### Opción 1: Interfaz web (recomendado)

1. Inicia el servidor: `python app.py`
2. Abre en el navegador: [http://localhost:5000/predictivo/](http://localhost:5000/predictivo/)
3. Completa los campos del formulario y haz clic en "Predecir condiciones"
4. Activa el checkbox **"Mostrar todas las variables fuera de rango"** para ver todas las alertas detectadas en una tabla

### Opción 2: API con curl

```bash
# Condiciones normales
curl -X POST http://localhost:5000/predictivo/predict \
  -H "Content-Type: application/json" \
  -d '{"cultivo":"lechuga","etapa":"Germinacion","temperature":20,"humidity":85,"soil_moisture":70,"sunlight":75}'

# Alerta por temperatura baja
curl -X POST http://localhost:5000/predictivo/predict \
  -H "Content-Type: application/json" \
  -d '{"cultivo":"tomate","etapa":"Floracion","temperature":12.5,"humidity":68,"soil_moisture":72,"sunlight":450}'

# Alerta por temperatura alta
curl -X POST http://localhost:5000/predictivo/predict \
  -H "Content-Type: application/json" \
  -d '{"cultivo":"lechuga","etapa":"CrecimientoVegetativo","temperature":30,"humidity":60,"soil_moisture":50,"sunlight":300}'

# Alerta con todas las variables fuera de rango
curl -X POST http://localhost:5000/predictivo/predict \
  -H "Content-Type: application/json" \
  -d '{"cultivo":"tomate","etapa":"Floracion","temperature":12.5,"humidity":30,"soil_moisture":20,"sunlight":50,"todas_las_variables":true}'
```

## Lógica de Inferencia

1. **Reglas determinísticas** (prioritarias): compara cada variable contra umbrales óptimos por cultivo+etapa. Si alguna variable está fuera de rango, genera alerta inmediata.
2. **Árbol de decisión** (respaldo): si las reglas no detectan alerta, el modelo ML evalúa combinaciones complejas de variables.

## Entrenamiento

```bash
python predictivo_ai/train.py
```

Genera dataset sintético (50,000 registros), entrena el árbol de decisión y guarda los `.pkl` en `models/`.
