from .constants import CULTIVOS_VALIDOS, ETAPAS_POR_CULTIVO

RANGOS = {
    'temperature': (0.0, 45.0),
    'humidity': (0.0, 100.0),
    'soil_moisture': (0.0, 100.0),
    'sunlight': (0.0, 1200.0)
}


def validar_cultivo(cultivo):
    if cultivo not in CULTIVOS_VALIDOS:
        raise ValueError(
            f"Cultivo '{cultivo}' no valido. Cultivos validos: {', '.join(CULTIVOS_VALIDOS)}"
        )


def validar_etapa(cultivo, etapa):
    etapas_validas = ETAPAS_POR_CULTIVO.get(cultivo, [])
    if etapa not in etapas_validas:
        raise ValueError(
            f"Etapa '{etapa}' no valida para cultivo '{cultivo}'. "
            f"Etapas validas: {', '.join(etapas_validas)}"
        )


def validar_rangos(temperature, humidity, soil_moisture, sunlight):
    valores = {
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': soil_moisture,
        'sunlight': sunlight
    }
    for nombre, (min_val, max_val) in RANGOS.items():
        valor = valores[nombre]
        if valor < min_val or valor > max_val:
            raise ValueError(
                f"{nombre} = {valor} fuera de rango permitido ({min_val}-{max_val})"
            )


import pandas as pd


def build_feature_vector(cultivo_cod, etapa_cod, temperature, humidity, soil_moisture, sunlight):
    return pd.DataFrame([[
        float(temperature),
        float(humidity),
        float(soil_moisture),
        float(sunlight),
        float(cultivo_cod),
        float(etapa_cod)
    ]], columns=['temperatura', 'humedad_relativa', 'humedad_suelo',
                 'iluminacion', 'cultivo_cod', 'etapa_cod'])
