ETAPAS_LECHUGA = [
    'Germinacion',
    'Plantula',
    'CrecimientoVegetativo',
    'FormacionCabeza',
    'Cosecha'
]

ETAPAS_TOMATE = [
    'Germinacion',
    'Plantula',
    'CrecimientoVegetativo',
    'Floracion',
    'CuajadoFrutos',
    'Maduracion'
]

ETAPAS_POR_CULTIVO = {
    'lechuga': ETAPAS_LECHUGA,
    'tomate': ETAPAS_TOMATE
}

UMBRALES = {
    'lechuga': {
        'Germinacion': {'temp': (18, 22), 'hr': (80, 95), 'suelo': (60, 80), 'luz': (50, 100)},
        'Plantula': {'temp': (15, 20), 'hr': (70, 85), 'suelo': (50, 70), 'luz': (100, 200)},
        'CrecimientoVegetativo': {'temp': (14, 18), 'hr': (60, 75), 'suelo': (40, 60), 'luz': (200, 400)},
        'FormacionCabeza': {'temp': (12, 16), 'hr': (55, 70), 'suelo': (35, 55), 'luz': (300, 500)},
        'Cosecha': {'temp': (10, 15), 'hr': (50, 65), 'suelo': (30, 45), 'luz': (200, 400)}
    },
    'tomate': {
        'Germinacion': {'temp': (22, 28), 'hr': (80, 90), 'suelo': (70, 85), 'luz': (50, 100)},
        'Plantula': {'temp': (20, 25), 'hr': (70, 80), 'suelo': (60, 75), 'luz': (150, 250)},
        'CrecimientoVegetativo': {'temp': (18, 24), 'hr': (65, 75), 'suelo': (55, 70), 'luz': (300, 500)},
        'Floracion': {'temp': (20, 25), 'hr': (60, 70), 'suelo': (60, 75), 'luz': (400, 600)},
        'CuajadoFrutos': {'temp': (18, 24), 'hr': (55, 65), 'suelo': (65, 80), 'luz': (500, 700)},
        'Maduracion': {'temp': (16, 22), 'hr': (50, 60), 'suelo': (50, 65), 'luz': (400, 600)}
    }
}

CULTIVOS_VALIDOS = ['lechuga', 'tomate']

VARIABLES = ['temperatura', 'humedad', 'suelo', 'luz']

CONSTANTES_ALERTA = {
    'SIN_ALERTA': 'Condiciones normales',
    'ACTIVAR_VENTILACION': 'Reducir temperatura o humedad',
    'ACTIVAR_CALEFACCION': 'Aumentar temperatura',
    'ACTIVAR_RIEGO': 'Aumentar humedad del suelo',
    'ACTIVAR_NEBULIZACION': 'Aumentar humedad ambiental',
    'ACTIVAR_DESHUMIDIFICADOR': 'Reducir humedad ambiental',
    'ACTIVAR_LED_SUPLEMENTAL': 'Aumentar iluminacion',
    'REDUCIR_LUZ': 'Disminuir iluminacion (sombreo)',
    'ACTIVAR_RIEGO_Y_VENTILACION': 'Combinacion riego + ventilacion',
    'ACTIVAR_CALEFACCION_Y_NEBULIZACION': 'Combinacion calor + humedad'
}

NIVELES_URGENCIA = {
    'BAJO': 'Monitorear, accion en 24-48h',
    'MEDIO': 'Accion en 6-12h',
    'ALTO': 'Accion en 1-2h',
    'CRITICO': 'Accion inmediata (<30 min)'
}

NOMBRES_VARIABLES = {
    'temp': 'temperatura',
    'hr': 'humedad',
    'suelo': 'humedad_suelo',
    'luz': 'iluminacion'
}
