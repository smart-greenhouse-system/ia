import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))

from predictivo_ai.model import PredictivoModel
from predictivo_ai.utils.preprocessing import (
    validar_cultivo,
    validar_etapa,
    validar_rangos,
    build_feature_vector
)
from predictivo_ai.utils.constants import (
    UMBRALES,
    CONSTANTES_ALERTA,
    NIVELES_URGENCIA,
)

import joblib


class PredictivoInference:

    def __init__(self):
        models_dir = BASE_DIR / "models"

        self.model = PredictivoModel()
        self.model.load(str(models_dir / "greenhouse_model.pkl"))

        self.le_cultivo = joblib.load(str(models_dir / "label_cultivo.pkl"))
        self.le_etapa = joblib.load(str(models_dir / "label_etapa.pkl"))
        self.le_alerta = joblib.load(str(models_dir / "label_alerta.pkl"))

        self.alerta_classes = self.le_alerta.classes_

    def _aplicar_reglas(self, temperature, humidity, soil_moisture, sunlight, umbral):
        t_min, t_max = umbral['temp']
        hr_min, hr_max = umbral['hr']
        s_min, s_max = umbral['suelo']
        l_min, l_max = umbral['luz']

        if temperature > t_max + 1:
            return 'ACTIVAR_VENTILACION', 'temperatura'
        if temperature < t_min - 1:
            return 'ACTIVAR_CALEFACCION', 'temperatura'
        if soil_moisture < s_min - 3:
            return 'ACTIVAR_RIEGO', 'humedad_suelo'
        if humidity < hr_min - 3:
            return 'ACTIVAR_NEBULIZACION', 'humedad'
        if humidity > hr_max + 3:
            return 'ACTIVAR_DESHUMIDIFICADOR', 'humedad'
        if sunlight < l_min - 30:
            return 'ACTIVAR_LED_SUPLEMENTAL', 'iluminacion'
        if sunlight > l_max + 50:
            return 'REDUCIR_LUZ', 'iluminacion'
        if temperature > t_max and soil_moisture < s_min:
            return 'ACTIVAR_RIEGO_Y_VENTILACION', 'multiple'
        if temperature < t_min and humidity < hr_min:
            return 'ACTIVAR_CALEFACCION_Y_NEBULIZACION', 'multiple'

        return 'SIN_ALERTA', None

    def _obtener_info_variable(self, variable, temperature, humidity, soil_moisture, sunlight, umbral):
        if variable is None or variable == 'multiple':
            return None, None, None, None

        mapa = {
            'temperatura': ('temp', temperature),
            'humedad': ('hr', humidity),
            'humedad_suelo': ('suelo', soil_moisture),
            'iluminacion': ('luz', sunlight)
        }
        clave, valor = mapa[variable]
        opt_min, opt_max = umbral[clave]
        return round(valor, 1), opt_min, opt_max

    def _calcular_nivel_urgencia(self, valor, opt_min, opt_max):
        if opt_min is None or opt_max is None:
            return 'MEDIO'
        medio = (opt_min + opt_max) / 2
        rango = opt_max - opt_min
        desviacion = abs(valor - medio)
        if desviacion > rango * 1.5:
            return 'CRITICO'
        elif desviacion > rango:
            return 'ALTO'
        return 'MEDIO'

    def _generar_mensaje(self, cultivo, etapa, accion, variable, valor_actual, opt_min, opt_max):
        nombre_cultivo = cultivo.upper()
        nombre_variable_map = {
            'temperatura': 'Temperatura',
            'humedad': 'Humedad relativa',
            'humedad_suelo': 'Humedad del suelo',
            'iluminacion': 'Iluminacion',
            'multiple': 'Multiples variables'
        }
        var_nombre = nombre_variable_map.get(variable, variable)
        if variable and variable != 'multiple' and valor_actual is not None and opt_min is not None:
            direccion = "por debajo del" if valor_actual < opt_min else "por encima del"
            parte_variable = f"{var_nombre} {valor_actual} {direccion} optimo ({opt_min}-{opt_max}). "
        else:
            parte_variable = ""

        return (f"{nombre_cultivo} en {etapa}: {parte_variable}"
                f"Accion: {CONSTANTES_ALERTA.get(accion, accion)}.")

    def predict(self, cultivo, etapa, temperature, humidity, soil_moisture, sunlight):
        validar_cultivo(cultivo)
        validar_etapa(cultivo, etapa)
        validar_rangos(temperature, humidity, soil_moisture, sunlight)

        umbral = UMBRALES[cultivo][etapa]

        accion_regla, variable = self._aplicar_reglas(
            temperature, humidity, soil_moisture, sunlight, umbral
        )

        if accion_regla == 'SIN_ALERTA':
            cultivo_cod = self.le_cultivo.transform([cultivo])[0]
            etapa_cod = self.le_etapa.transform([etapa])[0]
            X = build_feature_vector(cultivo_cod, etapa_cod, temperature, humidity, soil_moisture, sunlight)
            alerta_cod = self.model.model.predict(X)[0]
            accion_ml = self.alerta_classes[alerta_cod]

            if accion_ml == 'SIN_ALERTA':
                return {
                    "success": True,
                    "status": "OK",
                    "cultivo": cultivo,
                    "etapa": etapa
                }

            valor_actual, opt_min, opt_max = self._obtener_info_variable(
                variable, temperature, humidity, soil_moisture, sunlight, umbral
            )
            nivel_urgencia = self._calcular_nivel_urgencia(valor_actual, opt_min, opt_max) if variable else 'MEDIO'
            mensaje = self._generar_mensaje(cultivo, etapa, accion_ml, variable, valor_actual, opt_min, opt_max)

            result = {
                "success": True,
                "status": "alerta",
                "accion": accion_ml,
                "mensaje": mensaje,
                "nivel_urgencia": nivel_urgencia,
                "cultivo": cultivo,
                "etapa": etapa
            }
            if variable:
                result["variable"] = variable
                result["valor_actual"] = valor_actual
                result["valor_optimo_min"] = opt_min
                result["valor_optimo_max"] = opt_max
            return result

        valor_actual, opt_min, opt_max = self._obtener_info_variable(
            variable, temperature, humidity, soil_moisture, sunlight, umbral
        )
        nivel_urgencia = self._calcular_nivel_urgencia(valor_actual, opt_min, opt_max)
        mensaje = self._generar_mensaje(cultivo, etapa, accion_regla, variable, valor_actual, opt_min, opt_max)

        result = {
            "success": True,
            "status": "alerta",
            "accion": accion_regla,
            "mensaje": mensaje,
            "nivel_urgencia": nivel_urgencia,
            "cultivo": cultivo,
            "etapa": etapa
        }
        if variable:
            result["variable"] = variable
            result["valor_actual"] = valor_actual
            result["valor_optimo_min"] = opt_min
            result["valor_optimo_max"] = opt_max

        return result
