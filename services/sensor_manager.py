"""
Gestor de datos de sensores.
Almacena y gestiona información de sensores ambientales.
"""

from datetime import datetime
from typing import Dict, Any


class SensorManager:
    """
    Gestor de datos de sensores.
    Almacena la información de sensores y la muestra en consola.
    """
    
    def __init__(self):
        """Inicializa el almacenamiento de sensores."""
        # Variable para almacenar los datos de sensores actuales
        self.sensor_data = {}
    
    def store_sensor_data(self, temperatura, humedad_relativa, humedad_suelo, iluminacion, timestamp):
        """
        Almacena los datos de sensores en memoria.
        
        Args:
            temperatura (float): Temperatura en grados Celsius
            humedad_relativa (float): Humedad relativa en porcentaje
            humedad_suelo (float): Humedad del suelo en porcentaje
            iluminacion (float): Nivel de iluminación (unidad de medida del sensor)
            timestamp (str): Marca de tiempo ISO 8601
        """
        
        # Guardar datos en variable
        self.sensor_data = {
            "temperatura": temperatura,
            "humedad_relativa": humedad_relativa,
            "humedad_suelo": humedad_suelo,
            "iluminacion": iluminacion,
            "timestamp": timestamp,
            "almacenado_en": datetime.now().isoformat()
        }
        
        # Mostrar en consola
        self._print_sensor_data()
    
    def _print_sensor_data(self):
        """Imprime los datos de sensores en la consola."""
        print("\n" + "="*60)
        print("📊 DATOS DE SENSORES REGISTRADOS")
        print("="*60)
        print(f"  Temperatura:        {self.sensor_data['temperatura']}°C")
        print(f"  Humedad Relativa:   {self.sensor_data['humedad_relativa']}%")
        print(f"  Humedad Suelo:      {self.sensor_data['humedad_suelo']}%")
        print(f"  Iluminación:        {self.sensor_data['iluminacion']} lux")
        print(f"  Timestamp:          {self.sensor_data['timestamp']}")
        print("="*60 + "\n")
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Obtiene los datos de sensores almacenados.
        
        Returns:
            dict: Datos de sensores actuales
        """
        return self.sensor_data.copy()
    
    def get_sensor_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de los datos de sensores para incluir en respuestas.
        
        Returns:
            dict: Resumen de sensores
        """
        return {
            "temperatura": self.sensor_data.get("temperatura"),
            "humedad_relativa": self.sensor_data.get("humedad_relativa"),
            "humedad_suelo": self.sensor_data.get("humedad_suelo"),
            "iluminacion": self.sensor_data.get("iluminacion"),
            "timestamp": self.sensor_data.get("timestamp")
        }
