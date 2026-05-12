"""
Servicio de identificación de cultivos.
Módulo encargado de identificar qué cultivo contiene la imagen.

Lógica:
1. Intenta identificar tomate (usa modelo de tomate)
2. Si no es tomate, intenta identificar lechuga (usa modelo de lechuga)
3. Si no es ninguno, retorna error
"""

import sys
from pathlib import Path

# Importar modelos de inferencia
sys.path.insert(0, str(Path(__file__).parent.parent / "tomate_ai"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lechuga_ai"))

from tomate_ai.inference import TomatoInference
from lechuga_ai.inference import LechugaInference


class CropIdentifier:
    """
    Servicio para identificar el cultivo en una imagen.
    Prueba primero tomate, luego lechuga.
    """
    
    def __init__(self):
        """
        Inicializa los modelos de detección.
        Se cargan una sola vez al crear la instancia.
        """
        # Cargar modelo de tomate
        self.tomate_model = TomatoInference()
        
        # Cargar modelo de lechuga
        self.lechuga_model = LechugaInference()
    
    def identify(self, image_base64):
        """
        Identifica el cultivo en la imagen.
        
        Args:
            image_base64 (str): Imagen en formato Base64
            
        Returns:
            dict: Contiene crop_type, etapa, y detalles específicos del cultivo
                  O error si no identifica ningún cultivo
        """
        
        # PASO 1: Intentar detectar tomate
        tomate_result = self._try_tomate(image_base64)
        if tomate_result["success"]:
            return {
                "success": True,
                "cultivo": "tomate",
                "detalles": tomate_result
            }
        
        # PASO 2: Si no es tomate, intentar detectar lechuga
        lechuga_result = self._try_lechuga(image_base64)
        if lechuga_result["success"]:
            return {
                "success": True,
                "cultivo": "lechuga",
                "detalles": lechuga_result
            }
        
        # PASO 3: Si no detecta nada, retornar error
        return {
            "success": False,
            "cultivo": None,
            "error": "No se detectó tomate ni lechuga en la imagen",
            "detalle_tomate": tomate_result.get("message", ""),
            "detalle_lechuga": lechuga_result.get("message", "")
        }
    
    def _try_tomate(self, image_base64):
        """
        Intenta detectar tomate usando el modelo de tomate.
        
        Args:
            image_base64 (str): Imagen en formato Base64
            
        Returns:
            dict: Resultado de la predicción o error
        """
        try:
            result = self.tomate_model.predict_base64(image_base64)
            return result
        except Exception as e:
            return {
                "success": False,
                "message": f"Error al procesar con modelo de tomate: {str(e)}"
            }
    
    def _try_lechuga(self, image_base64):
        """
        Intenta detectar lechuga usando el modelo de lechuga.
        
        Args:
            image_base64 (str): Imagen en formato Base64
            
        Returns:
            dict: Resultado de la predicción o error
        """
        try:
            result = self.lechuga_model.predict_base64(image_base64)
            return result
        except Exception as e:
            return {
                "success": False,
                "message": f"Error al procesar con modelo de lechuga: {str(e)}"
            }
