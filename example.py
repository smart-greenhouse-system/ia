{
  "success": true,
  "message": "Predicción completada",
  "estrategia_usada": "clahe_sharpen",
  "errores": {
    "card_error": null,
    "tomato_error": null
  },
  "carta_detectada": true,
  "aviso_carta": null,
  "total_tomates": 2,
  "tomates": [
    {
      "id": 1,
      "estado": "fruit_maturation",
      "confianza": 0.942,
      "dias_cosecha": 14,
      "dias_texto": "Aproximadamente 14 días para la cosecha",
      "descripcion": "Maduración: el fruto cambia de color y acumula azúcares. Próximo a cosechar.",
      "medidas_reales": {
        "area_cm2": 9.17,
        "diametro_cm": 3.98,
        "ancho_cm": 4.11,
        "alto_cm": 4.02,
        "escala_real": true
      }
    },
    {
      "id": 2,
      "estado": "fully_grown",
      "confianza": 0.987,
      "dias_cosecha": 0,
      "dias_texto": "Listo para cosechar hoy",
      "descripcion": "Listo para cosecha: tomate maduro, color y firmeza óptimos para recolección.",
      "medidas_reales": {
        "area_cm2": 12.44,
        "diametro_cm": 4.56,
        "ancho_cm": 4.71,
        "alto_cm": 4.63,
        "escala_real": true
      }
    }
  ],
  "annotated_image": "BASE64_STRING_AQUI"
}