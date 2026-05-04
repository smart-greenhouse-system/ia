import shutil
import os
from pathlib import Path

# ============================================================
# COPIAR AUTOMÁTICAMENTE EL MEJOR MODELO ENTRENADO
# ============================================================
# Este script busca el archivo best.pt dentro de la carpeta
# runs/detect, sin importar el nombre exacto del experimento.
# Así evitas errores cuando YOLO crea carpetas como:
#
# - cherry_detector
# - cherry_detector2
# - cherry_detector3
#
# ============================================================

# Carpeta donde YOLO guarda los entrenamientos
runs_dir = Path("runs/detect")

# Buscar todos los archivos best.pt
best_models = list(runs_dir.glob("*/weights/best.pt"))

if not best_models:
    print("❌ No se encontró ningún archivo best.pt")
    print("Verifica que el entrenamiento haya finalizado correctamente.")
    exit()

# Seleccionar el más reciente
latest_model = max(best_models, key=lambda p: p.stat().st_mtime)

# Crear carpeta destino
destination_folder = Path("models")
destination_folder.mkdir(exist_ok=True)

# Ruta final
destination = destination_folder / "detector_best.pt"

# Copiar modelo
shutil.copy2(latest_model, destination)

print("✅ Modelo copiado correctamente.")
print(f"📥 Origen : {latest_model}")
print(f"📤 Destino: {destination}")