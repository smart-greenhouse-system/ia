import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from riego_ai.model import VaciadoModel
from riego_ai.utils.constants import CAUDAL_BOMBA_DEFECTO

BASE_DIR = Path(__file__).parent
N_SAMPLES = 200
CAUDAL = CAUDAL_BOMBA_DEFECTO
THRESHOLD = 20


def generar_dataset_sintetico(n_samples=N_SAMPLES, caudal=CAUDAL, threshold=THRESHOLD):
    np.random.seed(42)
    niveles_iniciales = np.random.uniform(30, 200, n_samples)
    ruido_pct = np.random.normal(0, 0.01, n_samples)

    datos = []
    for nivel in niveles_iniciales:
        volumen = nivel - threshold
        tiempo_real = (volumen / caudal) * (1 + ruido_pct[len(datos)])
        tiempo_real = max(0, tiempo_real)
        datos.append({
            'nivel_inicial': round(nivel, 1),
            'nivel_final': threshold,
            'volumen_utilizado': round(volumen, 1),
            'tiempo_real_segundos': round(tiempo_real, 2),
            'caudal_registrado': caudal
        })

    return pd.DataFrame(datos)


def main():
    print("Generando dataset sintetico...")
    df = generar_dataset_sintetico()

    df['ratio'] = df['volumen_utilizado'] / CAUDAL
    X = df[['ratio']]
    y = df['tiempo_real_segundos']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = VaciadoModel()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nEcuacion: tiempo = {modelo.coef_[0]:.4f} x ratio")
    print(f"Intercepto (sesgo): {modelo.intercept_:.4f}")
    print(f"Coeficiente ideal: 1.0000")
    print(f"Error absoluto medio: {mae:.2f} segundos")
    print(f"R²: {r2:.6f}")

    if abs(modelo.coef_[0] - 1.0) < 0.05:
        print("Modelo valido - Coeficiente cercano al esperado")
    else:
        print(f"Advertencia - Desvio del {abs(1.0 - modelo.coef_[0]) * 100:.2f}%")

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    modelo.save(str(models_dir / "modelo_vaciado.pkl"))
    print(f"Modelo guardado en: {models_dir / 'modelo_vaciado.pkl'}")

    pred = modelo.predict([[65.0 / CAUDAL]])[0]
    print(f"\nPrueba: nivel=85, threshold=20, caudal={CAUDAL}")
    print(f"  Ratio: {(85-20)/CAUDAL:.2f}")
    print(f"  Tiempo predicho: {pred:.1f} segundos ({int(pred//60)} min {int(pred%60)} seg)")


if __name__ == "__main__":
    main()
