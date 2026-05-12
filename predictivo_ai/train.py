import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from predictivo_ai.model import PredictivoModel
from predictivo_ai.utils.constants import UMBRALES, ETAPAS_LECHUGA, ETAPAS_TOMATE

BASE_DIR = Path(__file__).parent
N_SAMPLES = 50000


def determinar_alerta(temp, hr, suelo, luz, umbral):
    t_min, t_max = umbral['temp']
    hr_min, hr_max = umbral['hr']
    s_min, s_max = umbral['suelo']
    l_min, l_max = umbral['luz']

    if temp > t_max + 3:
        return 'ACTIVAR_VENTILACION'
    elif temp < t_min - 3:
        return 'ACTIVAR_CALEFACCION'
    elif suelo < s_min - 10:
        return 'ACTIVAR_RIEGO'
    elif hr < hr_min - 10:
        return 'ACTIVAR_NEBULIZACION'
    elif hr > hr_max + 10:
        return 'ACTIVAR_DESHUMIDIFICADOR'
    elif luz < l_min - 50:
        return 'ACTIVAR_LED_SUPLEMENTAL'
    elif luz > l_max + 100:
        return 'REDUCIR_LUZ'
    elif temp > t_max + 1 and suelo < s_min - 5:
        return 'ACTIVAR_RIEGO_Y_VENTILACION'
    elif temp < t_min - 1 and hr < hr_min - 5:
        return 'ACTIVAR_CALEFACCION_Y_NEBULIZACION'
    else:
        return 'SIN_ALERTA'


def generar_dataset(n_samples=N_SAMPLES):
    np.random.seed(42)
    datos = []

    for _ in range(n_samples):
        cultivo = np.random.choice(['lechuga', 'tomate'], p=[0.5, 0.5])

        if cultivo == 'lechuga':
            etapa = np.random.choice(ETAPAS_LECHUGA, p=[0.1, 0.15, 0.35, 0.25, 0.15])
        else:
            etapa = np.random.choice(ETAPAS_TOMATE, p=[0.1, 0.1, 0.3, 0.2, 0.2, 0.1])

        umbral = UMBRALES[cultivo][etapa]
        t_min, t_max = umbral['temp']
        hr_min, hr_max = umbral['hr']
        s_min, s_max = umbral['suelo']
        l_min, l_max = umbral['luz']

        if np.random.random() < 0.7:
            temp = np.random.uniform(t_min, t_max)
            hr = np.random.uniform(hr_min, hr_max)
            suelo = np.random.uniform(s_min, s_max)
            luz = np.random.uniform(l_min, l_max)
        else:
            if np.random.random() < 0.5:
                temp = np.random.uniform(t_max, t_max + 8)
                hr = np.random.uniform(hr_max, min(100, hr_max + 25))
                suelo = np.random.uniform(s_max, min(100, s_max + 20))
                luz = np.random.uniform(l_max, min(1200, l_max + 300))
            else:
                temp = np.random.uniform(max(0, t_min - 8), t_min)
                hr = np.random.uniform(max(0, hr_min - 25), hr_min)
                suelo = np.random.uniform(max(0, s_min - 20), s_min)
                luz = np.random.uniform(max(0, l_min - 300), l_min)

        temp = np.clip(temp, 0, 45)
        hr = np.clip(hr, 10, 100)
        suelo = np.clip(suelo, 0, 100)
        luz = np.clip(luz, 0, 1200)

        alerta = determinar_alerta(temp, hr, suelo, luz, umbral)

        datos.append({
            'cultivo': cultivo,
            'etapa': etapa,
            'temperatura': round(temp, 1),
            'humedad_relativa': round(hr, 1),
            'humedad_suelo': round(suelo, 1),
            'iluminacion': round(luz, 1),
            'alerta': alerta
        })

    return pd.DataFrame(datos)


def main():
    print("Generando dataset...")
    df = generar_dataset(N_SAMPLES)

    dataset_dir = BASE_DIR / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset_greenhouse.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Dataset guardado: {dataset_path}")
    print(f"Total registros: {len(df)}")
    print(f"Distribucion alertas:\n{df['alerta'].value_counts()}")

    le_cultivo = LabelEncoder()
    le_etapa = LabelEncoder()
    le_alerta = LabelEncoder()

    df['cultivo_cod'] = le_cultivo.fit_transform(df['cultivo'])
    df['etapa_cod'] = le_etapa.fit_transform(df['etapa'])
    df['alerta_cod'] = le_alerta.fit_transform(df['alerta'])

    X = df[['temperatura', 'humedad_relativa', 'humedad_suelo',
            'iluminacion', 'cultivo_cod', 'etapa_cod']]
    y = df['alerta_cod']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = PredictivoModel()
    modelo.fit(X_train, y_train)

    precision = modelo.model.score(X_test, y_test)
    print(f"\nPrecision del modelo: {precision:.4f}")
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, modelo.predict(X_test), target_names=le_alerta.classes_))

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    modelo.save(str(models_dir / "greenhouse_model.pkl"))
    joblib.dump(le_cultivo, str(models_dir / "label_cultivo.pkl"))
    joblib.dump(le_etapa, str(models_dir / "label_etapa.pkl"))
    joblib.dump(le_alerta, str(models_dir / "label_alerta.pkl"))

    print("Modelo y codificadores guardados en models/")

    importancia = pd.DataFrame({
        'variable': ['temperatura', 'humedad_relativa', 'humedad_suelo',
                     'iluminacion', 'cultivo', 'etapa'],
        'importancia': modelo.feature_importances()
    }).sort_values('importancia', ascending=False)
    print("\nImportancia de variables:")
    print(importancia)


if __name__ == "__main__":
    import joblib
    main()
