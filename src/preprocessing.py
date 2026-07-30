import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'casas_plusvalia.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'casas_limpias.csv')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocesador.pkl')

NUMERICAL_FEATURES = ['area_m2', 'habitaciones', 'banos', 'parqueaderos', 'antiguedad_anos']
CATEGORICAL_FEATURES = ['ubicacion', 'sector']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = 'precio_usd'

def cargar_datos_raw(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Carga los datos crudos desde el archivo CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo raw no existe en {filepath}. Ejecuta el scraper primero.")
    return pd.read_csv(filepath)

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza de datos:
    - Eliminación de duplicados.
    - Imputación/filtrado de valores nulos o inconsistentes.
    - Eliminación de outliers extremos mediante el método IQR.
    """
    df_clean = df.copy()

    # 1. Eliminar duplicados
    df_clean = df_clean.drop_duplicates()

    # 2. Filtrar valores nulos en columnas críticas
    df_clean = df_clean.dropna(subset=ALL_FEATURES + [TARGET_COL])

    # 3. Filtrar valores fuera de rangos lógicos
    df_clean = df_clean[
        (df_clean['area_m2'] >= 20) & (df_clean['area_m2'] <= 1000) &
        (df_clean['habitaciones'] >= 1) & (df_clean['habitaciones'] <= 10) &
        (df_clean['banos'] >= 1) & (df_clean['banos'] <= 10) &
        (df_clean['precio_usd'] >= 15000)
    ]

    # 4. Eliminación de Outliers en precio_usd con métrica IQR
    Q1 = df_clean[TARGET_COL].quantile(0.25)
    Q3 = df_clean[TARGET_COL].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    df_clean = df_clean[(df_clean[TARGET_COL] >= limite_inferior) & (df_clean[TARGET_COL] <= limite_superior)]

    return df_clean

def construir_pipeline_preprocesamiento():
    """
    Construye y retorna el ColumnTransformer para preprocesar
    variables numéricas (StandardScaler) y categóricas (OneHotEncoder).
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor

def preparar_y_guardar_datos_limpios():
    """Ejecuta la limpieza, guarda casas_limpias.csv y retorna df_clean."""
    df_raw = cargar_datos_raw()
    df_clean = limpiar_datos(df_raw)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[OK] Datos limpiados y guardados en: {PROCESSED_DATA_PATH}")
    print(f"[INFO] Registros procesados: {len(df_clean)} de {len(df_raw)} originales.")
    return df_clean

def preparar_datos_inferencia(dict_datos: dict, preprocessor) -> np.ndarray:
    """
    Convierte un diccionario con características de una vivienda en un array
    preprocesado listo para ser ingresado al modelo en el endpoint FastAPI.
    """
    df_single = pd.DataFrame([dict_datos])
    # Asegurar que existan todas las columnas requeridas
    for col in ALL_FEATURES:
        if col not in df_single.columns:
            raise ValueError(f"Falta el campo obligatorio: {col}")

    X_transformed = preprocessor.transform(df_single[ALL_FEATURES])
    return X_transformed

if __name__ == "__main__":
    preparar_y_guardar_datos_limpios()
