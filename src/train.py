import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import (
    limpiar_datos, cargar_datos_raw, construir_pipeline_preprocesamiento,
    ALL_FEATURES, TARGET_COL, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_precios.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocesador.pkl')

def entrenar_modelo():
    print("[INFO] Cargando y preprocesando datos para entrenamiento...")
    df_raw = cargar_datos_raw()
    df_clean = limpiar_datos(df_raw)

    X = df_clean[ALL_FEATURES]
    y = df_clean[TARGET_COL]

    # División train/test (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Preprocesamiento de variables
    preprocessor = construir_pipeline_preprocesamiento()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Entrenamiento del modelo Random Forest Regressor
    print("[INFO] Entrenando modelo Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train_trans, y_train)

    # Evaluacion en test set
    y_pred = model.predict(X_test_trans)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("       RESULTADOS DE EVALUACIÓN DEL MODELO       ")
    print("="*50)
    print(f"  * R² Score (Coef. de Determinación): {r2:.4f}")
    print(f"  * MAE (Error Absoluto Medio):        USD ${mae:,.2f}")
    print(f"  * RMSE (Raíz Error Cuadrático M.):   USD ${rmse:,.2f}")
    print("="*50 + "\n")

    # Guardar modelo y preprocesador
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print(f"[OK] Modelo guardado en: {MODEL_PATH}")
    print(f"[OK] Preprocesador guardado en: {PREPROCESSOR_PATH}")

    return model, preprocessor, r2, mae, rmse

if __name__ == "__main__":
    entrenar_modelo()
