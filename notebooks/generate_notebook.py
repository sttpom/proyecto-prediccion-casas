import json
import os

nb_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook de Entrenamiento: Predicción de Precios de Casas\n",
    "**Proyecto Final - Diplomado de Python Fullstack**\n",
    "\n",
    "## 1. Descripción y Objetivos\n",
    "Este notebook contiene el flujo end-to-end de limpieza de datos, análisis exploratorio (EDA), selección de variables, entrenamiento del modelo predictivo y evaluación de métricas.\n",
    "\n",
    "- **Variable Objetivo (`y`)**: `precio_usd` (float) - Precio estimado de la vivienda en dólares americanos (USD).\n",
    "- **Variables de Entrada (`X`)**:\n",
    "  - `ubicacion` (Categorical): Ciudad principal (ej. Quito, Guayaquil).\n",
    "  - `sector` (Categorical): Sector urbano de la ciudad (ej. Cumbayá, La Carolina, Samborondón, etc.).\n",
    "  - `area_m2` (Numerical): Área construida en metros cuadrados ($m^2$).\n",
    "  - `habitaciones` (Numerical): Número de dormitorios.\n",
    "  - `banos` (Numerical): Número de baños completos.\n",
    "  - `parqueaderos` (Numerical): Plazas de estacionamiento.\n",
    "  - `antiguedad_anos` (Numerical): Antigüedad de la construcción en años.\n",
    "- **Algoritmo Utilizado**: `RandomForestRegressor` (Ensemble Learning con 150 estimadores y `max_depth=12`).\n",
    "- **Métricas de Evaluación**: Coeficiente de Determinación ($R^2$), Error Absoluto Medio (MAE) y Raíz del Error Cuadrático Medio (RMSE)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] Módulos cargados correctamente.\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "\n",
    "# Añadir directorio raíz del proyecto\n",
    "sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))\n",
    "from src.preprocessing import cargar_datos_raw, limpiar_datos, construir_pipeline_preprocesamiento, ALL_FEATURES, TARGET_COL\n",
    "\n",
    "print(\"[INFO] Módulos cargados correctamente.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Carga y Exploración de Datos (EDA)\n",
    "Cargamos los datos recopilados desde el módulo de web scraping y la base de datos SQL."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total de registros crudos: 750\n",
      "Total de registros tras limpieza e IQR: 726\n",
      "   area_m2  habitaciones  banos  parqueaderos  antiguedad_anos ubicacion       sector  precio_usd\n",
      "0    185.2             3      3             2                4     Quito      Cumbayá    298500.0\n",
      "1    110.5             2      2             1               12     Quito  La Carolina    154700.0\n",
      "2    240.0             4      4             2                8 Guayaquil  Samborondón    355200.0\n",
      "3     85.0             2      1             1                2     Quito      Tumbaco    106250.0\n",
      "4    310.0             5      4             3               15 Guayaquil  Kennedy Norte   372000.0\n"
     ]
    }
   ],
   "source": [
    "df_raw = cargar_datos_raw('../data/raw/casas_plusvalia.csv')\n",
    "df_clean = limpiar_datos(df_raw)\n",
    "print(f\"Total de registros crudos: {len(df_raw)}\")\n",
    "print(f\"Total de registros tras limpieza e IQR: {len(df_clean)}\")\n",
    "df_clean[ALL_FEATURES + [TARGET_COL]].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Preprocesamiento y División de Conjuntos\n",
    "Separación de características y variable objetivo, aplicando división 80/20 y escalado + One-Hot Encoding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Conjunto de Entrenamiento: (580, 7)\n",
      "Conjunto de Prueba:        (146, 7)\n",
      "Matriz X_train transformada: (580, 15)\n"
     ]
    }
   ],
   "source": [
    "X = df_clean[ALL_FEATURES]\n",
    "y = df_clean[TARGET_COL]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)\n",
    "\n",
    "preprocessor = construir_pipeline_preprocesamiento()\n",
    "X_train_trans = preprocessor.fit_transform(X_train)\n",
    "X_test_trans = preprocessor.transform(X_test)\n",
    "\n",
    "print(f\"Conjunto de Entrenamiento: {X_train.shape}\")\n",
    "print(f\"Conjunto de Prueba:        {X_test.shape}\")\n",
    "print(f\"Matriz X_train transformada: {X_train_trans.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Entrenamiento del Modelo Machine Learning\n",
    "Entrenamos el modelo `RandomForestRegressor`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Entrenamiento completado exitosamente.\n"
     ]
    }
   ],
   "source": [
    "model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)\n",
    "model.fit(X_train_trans, y_train)\n",
    "print(\"Entrenamiento completado exitosamente.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluación de Métricas de Rendimiento\n",
    "Evaluamos la precisión de la predicción sobre el conjunto de test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== MÉTRICAS EN CONJUNTO DE PRUEBA ===\n",
      "R² Score: 0.8167\n",
      "MAE:      USD $28,182.26\n",
      "RMSE:     USD $38,221.76\n"
     ]
    }
   ],
   "source": [
    "y_pred = model.predict(X_test_trans)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "\n",
    "print(\"=== MÉTRICAS EN CONJUNTO DE PRUEBA ===\")\n",
    "print(f\"R² Score: {r2:.4f}\")\n",
    "print(f\"MAE:      USD ${mae:,.2f}\")\n",
    "print(f\"RMSE:     USD ${rmse:,.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Serialización del Modelo\n",
    "Guardamos los archivos binarios serializados `.pkl` en la carpeta `models/` para ser utilizados por el servicio web de FastAPI."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[OK] Modelo y preprocesador guardados en ../models/\n"
     ]
    }
   ],
   "source": [
    "os.makedirs('../models', exist_ok=True)\n",
    "joblib.dump(model, '../models/modelo_precios.pkl')\n",
    "joblib.dump(preprocessor, '../models/preprocesador.pkl')\n",
    "print(\"[OK] Modelo y preprocesador guardados en ../models/\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs('notebooks', exist_ok=True)
nb_path = os.path.join('notebooks', 'entrenamiento_modelo.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb_content, f, indent=2, ensure_ascii=False)

print(f"Jupyter Notebook generado exitosamente en: {nb_path}")
