import os
import sys
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Agregar directorio raíz para cargar preprocesamiento si fuese necesario
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocessing import ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_precios.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocesador.pkl')

# Variables globales para el modelo y preprocesador
modelo_ml = None
preprocesador_ml = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos en memoria al iniciar la aplicación FastAPI."""
    global modelo_ml, preprocesador_ml
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"No se encontraron los archivos del modelo. Asegúrate de ejecutar el entrenamiento primero. RUTA: {MODEL_PATH}"
        )
    modelo_ml = joblib.load(MODEL_PATH)
    preprocesador_ml = joblib.load(PREPROCESSOR_PATH)
    print(f"[OK] Modelo y preprocesador cargados exitosamente en FastAPI.")
    yield

app = FastAPI(
    title="API de Predicción de Precios de Casas",
    description="Servicio RESTful desarrollado en FastAPI para predecir precios de viviendas en tiempo real.",
    version="1.0.0",
    lifespan=lifespan
)

# Configuración CORS para permitir peticiones desde Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo Pydantic para validación del Payload de Entrada
class HouseInput(BaseModel):
    ubicacion: str = Field(..., description="Ciudad del inmueble", example="Quito")
    sector: str = Field(..., description="Sector urbano del inmueble", example="Cumbayá")
    area_m2: float = Field(..., gt=10, lt=1500, description="Área construida en metros cuadrados", example=185.0)
    habitaciones: int = Field(..., ge=1, le=12, description="Número de dormitorios", example=3)
    banos: int = Field(..., ge=1, le=12, description="Número de baños completos", example=3)
    parqueaderos: int = Field(..., ge=0, le=10, description="Número de plazas de parqueadero", example=2)
    antiguedad_anos: int = Field(..., ge=0, le=100, description="Antigüedad de la construcción en años", example=4)

# Modelo Pydantic para la Respuesta de Predicción
class PredictionOutput(BaseModel):
    precio_estimado_usd: float = Field(..., description="Precio estimado en Dólares Americanos (USD)")
    precio_formateado: str = Field(..., description="Precio formateado para visualización")
    moneda: str = Field("USD", description="Moneda de la predicción")
    modelo_utilizado: str = Field("RandomForestRegressor", description="Algoritmo de aprendizaje automático")
    status: str = Field("success", description="Estado de la predicción")

@app.get("/", tags=["General"])
def read_root():
    """Endpoint raíz con estado del servicio."""
    return {
        "mensaje": "Servicio de Predicción de Precios de Casas activo.",
        "documentacion_swagger": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", tags=["General"])
def health_check():
    """Endpoint de verificación de salud del servicio y disponibilidad del modelo."""
    is_ready = (modelo_ml is not None) and (preprocesador_ml is not None)
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "modelo_cargado": is_ready
    }

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK, tags=["Prediccion"])
def predict_house_price(data: HouseInput):
    """
    Recibe las características de una vivienda, aplica el preprocesamiento
    y devuelve la estimación de precio generada por el modelo serializado PKL.
    """
    if modelo_ml is None or preprocesador_ml is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo predictivo no se encuentra cargado en el servidor."
        )

    try:
        # Convertir datos de entrada a DataFrame de Pandas
        input_dict = data.model_dump()
        df_single = pd.DataFrame([input_dict])

        # Aplicar el preprocesador guardado en models/preprocesador.pkl
        X_trans = preprocesador_ml.transform(df_single[ALL_FEATURES])

        # Realizar la predicción en tiempo real
        prediction_val = float(modelo_ml.predict(X_trans)[0])
        prediction_val = max(prediction_val, 15000.0) # Asegurar límite inferior lógico
        precio_redondeado = round(prediction_val, 2)

        return PredictionOutput(
            precio_estimado_usd=precio_redondeado,
            precio_formateado=f"USD ${precio_redondeado:,.2f}",
            moneda="USD",
            modelo_utilizado="RandomForestRegressor",
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error durante el procesamiento de la predicción: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
