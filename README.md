# Predicción de Precios de Casas

Aplicación desarrollada como proyecto final para el **Diplomado de Python Fullstack**.

**Repositorio GitHub:**
`https://github.com/USUARIO/proyecto-prediccion-casas`

**Flujo del Sistema:**
`Web Scraping` ➔ `Almacenamiento CSV / SQL` ➔ `Entrenamiento ML` ➔ `Modelo PKL` ➔ `FastAPI REST` ➔ `Streamlit Web App`

---

## 1. Nombre y Descripción del Proyecto

El proyecto **Predicción de Precios de Casas** integra en una única solución reproducible y desacoplada la recolección, procesamiento, entrenamiento e inferencia en tiempo real para estimar el valor comercial de inmuebles residenciales en dólares americanos (USD).

**Resultado esperado:**
Una aplicación web interactiva en **Streamlit** donde el usuario ingresa las características de una vivienda (ubicación, sector, área en $m^2$, número de dormitorios, baños, parqueaderos y antigüedad). Streamlit transmite la información al servicio backend **FastAPI**, el cual preprocesa los datos, consulta el modelo de Machine Learning serializado en formato `.pkl` y retorna la estimación del precio en tiempo real.

---

## 2. Enlace al Repositorio

- **Repositorio Oficial de GitHub:** [https://github.com/USUARIO/proyecto-prediccion-casas](https://github.com/USUARIO/proyecto-prediccion-casas) *(Sustituir USUARIO por el usuario de GitHub correspondiente)*.

---

## 3. Arquitectura y Flujo de Datos

```
 +-------------------------+
 |   Scraping Plusvalía    | (Web scraping / extracción estructurada)
 +------------+------------+
              |
              v
 +-------------------------+
 | Almacenamiento de Datos | (data/raw/casas_plusvalia.csv & data/casas_db.sqlite)
 +------------+------------+
              |
              v
 +-------------------------+
 |  Preprocesamiento y EDA | (src/preprocessing.py & notebooks/entrenamiento_modelo.ipynb)
 +------------+------------+
              |
              v
 +-------------------------+
 |  Entrenamiento y PKL    | (models/modelo_precios.pkl & models/preprocesador.pkl)
 +------------+------------+
              |
              v
 +-------------------------+
 |    Backend FastAPI      | (api/main.py -> POST /predict)
 +------------+------------+
              |
              v
 +-------------------------+
 |   Frontend Streamlit    | (app/streamlit_app.py -> Interfaz visual interactiva)
 +-------------------------+
```

---

## 4. Requisitos del Sistema

- **Python**: 3.9, 3.10, 3.11 o 3.12 (probado en Python 3.10+)
- **Librerías principales**:
  - `fastapi` & `uvicorn`: Creación y despliegue del servicio API REST.
  - `streamlit`: Interfaz web interactiva.
  - `scikit-learn`: Algoritmo Random Forest Regressor y pipelines de preprocesamiento.
  - `pandas` & `numpy`: Manipulación y estructuración de datos.
  - `joblib`: Serialización y deserialización de modelos `.pkl`.
  - `requests` & `beautifulsoup4`: Extracción e integración HTTP.

---

## 5. Instalación

Siga los siguientes pasos desde la terminal ubicada en la raíz del proyecto:

### 1. Clonar el repositorio
```bash
git clone https://github.com/USUARIO/proyecto-prediccion-casas.git
cd proyecto-prediccion-casas
```

### 2. Crear y activar el entorno virtual
```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Windows (PowerShell / CMD):
.venv\Scripts\activate

# Activar en macOS / Linux:
source .venv/bin/activate
```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

---

## 6. Comandos de Ejecución

### Paso 1: Ejecutar el Scraping y Carga en Base de Datos (CSV & SQL)
```bash
python scraping/scraper_plusvalia.py
```
*Genera `data/raw/casas_plusvalia.csv` y la base de datos `data/casas_db.sqlite`.*

### Paso 2: Ejecutar Limpieza y Entrenamiento del Modelo
```bash
python src/train.py
```
*Genera `data/processed/casas_limpias.csv` y guarda los archivos serializados en `models/modelo_precios.pkl` y `models/preprocesador.pkl`.*

### Paso 3: Iniciar el Servicio API de FastAPI
```bash
python -m uvicorn api.main:app --reload --port 8000
```
*La API estará disponible en `http://127.0.0.1:8000` y la documentación interactiva Swagger en `http://127.0.0.1:8000/docs`.*

### Paso 4: Iniciar la Aplicación Web en Streamlit (En otra terminal)
```bash
python -m streamlit run app/streamlit_app.py
```
*La aplicación web se abrirá automáticamente en su navegador en `http://localhost:8501`.*

---

## 7. Descripción del Modelo de Machine Learning

- **Algoritmo Utilizado**: `RandomForestRegressor` (`n_estimators=150`, `max_depth=12`, `random_state=42`).
- **Variables de Entrada (`X`)**:
  - `ubicacion` (Categoría): Ciudad principal (Quito, Guayaquil).
  - `sector` (Categoría): Sector urbano (ej. Cumbayá, La Carolina, Samborondón, Tumbaco, etc.).
  - `area_m2` (Numérica): Área construida en metros cuadrados ($m^2$).
  - `habitaciones` (Numérica): Número de dormitorios ($1 - 10$).
  - `banos` (Numérica): Número de baños completos ($1 - 10$).
  - `parqueaderos` (Numérica): Número de plazas de estacionamiento ($0 - 8$).
  - `antiguedad_anos` (Numérica): Antigüedad del inmueble en años ($0 - 50$).
- **Variable Objetivo (`y`)**: `precio_usd` (Valor estimado de la vivienda en USD).
- **Métricas Principales de Evaluación (Test Set)**:
  - **Coeficiente de Determinación ($R^2$)**: `~0.8167` (El modelo explica más del 81% de la varianza en los precios).
  - **Error Absoluto Medio (MAE)**: `USD $28,182.26`
  - **Raíz del Error Cuadrático Medio (RMSE)**: `USD $38,221.76`

---

## 8. Uso de la API REST FastAPI

### Endpoint Principal: `POST /predict`

#### Ejemplo de Solicitud (Request Body JSON):
```json
{
  "ubicacion": "Quito",
  "sector": "Cumbayá",
  "area_m2": 185.0,
  "habitaciones": 3,
  "banos": 3,
  "parqueaderos": 2,
  "antiguedad_anos": 4
}
```

#### Ejemplo de Respuesta (Response JSON):
```json
{
  "precio_estimado_usd": 301344.62,
  "precio_formateado": "USD $301,344.62",
  "moneda": "USD",
  "modelo_utilizado": "RandomForestRegressor",
  "status": "success"
}
```

#### Prueba rápida vía `curl`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "ubicacion": "Quito",
  "sector": "Cumbayá",
  "area_m2": 185.0,
  "habitaciones": 3,
  "banos": 3,
  "parqueaderos": 2,
  "antiguedad_anos": 4
}'
```

---

## 9. Limitaciones y Consideraciones

1. **Cobertura Geográfica**: Los datos de entrenamiento corresponden a sectores urbanos seleccionados de Quito y Guayaquil (Cumbayá, La Carolina, Samborondón, Tumbaco, González Suárez, etc.).
2. **Factores Externos**: El modelo predice precios basados en las características físicas y ubicación. No incluye variables cualitativas como acabados de lujo específicos, vista panorámica o estado de conservación imprevisto.
3. **Frecuencia de Actualización**: Se recomienda re-ejecutar el script de scraping y reentrenar el modelo periódicamente para adaptar las predicciones a la inflación e índices inmobiliarios vigentes.
