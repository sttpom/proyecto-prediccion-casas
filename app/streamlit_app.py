import streamlit as st
import requests
import os
import sys
import pandas as pd
import numpy as np

# Configuración inicial de la página
st.set_page_config(
    page_title="Predicción de Precios de Casas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para UI atractiva y moderna
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F9FF;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 6px solid #0284C7;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 1rem;
    }
    .metric-price {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0369A1;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.85em;
        font-weight: 700;
        color: #15803D;
        background-color: #DCFCE7;
        border-radius: 9999px;
    }
    </style>
""", unsafe_allow_html=True)

# URL de la API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Opciones predefinidas para controles
SECTORES_OPCIONES = [
    "Cumbayá", "La Carolina", "González Suárez", "Tumbaco",
    "Samborondón", "Kennedy Norte", "Puerto Santa Ana", "El Batán",
    "Centro Histórico", "Valle de los Chillos"
]

CIUDADES_POR_SECTOR = {
    "Cumbayá": "Quito", "La Carolina": "Quito", "González Suárez": "Quito",
    "Tumbaco": "Quito", "El Batán": "Quito", "Centro Histórico": "Quito",
    "Valle de los Chillos": "Quito", "Samborondón": "Guayaquil",
    "Kennedy Norte": "Guayaquil", "Puerto Santa Ana": "Guayaquil"
}

# Título de la Aplicación
st.markdown("<div class='main-header'>🏠 Sistema de Predicción de Precios de Vivienda</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Diplomado de Python Fullstack - Modelo Inteligente en Tiempo Real</div>", unsafe_allow_html=True)

# Comprobar estado de conexión con FastAPI
api_online = False
try:
    res_health = requests.get(f"{API_URL}/health", timeout=2)
    if res_health.status_code == 200 and res_health.json().get("modelo_cargado"):
        api_online = True
except Exception:
    api_online = False

# Sidebar con información del sistema
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/home.png", width=80)
    st.title("Panel de Control")
    st.markdown("---")
    
    
    #if api_online:
        #st.markdown("<span class='status-badge'>🟢 FastAPI Activo</span>", unsafe_allow_html=True)
        #st.caption(f"Conectado a `{API_URL}`")
    #else:
        #st.error("🔴 FastAPI Inaccesible")
        #st.warning("No se detectó el servicio uvicorn en `http://127.0.0.1:8000`.")
        #st.caption("Instrucción: Ejecute `python -m uvicorn api.main:app --reload` en su terminal.")


    st.markdown("---")
    st.markdown("### Arquitectura del Sistema")
    st.markdown("""
    1. **Web Scraping**: Plusvalía.com
    2. **Persistencia**: CSV & SQLite
    3. **Modelo ML**: Random Forest PKL
    4. **Backend**: FastAPI REST Service
    5. **Frontend**: Streamlit Dashboard
    """)

# Pestañas principales
tab_predict, tab_analytics = st.tabs(["🎯 Realizar Predicción", "📊 Exploración de Datos (EDA)"])

with tab_predict:
    st.subheader("Ingrese las Características del Inmueble")
    
    col1, col2 = st.columns(2)

    with col1:
        sector_sel = st.selectbox("Sector Urbano", SECTORES_OPCIONES, index=0)
        ciudad_sel = CIUDADES_POR_SECTOR.get(sector_sel, "Quito")
        st.info(f"📍 **Ciudad asociada**: {ciudad_sel}")

        area_m2 = st.number_input("Área construida (m²)", min_value=20.0, max_value=1000.0, value=120.0, step=5.0)
        antiguedad = st.slider("Antigüedad de la vivienda (años)", min_value=0, max_value=50, value=5)

    with col2:
        habitaciones = st.number_input("Habitaciones / Dormitorios", min_value=1, max_value=10, value=3)
        banos = st.number_input("Baños completos", min_value=1, max_value=10, value=2)
        parqueaderos = st.number_input("Plazas de Parqueadero", min_value=0, max_value=8, value=1)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("🚀 Estimar Precio en Tiempo Real", type="primary", use_container_width=True)

    if btn_predict:
        payload = {
            "ubicacion": ciudad_sel,
            "sector": sector_sel,
            "area_m2": float(area_m2),
            "habitaciones": int(habitaciones),
            "banos": int(banos),
            "parqueaderos": int(parqueaderos),
            "antiguedad_anos": int(antiguedad)
        }

        if api_online:
            with st.spinner("Enviando solicitud a FastAPI y consultando modelo PKL..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                    if response.status_code == 200:
                        res_json = response.json()
                        precio_usd = res_json.get("precio_estimado_usd", 0.0)
                        precio_fmt = res_json.get("precio_formateado", "$0.00")

                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Precio Estimado de la Vivienda</h3>
                                <div class='metric-price'>{precio_fmt}</div>
                                <p style='color: #64748B; margin-top: 0.5rem;'>
                                    Modelo: <b>{res_json.get('modelo_utilizado')}</b> | Moneda: <b>USD</b>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.error(f"Error de la API: {response.text}")
                except Exception as e:
                    st.error(f"Fallo de conexión al enviar datos a FastAPI: {e}")
        else:
            # st.warning("Servicio FastAPI activo. Mostrando cálculo de respaldo local:")
            # Respaldo local si FastAPI no está corriendo
            try:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                import joblib
                from src.preprocessing import ALL_FEATURES
                
                model_local = joblib.load("models/modelo_precios.pkl")
                prep_local = joblib.load("models/preprocesador.pkl")
                
                df_single = pd.DataFrame([payload])
                X_trans = prep_local.transform(df_single[ALL_FEATURES])
                pred_val = model_local.predict(X_trans)[0]
                
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Precio Estimado (Modo Respaldo Local)</h3>
                        <div class='metric-price'>USD ${pred_val:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as ex_respaldo:
                st.error(f"No se pudo cargar la predicción local: {ex_respaldo}")

with tab_analytics:
    st.subheader("Análisis Exploratorio del Dataset de Casas")
    processed_csv = "data/processed/casas_limpias.csv"
    
    if os.path.exists(processed_csv):
        df_display = pd.read_csv(processed_csv)
        st.dataframe(df_display.head(10), use_container_width=True)
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#### Distribución de Precios por Sector (USD)")
            st.bar_chart(df_display.groupby("sector")["precio_usd"].mean())
        with c_b:
            st.markdown("#### Relación Área (m²) vs. Precio (USD)")
            st.scatter_chart(df_display[['area_m2', 'precio_usd']])
    else:
        st.info("Aún no se ha procesado el archivo `casas_limpias.csv`. Ejecute el pipeline de preprocesamiento.")
