import requests
import time
import subprocess
import os
import sys

def probar_api():
    print("[INFO] Probando backend FastAPI internamente...")
    from fastapi.testclient import TestClient
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from api.main import app

    with TestClient(app) as client:
        # 1. Probar GET /
        response_root = client.get("/")
        assert response_root.status_code == 200, f"Error en GET /: {response_root.text}"
        print("[OK] Endpoint GET / respondiendo correctamente:", response_root.json())

        # 2. Probar GET /health
        response_health = client.get("/health")
        assert response_health.status_code == 200, f"Error en GET /health: {response_health.text}"
        assert response_health.json()["status"] == "healthy", "El modelo no se reporta saludable"
        print("[OK] Endpoint GET /health respondiendo correctamente:", response_health.json())

        # 3. Probar POST /predict
        payload = {
            "ubicacion": "Quito",
            "sector": "Cumbayá",
            "area_m2": 185.0,
            "habitaciones": 3,
            "banos": 3,
            "parqueaderos": 2,
            "antiguedad_anos": 4
        }

        response_predict = client.post("/predict", json=payload)
        assert response_predict.status_code == 200, f"Error en POST /predict: {response_predict.text}"
        data = response_predict.json()
        print("\n" + "="*50)
        print("      PRUEBA DE PREDICCION CON CLIENTE DE API       ")
        print("="*50)
        print(f"  Payload Enviado:      {payload['sector']} - {payload['area_m2']} m2")
        print(f"  Precio Estimado USD:  {data['precio_estimado_usd']}")
        print(f"  Formato Visual:       {data['precio_formateado']}")
        print(f"  Estado del Servicio:  {data['status']}")
        print("="*50 + "\n")

if __name__ == "__main__":
    probar_api()
