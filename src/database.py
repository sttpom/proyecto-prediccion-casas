import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'casas_db.sqlite')

def init_db(db_path=DB_PATH):
    """Inicializa la base de datos SQLite y crea la tabla 'casas' si no existe."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS casas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT,
            ubicacion TEXT,
            sector TEXT,
            precio_usd REAL,
            area_m2 REAL,
            habitaciones INTEGER,
            banos INTEGER,
            parqueaderos INTEGER,
            antiguedad_anos INTEGER,
            fecha_extraccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def guardar_casas_en_db(df: pd.DataFrame, db_path=DB_PATH):
    """Guarda un DataFrame de pandas en la tabla 'casas' de la base de datos SQLite."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    # Reemplazar la tabla si se llama nuevamente para evitar duplicados en la demo
    df.to_sql('casas', conn, if_exists='replace', index=False)
    conn.close()
    print(f"[OK] Se guardaron {len(df)} registros en la base de datos: {db_path}")

def obtener_casas_de_db(db_path=DB_PATH) -> pd.DataFrame:
    """Obtiene todos los registros de viviendas desde la base de datos SQLite."""
    if not os.path.exists(db_path):
        init_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM casas", conn)
    conn.close()
    return df

if __name__ == "__main__":
    init_db()
    print("Base de datos inicializada correctamente.")
