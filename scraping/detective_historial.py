import pandas as pd

print("[SEARCH] ABRIENDO LA CAJA NEGRA DEL HISTORIAL COMPLETO...\n")

# 1. AHORA SÍ LEEMOS EL ARCHIVO FUSIONADO (Usa low_memory=False para evitar el warning de letras rojas)
ARCHIVO = "historialTenis.csv"
try:
    df = pd.read_csv(ARCHIVO, low_memory=False)
except FileNotFoundError:
    print(f"[FAIL] ERROR: No se encuentra el archivo {ARCHIVO}. Revisa que se llame así en tu carpeta.")
    exit()

# 2. BUSCAMOS POR APELLIDO (Así atrapamos a "Carlos Alcaraz" y a "Carlos Alcaraz Garfia")
apellido = "Cerundolo"

df_jugador = df[df['winner_name'].str.contains(apellido, case=False, na=False) | 
                df['loser_name'].str.contains(apellido, case=False, na=False)].copy()

# 3. IMPRIMIMOS LOS ÚLTIMOS 20 PARTIDOS
ultimos_20 = df_jugador.tail(20)
columnas_ver = ['tourney_id', 'tourney_name', 'tourney_date', 'round', 'winner_name', 'loser_name']

print(f"[ATP] Partidos encontrados para '{apellido}': {len(df_jugador)}\n")
print(ultimos_20[columnas_ver].to_string())