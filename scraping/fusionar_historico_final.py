import pandas as pd
import os

# --- CONFIGURACIÓN ---
ARCHIVO_HISTORICO = "historial_tenis.csv"       # Tu archivo original 2000-2024
ARCHIVO_NUEVO = "atp_matches_2025_2026_CORREGIDO.csv"   # El archivo nuevo enriquecido (o el raw)
ARCHIVO_SALIDA = "historial_tenis_COMPLETO.csv" # El resultado final

print("🧬 INICIANDO FUSIÓN FINAL...")

# 1. CARGAR ARCHIVOS
if not os.path.exists(ARCHIVO_HISTORICO):
    print(f"❌ Error: No encuentro '{ARCHIVO_HISTORICO}'")
    exit()

if not os.path.exists(ARCHIVO_NUEVO):
    print(f"❌ Error: No encuentro '{ARCHIVO_NUEVO}'")
    # Intenta buscar el raw si no encuentra el master
    if os.path.exists("atp_matches_2025_2026_raw.csv"):
        print("   ⚠️ Usando 'atp_matches_2025_2026_raw.csv' como respaldo.")
        ARCHIVO_NUEVO = "atp_matches_2025_2026_raw.csv"
    else:
        exit()

try:
    df_hist = pd.read_csv(ARCHIVO_HISTORICO)
    df_new = pd.read_csv(ARCHIVO_NUEVO)
    
    print(f"📂 Histórico: {len(df_hist)} partidos | Columnas: {len(df_hist.columns)}")
    print(f"📂 Nuevo:     {len(df_new)} partidos  | Columnas: {len(df_new.columns)}")

    # 2. NORMALIZAR COLUMNAS (El paso clave)
    # Hacemos que el nuevo tenga EXACTAMENTE las mismas columnas que el histórico
    columnas_hist = df_hist.columns.tolist()
    
    # Verificamos si hay columnas con nombres distintos y tratamos de arreglarlas
    # (A veces el scrape trae 'winner' y el historico 'winner_name')
    mapeo = {
        'winner': 'winner_name',
        'loser': 'loser_name',
        'tourney_id': 'tourney_id', # Asegurar que coincidan
        'surface': 'surface'
    }
    df_new.rename(columns=mapeo, inplace=True)

    # Creamos un DataFrame nuevo solo con las columnas del histórico
    df_new_aligned = pd.DataFrame(columns=columnas_hist)
    
    # Copiamos los datos que SÍ tenemos
    for col in df_new.columns:
        if col in columnas_hist:
            df_new_aligned[col] = df_new[col]
        else:
            print(f"   ⚠️ La columna '{col}' del nuevo archivo se ignorará (no existe en histórico).")
    
    # Rellenamos los datos faltantes (Stats de partido que no scrapeamos)
    # Ej: w_ace, w_df, minutes, etc.
    df_new_aligned.fillna(0, inplace=True)
    
    # 3. UNIR (CONCATENAR)
    print("🔄 Uniendo archivos...")
    df_total = pd.concat([df_hist, df_new_aligned], ignore_index=True)
    
    # 4. LIMPIEZA FINAL
    # Convertir tourney_date a formato numérico si es necesario
    df_total['tourney_date'] = pd.to_numeric(df_total['tourney_date'], errors='coerce').fillna(20260101).astype(int)
    
    # Ordenar por fecha (opcional)
    df_total.sort_values(by=['tourney_date', 'match_num'], inplace=True)

    # 5. GUARDAR
    df_total.to_csv(ARCHIVO_SALIDA, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 ¡FUSIÓN EXITOSA!")
    print(f"📊 Total partidos: {len(df_total)}")
    print(f"   (Histórico {len(df_hist)} + Nuevo {len(df_new)})")
    print(f"💾 Guardado en: {ARCHIVO_SALIDA}")
    print("="*50)

except Exception as e:
    print(f"❌ Error durante la fusión: {e}")
    # Diagnóstico de columnas
    print("\n🔍 DIAGNÓSTICO DE COLUMNAS:")
    print(f"Histórico (Primeras 5): {list(df_hist.columns)[:5]}")
    print(f"Nuevo (Primeras 5):     {list(df_new.columns)[:5]}")