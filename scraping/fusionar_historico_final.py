"""
Fusiona el historial completo con los datos scrapeados nuevos.

Toma historial_tenis_COMPLETO.csv (2000-2024) + los partidos nuevos (2025-2026)
y genera data/processed/historialTenis.csv listo para entrenamiento.
"""
import sys
from pathlib import Path

# Agregar src al path para usar el paquete
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pandas as pd
from atp_predictor.core.paths import get_scraping_dir, get_processed_data_dir, get_external_data_dir

# --- CONFIGURACION ---
scraping_dir = get_scraping_dir()
ARCHIVO_HISTORICO = get_external_data_dir() / "historial_tenis_COMPLETO.csv"

# Buscar el archivo de partidos nuevos (prioridad: corregido > indetectable > full)
ARCHIVO_NUEVO = None
for candidate in ["atp_matches_2026_corregido.csv", "atp_matches_2026_indetectable.csv", "atp_matches_2026_full.csv"]:
    if (scraping_dir / candidate).exists():
        ARCHIVO_NUEVO = scraping_dir / candidate
        break

ARCHIVO_SALIDA = get_processed_data_dir() / "historialTenis.csv"

print("[DNA] INICIANDO FUSION FINAL...")

# 1. CARGAR ARCHIVOS
if not ARCHIVO_HISTORICO.exists():
    print(f"[FAIL] Error: No encuentro '{ARCHIVO_HISTORICO}'")
    sys.exit(1)

if ARCHIVO_NUEVO is None:
    print(f"[FAIL] No encuentro archivos de partidos nuevos en {scraping_dir}")
    sys.exit(1)

try:
    df_hist = pd.read_csv(ARCHIVO_HISTORICO, low_memory=False)
    df_new = pd.read_csv(ARCHIVO_NUEVO, low_memory=False)

    print(f"[DATA] Historico: {len(df_hist)} partidos | Columnas: {len(df_hist.columns)}")
    print(f"[DATA] Nuevo:     {len(df_new)} partidos | Columnas: {len(df_new.columns)}")
    print(f"[DATA] Usando archivo nuevo: {ARCHIVO_NUEVO.name}")

    # 2. NORMALIZAR COLUMNAS
    columnas_hist = df_hist.columns.tolist()

    # Mapeo de columnas con nombres distintos
    mapeo = {
        'winner': 'winner_name',
        'loser': 'loser_name',
    }
    df_new.rename(columns=mapeo, inplace=True)

    # Alinear columnas del nuevo con el historico
    df_new_aligned = pd.DataFrame(columns=columnas_hist)
    for col in df_new.columns:
        if col in columnas_hist:
            df_new_aligned[col] = df_new[col].values
        else:
            print(f"   [WARN] Columna '{col}' del nuevo archivo se ignorara (no existe en historico).")

    # Rellenar datos faltantes
    df_new_aligned.fillna(0, inplace=True)

    # 3. UNIR (CONCATENAR)
    print("[REFRESH] Uniendo archivos...")
    df_total = pd.concat([df_hist, df_new_aligned], ignore_index=True)

    # 4. ELIMINAR DUPLICADOS
    dupe_cols = ['winner_name', 'loser_name', 'round', 'tourney_id']
    antes = len(df_total)
    df_total = df_total.drop_duplicates(subset=dupe_cols, keep='last')
    despues = len(df_total)
    print(f"[DATA] Duplicados eliminados: {antes - despues} partidos repetidos")
    print(f"[DATA] Partidos unicos: {despues}")

    # 4. LIMPIEZA FINAL Y ARREGLO DE FECHAS
    print("[CONFIG] Reparando linea de tiempo...")
    df_total['tourney_date'] = pd.to_numeric(df_total['tourney_date'], errors='coerce').fillna(0).astype(int)

    def arreglar_fecha(fila):
        fecha = fila['tourney_date']
        if fecha > 20000000:
            return fecha
        try:
            id_torneo = str(fila['tourney_id'])
            anio = id_torneo.split('-')[0]
            if len(anio) == 4 and anio.isdigit():
                return int(anio) * 10000 + 101
        except:
            pass
        return 20260101

    df_total['tourney_date'] = df_total.apply(arreglar_fecha, axis=1)

    # Ordenar por fecha
    df_total.sort_values(by=['tourney_date', 'match_num'], inplace=True)

    # 5. GUARDAR
    df_total.to_csv(ARCHIVO_SALIDA, index=False)

    print("\n" + "=" * 50)
    print(f"[DONE] FUSION EXITOSA!")
    print(f"[STATS] Total partidos: {len(df_total)}")
    print(f"   (Historico {len(df_hist)} + Nuevo {len(df_new)})")
    print(f"[SAVE] Guardado en: {ARCHIVO_SALIDA}")
    print("=" * 50)

except Exception as e:
    print(f"[FAIL] Error durante la fusion: {e}")
    import traceback
    traceback.print_exc()