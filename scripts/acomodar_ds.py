"""
Estandariza el CSV de historial de tenis.

Lee datos fusionados de data/processed/historialTenis.csv (o la fuente original),
normaliza rondas, corrige fechas usando un mapa construido del historial original,
limpia datos y genera data/processed/historialTenis.csv.
"""
import sys
from pathlib import Path

# Agregar src al path para usar el paquete
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from atp_predictor.core.paths import get_project_root, get_processed_data_dir, get_external_data_dir

project_root = get_project_root()
output_file = get_processed_data_dir() / "historialTenis.csv"

# Si ya existe el archivo fusionado (con datos nuevos), usarlo.
# Si no, leer de la fuente original.
merged_file = get_processed_data_dir() / "historialTenis.csv"
source_file = get_external_data_dir() / "historial_tenis_COMPLETO.csv"

if merged_file.exists():
    print(f"   [DATA] Re-estandarizando archivo fusionado: {merged_file}")
    df = pd.read_csv(merged_file, low_memory=False)
else:
    print(f"   [DATA] Leyendo fuente original: {source_file}")
    df = pd.read_csv(source_file)

# ==============================================================================
# 0. CONSTRUIR MAPA DE FECHAS a partir del historial original
# ==============================================================================
# El archivo original tiene fechas correctas para los torneos historicos.
# Lo usamos para construir un mapa tourney_id -> fecha_mas_frecuente.
df_original = pd.read_csv(source_file, low_memory=False)
df_fecha_valida = df_original[df_original['tourney_date'] > 19900000]
fecha_por_tourney_id = {}
for tid, grp in df_fecha_valida.groupby('tourney_id'):
    fecha_moda = grp['tourney_date'].mode().iloc[0]
    fecha_por_tourney_id[str(tid)] = int(fecha_moda)
print(f"   [DATA] Mapa de fechas: {len(fecha_por_tourney_id)} torneos con fecha conocida")

# ==============================================================================
# 1. NORMALIZACION DE RONDAS (R16, QF, SF, F...)
# ==============================================================================
print("   -> Estandarizando nombres de rondas...")

mapa_rondas = {
    'Round of 128': 'R128', 'Round of 64': 'R64', 'Round of 32': 'R32',
    'Round of 16': 'R16', 'Quarterfinals': 'QF', 'Semifinals': 'SF',
    'Semifinal': 'SF', 'Finals': 'F', 'The Final': 'F', 'Final': 'F',
    'Winner': 'W', 'Round Robin': 'RR',
    '1st Round Qualifying': 'Q1', '2nd Round Qualifying': 'Q2',
    '3rd Round Qualifying': 'Q3', 'Qualifying Round': 'Q1',
    'Quarter': 'QF', 'Semi': 'SF', 'BR': 'BR',
    'Round Robin Day 1': 'RR', 'Round Robin Day 2': 'RR', 'Round Robin Day 3': 'RR',
    'Round Robin Day 4': 'RR', 'Round Robin Day 5': 'RR', 'Round Robin Day 6': 'RR',
}

df['round'] = df['round'].astype(str).str.strip().replace(mapa_rondas)

# ==============================================================================
# 2. ARREGLAR FECHAS
# ==============================================================================
print("   -> Reparando fechas...")

meses_torneos = {
    'australian': '0115', 'garros': '0528', 'wimbledon': '0701', 'us-open': '0828',
    'indian': '0310', 'miami': '0325', 'monte': '0407', 'madrid': '0501',
    'rome': '0515', 'canada': '0807', 'cincinnati': '0815', 'shanghai': '1005',
    'paris': '1030', 'finals': '1115', 'nitto': '1115',
    'brisbane': '0101', 'adelaide': '0106', 'auckland': '0106', 'hobart': '0110',
    'dallas': '0205', 'delray': '0210', 'rio-de-janeiro': '0215',
    'buenos-aires': '0210', 'rotterdam': '0210', 'marseille': '0215',
    'doha': '0220', 'dubai': '0225', 'acapulco': '0225', 'santiago': '0225',
    'houston': '0405', 'estoril': '0405', 'marrakech': '0405',
    'barcelona': '0415', 'munich': '0420', 'belgrade': '0420', 'srb': '0420',
    'lyon': '0518', 'geneva': '0518', 's-hertogenbosch': '0610',
    'stuttgart': '0610', 'halle': '0615', 'queen': '0615',
    'eastbourne': '0625', 'mallorca': '0625',
    'newport': '0710', 'bastad': '0715', 'swedish': '0715',
    'gstaad': '0720', 'hamburg': '0720', 'umag': '0725', 'kitzbuhel': '0725',
    'washington': '0730', 'atlanta': '0725', 'los-cabos': '0805',
    'winston-salem': '0820',
    'chengdu': '0920', 'zhuhai': '0920', 'astana': '0925', 'almaty': '0925',
    'tokyo': '0925', 'stockholm': '1010', 'antwerp': '1015', 'basel': '1020',
    'vienna': '1020', 'naples': '0925', 'seoul': '0925', 'tel': '0925',
    'metz': '0920', 'st-petersburg': '0920', 'san-diego': '0920',
    'sofia': '1105', 'sydney': '0106', 'perth': '0101', 'montpellier': '0205',
    'bucharest': '0405', 'rio': '0215', 'brussels': '0515', 'beijing': '0925',
    'laver': '0920', 'hangzhou': '0920', 'hong-kong': '0106',
    'cup': '0115', 'davis': '0201', 'united': '0101', 'hopman': '0105',
}

def corregir_fecha(row):
    fecha_actual = row['tourney_date']
    # Solo conservar fechas validas que NO sean el generico 20250101/20260101
    # Esas fechas genericas se asignaron en la fusion y necesitan correccion
    if pd.notna(fecha_actual) and float(fecha_actual) > 19900000:
        fecha_int = int(fecha_actual)
        # Fechas genericas: YYYY0101 (enero 1) son sospechosas
        if fecha_int % 10000 != 101:  # No es 1 de enero generico
            return fecha_int
        # Es 20250101 o 20260101 -> verificar si es real o generico
        # Solo Australian Open ( Brisbame, Adelaide) son en enero
        tourney_id = str(row['tourney_id']).lower()
        for key in ['australian', 'brisbane', 'adelaide', 'auckland', 'perth', 'sydney', 'hobart', 'cup', 'united']:
            if key in tourney_id:
                return fecha_int  # Si, es real en enero
        # No es un torneo de enero -> necesita correccion, seguir abajo
    
    tourney_id = str(row['tourney_id'])
    
    # 1. Buscar en el mapa directo (construido del historial original)
    if tourney_id in fecha_por_tourney_id:
        return fecha_por_tourney_id[tourney_id]
    
    # 2. Inferir del tourney_id usando el diccionario de slugs
    try:
        id_parts = tourney_id.split('-')
        year = id_parts[0]
        if year.isdigit() and len(year) == 4:
            resto_id = tourney_id.lower()
            mes_dia = "0101"  # default: enero
            for key, val in meses_torneos.items():
                if key in resto_id:
                    mes_dia = val
                    break
            return int(year + mes_dia)
    except:
        pass
    return 0

df['tourney_date'] = df.apply(corregir_fecha, axis=1)

# ==============================================================================
# 3. LIMPIEZA DE CEROS EN BIO (Edad, Altura, Pais)
# ==============================================================================
print("   -> Limpiando ceros en datos biograficos...")
cols_bio = ['winner_age', 'winner_ht', 'winner_ioc', 'loser_age', 'loser_ht', 'loser_ioc']

for col in cols_bio:
    if col in df.columns:
        df[col] = df[col].replace([0, 0.0, '0'], np.nan)

# ==============================================================================
# 4. ORDENAMIENTO FINAL Y GUARDADO
# ==============================================================================
orden_ronda = {
    'Q1':1, 'Q2':2, 'Q3':3, 'BR':6,
    'R128':10, 'R64':20, 'R32':30, 'R16':40, 'QF':50, 'SF':60, 'F':70, 'W':80, 
    'RR': 5
}
df['orden_temp'] = df['round'].map(orden_ronda).fillna(0)

df = df.sort_values(by=['tourney_date', 'tourney_id', 'orden_temp'])
df = df.drop(columns=['orden_temp'])

df.to_csv(output_file, index=False)

# Reporte
dates = pd.to_numeric(df['tourney_date'], errors='coerce')
generic = ((dates == 20250101) | (dates == 20260101)).sum()
total = len(df)
print(f"\n[OK] LISTO! CSV Estandarizado generado en:")
print(f"   [DATA] {output_file}")
print(f"   [STATS] {total} partidos, {generic} con fecha aproximada")
print("   [>] Ahora las rondas son: R128, R64, R16, QF, SF, F")