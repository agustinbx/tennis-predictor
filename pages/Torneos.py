"""
Página de torneos y partidos.
Muestra próximos torneos y resultados recientes.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from streamlit_app.utils import apply_style, get_project_root
except ImportError:
    def apply_style():
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)
    
    def get_project_root():
        return Path(__file__).parent.parent


st.set_page_config(page_title="Próximos Torneos", page_icon="📅")

st.title("📅 Próximos Partidos y Resultados")

apply_style()

# Cargar datos recientes
@st.cache_data
def cargar_partidos():
    project_root = get_project_root()
    possible_files = [
        project_root / "scraping" / "atp_matches_2025_2026_unidos.csv",
        project_root / "scraping" / "atp_matches_2026_full.csv",
        project_root / "scraping" / "atp_matches_2026_corregido.csv",
    ]
    
    for path in possible_files:
        if path.exists():
            return pd.read_csv(path)
    
    return pd.DataFrame()


df = cargar_partidos()

if not df.empty:
    # Filtros
    torneos = df['tourney_name'].unique() if 'tourney_name' in df.columns else []
    
    if len(torneos) > 0:
        torneo_sel = st.selectbox("Selecciona un Torneo:", torneos)
        
        # Filtrar por torneo
        df_t = df[df['tourney_name'] == torneo_sel]
        
        st.write(f"Partidos encontrados: {len(df_t)}")
        
        for index, row in df_t.iterrows():
            winner = row.get('winner_name', 'N/A')
            loser = row.get('loser_name', 'N/A')
            score = row.get('score', 'N/A')
            surface = row.get('surface', 'N/A')
            round_name = row.get('round', 'N/A')
            
            with st.expander(f"{round_name}: {winner} vs {loser}"):
                st.write(f"**Resultado:** {score}")
                st.write(f"**Superficie:** {surface}")
                st.info("Ve a la pestaña 'Predictor' para analizar este matchup en detalle.")
    else:
        st.warning("No se encontraron torneos en los datos.")
else:
    st.warning("No se encontraron partidos recientes cargados.")
    st.info("Ejecuta el scraper para obtener datos: `python scraping/scraper_2026_final.py`")
