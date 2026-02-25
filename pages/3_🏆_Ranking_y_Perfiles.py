import streamlit as st
import pandas as pd
import joblib
import sys
import os

# --- MAPA HACIA LA CARPETA SCRAPING ---
# Subimos un nivel (a la carpeta principal) y luego bajamos a la carpeta 'scraping'
ruta_scraping = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scraping'))

if ruta_scraping not in sys.path:
    sys.path.append(ruta_scraping)


st.set_page_config(page_title="Ranking ATP", page_icon="🏆", layout="wide")

st.title("🏆 Ranking ATP en Vivo & Perfiles")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} /* Oculta los 3 puntitos de arriba a la derecha */
            footer {visibility: hidden;} /* Oculta el "Made with Streamlit" de abajo */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- BOTÓN DE ACTUALIZACIÓN ---
# Importamos el actualizador que acabamos de crear
from actualizador_maestro import ejecutar_pipeline

if st.button("🔄 Buscar Nuevos Partidos y Actualizar Ranking", type="primary"):
    with st.spinner("Los robots están trabajando... Esto puede tardar unos minutos."):
        exito = ejecutar_pipeline()
        if exito:
            st.success("¡Base de datos actualizada con éxito!")
            st.rerun() # Recarga la página para mostrar los datos frescos
        else:
            st.error("Hubo un error en la actualización. Revisa la consola.")

st.markdown("---")

# --- CARGAR DATOS ---
ruta_scraping = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scraping'))
ruta_ranking = os.path.join(ruta_scraping, "ranking_2026.csv")
ruta_perfiles = os.path.join(ruta_scraping, "perfiles_jugadores.pkl")

try:
    df_ranking = pd.read_csv(ruta_ranking)
    perfiles = joblib.load(ruta_perfiles)
except Exception as e:
    st.warning(f"No se encontraron los datos en la carpeta scraping. ¿Ya corriste la actualización?")
    st.error(f"Error técnico: {e}") # Esto nos dirá exactamente qué falta si vuelve a fallar
    st.stop()

col1, col2 = st.columns([1, 1])

# --- TRUCO MAGISTRAL: Nombres Completos ---
# Extraemos el nombre real de la URL igual que en tu generador de perfiles
def extraer_nombre_real(url):
    try:
        slug = str(url).split('/')[5] 
        return slug.replace('-', ' ').title()
    except:
        return ""

# Creamos la columna mágica antes de dividir la pantalla
df_ranking['Nombre Completo'] = df_ranking['url_perfil'].apply(extraer_nombre_real)

# --- COLUMNA 1: TABLA DE RANKING ---
with col1:
    st.subheader("Clasificación Mundial")
    
    # Armamos la tabla bonita usando el Nombre Completo
    df_mostrar = df_ranking[['rank', 'Nombre Completo', 'points']].copy()
    df_mostrar.columns = ['Rank', 'Jugador', 'Puntos']
    st.dataframe(df_mostrar.set_index('Rank'), height=600, use_container_width=True)

# --- COLUMNA 2: PERFIL DEL JUGADOR ---
with col2:
    st.subheader("🔍 Analizador de Perfil")
    
    # 🧠 Pasamos la lista de Nombres Completos al buscador
    lista_jugadores = df_ranking['Nombre Completo'].tolist()
    
    # Filtramos los vacíos por si acaso
    lista_jugadores = [j for j in lista_jugadores if j != ""] 
    
    jugador_seleccionado = st.selectbox("Buscar jugador:", lista_jugadores)
    
    # Ahora sí, buscará "Jannik Sinner" en los perfiles
    if jugador_seleccionado in perfiles:
        p = perfiles[jugador_seleccionado]
        
        # Panel de métricas principales
        c1, c2, c3 = st.columns(3)
        c1.metric("Ranking", int(p.get('rank', 0)))
        c2.metric("Puntos", f"{int(p.get('points', 0)):,}".replace(",", "."))
        c3.metric("País", p.get('ioc', 'UNK'))
        
        st.markdown("#### 🧬 Atributos Biométricos")
        c4, c5 = st.columns(2)
        c4.info(f"**Edad:** {p.get('age', 'N/A')} años")
        c5.info(f"**Altura:** {p.get('ht', 'N/A')} cm")
        
        st.markdown("#### 🎾 Estadísticas Avanzadas")
        st.write(f"**1er Saque Ganado:** {p.get('serve_win', 0)}%")
        st.write(f"**Break Points Salvados:** {p.get('bp_saved', 0)}%")
        st.write(f"**Aces por partido:** {p.get('aces', 0):.1f}")
        st.write(f"**Momentum (Últimos 5):** {int(p.get('momentum', 0) * 100)}%")
        
    else:
        st.info("No hay estadísticas avanzadas para este jugador todavía.")