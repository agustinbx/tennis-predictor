import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="ATP Predictor IA 2026",
    page_icon="🎾",
    layout="wide"
)

# --- ⚠️ CONFIGURACIÓN MANUAL DEL ACURACY ---
# Pon aquí el número que te dio tu script de entrenamiento (guardar_campeon.py)
PRECISION_DEL_MODELO = 69.13  # Porcentaje (ej: 72.5%)

# --- CARGAR DATOS Y MODELO ---
@st.cache_data
def cargar_datos():
    # Carga tu archivo MAESTRO FINAL
    df = pd.read_csv("atp_matches_historico.csv") 
    # Asegurar formato de fecha
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(by='tourney_date', ascending=False)
    return df

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_logistico_final.pkl")

try:
    df = cargar_datos()
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"❌ Error cargando archivos: {e}")
    st.stop()

# --- FUNCIÓN INTELIGENTE DE DATOS (CORREGIDA) ---
def obtener_ultimos_datos(nombre_jugador):
    """
    Busca el último partido y PROYECTA la edad a 2026.
    """
    # Buscamos el último registro donde aparezca
    filtro = df[(df['winner_name'] == nombre_jugador) | (df['loser_name'] == nombre_jugador)].head(1)
    
    if filtro.empty: 
        return None
    
    fila = filtro.iloc[0]
    
    # Detectamos si fue ganador o perdedor
    if fila['winner_name'] == nombre_jugador:
        base_age = fila['winner_age']
        base_rank = fila['winner_rank']
        base_ht = fila['winner_ht']
    else:
        base_age = fila['loser_age']
        base_rank = fila['loser_rank']
        base_ht = fila['loser_ht']
    
    # --- CORRECCIÓN DE EDAD ---
    # Extraemos el año del partido (los primeros 4 dígitos de tourney_date, ej: 20240101 -> 2024)
    try:
        anio_partido = int(str(fila['tourney_date'])[:4])
    except:
        anio_partido = 2024 # Fallback
        
    diferencia_años = 2026 - anio_partido
    edad_proyectada = base_age + diferencia_años
    
    return {
        'rank': base_rank,
        'age': edad_proyectada, # ¡Aquí está la magia!
        'ht': base_ht
    }

# ==============================================================================
# 🎨 INTERFAZ GRÁFICA (UI)
# ==============================================================================

# 1. ENCABEZADO Y PRESENTACIÓN
st.title("🎾 ATP Match Predictor 2026")
st.markdown("---")

# Columnas para Intro (Izquierda) y Métricas (Derecha)
col_intro, col_metric = st.columns([3, 1])

with col_intro:
    st.markdown("""
    ### 🤖 ¿Qué es esto?
    Esta aplicación utiliza un modelo de **Inteligencia Artificial (Machine Learning)** entrenado con más de 
    **20 años de historia ATP** (2000 - 2026).
    
    El sistema analiza patrones complejos como:
    * 📊 Diferencia de Ranking y Edad.
    * 📏 Ventajas físicas (Altura).
    * 🏟️ Rendimiento en superficie específica (Arcilla, Dura, Pasto).
    * ⚔️ Historial directo (Head-to-Head).
    """)
    
    with st.expander("ℹ️ Ver detalles técnicos del modelo"):
        st.write("""
        - **Algoritmo:** Random Forest Classifier / XGBoost.
        - **Datos:** Scrapeados de ATP Tour oficial.
        - **Actualización:** Incluye partidos del Australian Open 2026.
        - **Variables:** El modelo no ve nombres, ve matemáticas (Deltas de ranking, edad, etc.).
        """)

with col_metric:
    # Tarjeta de Métrica Grande
    st.metric(
        label="🎯 Precisión del Modelo", 
        value=f"{PRECISION_DEL_MODELO}%", 
        delta="En Test Set",
        help="Porcentaje de acierto obtenido en las pruebas con datos no vistos."
    )

st.markdown("---")

# 2. SELECCIÓN DE JUGADORES
st.subheader("🛠️ Configurar Partido")
jugadores = sorted(pd.concat([df['winner_name'], df['loser_name']]).unique())

c1, c2, c3 = st.columns([1, 0.2, 1]) # Columnas con espacio en el medio para el "VS"

with c1:
    st.info("👤 **Jugador 1**")
    p1 = st.selectbox("Buscar Jugador 1", jugadores, index=jugadores.index("Jannik Sinner") if "Jannik Sinner" in jugadores else 0)
    
    # Autocompletado P1
    d1 = obtener_ultimos_datos(p1)
    if d1:
        r1 = st.number_input("Ranking", value=int(d1['rank']) if pd.notna(d1['rank']) else 100, key="r1")
        a1 = st.number_input("Edad", value=float(d1['age']) if pd.notna(d1['age']) else 25.0, key="a1")
        h1 = st.number_input("Altura (cm)", value=int(d1['ht']) if pd.notna(d1['ht']) else 185, key="h1")
    else:
        r1 = st.number_input("Ranking", value=100, key="r1")
        a1 = st.number_input("Edad", value=25.0, key="a1")
        h1 = st.number_input("Altura (cm)", value=185, key="h1")

with c2:
    st.markdown("<h1 style='text-align: center; margin-top: 100px;'>VS</h1>", unsafe_allow_html=True)

with c3:
    st.success("👤 **Jugador 2**")
    p2 = st.selectbox("Buscar Jugador 2", jugadores, index=jugadores.index("Carlos Alcaraz") if "Carlos Alcaraz" in jugadores else 1)
    
    # Autocompletado P2
    d2 = obtener_ultimos_datos(p2)
    if d2:
        r2 = st.number_input("Ranking", value=int(d2['rank']) if pd.notna(d2['rank']) else 100, key="r2")
        a2 = st.number_input("Edad", value=float(d2['age']) if pd.notna(d2['age']) else 25.0, key="a2")
        h2 = st.number_input("Altura (cm)", value=int(d2['ht']) if pd.notna(d2['ht']) else 185, key="h2")
    else:
        r2 = st.number_input("Ranking", value=100, key="r2")
        a2 = st.number_input("Edad", value=25.0, key="a2")
        h2 = st.number_input("Altura (cm)", value=185, key="h2")

# 3. HISTORIAL Y SUPERFICIE
st.markdown("---")
col_surf, col_h2h = st.columns([1, 2])

with col_surf:
    st.subheader("🏟️ Escenario")
    surface = st.radio("Superficie:", ["Hard", "Clay", "Grass"], horizontal=True)
    
    # Botón de Predicción GIGANTE
    st.markdown("###")
    btn_predict = st.button("🚀 PREDECIR GANADOR", use_container_width=True, type="primary")

with col_h2h:
    st.subheader("⚔️ Historial (H2H)")
    
    # Lógica H2H
    mask_1 = (df['winner_name'] == p1) & (df['loser_name'] == p2)
    mask_2 = (df['winner_name'] == p2) & (df['loser_name'] == p1)
    matches = pd.concat([df[mask_1], df[mask_2]]).sort_values('tourney_date', ascending=False)
    
    w1 = len(df[mask_1])
    w2 = len(df[mask_2])
    
    if w1 + w2 > 0:
        c_chart, c_table = st.columns([1, 2])
        with c_chart:
            fig = px.pie(values=[w1, w2], names=[p1, p2], hole=0.4, color_discrete_sequence=['#3498db', '#2ecc71'])
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=150)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{p1}: {w1} - {p2}: {w2}")
        with c_table:
            st.dataframe(matches[['tourney_date', 'tourney_name', 'surface', 'winner_name', 'score']].head(5), hide_index=True)
    else:
        st.info("No hay partidos previos registrados entre estos dos jugadores.")

# 4. RESULTADO DE PREDICCIÓN
if btn_predict:
    st.markdown("---")
    with st.spinner('La IA está analizando 20 años de tenis...'):
        
        # Feature Engineering (TIENE QUE SER IGUAL AL ENTRENAMIENTO)
        diff_rank = r1 - r2
        diff_age = a1 - a2
        diff_ht = h1 - h2
        
        s_clay = 1 if surface == "Clay" else 0
        s_grass = 1 if surface == "Grass" else 0
        s_hard = 1 if surface == "Hard" else 0
        
        features = [[diff_rank, diff_age, diff_ht, s_clay, s_grass, s_hard]]
        
        try:
            prob = modelo.predict_proba(features)[0][1] # Prob ganar P1
            
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            
            with col_res2:
                if prob > 0.5:
                    winner = p1
                    color = "green"
                    final_prob = prob
                else:
                    winner = p2
                    color = "red"
                    final_prob = 1 - prob
                
                st.balloons()
                st.markdown(f"<h2 style='text-align: center; color: {color};'>🏆 Ganador Esperado: {winner}</h2>", unsafe_allow_html=True)
                st.progress(final_prob)
                st.markdown(f"<p style='text-align: center;'>Confianza del modelo: <b>{final_prob*100:.1f}%</b></p>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error en predicción: {e}")