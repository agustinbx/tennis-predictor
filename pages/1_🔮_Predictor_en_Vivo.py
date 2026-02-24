import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(page_title="ATP Predictor 2026", page_icon="🎾", layout="wide")

st.title("🎾 ATP Prediction Pro 2026")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
Esta aplicación utiliza un modelo de **Inteligencia Artificial** entrenado con datos históricos (2000-2024) y actualizado con el **Ranking 2026**.
El sistema analiza:
* 📊 **Jerarquía Actual:** Ranking 2026, Edad y Altura.
* ⚔️ **Historial:** Enfrentamientos previos (H2H).
* 🧠 **Momentum:** Racha reciente y fatiga.
""")

st.write("---")

# CARGAR ARCHIVOS 
@st.cache_resource
def cargar_todo():
    # Detectar la carpeta principal del proyecto (Subimos un nivel desde /pages)
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_proyecto = os.path.dirname(ruta_script) 
    
    # --- LA BIFURCACIÓN: Definimos las dos carpetas ---
    ruta_prediccion = os.path.join(ruta_proyecto, "prediccion")
    ruta_scraping = os.path.join(ruta_proyecto, "scraping") 

    # Funciones auxiliares para buscar en la carpeta correcta
    def get_path_pred(archivo):
        return os.path.join(ruta_prediccion, archivo)
        
    def get_path_scrap(archivo):
        return os.path.join(ruta_scraping, archivo)

    try:
        # 🧠 1. MODELOS ESTÁTICOS (Leen de /prediccion)
        model_xgb = joblib.load(get_path_pred('modelo_xgboost_final.pkl'))
        model_log = joblib.load(get_path_pred('modelo_logistico_final.pkl'))
        scaler = joblib.load(get_path_pred('scaler_final.pkl'))
        stats_dict = joblib.load(get_path_pred('stats_superficie_v2.pkl'))
        
        # 🎾 2. DATOS VIVOS (Leen de /scraping)
        perfiles = joblib.load(get_path_scrap('perfiles_jugadores.pkl'))
        
    except FileNotFoundError as e:
        st.error(f"Faltan archivos fundamentales. Error técnico: {e}")
        st.stop()

    # 📊 3. HISTORIAL Y RANKING (Leen de /scraping)
    try:
        df_history = pd.read_csv(get_path_scrap("historialTenis.csv"), low_memory=False)
    except:
        df_history = pd.DataFrame() 
    
    try:
        df_rank_26 = pd.read_csv(get_path_scrap("ranking_2026.csv"))
        # (Asegúrate de que 'player_slug' exista en tu CSV de ranking, 
        # o cámbialo por 'player' / 'Nombre Completo' según como lo hayas dejado en tu scraper)
        ranking_2026_dict = dict(zip(df_rank_26['player_slug'], df_rank_26['rank']))
    except:
        ranking_2026_dict = {}

    return model_xgb, model_log, scaler, stats_dict, perfiles, df_history, ranking_2026_dict

# Desempaquetamos todo
model_xgb, model_log, scaler, stats_dict, perfiles, df_history, ranking_2026_dict = cargar_todo()


def get_skill(p, s): return stats_dict.get((p, s), 0.5)

def mostrar_historial_detallado(lista_partidos):
    if not lista_partidos:
        st.caption("No hay datos recientes.")
        return

    # Invertimos la lista para mostrar el MÁS RECIENTE arriba
    for partido in reversed(lista_partidos):
        resultado = partido['resultado']
        rival = partido['rival']
        score = partido['score']
        ronda = partido.get('ronda', '??')
        
        icono = "✅" if resultado == 'W' else "🔴"
        
        # Formato 
        st.markdown(f"{icono} **{ronda}**: vs {rival}")
        st.caption(f"Score: {score}")
        st.divider() # Línea separadora

# FUNCIÓN H2H 
def calcular_h2h(p1, p2):
    if df_history.empty: return 0, 0
    # Partidos donde ganó P1 contra P2
    wins1 = len(df_history[(df_history['winner_name'] == p1) & (df_history['loser_name'] == p2)])
    # Partidos donde ganó P2 contra P1
    wins2 = len(df_history[(df_history['winner_name'] == p2) & (df_history['loser_name'] == p1)])
    return wins1, wins2

def mostrar_racha_visual(lista_racha):
    if not lista_racha: return "Sin datos"
    # Convertimos 1 en Check verde y 0 en X roja
    iconos = ["✅" if x == 1 else "🔴" for x in lista_racha]
    return " ".join(iconos)

def grafico_radar(j1, j2, perfiles, stats_sup):
    
    d1 = perfiles.get(j1, {}); d2 = perfiles.get(j2, {})
    
    # --- 1. NORMALIZACIÓN DE ESTADÍSTICAS (Escala 0 a 1) ---
    
    # A. Aces (Asumimos que ~1000 aces es el nivel élite máximo)
    aces1 = d1.get('aces', 0); aces2 = d2.get('aces', 0)
    score_aces1 = min(1, aces1 / 1000)
    score_aces2 = min(1, aces2 / 1000)

    # B. Control / Dobles Faltas (INVERSO: 0 DF es 1.0, 400 DF es 0.0)
    df1 = d1.get('df', 0); df2 = d2.get('df', 0)
    score_df1 = max(0, 1 - (df1 / 400))
    score_df2 = max(0, 1 - (df2 / 400))

    # C. Potencia (1er Saque Ganado) - Rango de 60% a 85%
    srv1 = d1.get('serve_win', 65); srv2 = d2.get('serve_win', 65)
    score_srv1 = max(0, min(1, (srv1 - 60) / 25)) 
    score_srv2 = max(0, min(1, (srv2 - 60) / 25))

    # D. Mentalidad (Break Points Salvados) - Rango de 50% a 75%
    bp1 = d1.get('bp_saved', 60); bp2 = d2.get('bp_saved', 60)
    score_bp1 = max(0, min(1, (bp1 - 50) / 25))
    score_bp2 = max(0, min(1, (bp2 - 50) / 25))
    
    # E. Solidez (Juegos de Servicio Ganados) - Rango de 65% a 90%
    hold1 = d1.get('service_hold', 75); hold2 = d2.get('service_hold', 75)
    score_hold1 = max(0, min(1, (hold1 - 65) / 25))
    score_hold2 = max(0, min(1, (hold2 - 65) / 25))

    # F. Superficies (Rango 0% a 100%)
    hard1 = stats_sup.get((j1, 'Hard'), 0.5); hard2 = stats_sup.get((j2, 'Hard'), 0.5)
    clay1 = stats_sup.get((j1, 'Clay'), 0.5); clay2 = stats_sup.get((j2, 'Clay'), 0.5)
    grass1 = stats_sup.get((j1, 'Grass'), 0.5); grass2 = stats_sup.get((j2, 'Grass'), 0.5)

    # --- 2. ARMADO DEL GRÁFICO ---
    categories = [
        'Aces', 'Control (Pocas DF)', 'Potencia (1st Serve)', 
        'Mentalidad (BP Saved)', 'Solidez (Service Hold)', 
        'Hard', 'Clay', 'Grass', 'Aces'
    ]
    
    values_j1 = [score_aces1, score_df1, score_srv1, score_bp1, score_hold1, hard1, clay1, grass1, score_aces1]
    values_j2 = [score_aces2, score_df2, score_srv2, score_bp2, score_hold2, hard2, clay2, grass2, score_aces2]
    
    # Textos reales al pasar el mouse
    hover_j1 = [f"{aces1} Aces", f"{df1} D. Faltas", f"{srv1}%", f"{bp1}%", f"{hold1}%", f"{hard1:.0%} win", f"{clay1:.0%} win", f"{grass1:.0%} win", f"{aces1} Aces"]
    hover_j2 = [f"{aces2} Aces", f"{df2} D. Faltas", f"{srv2}%", f"{bp2}%", f"{hold2}%", f"{hard2:.0%} win", f"{clay2:.0%} win", f"{grass2:.0%} win", f"{aces2} Aces"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_j1, theta=categories, fill='toself', name=j1,
        hovertext=hover_j1, hoverinfo="text+name", line_color='#00CC96'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values_j2, theta=categories, fill='toself', name=j2,
        hovertext=hover_j2, hoverinfo="text+name", line_color='#AB63FA'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', showlegend=True, height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )
    return fig

# LISTA DE JUGADORES 
lista_jugadores = sorted(list(perfiles.keys()))

# FUNCIONES DE ACTUALIZACIÓN 
def actualizar_j1():
    nombre = st.session_state.sel_j1
    datos = perfiles[nombre]
    
    # Si el jugador está en el ranking 2026, usamos ese. Si no, el del perfil viejo.
    rank_real = ranking_2026_dict.get(nombre, int(datos['rank']))
    st.session_state.r1 = rank_real
    
    # Si el perfil dice 2024, le sumamos 2 años mentalmente o lo dejamos como viene en el perfil
    st.session_state.a1 = float(datos['age']) 
    
    st.session_state.h1 = int(datos['ht'])
    st.session_state.m1 = int(datos['momentum'] * 100)
    st.session_state.nac1 = datos['ioc']
    st.session_state.l5_1 = datos.get('last_5', [])

def actualizar_j2():
    nombre = st.session_state.sel_j2
    datos = perfiles[nombre]
    
    rank_real = ranking_2026_dict.get(nombre, int(datos['rank']))
    st.session_state.r2 = rank_real
    
    st.session_state.a2 = float(datos['age'])
    st.session_state.h2 = int(datos['ht'])
    st.session_state.m2 = int(datos['momentum'] * 100)
    st.session_state.nac2 = datos['ioc']
    st.session_state.l5_2 = datos.get('last_5', [])

# INTERFAZ
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # --- SELECTOR DE MODELO (NUEVO) ---
    st.subheader("🧠 Cerebro de la IA")
    modelo_seleccionado = st.radio(
        "Elige el algoritmo:",
        ["XGBoost (Recomendado)", "Regresión Logística"],
        captions=["Mayor precisión (72%)", "Más simple y clásico (69%)"]
    )
    
    # Asignamos el modelo activo según la elección
    if "XGBoost" in modelo_seleccionado:
        active_model = model_xgb
        st.info("Usando: **Árboles de Decisión Avanzados**")
    else:
        active_model = model_log
        st.info("Usando: **Estadística Lineal Clásica**")
        
    st.divider()
    
    superficie = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    pais_torneo = st.selectbox("País Sede", ["NEUTRAL", "ARG", "ESP", "FRA", "USA", "GBR", "AUS"])

col1, col2 = st.columns(2)

# ================= JUGADOR 1 =================
with col1:
    st.markdown("### 👤 Jugador 1")
    
    def_nom1 = "Carlos Alcaraz" if "Carlos Alcaraz" in lista_jugadores else lista_jugadores[0]
    
    # Inicialización la primera vez
    if 'nac1' not in st.session_state:
        d = perfiles[def_nom1]
        # Intentamos buscar ranking 2026 inicial
        st.session_state.r1 = ranking_2026_dict.get(def_nom1, int(d['rank']))
        st.session_state.a1 = float(d['age'])
        st.session_state.h1 = int(d['ht'])
        st.session_state.m1 = int(d['momentum'] * 100)
        st.session_state.nac1 = d['ioc']
        st.session_state.l5_1 = d.get('last_5', [])

    # Selector
    nombre1 = st.selectbox("Seleccionar:", lista_jugadores, index=lista_jugadores.index(def_nom1), key="sel_j1", on_change=actualizar_j1)
    
    # País
    nac1 = st.text_input("País", disabled=True, key="nac1")

    # RANKING Y PUNTOS JUNTOS
    puntos1 = perfiles.get(nombre1, {}).get('points', 0) # Extraemos los puntos actuales

    c_rank1, c_pts1 = st.columns(2)
    with c_rank1:
        r1 = st.number_input("Ranking (2026)", 1, 5000, key="r1")
    with c_pts1:
        st.metric(label="🏆 Puntos ATP", value=f"{int(puntos1):,}".replace(",", "."))
    
    
    # Inputs numéricos
    #r1 = st.number_input("Ranking (2026)", 1, 5000, key="r1") # Este tomará el valor actualizado
    a1 = st.number_input("Edad", 15.0, 50.0, step=0.5, key="a1")
    h1 = st.number_input("Altura", 150, 230, key="h1")
    
    st.markdown("##### ⚡ Estado")
    mom1 = st.slider("Momentum (%)", 0, 100, key="m1") / 100

    historial_j1 = st.session_state.get('l5_1', [])
    with st.expander("Ver últimos 5 partidos"):
        mostrar_historial_detallado(historial_j1)

    fat1 = st.number_input("Fatiga (min)", 0, 1000, 0, key="f1")

# ================= JUGADOR 2 =================
with col2:
    st.markdown("### 👤 Jugador 2")
    
    def_nom2 = "Novak Djokovic" if "Novak Djokovic" in lista_jugadores else lista_jugadores[1]
    
    if 'nac2' not in st.session_state:
        d = perfiles[def_nom2]
        st.session_state.r2 = ranking_2026_dict.get(def_nom2, int(d['rank']))
        st.session_state.a2 = float(d['age'])
        st.session_state.h2 = int(d['ht'])
        st.session_state.m2 = int(d['momentum'] * 100)
        st.session_state.nac2 = d['ioc']
        st.session_state.l5_2 = d.get('last_5', [])

    nombre2 = st.selectbox("Seleccionar:", lista_jugadores, index=lista_jugadores.index(def_nom2), key="sel_j2", on_change=actualizar_j2)
    
    nac2 = st.text_input("País", disabled=True, key="nac2")

    puntos2 = perfiles.get(nombre2, {}).get('points', 0) 
    
    c_rank2, c_pts2 = st.columns(2)
    with c_rank2:
        r2 = st.number_input("Ranking (2026)", 1, 5000, key="r2")
    with c_pts2:
        st.metric(label="🏆 Puntos ATP", value=f"{int(puntos2):,}".replace(",", "."))
    
    #r2 = st.number_input("Ranking (2026)", 1, 5000, key="r2")
    a2 = st.number_input("Edad", 15.0, 50.0, step=0.5, key="a2")
    h2 = st.number_input("Altura", 150, 230, key="h2")
    
    st.markdown("##### ⚡ Estado")
    mom2 = st.slider("Momentum (%)", 0, 100, key="m2") / 100

    historial_j2 = st.session_state.get('l5_2', [])
    with st.expander("Ver últimos 5 partidos"):
        mostrar_historial_detallado(historial_j2)

    fat2 = st.number_input("Fatiga (min)", 0, 1000, 0, key="f2")

# ================= SECCIÓN H2H =================
# --- H2H Y RADAR ---
st.divider()

# Columnas: H2H a la izquierda, Radar en el centro/derecha
c_h2h, c_radar = st.columns([1, 2])

with c_h2h:
    st.subheader("⚔️ Historial")
    wins_p1, wins_p2 = calcular_h2h(nombre1, nombre2)
    st.metric(f"Victorias {nombre1}", wins_p1)
    st.metric(f"Victorias {nombre2}", wins_p2)
    st.caption("Partidos previos registrados en la base de datos.")

with c_radar:
    st.subheader("🕸️ Análisis Técnico 360°")
    try:
        # Llamamos al gráfico pasando solo los perfiles y estadísticas de superficie
        fig_radar = grafico_radar(nombre1, nombre2, perfiles, stats_dict)
        st.plotly_chart(fig_radar, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo generar el radar: {e}")
        
st.divider()



# ================= PREDICCIÓN =================
st.divider()

boton_texto = f"🔮 PREDECIR con {modelo_seleccionado.split(' ')[0]}"

if st.button(boton_texto, type="primary", use_container_width=True):
    
    skill1 = get_skill(nombre1, superficie)
    skill2 = get_skill(nombre2, superficie)
    
    home1 = 1 if nac1 == pais_torneo else 0
    home2 = 1 if nac2 == pais_torneo else 0

    pts1 = st.session_state.get('p1_points', 0)
    pts2 = st.session_state.get('p2_points', 0)
    diff_rank_points = pts1 - pts2  # <--- ¡ESTO FALTABA!
    
    diff_h2h = wins_p1 - wins_p2

    # DataFrame de entrada
    input_data = pd.DataFrame([{
        'diff_rank': r2 - r1, # Ranking P2 - Ranking P1 (Si P1 es #1 y P2 es #10, la dif es 9, positivo para P1)
        'diff_rank_points': diff_rank_points,
        'diff_age': a1 - a2,
        'diff_ht': h1 - h2,
        'diff_skill': skill1 - skill2,
        'diff_home': home1 - home2,
        'diff_fatigue': fat1 - fat2,
        'diff_momentum': mom1 - mom2,
        'diff_h2h': diff_h2h
    }])
    
    try:
        input_scaled = scaler.transform(input_data)
        
        # USAMOS EL MODELO ACTIVO SELECCIONADO 
        prob = active_model.predict_proba(input_scaled)[0]
        prob_j1 = prob[1]
        
        st.divider()
        col_res_izq, col_res_der = st.columns([1, 3])
        
        with col_res_izq:
            if prob_j1 > 0.5:
                st.markdown("## 🎾")
                
        with col_res_der:
            if prob_j1 > 0.5:
                st.success(f"🏆 Ganador: **{nombre1}**")
                st.metric("Confianza", f"{prob_j1:.1%}", delta=f"Modelo: {modelo_seleccionado.split(' ')[0]}")
            else:
                st.error(f"🏆 Ganador: **{nombre2}**")
                st.metric("Confianza", f"{(1-prob_j1):.1%}", delta=f"Modelo: {modelo_seleccionado.split(' ')[0]}")
            
    except Exception as e:
        st.error(f"⚠️ Error en predicción: {e}")