import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="ATP Predictor 2026", page_icon="🎾", layout="wide")

st.title("🎾 ATP Prediction Pro 2026")

st.markdown("""
Esta aplicación utiliza un modelo de **Inteligencia Artificial** entrenado con datos históricos (2000-2024) y actualizado con el **Ranking 2026**.
El sistema analiza:
* 📊 **Jerarquía Actual:** Ranking 2026, Edad y Altura.
* ⚔️ **Historial:** Enfrentamientos previos (H2H).
* 🧠 **Momentum:** Racha reciente y fatiga.
""")

st.write("---")

# --- 1. CARGAR ARCHIVOS ---
@st.cache_resource
def cargar_todo():
    # Detectar rutas
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_script)
    
    ruta_raiz = os.path.join(ruta_raiz, "prediccion") # <--- Descomentar si usas carpeta ordenada

    def get_path(archivo):
        return os.path.join(ruta_raiz, archivo)

    try:
        # Cargamos AMBOS modelos
        model_xgb = joblib.load(get_path('modelo_xgboost_final.pkl'))
        model_log = joblib.load(get_path('modelo_logistico_final.pkl'))
        
        # El scaler y los datos son compartidos (asumiendo que entrenaste ambos con las mismas variables)
        scaler = joblib.load(get_path('scaler_final.pkl'))
        stats_dict = joblib.load(get_path('stats_superficie_v2.pkl'))
        perfiles = joblib.load(get_path('perfiles_jugadores.pkl'))
        
    except FileNotFoundError as e:
        st.error(f"⚠️ Faltan archivos. Asegúrate de tener 'modelo_xgboost_final.pkl' y 'modelo_logistico_final.pkl'. Error: {e}")
        st.stop()

    try:
        df_history = pd.read_csv(get_path("historial_tenis.csv"))
    except:
        df_history = pd.DataFrame() 
    
    try:
        df_rank_26 = pd.read_csv(get_path("ranking_actual_2026.csv"))
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
        
        # Color del icono
        icono = "✅" if resultado == 'W' else "🔴"
        
        # Formato bonito: "✅ vs Nadal (6-4 6-3)"
        st.markdown(f"{icono} **{ronda}**: vs {rival}")
        st.caption(f"Score: {score}")
        st.divider() # Línea separadora

# --- FUNCIÓN H2H ---
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

# --- LISTA DE JUGADORES ---
lista_jugadores = sorted(list(perfiles.keys()))

# --- FUNCIONES DE ACTUALIZACIÓN (MODIFICADAS PARA 2026) ---
def actualizar_j1():
    nombre = st.session_state.sel_j1
    datos = perfiles[nombre]
    
    # 1. RANKING: Prioridad al archivo 2026
    # Si el jugador está en el ranking 2026, usamos ese. Si no, el del perfil viejo.
    rank_real = ranking_2026_dict.get(nombre, int(datos['rank']))
    st.session_state.r1 = rank_real
    
    # 2. EDAD: Ajuste simple si el perfil es viejo (opcional, pero recomendado)
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

# --- BARRA LATERAL ---
#with st.sidebar:
#    st.header("Configuración")
#    superficie = st.selectbox("Superficie", ["Clay", "Hard", "Grass"])
#    pais_torneo = st.selectbox("País Sede", ["NEUTRAL", "ARG", "ESP", "FRA", "USA", "GBR", "AUS"])

#col1, col2 = st.columns(2)

# --- INTERFAZ ---
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
    
    # Inputs numéricos
    r1 = st.number_input("Ranking (2026)", 1, 5000, key="r1") # Este tomará el valor actualizado
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
    
    r2 = st.number_input("Ranking (2026)", 1, 5000, key="r2")
    a2 = st.number_input("Edad", 15.0, 50.0, step=0.5, key="a2")
    h2 = st.number_input("Altura", 150, 230, key="h2")
    
    st.markdown("##### ⚡ Estado")
    mom2 = st.slider("Momentum (%)", 0, 100, key="m2") / 100

    historial_j2 = st.session_state.get('l5_2', [])
    with st.expander("Ver últimos 5 partidos"):
        mostrar_historial_detallado(historial_j2)

    fat2 = st.number_input("Fatiga (min)", 0, 1000, 0, key="f2")

# ================= SECCIÓN H2H =================
st.divider()
st.subheader("⚔️ Historial (Head to Head)")

wins_p1, wins_p2 = calcular_h2h(nombre1, nombre2)

c_h1, c_h2, c_h3 = st.columns([1, 2, 1])
with c_h1:
    st.metric(label=f"Victorias {nombre1}", value=wins_p1)
with c_h2:
    total_matches = wins_p1 + wins_p2
    if total_matches > 0:
        st.progress(wins_p1 / total_matches)
    else:
        st.progress(0.5)
    st.caption("Distribución de victorias históricas")
with c_h3:
    st.metric(label=f"Victorias {nombre2}", value=wins_p2)

# ================= PREDICCIÓN =================
st.divider()

# --- PREDICCIÓN ---
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
        
        # USAMOS EL MODELO ACTIVO SELECCIONADO EN EL SIDEBAR
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