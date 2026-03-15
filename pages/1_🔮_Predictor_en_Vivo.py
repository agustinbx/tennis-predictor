import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="ATP Predictor 2026", page_icon="🎾", layout="wide")

st.title("🎾 ATP Prediction Pro 2026")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} /* Oculta los 3 puntitos de arriba a la derecha */
            footer {visibility: hidden;} /* Oculta el "Made with Streamlit" de abajo */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
Esta aplicación utiliza un modelo de **Inteligencia Artificial** servido mediante **FastAPI**, entrenado con datos históricos (2000-2024) y actualizado con el **Ranking 2026**.
El sistema analiza:
* 📊 **Jerarquía Actual:** Ranking 2026, Edad y Altura.
* ⚔️ **Historial:** Enfrentamientos previos (H2H).
* 🧠 **Momentum:** Racha reciente y fatiga.
""")

st.write("---")

API_URL = "http://localhost:8002"

# --- CONSUMO DE LA API ---

@st.cache_data
def obtener_jugadores():
    try:
        response = requests.get(f"{API_URL}/players/")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"No se pudo conectar a la API ({API_URL}). Asegúrate de que FastAPI esté corriendo.")
        return []

def obtener_perfil(nombre):
    try:
        res = requests.get(f"{API_URL}/players/{nombre}")
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return {}

def obtener_skill(nombre, superficie):
    try:
        res = requests.get(f"{API_URL}/stats/{nombre}/{superficie}")
        if res.status_code == 200:
            return res.json()["win_rate"]
    except:
        pass
    return 0.5


lista_jugadores = obtener_jugadores()

if not lista_jugadores:
    st.warning("⚠️ Esperando a la API...")
    st.stop()
    
# FUNCIONES DE UI (Graficos, historiales, etc)

def mostrar_historial_detallado(lista_partidos):
    if not lista_partidos:
        st.caption("No hay datos recientes.")
        return
    for partido in reversed(lista_partidos):
        resultado = partido.get('resultado', '?')
        rival = partido.get('rival', '?')
        score = partido.get('score', '')
        ronda = partido.get('ronda', '??')
        icono = "✅" if resultado == 'W' else "🔴"
        st.markdown(f"{icono} **{ronda}**: vs {rival}")
        st.caption(f"Score: {score}")
        st.divider()

def grafico_radar(d1, d2, j1, j2, skill_hard1, skill_hard2, skill_clay1, skill_clay2, skill_grass1, skill_grass2):
    # A. Aces 
    aces1 = d1.get('aces') or 0; aces2 = d2.get('aces') or 0
    score_aces1 = min(1, aces1 / 1000); score_aces2 = min(1, aces2 / 1000)

    # B. Control / Dobles Faltas 
    df1 = d1.get('df') or  0; df2 = d2.get('df') or 0
    score_df1 = max(0, 1 - (df1 / 400)); score_df2 = max(0, 1 - (df2 / 400))

    # C. Potencia 
    srv1 = d1.get('serve_win') or 65; srv2 = d2.get('serve_win') or 65
    score_srv1 = max(0, min(1, (srv1 - 60) / 25)); score_srv2 = max(0, min(1, (srv2 - 60) / 25))

    # D. Mentalidad 
    bp1 = d1.get('bp_saved') or 60; bp2 = d2.get('bp_saved') or 60
    score_bp1 = max(0, min(1, (bp1 - 50) / 25)); score_bp2 = max(0, min(1, (bp2 - 50) / 25))
    
    # E. Solidez 
    hold1 = d1.get('service_hold') or 75; hold2 = d2.get('service_hold') or 75
    score_hold1 = max(0, min(1, (hold1 - 65) / 25)); score_hold2 = max(0, min(1, (hold2 - 65) / 25))

    # --- 2. ARMADO DEL GRÁFICO ---
    categories = ['Aces', 'Control', 'Potencia (1st)', 'Mentalidad (BP)', 'Solidez (Serve)', 'Hard', 'Clay', 'Grass', 'Aces']
    
    values_j1 = [score_aces1, score_df1, score_srv1, score_bp1, score_hold1, skill_hard1, skill_clay1, skill_grass1, score_aces1]
    values_j2 = [score_aces2, score_df2, score_srv2, score_bp2, score_hold2, skill_hard2, skill_clay2, skill_grass2, score_aces2]
    
    hover_j1 = [f"{aces1} Aces", f"{df1} D.F.", f"{srv1}%", f"{bp1}%", f"{hold1}%", f"{skill_hard1:.0%} w", f"{skill_clay1:.0%} w", f"{skill_grass1:.0%} w", f"{aces1} Aces"]
    hover_j2 = [f"{aces2} Aces", f"{df2} D.F.", f"{srv2}%", f"{bp2}%", f"{hold2}%", f"{skill_hard2:.0%} w", f"{skill_clay2:.0%} w", f"{skill_grass2:.0%} w", f"{aces2} Aces"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_j1, theta=categories, fill='toself', name=j1, hovertext=hover_j1, hoverinfo="text+name", line_color='#00CC96'))
    fig.add_trace(go.Scatterpolar(r=values_j2, theta=categories, fill='toself', name=j2, hovertext=hover_j2, hoverinfo="text+name", line_color='#AB63FA'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'),
                      paper_bgcolor='rgba(0,0,0,0)', showlegend=True, height=450,
                      legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    return fig

# INTERFAZ
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("🧠 Cerebro de la IA")
    modelo_seleccionado = st.radio(
        "Elige el algoritmo remoto:",
        ["XGBoost (Recomendado)", "Regresión Logística"],
        captions=["Mayor precisión (72%)", "Más simple y clásico (69%)"]
    )
    
    st.divider()
    superficie = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    pais_torneo = st.selectbox("País Sede", ["NEUTRAL", "ARG", "ESP", "FRA", "USA", "GBR", "AUS"])

col1, col2 = st.columns(2)

# ================= JUGADOR 1 =================
with col1:
    st.markdown("### 👤 Jugador 1")
    def_nom1 = "Carlos Alcaraz" if "Carlos Alcaraz" in lista_jugadores else lista_jugadores[0]
    nombre1 = st.selectbox("Seleccionar:", lista_jugadores, index=lista_jugadores.index(def_nom1), key="sel_j1")
    
    perfil1 = obtener_perfil(nombre1)
    
    nac1 = st.text_input("País", value=perfil1.get('nacionalidad', ''), disabled=True, key="nac1")

    puntos1 = perfil1.get('puntos') or 0
    c_rank1, c_pts1 = st.columns(2)
    with c_rank1:
        r1 = st.number_input("Ranking (2026)", 1, 5000, value=perfil1.get('ranking') or 9999, key="r1")
    with c_pts1:
        st.metric(label="🏆 Puntos ATP", value=f"{int(puntos1):,}".replace(",", "."))
    
    a1 = st.number_input("Edad", 15.0, 50.0, step=0.5, value=float(perfil1.get('edad') or 25.0), key="a1")
    h1 = st.number_input("Altura", 150, 230, value=int(perfil1.get('altura') or 180), key="h1")
    
    st.markdown("##### ⚡ Estado")
    m_val1 = perfil1.get('momentum') or 0.5
    mom1 = st.slider("Momentum (%)", 0, 100, value=int(m_val1 * 100), key="m1") / 100

    historial_j1 = perfil1.get('last_5', [])
    with st.expander("Ver últimos 5 partidos"):
        mostrar_historial_detallado(historial_j1)

    fat1 = st.number_input("Fatiga (min)", 0, 1000, 0, key="f1")
    h2h_1 = st.number_input(f"H2H (Victorias de {nombre1})", 0, 50, 0, key="h2h1")

# ================= JUGADOR 2 =================
with col2:
    st.markdown("### 👤 Jugador 2")
    def_nom2 = "Novak Djokovic" if "Novak Djokovic" in lista_jugadores else lista_jugadores[1]
    nombre2 = st.selectbox("Seleccionar:", lista_jugadores, index=lista_jugadores.index(def_nom2), key="sel_j2")
    
    perfil2 = obtener_perfil(nombre2)
    
    nac2 = st.text_input("País", value=perfil2.get('nacionalidad', ''), disabled=True, key="nac2")

    puntos2 = perfil2.get('puntos') or 0 
    c_rank2, c_pts2 = st.columns(2)
    with c_rank2:
        r2 = st.number_input("Ranking (2026)", 1, 5000, value=perfil2.get('ranking') or 9999, key="r2")
    with c_pts2:
        st.metric(label="🏆 Puntos ATP", value=f"{int(puntos2):,}".replace(",", "."))
    
    a2 = st.number_input("Edad", 15.0, 50.0, step=0.5, value=float(perfil2.get('edad') or 25.0), key="a2")
    h2 = st.number_input("Altura", 150, 230, value=int(perfil2.get('altura') or 180), key="h2")
    
    st.markdown("##### ⚡ Estado")
    m_val2 = perfil2.get('momentum') or 0.5
    mom2 = st.slider("Momentum (%)", 0, 100, value=int(m_val2 * 100), key="m2") / 100

    historial_j2 = perfil2.get('last_5', [])
    with st.expander("Ver últimos 5 partidos"):
        mostrar_historial_detallado(historial_j2)

    fat2 = st.number_input("Fatiga (min)", 0, 1000, 0, key="f2")
    h2h_2 = st.number_input(f"H2H (Victorias de {nombre2})", 0, 50, 0, key="h2h2")

st.divider()

# Columnas: Radar en el centro/derecha
st.subheader("🕸️ Análisis Técnico 360°")
try:
    s_h1 = obtener_skill(nombre1, "Hard"); s_c1 = obtener_skill(nombre1, "Clay"); s_g1 = obtener_skill(nombre1, "Grass")
    s_h2 = obtener_skill(nombre2, "Hard"); s_c2 = obtener_skill(nombre2, "Clay"); s_g2 = obtener_skill(nombre2, "Grass")
    
    fig_radar = grafico_radar(perfil1, perfil2, nombre1, nombre2, s_h1, s_h2, s_c1, s_c2, s_g1, s_g2)
    st.plotly_chart(fig_radar, use_container_width=True)
except Exception as e:
    st.warning(f"No se pudo generar el radar: {e}")
        
st.divider()

# ================= PREDICCIÓN CON API =================

boton_texto = f"🔮 SOLICITAR PREDICCIÓN AL SERVIDOR"

if st.button(boton_texto, type="primary", use_container_width=True):
    
    payload = {
        "jugador_1": nombre1,
        "jugador_2": nombre2,
        "superficie": superficie,
        "pais_torneo": pais_torneo,
        "modelo": "XGBoost" if "XGBoost" in modelo_seleccionado else "Logistic Regression",
        "fatiga_1": fat1,
        "fatiga_2": fat2,
        "h2h_1": h2h_1,
        "h2h_2": h2h_2
    }
    
    try:
        with st.spinner("Conectando con la API..."):
            response = requests.post(f"{API_URL}/predict", json=payload)
            
        if response.status_code == 200:
            resultado = response.json()
            
            st.divider()
            col_res_izq, col_res_der = st.columns([1, 3])
            
            with col_res_izq:
                st.markdown("## 🎾")
                    
            with col_res_der:
                if resultado["ganador"] == nombre1:
                    st.success(f"🏆 Ganador estimado: **{nombre1}**")
                else:
                    st.error(f"🏆 Ganador estimado: **{nombre2}**")
                    
                st.metric("Confianza de la IA", f"{resultado['confianza']:.1%}", delta=f"Modelo Web: {resultado['modelo_utilizado']}")
        else:
            st.error(f"Error en la API: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error(f"❌ No se pudo conectar a la API en {API_URL}. ¿Está encendido el servidor FastAPI?")