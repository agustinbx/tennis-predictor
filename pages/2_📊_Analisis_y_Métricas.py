"""
Página de análisis y métricas de modelos.
Muestra comparación de modelos y feature importance.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from streamlit_app.utils import apply_style, load_comparison_results, load_feature_importance
except ImportError:
    # Fallback
    PROJECT_ROOT = Path(__file__).parent.parent
    
    def apply_style():
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)
    
    def load_comparison_results():
        path = PROJECT_ROOT / "resultados_comparacion.csv"
        if path.exists():
            return pd.read_csv(path)
        return None
    
    def load_feature_importance():
        path = PROJECT_ROOT / "importancia_real.csv"
        if path.exists():
            return pd.read_csv(path)
        return None


st.set_page_config(page_title="Laboratorio IA", page_icon="🧠", layout="wide")

st.title("🧠 Laboratorio de Datos")

apply_style()

st.markdown("### ⚔️ La Batalla de los Modelos")
st.info("Para elegir el 'cerebro' de esta aplicación, pusimos a competir a 3 algoritmos diferentes. Aquí te explicamos los resultados de forma sencilla.")

# Cargar datos
df_res = load_comparison_results()

if df_res is None:
    st.error("⚠️ Faltan los resultados. Ejecuta primero el pipeline de entrenamiento avanzado.")
    st.stop()

df_res['Accuracy %'] = (df_res['Accuracy'] * 100).round(2)
df_res = df_res.sort_values('Accuracy', ascending=False)

# Gráfico del modelo ganador
col_graf, col_tabla = st.columns([2, 1])

with col_graf:
    st.subheader("🏆 ¿Quién acertó más?")
    
    fig = px.bar(
        df_res,
        x='Accuracy %',
        y='Modelo',
        orientation='h',
        text='Accuracy %',
        title="Porcentaje de Acierto en Partidos Nuevos",
    )
    fig.update_traces(marker_color='#2563EB', textposition='outside')
    fig.update_layout(xaxis_range=[60, 80], xaxis_title="Porcentaje de Acierto")
    st.plotly_chart(fig, use_container_width=True)

with col_tabla:
    st.subheader("🥇 El Ganador")
    ganador = df_res.iloc[0]
    st.success(f"El modelo **{ganador['Modelo']}** fue el mejor.")
    st.markdown(f"""
    Logró predecir correctamente el **{ganador['Accuracy %']}%** de los partidos de prueba.
    
    Por eso, es el motor elegido para esta App.
    """)

st.divider()

# Explicación de modelos
st.header("🤓 ¿Cómo 'piensa' cada modelo?")
st.markdown("Imagina que tienes que adivinar quién gana un partido. Estos tres modelos son como tres tipos de personas diferentes intentando adivinar:")

c1, c2, c3 = st.columns(3)

with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/2645/2645897.png", width=80)
    st.subheader("1. La Balanza")
    st.caption("(Regresión Logística)")
    st.write("""
    **¿Cómo funciona?**
    Funciona sumando y restando puntos, como una balanza antigua.
    
    * *"Si el Ranking es bueno, suma 10 puntos."*
    * *"Si está cansado, resta 5 puntos."*
    
    Si la suma total es positiva, dice que **GANA**. Si es negativa, dice que **PIERDE**.
    
    **Veredicto:** Es rápido y lógico, pero a veces el tenis es más complejo que una simple suma.
    """)

with c2:
    st.image("https://cdn-icons-png.flaticon.com/512/1534/1534938.png", width=80)
    st.subheader("2. La Democracia")
    st.caption("(Random Forest)")
    st.write("""
    **¿Cómo funciona?**
    En lugar de decidir solo, crea **100 pequeños expertos** (árboles) y les hace votar.
    
    * Experto 1: *"Gana Nadal porque es zurdo".*
    * Experto 2: *"Gana Federer porque juega en pasto".*
    * Experto 3: *"Gana Nadal por el historial".*
    
    Al final, **gana la mayoría**.
    
    **Veredicto:** Muy seguro y estable, pero a veces le cuesta ver patrones sutiles.
    """)

with c3:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=80)
    st.subheader("3. El Perfeccionista")
    st.caption("(XGBoost - El Campeón)")
    st.write("""
    **¿Cómo funciona?**
    Es como un alumno que aprende de sus errores paso a paso.
    
    1. Hace una predicción inicial.
    2. Mira en qué partidos se equivocó.
    3. Crea un nuevo "mini-modelo" enfocado **exclusivamente** en corregir esos errores difíciles.
    4. Repite esto cientos de veces hasta pulir el resultado.
    
    **Veredicto:** Es el más inteligente porque aprende de sus propias fallas. Por eso ganó.
    """)

st.divider()

# Análisis de variables
st.subheader("👀 ¿Qué es lo que más mira la IA?")
st.write("Analizamos matemáticamente qué peso le da el modelo **XGBoost** a cada dato. Aquí están los porcentajes reales:")

df_imp = load_feature_importance()

if df_imp is not None:
    # Filtrar solo XGBoost
    df_xgboost = df_imp[df_imp['Modelo'] == 'XGBoost'].sort_values('Importancia', ascending=False)
    
    # Nombres bonitos
    nombres_bonitos = {
        'diff_elo': 'ELO Rating',
        'diff_rank_points': 'Puntos ATP',
        'diff_rank': 'Ranking ATP',
        'diff_h2h': 'Historial (H2H)',
        'diff_age': 'Edad',
        'diff_skill': 'Efectividad Superficie',
        'diff_fatigue': 'Fatiga',
        'diff_momentum': 'Racha (Momentum)',
        'diff_clutch': 'Factor Clutch',
        'diff_ht': 'Altura',
        'diff_home': 'Localía'
    }
    df_xgboost['Nombre'] = df_xgboost['Variable'].map(nombres_bonitos)
    
    fig_imp = px.bar(
        df_xgboost,
        x='Importancia',
        y='Nombre',
        orientation='h',
        text_auto='.1f',
        title="Impacto de cada variable en la decisión final (%)",
        color='Importancia',
        color_continuous_scale='Viridis'
    )
    
    fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
    
    top_var = df_xgboost.iloc[0]['Nombre']
    st.info(f"💡 **Conclusión:** El modelo confirma que **{top_var}** es el factor más determinante para predecir al ganador hoy en día.")
else:
    st.warning("⚠️ Aún no has generado el análisis de variables. Ejecuta el pipeline de entrenamiento avanzado de nuevo.")
