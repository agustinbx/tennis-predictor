import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Laboratorio IA", page_icon="🧠", layout="wide")

st.title("🧠 Laboratorio de Datos")
st.markdown("### ⚔️ La Batalla de los Modelos")

st.info("Para elegir el 'cerebro' de esta aplicación, pusimos a competir a 3 algoritmos diferentes. Aquí te explicamos los resultados de forma sencilla.")

# --- 1. CARGA DE DATOS (Igual que antes) ---
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_script)
# ruta_raiz = os.path.join(ruta_raiz, "archivos_modelo") # <--- Descomentar si usas la carpeta

try:
    path_csv = os.path.join(ruta_raiz, "resultados_comparacion.csv")
    df_res = pd.read_csv(path_csv)
    df_res['Accuracy %'] = (df_res['Accuracy'] * 100).round(2)
    df_res = df_res.sort_values('Accuracy', ascending=False)
except FileNotFoundError:
    st.error("⚠️ Faltan los resultados. Ejecuta primero 'comparar_modelos.py'.")
    st.stop()

# --- 2. EL GANADOR (GRÁFICO) ---
col_graf, col_tabla = st.columns([2, 1])

with col_graf:
    st.subheader("🏆 ¿Quién acertó más?")
    
    # Colores: Dorado para el ganador, Gris para el resto
    colors = ['#FFD700' if x == df_res.iloc[0]['Modelo'] else '#E5E7EB' for x in df_res['Modelo']]
    
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

# --- 3. EXPLICACIÓN PARA HUMANOS (SIN MATEMÁTICA) ---
st.header("🤓 ¿Cómo 'piensa' cada modelo?")
st.markdown("Imagina que tienes que adivinar quién gana un partido. Estos tres modelos son como tres tipos de personas diferentes intentando adivinar:")

c1, c2, c3 = st.columns(3)

# --- A. REGRESIÓN LOGÍSTICA ---
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

# --- B. RANDOM FOREST ---
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

# --- C. XGBOOST ---
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

# --- 4. ANÁLISIS REAL DE VARIABLES (NUEVO) ---
st.subheader("👀 ¿Qué es lo que más mira la IA?")
st.write("Analizamos matemáticamente qué peso le da el modelo **XGBoost** a cada dato. Aquí están los porcentajes reales:")

# Cargar el archivo de importancia real
try:
    path_imp = os.path.join(ruta_raiz, "importancia_real.csv")
    df_imp = pd.read_csv(path_imp)
    
    # Filtrar solo para el modelo ganador (XGBoost)
    df_xgboost = df_imp[df_imp['Modelo'] == 'XGBoost'].sort_values('Importancia', ascending=False)
    
    # Diccionario para nombres bonitos en el gráfico
    nombres_bonitos = {
        'diff_rank_points': 'Diferencia de Puntos',
        'diff_rank': 'Ranking ATP',
        'diff_h2h': 'Historial (H2H)',
        'diff_age': 'Edad',
        'diff_skill': 'Efectividad Superficie',
        'diff_fatigue': 'Fatiga',
        'diff_momentum': 'Racha (Momentum)',
        'diff_ht': 'Altura',
        'diff_home': 'Localía'
    }
    df_xgboost['Nombre'] = df_xgboost['Variable'].map(nombres_bonitos)
    
    # Gráfico de Barras Horizontal
    fig_imp = px.bar(
        df_xgboost, 
        x='Importancia', 
        y='Nombre', 
        orientation='h',
        text_auto='.1f', # Muestra el valor con 1 decimal
        title="Impacto de cada variable en la decisión final (%)",
        color='Importancia',
        color_continuous_scale='Viridis' # Colores profesionales
    )
    
    # Invertir eje Y para que el más importante salga arriba
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Mensaje inteligente según el dato más alto
    top_var = df_xgboost.iloc[0]['Nombre']
    st.info(f"💡 **Conclusión:** El modelo confirma que **{top_var}** es el factor más determinante para predecir al ganador hoy en día.")

except FileNotFoundError:
    st.warning("⚠️ Aún no has generado el análisis de variables. Ejecuta 'comparar_modelos.py' de nuevo.")