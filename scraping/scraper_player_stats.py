import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chrome_helper import create_chrome_driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import re

print("[DETECTIVE] INICIANDO ROBO DE ESTADISTICAS (MODO LECTOR DE TEXTO VISIBLE)...")

try:
    df_rank = pd.read_csv("ranking_actual_2026.csv")
except FileNotFoundError:
    print("[FAIL] Falta el archivo 'ranking_actual_2026.csv'")
    exit()

driver = create_chrome_driver()

stats_data = []

def buscar_numero_en_texto(texto, palabra_clave):
    # Esta magia busca la palabra clave, luego espacios/saltos de línea, y captura el número
    # Ej: Si lee "Aces \n 3,456", atrapa el 3456.
    patron = re.escape(palabra_clave) + r'\s*([\d\,\.]+)\s*\%?'
    coincidencia = re.search(patron, texto, re.IGNORECASE)
    if coincidencia:
        numero_limpio = coincidencia.group(1).replace(',', '')
        try:
            return float(numero_limpio)
        except:
            return 0.0
    return 0.0

for index, row in df_rank.iterrows():
    nombre = row['player']
    url_stats = row['url_perfil'].replace("overview", "player-stats")
    
    print(f"[{index+1}] Escaneando a {nombre}...")
    
    try:
        driver.get(url_stats)
        time.sleep(4.0) # Damos tiempo a que carguen los números
        
        # --- [SHIELD] EL ASESINO DE COOKIES ---
        try:
            btn_cookies = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            btn_cookies.click()
            time.sleep(1)
        except:
            driver.execute_script("""
                var cookieBanner = document.getElementById('onetrust-consent-sdk');
                if(cookieBanner) cookieBanner.remove();
                document.body.style.overflow = 'scroll';
            """)

        # Hacemos un poco de scroll para asegurar que la web "despierte" los números
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(1.0)

        # [BRAIN] LA MAGIA: Extraemos TODO el texto visible de la página, ignorando el HTML
        texto_pantalla = driver.find_element(By.TAG_NAME, "body").text
        
        # DEBUG: Guardar pruebas del primer escaneo
        if index == 0:
            driver.save_screenshot("pantalla_robot.png")
            with open("texto_leido.txt", "w", encoding="utf-8") as f:
                f.write(texto_pantalla)
            print("   [DOC] Texto visible guardado en 'texto_leido.txt' para depuración.")
        
        # Buscamos las métricas leyendo el texto puro
        aces = buscar_numero_en_texto(texto_pantalla, "Aces")
        dfaults = buscar_numero_en_texto(texto_pantalla, "Double Faults")
        
        # A veces la ATP lo escribe diferente, probamos variantes
        srv_win = buscar_numero_en_texto(texto_pantalla, "1st Serve Points Won")
        if srv_win == 0.0: srv_win = buscar_numero_en_texto(texto_pantalla, "1st Serve Won")
            
        bp_saved = buscar_numero_en_texto(texto_pantalla, "Break Points Saved")
        srv_games = buscar_numero_en_texto(texto_pantalla, "Service Games Won")
        
        print(f"   -> Aces: {aces} | 1er Saque: {srv_win}% | BP Salvados: {bp_saved}%")
        
        stats_data.append({
            'player': nombre,
            'aces_avg': aces,
            'df_avg': dfaults,
            'serve_win_pct': srv_win,
            'bp_saved_pct': bp_saved,
            'service_hold_pct': srv_games
        })
        
    except Exception as e:
        print(f"   [WARN] Error de conexión con {nombre}: {e}")

df_stats = pd.DataFrame(stats_data)
df_stats.to_csv("estadisticas_jugadores_avanzadas.csv", index=False)
driver.quit()

print("\n[OK] ¡Escaneo finalizado!")