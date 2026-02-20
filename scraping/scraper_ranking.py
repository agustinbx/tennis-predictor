import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

URL_RANKING = "https://www.atptour.com/en/rankings/singles?rankRange=1-400"
ARCHIVO_SALIDA = "ranking_2026.csv"

print("🏆 RE-ESCANEO DE RANKING (Modo Rastreador por Nombre)...")

options = uc.ChromeOptions()
options.add_argument("--start-maximized")
driver = uc.Chrome(options=options, version_main=144)

data_ranking = []

try:
    driver.get(URL_RANKING)
    print("⏳ Esperando carga de la página...")
    time.sleep(4)
    
    # --- 🛡️ EL ASESINO DE COOKIES ---
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

    # --- 📜 AUTO-SCROLL ---
    print("📜 Deslizando para despertar a la tabla...")
    for _ in range(6):
        driver.execute_script("window.scrollBy(0, 1500);")
        time.sleep(1.5)
        
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    tabla = soup.find('table', class_='mega-table')
    
    if tabla:
        filas = tabla.find('tbody').find_all('tr')
        print(f"✅ Encontrados {len(filas)} jugadores. Extrayendo datos...")
        
        for index, row in enumerate(filas):
            try:
                cols = row.find_all('td')
                
                # CHIVATO: Imprimir exactamente qué celdas ve el robot en el primer jugador
                if index == 0:
                    textos_celdas = [td.get_text(strip=True) for td in cols]
                    print(f"\n🔍 [DEBUG FILA 1] Esto es lo que lee el robot en cada celda:\n{textos_celdas}\n")

                # 1. Ranking (Extraemos el número de la primera columna)
                rank_limpio = re.sub(r'\D', '', cols[0].get_text(strip=True))
                rank = int(rank_limpio) if rank_limpio else 999

                # 2. Encontrar el Nombre y el índice de su columna
                index_nombre = -1
                full_link = ""
                nombre = ""
                
                for i, td in enumerate(cols):
                    link_tag = td.find('a', href=True)
                    if link_tag:
                        full_link = f"https://www.atptour.com{link_tag['href']}"
                        nombre = link_tag.get_text(strip=True)
                        index_nombre = i
                        break
                
                if index_nombre == -1: continue # Si no hay nombre, saltamos

                # 3. 🧠 EXTRAER PUNTOS (Buscando en las celdas a la derecha del nombre)
                puntos = 0
                # Escaneamos las siguientes 4 celdas después del nombre
                for td in cols[index_nombre+1 : index_nombre+5]:
                    texto_celda = td.get_text(strip=True).replace(',', '') # Quitamos la coma (11,520 -> 11520)
                    if texto_celda.isdigit():
                        valor = int(texto_celda)
                        # Edad y Torneos son pequeños. Los puntos son el valor más grande.
                        if valor > puntos:
                            puntos = valor

                # Guardamos
                data_ranking.append({
                    'player': nombre,
                    'rank': rank,
                    'points': puntos,
                    'url_perfil': full_link
                })
                
                # Imprimir solo los 3 primeros
                if index < 3:
                    print(f"   -> [OK] {nombre} | Rank: {rank} | Puntos: {puntos}")

            except Exception as e:
                continue
            
    # --- GUARDADO ---
    if len(data_ranking) > 0:
        pd.DataFrame(data_ranking).to_csv(ARCHIVO_SALIDA, index=False)
        print(f"\n🎉 ¡Ganamos! Archivo '{ARCHIVO_SALIDA}' generado correctamente.")
    else:
        print("\n❌ CRÍTICO: No se extrajo nada.")

except Exception as e:
    print(f"❌ Error General: {e}")
finally:
    driver.quit()