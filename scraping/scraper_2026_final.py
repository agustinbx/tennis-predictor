import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chrome_helper import create_chrome_driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random

import os

ruta_script = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_ENTRADA = os.path.join(ruta_script, "atp_torneos_2026_final.csv")
ARCHIVO_SALIDA = os.path.join(ruta_script, "atp_matches_2026_indetectable.csv")
ANIO = 2026

print(f"[NINJA] INICIANDO MODO INDETECTABLE ({ANIO})...")

# 1. PREPARAR DATOS Y LISTAR TORNEOS COMPLETADOS
try:
    df = pd.read_csv(ARCHIVO_ENTRADA)
    urls = []
    names = []
    print("[LINK] Procesando enlaces...")
    for link in df['Link_Resultados']:
        parts = link.strip().split('/')
        try:
            if 'tournaments' in parts:
                idx = parts.index('tournaments')
                nombre = parts[idx+1]
                id_t = parts[idx+2]
            elif 'archive' in parts:
                idx = parts.index('archive')
                nombre = parts[idx+1]
                id_t = parts[idx+2]
            else:
                continue # Saltamos links raros
            
            # Usamos EN (Inglés) para mayor estabilidad
            new_link = f"https://www.atptour.com/en/scores/archive/{nombre}/{id_t}/{ANIO}/results"
            urls.append(new_link)
            names.append(nombre)
        except:
            pass
except Exception as e:
    print(f"[FAIL] Error leyendo CSV origen: {e}")
    exit()

# --- NUEVO: FILTRO DE TORNEOS YA DESCARGADOS ---
torneos_descargados = set()
try:
    df_existente = pd.read_csv(ARCHIVO_SALIDA)
    torneos_descargados = set(df_existente['tourney_name'].unique())
    print(f"[OK] Se encontraron {len(torneos_descargados)} torneos ya descargados en la base de datos local.")
except FileNotFoundError:
    print("ℹ No se encontró historial previo de 2026. Se creará desde cero.")

# Filtrar listas
urls_nuevas = []
names_nuevos = []
for u, n in zip(urls, names):
    if n not in torneos_descargados:
        urls_nuevas.append(u)
        names_nuevos.append(n)
        
print(f"[SKIP] Torneos a descargar HOY: {len(urls_nuevas)} (Omitiendo {len(urls) - len(urls_nuevas)} ya guardados)")

if not urls_nuevas:
    print("[START] ¡Todo está actualizado! No hay torneos nuevos para descargar.")
    exit()

# 2. INICIAR NAVEGADOR INDETECTABLE
# OJO: Esto abre una ventana de Chrome que NO dice "Chrome está siendo controlado..."

print("[START] Lanzando Chrome parcheado (puede tardar unos segundos)...")
driver = create_chrome_driver()

# 3. BUCLE DE TORNEOS
for i, url in enumerate(urls_nuevas):
    torneo = names_nuevos[i]
    print(f"\n[GLOBE] [{i+1}/{len(urls_nuevas)}] {torneo}")
    print(f"   Link: {url}")
    
    torneo_matches_temp = []
    
    try:
        driver.get(url)
        
        # TIEMPO DE SEGURIDAD PARA CLOUDFLARE
        # Si sale el challenge, undetected-chromedriver suele pasarlo solo,
        # o te deja hacer clic sin bloquearte.
        time.sleep(5) 
        
        # CHEQUEO DE BLOQUEO MANUAL
        if "just a moment" in driver.title.lower():
            print("[STOP] Cloudflare detectado. Tienes 10 segundos para hacer clic si es necesario...")
            time.sleep(10)

        # SCROLL PARA CARGAR PARTIDOS
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # PARSEO
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        matches = soup.find_all('div', class_='match')
        
        if not matches:
            print("   [WARN] 0 partidos. Se asume que llegamos a torneos futuros.")
            print("   [STOP] ¡Deteniendo el scraper para ahorrar tiempo!")
            break  # Esto rompe el bucle y pasa directo a guardar
            
        print(f"   [OK] ¡{len(matches)} PARTIDOS!")
        
        count = 0
        for m in matches:
            try:
                # Extracción rápida
                round_txt = m.find('div', class_='match-header').get_text(strip=True).split("-")[0]
                
                players = m.find_all('div', class_='stats-item')
                if len(players) < 2: continue
                
                p1 = players[0].find('div', class_='name').get_text(strip=True).split("(")[0].strip()
                p2 = players[1].find('div', class_='name').get_text(strip=True).split("(")[0].strip()
                
                if players[0].find('div', class_='winner'):
                    winner, loser = p1, p2
                    w_node, l_node = players[0], players[1]
                else:
                    winner, loser = p2, p1
                    w_node, l_node = players[1], players[0]
                
                # Score simple
                score_parts = []
                sw = w_node.select('.score-item span')
                sl = l_node.select('.score-item span')
                for k in range(min(len(sw), len(sl))):
                    v1, v2 = sw[k].get_text(strip=True), sl[k].get_text(strip=True)
                    if v1 and v2: score_parts.append(f"{v1}-{v2}")
                
                torneo_matches_temp.append({
                    'tourney_id': f"{ANIO}-{torneo}-{i}",
                    'tourney_name': torneo,
                    'surface': 'Hard',
                    'winner_name': winner,
                    'loser_name': loser,
                    'score': " ".join(score_parts),
                    'round': round_txt,
                    'minutes': 100
                })
                count += 1
            except: continue
            
        # GUARDADO PARCIAL INMEDIATO DESPUÉS DE CADA TORNEO EXITOSO
        if torneo_matches_temp:
            df_new = pd.DataFrame(torneo_matches_temp)
            # Rellenar columnas extra para compatibilidad
            cols = ['draw_size','tourney_level','tourney_date','match_num','winner_id','winner_seed','winner_entry','winner_hand','winner_ht','winner_ioc','winner_age','loser_id','loser_seed','loser_entry','loser_hand','loser_ht','loser_ioc','loser_age','best_of','winner_rank','winner_rank_points','loser_rank','loser_rank_points']
            for c in cols: df_new[c] = 0
            
            import os
            es_nuevo = not os.path.exists(ARCHIVO_SALIDA)
            df_new.to_csv(ARCHIVO_SALIDA, mode='a', header=es_nuevo, index=False)
            print(f"   [SAVE] Progreso guardado: {len(df_new)} partidos añadidos.")
            
    except Exception as e:
        print(f"   [FAIL] Error: {e}")

driver.quit()
print("\n[DONE] ¡PROCESO DE SCRAPING FINALIZADO!")