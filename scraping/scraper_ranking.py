import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import time

URL_RANKING = "https://www.atptour.com/en/rankings/singles?rankRange=1-500"

print("🏆 INICIANDO SCRAPER DE RANKINGS (VERSIÓN 4 COLUMNAS)...")

options = uc.ChromeOptions()
options.add_argument("--start-maximized")
# Forzamos versión 144 para evitar tu error de driver
driver = uc.Chrome(options=options, version_main=144) 

data_ranking = []

try:
    driver.get(URL_RANKING)
    print("⏳ Esperando carga...")
    time.sleep(7)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Buscamos la tabla (ya sabemos que está ahí)
    tabla = soup.find('table', class_='mega-table')
    if not tabla: tabla = soup.find('table')
    
    if tabla:
        filas = tabla.find('tbody').find_all('tr')
        print(f"✅ Procesando {len(filas)} jugadores...")
        
        for row in filas:
            try:
                cols = row.find_all('td')
                if len(cols) < 3: continue 
                
                # 1. RANKING (Columna 0)
                rank_txt = cols[0].get_text(strip=True).replace("T", "")
                
                # 2. PUNTOS (Columna 2)
                points_txt = cols[2].get_text(strip=True).replace(",", "")
                
                # 3. NOMBRE REAL (Sacado del link en Columna 2 o 1)
                # El debug dijo que el link estaba en Col 2
                link_tag = cols[2].find('a')
                if not link_tag: link_tag = cols[1].find('a') # Por si acaso
                
                if link_tag:
                    href = link_tag.get('href') # Ej: /en/players/carlos-alcaraz/a0e2/...
                    slug = href.split('/')[3]   # "carlos-alcaraz"
                    # Convertimos "carlos-alcaraz" a "Carlos Alcaraz"
                    full_name = slug.replace('-', ' ').title()
                    
                    # Correcciones manuales comunes
                    full_name = full_name.replace("De Minaur", "de Minaur") # A veces falla la mayúscula
                else:
                    # Si no hay link, usamos el texto de Col 1 (ej: "C. Alcaraz")
                    full_name = cols[1].get_text(strip=True)

                data_ranking.append({
                    'player_slug': full_name,
                    'rank': int(rank_txt) if rank_txt.isdigit() else 9999,
                    'age': 0, # No aparece en esta tabla, lo llenaremos con el histórico
                    'points': int(points_txt) if points_txt.isdigit() else 0
                })
            except Exception as e:
                continue

    # GUARDAR
    if data_ranking:
        df = pd.DataFrame(data_ranking)
        df.to_csv("ranking_actual_2026.csv", index=False)
        print(f"\n🎉 ¡ÉXITO! Guardados {len(df)} jugadores en 'ranking_actual_2026.csv'")
    else:
        print("❌ No se extrajeron datos.")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    try:
        driver.quit()
    except:
        pass # Ignoramos el error de cierre de Windows