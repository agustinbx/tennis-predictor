"""
Detección de superficie de torneos ATP a partir del nombre del torneo.

Único punto de verdad: antes existían dos funciones divergentes
(en enriquecer_2026.py y corregir_superficie_ranking.py) con listas de
palabras clave distintas para el mismo problema, lo que podía producir
resultados distintos para el mismo torneo según qué script corriera último.
"""

# Torneos conocidos con nombre exacto (cubre casos donde el nombre no
# contiene ninguna palabra clave reconocible, p.ej. "Canada" o "Shanghai").
SUPERFICIE_POR_TORNEO = {
    'australian open': 'Hard',
    'roland garros': 'Clay',
    'wimbledon': 'Grass',
    'us open': 'Hard',
    'indian wells': 'Hard',
    'miami': 'Hard',
    'monte carlo': 'Clay',
    'madrid': 'Clay',
    'rome': 'Clay',
    'cincinnati': 'Hard',
    'canada': 'Hard',
    'shanghai': 'Hard',
    'paris': 'Hard',
    'rotterdam': 'Hard',
    'perth-sydney': 'Hard',
    'brisbane': 'Hard',
    'hong-kong': 'Hard',
    'adelaide': 'Hard',
    'auckland': 'Hard',
    'montpellier': 'Clay',
    'dallas': 'Hard',
    'buenos-aires': 'Clay',
    'doha': 'Hard',
    'rio-de-janeiro': 'Clay',
    'delray-beach': 'Hard',
}

# Palabras clave para detectar superficie cuando el torneo no está en el
# mapa exacto de arriba (cubre variantes de nombre del histórico Sackmann).
CLAY_KEYWORDS = [
    'roland garros', 'madrid', 'rome', 'roma', 'monte carlo', 'barcelona',
    'rio', 'buenos aires', 'cordoba', 'santiago', 'estoril', 'munich',
    'geneva', 'lyon', 'hamburg', 'bastad', 'gstaad', 'umag', 'kitzbuhel',
    'montpellier',
]
GRASS_KEYWORDS = [
    'wimbledon', 'queen', 'halle', 'mallorca', 'eastbourne',
    'stuttgart', 'hertogenbosch', 'newport',
]


def detectar_superficie(nombre_torneo: str) -> str:
    """
    Detecta la superficie (Hard/Clay/Grass) a partir del nombre del torneo.

    Orden de resolución:
    1. Coincidencia exacta en SUPERFICIE_POR_TORNEO (case-insensitive).
    2. Palabra clave de Clay o Grass en el nombre.
    3. Hard por defecto (cemento/indoor/carpet, la superficie más común).
    """
    nombre = str(nombre_torneo).strip().lower()

    if nombre in SUPERFICIE_POR_TORNEO:
        return SUPERFICIE_POR_TORNEO[nombre]

    if any(k in nombre for k in CLAY_KEYWORDS):
        return 'Clay'
    if any(k in nombre for k in GRASS_KEYWORDS):
        return 'Grass'

    return 'Hard'
