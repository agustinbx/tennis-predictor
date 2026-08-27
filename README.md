# 🎾 ATP Tennis Predictor AI

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B)
![XGBoost](https://img.shields.io/badge/modelo-XGBoost-1f77b4)

Sistema end-to-end de predicción de partidos de tenis ATP: scraping en vivo de [atptour.com](https://www.atptour.com), feature engineering con trackers online (ELO, H2H, momentum, descanso), un modelo XGBoost con split temporal libre de data leakage, servido por una API REST y una app Streamlit interactiva.

## 🎯 Resultados

| Métrica                       | Valor              |
| ------------------------------ | ------------------- |
| **Accuracy (test temporal)**   | **66.21%**          |
| ROC-AUC                        | 0.728               |
| Log-loss                       | 0.605               |
| CV Accuracy (5-fold)           | 66.64% ± 0.84%      |
| Baseline "favorito por ranking"| 65.91%              |
| Partidos analizados            | ~33.500 (2000–2026) |

El número que importa no es el 66% aislado, sino la comparación: el modelo le gana a la heurística ingenua de "siempre apuesta al mejor rankeado" — el techo real en predicción de tenis 1 vs 1 ronda los 65-70% incluso en sistemas publicados (hay un componente de azar genuino en el deporte), así que ese margen es la señal real que aporta el modelo.

Estos números salen de correr `python models/predict_xgboost.py --advanced` de punta a punta, sin atajos: podés reproducirlos vos mismo.

## ✨ Funcionalidades

- **Predicción en vivo** de cualquier enfrentamiento hipotético, con explicación narrativa de por qué el modelo se inclina por un jugador
- **Scraping automatizado** de rankings, resultados de torneos y perfiles de jugadores
- **API REST** (FastAPI) y **frontend interactivo** (Streamlit multi-página)
- **Pipeline ETL** orquestado y corrido diariamente vía GitHub Actions

## 🧠 Metodología

Lo que distingue a este proyecto de un notebook de Kaggle es el cuidado puesto en no engañarse con las métricas:

- **Split temporal agrupado por partido, no aleatorio.** El dataset genera dos filas por partido (ganador/perdedor) con features espejadas; un split aleatorio dejaría que el modelo "viera" la versión invertida del mismo partido que después se evalúa. El split real reserva los partidos más recientes como test — igual que en producción, donde solo se conoce el pasado.
- **Baseline explícito.** Antes de confiar en el accuracy del modelo, se lo compara contra heurísticas triviales (favorito por ranking, favorito por ELO) sobre el mismo test set. Sin esto, un número aislado no dice nada.
- **Feature engineering con trackers online**, todos actualizados partido a partido *antes* de conocer el resultado (para no filtrar información del futuro): ELO global y específico por superficie, H2H bruto y ponderado por recencia (decaimiento exponencial), momentum general y por superficie, fatiga dentro del torneo y descanso entre torneos.
- **Calibración de probabilidades** (isotonic regression) y medición con Brier score — que el modelo diga "70% de confianza" solo sirve si ese 70% es real.
- **Búsqueda de hiperparámetros** con Grid Search + Cross Validation, no valores elegidos a mano.

### Importancia de variables (XGBoost)

| Feature                       | Importancia | Descripción                                          |
| ------------------------------ | ----------- | ----------------------------------------------------- |
| `diff_points`                  | 37.3%       | Diferencia de puntos ATP                              |
| `diff_elo`                     | 25.0%       | Diferencia de rating ELO global                       |
| `diff_elo_surface`             | 7.1%        | ELO específico de la superficie del partido           |
| `diff_rank`                    | 6.3%        | Diferencia de ranking ATP                              |
| `diff_skill`                   | 3.4%        | Win rate histórico por superficie                     |
| `diff_clutch`                  | 3.0%        | Rendimiento en break points / bajo presión            |
| `diff_age`                     | 2.9%        | Diferencia de edad                                    |
| `diff_descanso`                | 2.5%        | Días de descanso desde el último partido              |
| `diff_h2h_reciente`            | 2.3%        | Historial directo ponderado por recencia              |
| `diff_home`                    | 2.2%        | Ventaja de jugar en su país                            |
| `diff_momentum`                | 1.9%        | Racha de los últimos 5 partidos                       |
| `diff_h2h`                     | 1.7%        | Historial directo (conteo bruto)                      |
| `diff_ht`                      | 1.6%        | Diferencia de altura                                   |
| `diff_fatigue`                 | 1.5%        | Sets jugados en el torneo actual                       |
| `diff_momentum_superficie`     | 1.3%        | Racha reciente específica de la superficie             |

## 🏗️ Arquitectura

```
┌────────────────────────────────────────────────────────────────────┐
│                          ATP PREDICTOR                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Scraping] ──► [ETL] ──► [Feature Eng. + ML] ──► [API] ──► [Streamlit]
│                                                                      │
│  atptour.com    fusión +     trackers online      FastAPI   Frontend│
│  (Selenium)     limpieza     + XGBoost + CV        REST API interactivo
│                                                                      │
│  Datos: SQLite + modelos .pkl (ELO, H2H, momentum, calibración)      │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
Proyecto DS/
├── src/atp_predictor/          # Paquete Python principal
│   ├── core/                   # Feature engineering (trackers) y manejo de paths
│   ├── config/                 # Configuración (pydantic-settings)
│   ├── ml/                     # Entrenamiento del modelo (train.py)
│   └── api/                    # Config/ORM/schemas de la API
│
├── api/                        # API REST (FastAPI, entrypoint real: api/main.py)
├── streamlit_app/ + pages/     # Frontend Streamlit multi-página
├── scripts/                    # Orquestación del pipeline ETL
├── scraping/                   # Scrapers de atptour.com
├── models/                     # Modelo entrenado (.pkl) + script de entrenamiento
├── data/
│   ├── external/                # Dataset histórico externo (Jeff Sackmann)
│   └── processed/                # Dataset final listo para entrenamiento
├── tests/                      # Tests unitarios (pytest)
│
├── pyproject.toml              # Configuración del paquete y dependencias
└── README.md
```

## 🛠️ Stack Tecnológico

| Capa            | Tecnología                              |
| ---------------- | ---------------------------------------- |
| Scraping         | Selenium, `undetected_chromedriver`, BeautifulSoup |
| ML               | XGBoost, scikit-learn, pandas             |
| API              | FastAPI, SQLAlchemy, Pydantic             |
| Frontend         | Streamlit, Plotly                         |
| Testing          | pytest                                    |
| CI/CD            | GitHub Actions                            |

## 🚀 Instalación

### Prerrequisitos

- Python 3.11 o superior
- pip

### Instalación rápida

```bash
git clone <repo-url>
cd "Proyecto DS"

python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/Mac

# Instalar el paquete en modo desarrollo (única fuente de dependencias)
pip install -e ".[dev]"
```

## ▶️ Uso

### 1. Iniciar la API

```bash
python -m uvicorn api.main:app --reload --port 8002
```

### 2. Iniciar el Frontend

```bash
streamlit run 0_🏠_Inicio.py
```

### 3. Ejecutar el Pipeline ETL

```bash
# Pipeline completo (scraping + fusión + entrenamiento + carga a DB)
python scripts/run_etl_pipeline.py

# O solo entrenamiento, con Cross Validation + Grid Search + Calibración
python models/predict_xgboost.py --advanced
```

## 🧪 Tests

```bash
pytest tests/ -v

# Con coverage
pytest tests/ -v --cov=src/atp_predictor
```

## 🔄 CI/CD

- **CI** (`.github/workflows/ci.yml`): tests y linting en cada PR
- **ETL Pipeline** (`.github/workflows/atp_etl_pipeline.yml`): scraping + reentrenamiento diario vía cron

## 📌 Limitaciones y próximos pasos

- Los partidos scrapeados de 2025-2026 no incluyen estadísticas de saque, nivel del torneo (`tourney_level`) ni mano hábil — el scraper de `atptour.com` no captura esos datos por partido, a diferencia del histórico de Jeff Sackmann (2000-2024). Enriquecer el scraper para capturarlos destrabaría esas features.
- El techo de accuracy en predicción de tenis 1 vs 1 está acotado por la aleatoriedad propia del deporte; no se espera superar significativamente el 68-70%.

## 📝 Licencia

MIT License — ver [LICENSE](LICENSE)

## 🙏 Créditos

- Datos históricos: [Jeff Sackmann / Tennis Abstract](http://www.tennisabstract.com/) — [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Datos en vivo: [ATP Tour](https://www.atptour.com/)

---

Desarrollado con ❤️ para los amantes del tenis y el Machine Learning.
