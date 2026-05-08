# 🎾 ATP Tennis Predictor AI

Sistema de predicción de partidos de tenis ATP utilizando Machine Learning (XGBoost).

## 📊 Características

- **Predicción en tiempo real** de partidos ATP
- **API REST** con FastAPI para predicciones
- **Interfaz web** con Streamlit
- **Pipeline ETL automatizado** con GitHub Actions
- **Modelos comparados**: XGBoost (72% precisión) vs Regresión Logística (69%)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                      ATP PREDICTOR                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Scraping] ──► [ETL] ──► [ML Training] ──► [API] ──► [Streamlit]│
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ atp_tour │  │ XGBoost  │  │ FastAPI  │  │ Streamlit│         │
│  │ (web)    │  │ Model    │  │ REST API │  │ Frontend │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                                                                 │
│  Datos: SQLite + PKL Models                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
Proyecto DS/
├── src/atp_predictor/          # Paquete Python principal
│   ├── core/                   # Módulo core (features, paths)
│   │   ├── features.py         # Feature engineering (ELO, H2H, etc.)
│   │   └── paths.py            # Manejo de paths
│   ├── config/                 # Configuración
│   │   └── settings.py         # Settings con pydantic-settings
│   ├── ml/                     # Machine Learning
│   │   └── train.py            # Entrenamiento de modelos
│   ├── api/                    # API REST
│   │   ├── main.py             # FastAPI app
│   │   ├── database.py         # SQLAlchemy config
│   │   ├── models.py           # ORM models
│   │   └── schemas.py          # Pydantic schemas
│   └── __init__.py
│
├── streamlit_app/              # Frontend Streamlit
│   └── utils.py                # Utilidades compartidas
│
├── pages/                      # Páginas de Streamlit
│   ├── 1_🔮_Predictor_en_Vivo.py
│   ├── 2_📊_Analisis_y_Métricas.py
│   └── 3_🏆_Ranking_y_Perfiles.py
│
├── scripts/                    # Scripts de utilidad
│   └── run_etl_pipeline.py     # Orquestador ETL
│
├── scraping/                   # Scripts de scraping
│   ├── scraper_2026_final.py   # Scraper de ATP Tour
│   └── generar_perfiles.py     # Generador de perfiles
│
├── prediccion/                 # Modelos entrenados (.pkl)
│
├── api/                        # API legacy (redirige al paquete)
│
├── tests/                      # Tests unitarios
│   ├── test_features.py
│   └── test_train.py
│
├── pyproject.toml              # Configuración del paquete
├── requirements.txt            # Dependencias de runtime
├── requirements-dev.txt        # Dependencias de desarrollo
└── README.md
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.11 o superior
- pip

### Instalación rápida

```bash
# Clonar el repositorio
git clone <repo-url>
cd "Proyecto DS"

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate   # Windows

# Instalar el paquete en modo desarrollo
pip install -e ".[dev]"
```

## 🏃 Uso

### 1. Iniciar la API

```bash
# Opción A: Con uvicorn directamente
python -m uvicorn api.main:app --reload --port 8002

# Opción B: Con el script del paquete
python -m atp_predictor.api.main
```

### 2. Iniciar el Frontend

```bash
streamlit run 0_🏠_Inicio.py
```

### 3. Ejecutar el Pipeline ETL

```bash
# Pipeline completo
python scripts/run_etl_pipeline.py

# O solo entrenamiento
python prediccion/predict_xgboost.py
```

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con coverage
pytest tests/ -v --cov=src/atp_predictor

# Solo tests de features
pytest tests/test_features.py -v
```

## 📊 Modelos

| Modelo              | Precisión | Descripción                       |
| ------------------- | --------- | --------------------------------- |
| XGBoost             | 72%       | Gradient Boosting con 100 árboles |
| Regresión Logística | 69%       | Modelo lineal clásico             |

### Features utilizadas

| Feature         | Descripción                      |
| --------------- | -------------------------------- |
| `diff_elo`      | Diferencia de rating ELO         |
| `diff_rank`     | Diferencia de ranking ATP        |
| `diff_points`   | Diferencia de puntos ATP         |
| `diff_skill`    | Win rate por superficie          |
| `diff_h2h`      | Historial directo (head-to-head) |
| `diff_momentum` | Racha de últimos 5 partidos      |
| `diff_fatigue`  | Fatiga acumulada (sets jugados)  |
| `diff_clutch`   | Capacidad bajo presión           |
| `diff_home`     | Ventaja de localía               |

## 🔄 CI/CD

El proyecto incluye workflows de GitHub Actions:

- **CI** (`.github/workflows/ci.yml`): Tests y linting en cada PR
- **ETL Pipeline** (`.github/workflows/atp_etl_pipeline.yml`): Actualización automática de datos

## 📝 Licencia

MIT License

## 🙏 Créditos

- Datos de: [Jeff Sackmann / Tennis Abstract](http://www.tennisabstract.com/)
- Licencia de datos: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 🤝 Contribuir

1. Fork del repositorio
2. Crear branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commitear cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Abrir Pull Request

---

venv/Scripts/python.exe -X utf8 scraping/scraper_ranking.py
venv/Scripts/python.exe -X utf8 scraping/scraper_2026_final.py
venv/Scripts/python.exe -X utf8 scraping/enriquecer_2026.py
venv/Scripts/python.exe -X utf8 scraping/corregir_superficie_ranking.py
venv/Scripts/python.exe -X utf8 scraping/juntar_scrapings.py
venv/Scripts/python.exe -X utf8 scraping/fusionar_historico_final.py
venv/Scripts/python.exe -X utf8 scraping/generar_perfiles.py
venv/Scripts/python.exe -X utf8 analisis/acomodar_ds.py
venv/Scripts/python.exe -X utf8 prediccion/predict_xgboost.py --advanced

Desarrollado con ❤️ para los amantes del tenis y el Machine Learning.
