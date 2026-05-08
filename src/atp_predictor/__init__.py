"""
ATP Predictor - Sistema de predicción de partidos de tenis ATP.

Este paquete proporciona:
- Modelos de Machine Learning para predicción de partidos
- API REST para consultar predicciones
- Pipeline ETL para actualización de datos
"""

__version__ = "1.0.0"
__author__ = "Agustin Baldassarri"

from atp_predictor.config import get_settings

__all__ = ["__version__", "get_settings"]
