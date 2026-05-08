"""
Backward compatibility - usar el módulo del paquete.
"""
from atp_predictor.api.database import *

# Re-export everything for backward compatibility
__all__ = ["get_db", "engine", "Base", "SessionLocal"]
