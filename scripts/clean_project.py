#!/usr/bin/env python
"""
Script de limpieza del proyecto ATP Predictor.

Elimina archivos duplicados, intermedios y temporales que ya no se necesitan.
Mantiene solo los archivos esenciales para el funcionamiento.
"""
import os
import shutil
from pathlib import Path

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Archivos a eliminar (duplicados, intermedios, temporales)
FILES_TO_DELETE = [
    # CSVs duplicados en analisis/
    "analisis/historialTenis.csv",
    
    # CSVs duplicados en prediccion/
    "prediccion/historialTenis.csv",
    "prediccion/historial_tenis.csv",
    "prediccion/historial_tenis_COMPLETO.csv",
    "prediccion/ranking_2026.csv",
    "prediccion/ranking_actual_2026.csv",
    "prediccion/estadisticas_jugadores_avanzadas.csv",
    "prediccion/resultados_comparacion.csv",
    
    # CSVs intermedios en scraping/
    "scraping/historialTenis.csv",
    "scraping/historial_tenis.csv",
    # NOTA: NO borrar historial_tenis_COMPLETO.csv - es necesario para el entrenamiento
    "scraping/atp_matches.csv",
    "scraping/atp_matches_2025.csv",
    "scraping/atp_matches_2025_2026_unidos.csv",
    "scraping/atp_matches_2026_corregido.csv",
    "scraping/atp_matches_historico.csv",
    
    # PKLs duplicados/antiguos en prediccion/
    "prediccion/elo_rating.pkl",
    "prediccion/elo_rating_surface.pkl",
    "prediccion/modelo_elo_xgboost.pkl",
    "prediccion/scaler_elo.pkl",
    "prediccion/stats_superficie.pkl",
    "prediccion/perfiles_jugadores.pkl",
    
    # Outputs intermedios
    "prediccion/out_xgboost.txt",
    "prediccion/output.txt",
    
    # CSVs en raíz
    "importancia_real.csv",
    "resultados_comparacion.csv",
    
    # Archivo de documentación viejo
    "doc.md",
    
    # Test temporal
    "test_streamlit_imports.py",
]

# Directorios a eliminar completamente
DIRS_TO_DELETE = [
    "models",
    "data", 
    ".pytest_cache",
]


def main():
    """Ejecuta la limpieza."""
    print("="*60)
    print("LIMPIEZA DEL PROYECTO ATP PREDICTOR")
    print("="*60)
    
    deleted_count = 0
    freed_space = 0
    
    for file_rel in FILES_TO_DELETE:
        file_path = PROJECT_ROOT / file_rel
        
        if file_path.exists():
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                freed_space += size
                print(f"[ELIMINADO] {file_rel}")
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] {file_rel}: {e}")
        else:
            print(f"[NO EXISTE] {file_rel}")
    
    # Eliminar directorios vacíos
    for dir_rel in DIRS_TO_DELETE:
        dir_path = PROJECT_ROOT / dir_rel
        
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"[DIR ELIMINADO] {dir_rel}")
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] {dir_rel}: {e}")
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"Archivos eliminados: {deleted_count}")
    print(f"Espacio liberado: {freed_space / 1024 / 1024:.2f} MB")
    
    # Verificar estructura final
    print("\nESTRUCTURA FINAL:")
    print("\nARCHIVOS ESENCIALES:")
    
    essential_files = [
        "prediccion/modelo_xgboost_final.pkl",
        "prediccion/scaler_final.pkl",
        "prediccion/h2h_tracker.pkl",
        "prediccion/elo_tracker.pkl",
        "prediccion/clutch_tracker.pkl",
        "prediccion/stats_superficie_v2.pkl",
        "scraping/perfiles_jugadores.pkl",
        "scraping/historial_tenis_COMPLETO.csv",
        "scraping/atp_matches_2026_indetectable.csv",
        "scraping/ranking_2026.csv",
        "atp_tennis.db",
    ]
    
    for f in essential_files:
        path = PROJECT_ROOT / f
        if path.exists():
            print(f"   [OK] {f}")
        else:
            print(f"   [FALTA] {f}")
    
    print("\n" + "="*60)
    print("LIMPIEZA COMPLETADA")
    print("="*60)


if __name__ == "__main__":
    main()