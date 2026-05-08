#!/usr/bin/env python
"""
Orquestador del Pipeline ETL y ML.

Ejecuta los scripts en secuencia para actualizar:
1. Datos de scraping
2. Entrenamiento del modelo
3. Perfiles de jugadores
4. Base de datos
"""
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from atp_predictor.core.paths import get_project_root, get_analisis_dir, get_scraping_dir, get_models_dir

# Agregar src al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_script(script_path: Path) -> bool:
    """
    Ejecuta un script de Python y retorna True si fue exitoso.
    
    Args:
        script_path: Ruta al script a ejecutar    
    Returns:
        True si el script terminó exitosamente, False si hubo error
    """
    logger.info(f"{'='*60}")
    logger.info(f"[START] EJECUTANDO: {script_path.name}")
    logger.info(f"{'='*60}")
    
    python_exe = sys.executable
    
    try:
        result = subprocess.run(
            [python_exe, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(get_project_root())  # Ejecutar desde raíz del proyecto
        )
        
        if result.stdout:
            print(result.stdout)
        
        logger.info(f"[OK] EXITOSO: {script_path.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] ERROR en {script_path.name}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Ejecuta el pipeline completo."""
    start_time = datetime.now()
    
    logger.info(f"Directorio de trabajo: {get_project_root()}")
    logger.info(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Definir pasos del pipeline
    # Formato: (directorio, nombre_archivo, descripcion)
    steps = [
        (get_scraping_dir(), "corregir_superficie_ranking.py", "Corrección de superficie y ranking"),
        (get_scraping_dir(), "fusionar_historico_final.py", "Fusión de datos históricos"),
        (get_analisis_dir(), "acomodar_ds.py", "Estandarización de datos"),
        (get_models_dir(), "predict_xgboost.py", "Entrenamiento XGBoost"),
        (get_scraping_dir(), "generar_perfiles.py", "Generación de perfiles"),
        (get_project_root() / "api", "cargar_datos_db.py", "Carga a base de datos"),
    ]
    
    # Ejecutar cada paso
    failed_steps = []
    
    for directory, filename, description in steps:
        script_path = directory / filename
        
        if not script_path.exists():
            logger.error(f"[FAIL] ARCHIVO NO ENCONTRADO: {script_path}")
            failed_steps.append((filename, "No encontrado"))
            continue
        
        logger.info(f"[LIST] {description}")
        
        if not run_script(script_path):
            failed_steps.append((filename, "Error de ejecución"))
            # Continuar con los siguientes pasos (no detener todo el pipeline)
    
    # Resumen final
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"{'='*60}")
    logger.info("[STATS] RESUMEN DEL PIPELINE")
    logger.info(f"{'='*60}")
    logger.info(f"Duración total: {duration}")
    
    if failed_steps:
        logger.error(f"[FAIL] Pasos fallidos ({len(failed_steps)}):")
        for filename, reason in failed_steps:
            logger.error(f"   - {filename}: {reason}")
        sys.exit(1)
    else:
        logger.info("[DONE] ¡PIPELINE COMPLETADO EXITOSAMENTE!")
        sys.exit(0)


if __name__ == "__main__":
    main()
