import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n==============================================")
    print(f"🚀 EJECUTANDO: {script_path}")
    print(f"==============================================\n")
    
    # Usar el ejecutable de python actual
    python_exe = sys.executable 
    
    try:
        resultado = subprocess.run(
            [python_exe, script_path], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("✅ SALIDA EXITOSA:")
        print(resultado.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ ERROR DURANTE LA EJECUCIÓN:")
        print(e.stdout)
        print(e.stderr)
        print(f"\n[!] El pipeline se ha detenido en el paso: {script_path}")
        sys.exit(1)

def main():
    # Asegurar que el entorno de trabajo sea la raíz del proyecto
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_proyecto = os.path.dirname(ruta_script)
    os.chdir(ruta_proyecto)
    
    print(f"Directorio de trabajo actual: {os.getcwd()}")
    
    # Secuencia ETL y ML
    pasos = [
        "scraping/corregir_superficie_ranking.py",
        "scraping/fusionar_historico_final.py",
        "etl/acomodar_ds.py",
        "prediccion/predict_xgboost.py",
        "api/cargar_datos_db.py" # Actualiza SQLite
    ]
    
    for paso in pasos:
        ruta_completa = os.path.join(ruta_proyecto, paso.replace('/', os.sep))
        if os.path.exists(ruta_completa):
            run_script(ruta_completa)
        else:
            print(f"❌ ARCHIVO NO ENCONTRADO: {ruta_completa}")
            sys.exit(1)

    print("\n🎉 ¡PIPELINE ETL ESTÁNDAR COMPLETADO EXITOSAMENTE!")

if __name__ == "__main__":
    main()
