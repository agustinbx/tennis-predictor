#!/usr/bin/env python
"""
Script de entrenamiento del modelo XGBoost.

Este script es un wrapper que usa el módulo atp_predictor.ml.train.

MODO DE USO:
    python models/predict_xgboost.py              # Pipeline básico
    python models/predict_xgboost.py --advanced   # Pipeline con CV, GridSearch y Calibración
    python models/predict_xgboost.py --no-grid   # CV + Calibración sin GridSearch
"""
import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atp_predictor.ml.train import (
    run_training_pipeline,
    run_training_pipeline_advanced
)


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo de predicción de tenis ATP'
    )
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Usar pipeline avanzado con CV, GridSearch y Calibración'
    )
    parser.add_argument(
        '--no-grid',
        action='store_true',
        help='En modo avanzado, saltar GridSearch (más rápido)'
    )
    parser.add_argument(
        '--no-calibration',
        action='store_true',
        help='En modo avanzado, saltar calibración de probabilidades'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Número de folds para Cross Validation (default: 5)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("[ATP] ATP PREDICTOR - ENTRENAMIENTO")
    print("="*70)
    
    if args.advanced:
        print(f"\n[STATS] Modo: AVANZADO (CV + {'GridSearch + ' if not args.no_grid else ''}{'Calibración' if not args.no_calibration else 'Sin calibración'})")
        
        results = run_training_pipeline_advanced(
            use_grid_search=not args.no_grid,
            use_calibration=not args.no_calibration,
            cv_folds=args.cv_folds
        )
        
        print(f"\n{'='*70}")
        print("[OK] ENTRENAMIENTO AVANZADO COMPLETADO")
        print("="*70)
        print(f"   Precisión Test:          {results['accuracy']*100:.2f}%")
        print(f"   ROC-AUC Test:            {results['roc_auc']:.4f}")
        print(f"   Log-loss Test:           {results['log_loss']:.4f}")
        print(f"   CV Accuracy:             {results['cv_accuracy']*100:.2f}% ± {results['cv_std']*100:.2f}%")
        print(f"   Intervalo confianza 95%: [{results['cv_ci_lower']*100:.2f}%, {results['cv_ci_upper']*100:.2f}%]")
        print(f"\n   Baselines de referencia:")
        for name, acc in results['baselines'].items():
            print(f"      - {name}: {acc*100:.2f}%")

        if 'best_params' in results and results.get('best_params'):
            print(f"\n   Mejores hiperparámetros:")
            for param, value in results['best_params'].items():
                print(f"      - {param}: {value}")
        
    else:
        print(f"\n[STATS] Modo: BÁSICO (sin optimizaciones)")
        
        results = run_training_pipeline()
        
        print(f"\n{'='*70}")
        print("[OK] ENTRENAMIENTO BÁSICO COMPLETADO")
        print("="*70)
        print(f"   Precisión Test: {results['accuracy']*100:.2f}%")
        print(f"   ROC-AUC Test:   {results['roc_auc']:.4f}")
        print(f"   Log-loss Test:  {results['log_loss']:.4f}")
        print(f"\n   Baselines de referencia:")
        for name, acc in results['baselines'].items():
            print(f"      - {name}: {acc*100:.2f}%")
        print(f"\n   Tip: Usa --advanced para CV, GridSearch y Calibración")
    
    print(f"\n   Muestras totales: {results['n_samples']}")
    print(f"   Artefactos guardados en: {Path(results['artifacts']['model']).parent}")


if __name__ == "__main__":
    main()