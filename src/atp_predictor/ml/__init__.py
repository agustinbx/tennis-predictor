"""
Módulo de Machine Learning para ATP Predictor.

Funciones principales:
- run_training_pipeline: Pipeline básico de entrenamiento
- run_training_pipeline_advanced: Pipeline con CV, GridSearch y Calibración
- evaluate_with_cross_validation: Evaluación robusta con Cross Validation
- tune_hyperparameters: Búsqueda de mejores hiperparámetros
- calibrate_model: Calibración de probabilidades
"""
from .train import (
    train_model,
    load_training_data,
    prepare_features,
    run_training_pipeline,
    run_training_pipeline_advanced,
    evaluate_with_cross_validation,
    tune_hyperparameters,
    calibrate_model,
)

__all__ = [
    # Funciones básicas
    "train_model",
    "load_training_data",
    "prepare_features",
    "run_training_pipeline",
    # Funciones avanzadas
    "run_training_pipeline_advanced",
    "evaluate_with_cross_validation",
    "tune_hyperparameters",
    "calibrate_model",
]