"""
Entrenamiento del modelo de predicción de tenis.

Este módulo refactoriza predict_xgboost.py para usar:
- Core features module (EloTracker, H2HTracker, etc.)
- Centralized configuration
- Modular structure
- Cross Validation para evaluación robusta
- Grid Search para optimización de hiperparámetros
- Calibración de probabilidades
"""
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

from atp_predictor.core.paths import get_project_root, get_models_dir
from atp_predictor.core.features import (
    EloTracker,
    H2HTracker,
    SurfaceStatsTracker,
    MomentumTracker,
    get_clutch_score,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Features utilizados por el modelo
FEATURES = [
    'diff_elo', 'diff_rank', 'diff_points', 'diff_clutch',
    'diff_age', 'diff_ht', 'diff_skill', 'diff_fatigue',
    'diff_momentum', 'diff_h2h', 'diff_home'
]


class MatchProcessor:
    """
    Procesador de partidos para feature engineering.
    
    Implementa el patrón de "online learning" donde las estadísticas
    se actualizan partido a partido para evitar data leakage.
    """
    
    def __init__(self):
        self.elo_tracker = EloTracker()
        self.h2h_tracker = H2HTracker()
        self.surface_tracker = SurfaceStatsTracker()
        self.momentum_tracker = MomentumTracker(window_size=5)
        self.fatigue_tracker: Dict[Tuple[str, str], int] = {}  # (tournament_id, player) -> sets_played
        self.clutch_tracker: Dict[str, list] = {}  # player -> [bp_saved, bp_faced, sv_hold, sv_games]
    
    def process_match(self, row: pd.Series) -> Dict[str, Any]:
        """
        Procesa un partido y retorna las features calculadas.
        
        Args:
            row: Fila del DataFrame con datos del partido
            
        Returns:
            Diccionario con las features y el target
        """
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row.get('surface', 'Hard')
        tourney_id = row.get('tourney_id', '')
        score = str(row.get('score', ''))
        minutes = row.get('minutes', 100)
        
        # Obtener valores actuales ANTES de actualizar
        elo_w = self.elo_tracker.get_rating(winner)
        elo_l = self.elo_tracker.get_rating(loser)
        
        h2h_diff = self.h2h_tracker.get_h2h_diff(winner, loser)
        
        skill_w = self.surface_tracker.get_win_rate(winner, surface)
        skill_l = self.surface_tracker.get_win_rate(loser, surface)
        
        momentum_w = self.momentum_tracker.get_momentum(winner)
        momentum_l = self.momentum_tracker.get_momentum(loser)
        
        # Fatiga
        f_w = self.fatigue_tracker.get((tourney_id, winner), 0)
        f_l = self.fatigue_tracker.get((tourney_id, loser), 0)
        
        # Clutch
        clutch_w = self._get_clutch_score(winner)
        clutch_l = self._get_clutch_score(loser)
        
        # Datos del partido
        r_w = row.get('winner_rank', 9999) if pd.notna(row.get('winner_rank')) else 9999
        r_l = row.get('loser_rank', 9999) if pd.notna(row.get('loser_rank')) else 9999
        
        p_w = row.get('winner_rank_points', 0) if pd.notna(row.get('winner_rank_points')) else 0
        p_l = row.get('loser_rank_points', 0) if pd.notna(row.get('loser_rank_points')) else 0
        
        age_w = row.get('winner_age', 25) if pd.notna(row.get('winner_age')) else 25
        age_l = row.get('loser_age', 25) if pd.notna(row.get('loser_age')) else 25
        
        ht_w = row.get('winner_ht', 185) if pd.notna(row.get('winner_ht')) else 185
        ht_l = row.get('loser_ht', 185) if pd.notna(row.get('loser_ht')) else 185
        
        # Localía
        ioc_w = str(row.get('winner_ioc', '')).strip().upper()
        ioc_l = str(row.get('loser_ioc', '')).strip().upper()
        t_country = self._detect_country(row.get('tourney_name', ''))
        
        home_w = 1 if ioc_w == t_country else 0
        home_l = 1 if ioc_l == t_country else 0
        
        # Features como diferencias
        features = {
            'diff_elo': elo_w - elo_l,
            'diff_rank': r_l - r_w,  # Invertido: menor ranking = mejor
            'diff_points': p_w - p_l,
            'diff_clutch': clutch_w - clutch_l,
            'diff_age': age_w - age_l,
            'diff_ht': ht_w - ht_l,
            'diff_skill': skill_w - skill_l,
            'diff_fatigue': f_w - f_l,
            'diff_momentum': momentum_w - momentum_l,
            'diff_h2h': h2h_diff,
            'diff_home': home_w - home_l
        }
        
        # Ahora ACTUALIZAR los trackers para el siguiente partido
        self.elo_tracker.update(winner, loser)
        self.h2h_tracker.update(winner, loser)
        self.surface_tracker.update(winner, loser, surface)
        self.momentum_tracker.update(winner, loser)
        self._update_fatigue(tourney_id, winner, loser, score)
        self._update_clutch(winner, loser, row)
        
        return features
    
    def _get_clutch_score(self, player: str) -> float:
        """Obtiene el clutch score de un jugador."""
        if player not in self.clutch_tracker:
            return 0.5
        return get_clutch_score(self.clutch_tracker[player])
    
    def _update_fatigue(self, tourney_id: str, winner: str, loser: str, score: str) -> None:
        """Actualiza la fatiga acumulada."""
        # Contar sets del score
        if not score or pd.isna(score) or score.strip() == '':
            sets_played = 2
        else:
            sets_played = len(score.split(' '))
            sets_played = max(1, min(sets_played, 5))
        
        key_w = (tourney_id, winner)
        key_l = (tourney_id, loser)
        
        self.fatigue_tracker[key_w] = self.fatigue_tracker.get(key_w, 0) + sets_played
        self.fatigue_tracker[key_l] = self.fatigue_tracker.get(key_l, 0) + sets_played
    
    def _update_clutch(self, winner: str, loser: str, row: pd.Series) -> None:
        """Actualiza estadísticas de clutch."""
        # Inicializar si no existe
        if winner not in self.clutch_tracker:
            self.clutch_tracker[winner] = [0, 0, 0, 0]
        if loser not in self.clutch_tracker:
            self.clutch_tracker[loser] = [0, 0, 0, 0]
        
        # Acumular stats si están disponibles
        for player, prefix in [(winner, 'w'), (loser, 'l')]:
            bp_saved = row.get(f'{prefix}_bpSaved', 0) or 0
            bp_faced = row.get(f'{prefix}_bpFaced', 0) or 0
            sv_gms = row.get(f'{prefix}_SvGms', 0) or 0
            
            self.clutch_tracker[player][0] += bp_saved
            self.clutch_tracker[player][1] += bp_faced
            self.clutch_tracker[player][2] += sv_gms
            self.clutch_tracker[player][3] += sv_gms
    
    def _detect_country(self, tourney_name: str) -> str:
        """Detecta el país del torneo desde el nombre."""
        name = str(tourney_name).upper()
        mapping = {
            'MADRID': 'ESP', 'BARCELONA': 'ESP', 'VALENCIA': 'ESP',
            'PARIS': 'FRA', 'ROLAND': 'FRA', 'MONTE CARLO': 'FRA', 'LYON': 'FRA',
            'US OPEN': 'USA', 'INDIAN': 'USA', 'MIAMI': 'USA', 'CINCINNATI': 'USA',
            'WIMBLEDON': 'GBR', 'QUEENS': 'GBR', 'LONDON': 'GBR',
            'AUSTRALIAN': 'AUS', 'SYDNEY': 'AUS', 'MELBOURNE': 'AUS', 'BRISBANE': 'AUS',
            'ROME': 'ITA', 'MILAN': 'ITA', 'TURIN': 'ITA',
            'BUENOS AIRES': 'ARG', 'RIO': 'ARG',
        }
        for key, country in mapping.items():
            if key in name:
                return country
        return 'NEUTRAL'
    
    def get_trackers(self) -> Dict[str, Any]:
        """Retorna todos los trackers para exportar."""
        return {
            'elo': self.elo_tracker.to_dict(),
            'h2h': self.h2h_tracker.to_dict(),
            'surface': self.surface_tracker.to_win_rates_dict(),
            'clutch': {p: get_clutch_score(s) for p, s in self.clutch_tracker.items()},
        }


def load_training_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Carga los datos de entrenamiento desde CSV.
    
    Args:
        csv_path: Ruta al archivo CSV. Si es None, busca en ubicaciones por defecto.    
    Returns:
        DataFrame con los datos de entrenamiento
    """
    if csv_path is None:
        # Buscar en ubicaciones por defecto (data/processed/ primero)
        project_root = get_project_root()
        from atp_predictor.core.paths import get_processed_data_dir
        possible_paths = [
            get_processed_data_dir() / "historialTenis.csv",
            project_root / "historialTenis.csv",
            project_root / "scraping" / "historialTenis.csv",
            project_root / "prediccion" / "historialTenis.csv",
        ]
        
        for path in possible_paths:
            if path.exists():
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError("No se encontró historialTenis.csv")
    
    logger.info(f"Cargando datos desde: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False)
    df['minutes'] = df['minutes'].fillna(100)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(by=['tourney_date', 'match_num'])
    
    # Rellenar estadísticas de saque
    cols_clutch = ['w_bpSaved', 'w_bpFaced', 'w_SvGms', 'l_bpSaved', 'l_bpFaced', 'l_SvGms']
    for col in cols_clutch:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    logger.info(f"Datos cargados: {len(df)} partidos")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MatchProcessor]:
    """
    Prepara las features para entrenamiento.
    
    Args:
        df: DataFrame con los datos de partidos
    
    Returns:
        Tuple de (X, y, processor)
    """
    logger.info("Generando features...")
    
    processor = MatchProcessor()
    data_rows = []
    
    for idx, row in df.iterrows():
        # Procesar partido
        features = processor.process_match(row)
        
        # Crear dos ejemplos: uno desde perspectiva del ganador, otro del perdedor
        features_winner = features.copy()
        features_winner['target'] = 1
        data_rows.append(features_winner)
        
        features_loser = {k: -v for k, v in features.items()}
        features_loser['target'] = 0
        data_rows.append(features_loser)
    
    df_train = pd.DataFrame(data_rows).dropna()
    
    X = df_train[FEATURES]
    y = df_train['target']
    
    logger.info(f"Features preparadas: {len(X)} ejemplos")
    return X, y, processor


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5
) -> Tuple[Any, StandardScaler, float]:
    """
    Entrena el modelo XGBoost.
    
    Args:
        X_train, y_train: Datos de entrenamiento escalados
        X_test, y_test: Datos de test escalados
        n_estimators: Número de árboles
        learning_rate: Tasa de aprendizaje
        max_depth: Profundidad máxima
    
    Returns:
        Tuple de (model, scaler, accuracy)
    """
    logger.info(f"Entrenando XGBoost con {len(X_train)} datos...")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Precisión en test: {accuracy * 100:.2f}%")
    
    return model, accuracy


def evaluate_with_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Evalúa el modelo usando Cross Validation.
    
    Esto da una estimación más robusta de la precisión real,
    probando en diferentes subsets de datos.
    
    Args:
        model: Modelo a evaluar (puede ser pipeline con scaler)
        X: Features completas (sin escalar)
        y: Target
        cv_folds: Número de folds para cross validation
    
    Returns:
        Dict con mean_accuracy, std_accuracy, y scores por fold
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[STATS] EVALUACION CON CROSS VALIDATION ({cv_folds} FOLDS)")
    logger.info(f"{'='*60}")
    
    # Crear un scaler para usar dentro del CV
    scaler = StandardScaler()
    
    # Usar StratifiedKFold para mantener proporción de clases
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross validation scores
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    mean_acc = scores.mean()
    std_acc = scores.std()
    
    logger.info(f"Scores por fold: {[f'{s:.4f}' for s in scores]}")
    logger.info(f"")
    logger.info(f"┌─────────────────────────────────────────────┐")
    logger.info(f"│ PRECISIÓN CROSS VALIDATION:                │")
    logger.info(f"│   Media: {mean_acc*100:.2f}%                              │")
    logger.info(f"│   Desv. Estándar: ±{std_acc*100:.2f}%                        │")
    logger.info(f"│   Intervalo 95%: [{(mean_acc-1.96*std_acc)*100:.2f}%, {(mean_acc+1.96*std_acc)*100:.2f}%]    │")
    logger.info(f"└─────────────────────────────────────────────┘")
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'scores': scores.tolist(),
        'ci_lower': mean_acc - 1.96 * std_acc,
        'ci_upper': mean_acc + 1.96 * std_acc
    }


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 3
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Busca los mejores hiperparámetros usando Grid Search.
    
    PRUEBA DIFERENTES COMBINACIONES:
    - n_estimators: [50, 100, 200]  → Cuántos árboles
    - max_depth: [3, 5, 7]         → Profundidad de cada árbol
    - learning_rate: [0.01, 0.05, 0.1]  → Qué tan rápido aprende
    
    Con 3×3×3 = 27 combinaciones y CV=3: 81 entrenamientos
    
    Args:
        X_train: Datos de entrenamiento escalados
        y_train: Target de entrenamiento
        X_test: Datos de test escalados
        y_test: Target de test
        cv_folds: Folds para validación cruzada
    
    Returns:
        Tuple de (best_model, best_params, best_score)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[SEARCH] GRID SEARCH: BUSCANDO MEJORES HIPERPARAMETROS")
    logger.info(f"{'='*60}")
    
    # Definir espacio de búsqueda
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],  # Agregado: fracción de muestras por árbol
        'colsample_bytree': [0.8, 1.0]  # Agregado: fracción de features por árbol
    }
    
    # Modelo base
    base_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1  # Usar todos los cores
    )
    
    # Grid Search con CV
    logger.info(f"Probando {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinaciones...")
    logger.info(f"Con CV={cv_folds}, esto tomará tiempo. [WAIT]")
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    # Evaluar en test con el mejor modelo
    y_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    
    logger.info(f"\n[OK] MEJORES PARÁMETROS ENCONTRADOS:")
    for param, value in best_params.items():
        logger.info(f"   - {param}: {value}")
    
    logger.info(f"\n[STATS] RESULTADOS:")
    logger.info(f"   - CV Score: {best_cv_score*100:.2f}%")
    logger.info(f"   - Test Score: {test_score*100:.2f}%")
    
    return best_model, best_params, test_score


def calibrate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str = 'isotonic'
) -> Tuple[Any, Dict[str, float]]:
    """
    Calibra las probabilidades del modelo.
    
    POR QUÉ ES IMPORTANTE:
    Un modelo puede predecir "70% probabilidad de que gane Nadal",
    pero ¿ese 70% es real? A veces los modelos están mal calibrados.
    
    - Modelo calibrado: Cuando predice 70%, gana 70% de las veces
    - Modelo no calibrado: Predice 70% pero solo gana 60%
    
    La calibración ajusta las probabilidades para que sean confiables.
    
    Args:
        model: Modelo a calibrar
        X_train: Datos de entrenamiento
        y_train: Target de entrenamiento
        X_test: Datos de test
        y_test: Target de test
        method: 'isotonic' (más flexible) o 'sigmoid' (Platt scaling)
    
    Returns:
        Tuple de (calibrated_model, metrics)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[CALIB] CALIBRANDO PROBABILIDADES ({method})")
    logger.info(f"{'='*60}")
    
    # Medir accuracy antes de calibrar
    y_pred_before = model.predict(X_test)
    acc_before = accuracy_score(y_test, y_pred_before)
    
    # Probabilidades antes de calibrar (si el modelo las soporta)
    try:
        prob_before = model.predict_proba(X_test)
        # ¿Qué tan calibrado está? Promedio de probabilidades predichas
        avg_prob_before = prob_before[:, 1].mean()
    except:
        prob_before = None
        avg_prob_before = None
    
    # Calibrar el modelo
    logger.info(f"   Ajustando calibración con método '{method}'...")
    
    # sklearn >=1.6 removio cv='prefit'. Usamos StratifiedKFold(5) 
    # que re-entrena en cada fold y produce probabilidades calibradas.
    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    calibrated_model.fit(X_train, y_train)
    
    # Medir accuracy después de calibrar
    y_pred_after = calibrated_model.predict(X_test)
    acc_after = accuracy_score(y_test, y_pred_after)
    
    # Probabilidades después de calibrar
    prob_after = calibrated_model.predict_proba(X_test)
    avg_prob_after = prob_after[:, 1].mean()
    
    logger.info(f"\n[STATS] COMPARACIÓN ANTES/DESPUÉS:")
    logger.info(f"   ┌────────────────────────────────────────┐")
    logger.info(f"   │                ANTES    DESPUÉS        │")
    logger.info(f"   │ Accuracy:     {acc_before*100:.2f}%    {acc_after*100:.2f}%           │")
    if avg_prob_before:
        logger.info(f"   │ Prob media:  {avg_prob_before*100:.2f}%    {avg_prob_after*100:.2f}%           │")
    logger.info(f"   └────────────────────────────────────────┘")
    
    # Nota: La calibración puede cambiar ligeramente el accuracy
    # pero hace que las probabilidades sean más confiables
    
    metrics = {
        'accuracy_before': acc_before,
        'accuracy_after': acc_after,
        'avg_prob_before': avg_prob_before,
        'avg_prob_after': avg_prob_after,
        'method': method
    }
    
    return calibrated_model, metrics


def save_artifacts(
    model: Any,
    scaler: StandardScaler,
    processor: MatchProcessor,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Guarda todos los artefactos del modelo.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler ajustado
        processor: Procesador con los trackers
        output_dir: Directorio de salida
    
    Returns:
        Dict con las rutas de los archivos guardados
    """
    if output_dir is None:
        output_dir = get_models_dir()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exportar trackers
    trackers = processor.get_trackers()
    
    artifacts = {
        'model': output_dir / 'modelo_xgboost_final.pkl',
        'scaler': output_dir / 'scaler_final.pkl',
        'elo': output_dir / 'elo_tracker.pkl',
        'h2h': output_dir / 'h2h_tracker.pkl',
        'surface': output_dir / 'stats_superficie_v2.pkl',
        'clutch': output_dir / 'clutch_tracker.pkl',
    }
    
    # Guardar modelo y scaler
    joblib.dump(model, artifacts['model'])
    joblib.dump(scaler, artifacts['scaler'])
    
    # Guardar trackers
    joblib.dump(trackers['elo'], artifacts['elo'])
    joblib.dump(trackers['h2h'], artifacts['h2h'])
    joblib.dump(trackers['surface'], artifacts['surface'])
    joblib.dump(trackers['clutch'], artifacts['clutch'])
    
    logger.info(f"Artefactos guardados en: {output_dir}")
    for name, path in artifacts.items():
        logger.info(f"  - {name}: {path}")
    
    return artifacts


def run_training_pipeline(csv_path: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de entrenamiento.
    
    Args:
        csv_path: Ruta al archivo de datos (opcional)
        output_dir: Directorio de salida (opcional)
    
    Returns:
        Dict con métricas y rutas de artefactos
    """
    logger.info("="*60)
    logger.info("[START] INICIANDO PIPELINE DE ENTRENAMIENTO")
    logger.info("="*60)
    
    # Cargar datos
    df = load_training_data(csv_path)
    
    # Preparar features
    X, y, processor = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar
    model, accuracy = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Feature importance
    logger.info("\n[STATS] IMPORTANCIA DE VARIABLES:")
    importances = pd.DataFrame({
        'Variable': FEATURES,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    for _, row in importances.iterrows():
        logger.info(f"  - {row['Variable']}: {row['Importancia']*100:.2f}%")
    
    # Guardar
    artifacts = save_artifacts(model, scaler, processor, output_dir)
    
    logger.info("="*60)
    logger.info(f"[TOP] PRECISIÓN FINAL: {accuracy*100:.2f}%")
    logger.info("="*60)
    
    return {
        'accuracy': accuracy,
        'feature_importance': importances.to_dict('records'),
        'artifacts': {k: str(v) for k, v in artifacts.items()},
        'n_samples': len(X),
    }


def run_training_pipeline_advanced(
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_grid_search: bool = True,
    use_calibration: bool = True,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Pipeline de entrenamiento avanzado con CV, Grid Search y Calibración.
    
    Este pipeline incluye TODAS las mejoras:
    1. Cross Validation para evaluación robusta
    2. Grid Search para optimizar hiperparámetros
    3. Calibración de probabilidades
    
    Args:
        csv_path: Ruta al archivo de datos (opcional)
        output_dir: Directorio de salida (opcional)
        use_grid_search: Si True, busca mejores hiperparámetros
        use_calibration: Si True, calibra probabilidades
        cv_folds: Número de folds para cross validation
    
    Returns:
        Dict con métricas, mejores parámetros y rutas de artefactos
    """
    logger.info("="*70)
    logger.info("[START] PIPELINE DE ENTRENAMIENTO AVANZADO")
    logger.info("="*70)
    logger.info(f"Configuración:")
    logger.info(f"   - Cross Validation: {cv_folds} folds")
    logger.info(f"   - Grid Search: {'Sí' if use_grid_search else 'No'}")
    logger.info(f"   - Calibración: {'Sí' if use_calibration else 'No'}")
    logger.info("="*70)
    
    # ===========================================
    # PASO 1: Cargar y preparar datos
    # ===========================================
    logger.info("\n[DATA] PASO 1: Cargando datos...")
    df = load_training_data(csv_path)
    X, y, processor = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"   - Train: {len(X_train)} muestras")
    logger.info(f"   - Test: {len(X_test)} muestras")
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===========================================
    # PASO 2: Cross Validation del modelo base
    # ===========================================
    logger.info("\n[STATS] PASO 2: Cross Validation (modelo base)...")
    
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        eval_metric='logloss',
        random_state=42
    )
    
    # CV requiere datos sin escalar (creamos un pipeline simple)
    # Pero para XGBoost el escalado ayuda, así que usamos los escalados
    cv_results = evaluate_with_cross_validation(base_model, X_train_scaled, y_train, cv_folds=cv_folds)
    
    # ===========================================
    # PASO 3: Grid Search (opcional)
    # ===========================================
    if use_grid_search:
        logger.info("\n[SEARCH] PASO 3: Grid Search para hiperparámetros...")
        best_model, best_params, test_score = tune_hyperparameters(
            X_train_scaled, y_train, X_test_scaled, y_test, cv_folds=3
        )
        model = best_model
    else:
        logger.info("\n[SKIP] PASO 3: Saltando Grid Search (use_grid_search=False)")
        # Entrenar modelo base
        base_model.fit(X_train_scaled, y_train)
        model = base_model
        best_params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5
        }
        test_score = accuracy_score(y_test, model.predict(X_test_scaled))
    
    # ===========================================
    # PASO 4: Calibración (opcional)
    # ===========================================
    if use_calibration:
        logger.info("\n[CALIB] PASO 4: Calibrando probabilidades...")
        final_model, calib_metrics = calibrate_model(
            model, X_train_scaled, y_train, X_test_scaled, y_test
        )
        calibration_info = calib_metrics
    else:
        logger.info("\n[SKIP] PASO 4: Saltando Calibración (use_calibration=False)")
        final_model = model
        calibration_info = {'method': 'none'}
    
    # ===========================================
    # PASO 5: Métricas finales
    # ===========================================
    logger.info("\n"+"="*70)
    logger.info("[STATS] MÉTRICAS FINALES")
    logger.info("="*70)
    
    # Predicciones finales
    y_pred = final_model.predict(X_test_scaled)
    final_accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    logger.info("\n[STATS] IMPORTANCIA DE VARIABLES:")
    importances_df = pd.DataFrame({
        'Variable': FEATURES,
        'Importancia': final_model.feature_importances_ if hasattr(final_model, 'feature_importances_') else [0]*len(FEATURES)
    }).sort_values('Importancia', ascending=False)
    
    for _, row in importances_df.iterrows():
        logger.info(f"   - {row['Variable']}: {row['Importancia']*100:.2f}%")
    
    # Resumen
    logger.info("\n" + "="*70)
    logger.info("[TOP] RESUMEN DEL ENTRENAMIENTO")
    logger.info("="*70)
    logger.info(f"│ CV Accuracy (modelo base):     {cv_results['mean_accuracy']*100:.2f}% ± {cv_results['std_accuracy']*100:.2f}%")
    logger.info(f"│ CV Accuracy (mejor modelo):    {test_score*100:.2f}%" if use_grid_search else f"│ CV Accuracy (sin GS):           {cv_results['mean_accuracy']*100:.2f}%")
    logger.info(f"│ Test Accuracy (final):          {final_accuracy*100:.2f}%")
    logger.info(f"│ Intervalo de confianza 95%:    [{cv_results['ci_lower']*100:.2f}%, {cv_results['ci_upper']*100:.2f}%]")
    logger.info("="*70)
    
    # Guardar artefactos
    artifacts = save_artifacts(final_model, scaler, processor, output_dir)
    
    # Si se calibró, guardar también el modelo sin calibrar para referencia
    if use_calibration and hasattr(final_model, 'calibrated_classifiers_'):
        artifacts['model_calibrated'] = artifacts['model']
    
    return {
        'accuracy': final_accuracy,
        'cv_accuracy': cv_results['mean_accuracy'],
        'cv_std': cv_results['std_accuracy'],
        'cv_ci_lower': cv_results['ci_lower'],
        'cv_ci_upper': cv_results['ci_upper'],
        'best_params': best_params,
        'calibration': calibration_info,
        'feature_importance': importances_df.to_dict('records'),
        'artifacts': {k: str(v) for k, v in artifacts.items()},
        'n_samples': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }


if __name__ == "__main__":
    # Ejecutar pipeline avanzado por defecto
    results = run_training_pipeline_advanced(
        use_grid_search=True,
        use_calibration=True,
        cv_folds=5
    )
    
    print(f"\n[OK] Entrenamiento completado.")
    print(f"   Precisión final: {results['accuracy']*100:.2f}%")
    print(f"   CV Accuracy: {results['cv_accuracy']*100:.2f}% ± {results['cv_std']*100:.2f}%")
    print(f"   Intervalo 95%: [{results['cv_ci_lower']*100:.2f}%, {results['cv_ci_upper']*100:.2f}%]")
