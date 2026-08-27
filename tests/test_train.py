"""
Tests para el módulo de entrenamiento.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from atp_predictor.core.features import EloTracker, H2HTracker
from atp_predictor.ml.train import (
    MatchProcessor,
    FEATURES,
    prepare_features,
    temporal_train_test_split,
    evaluate_baselines,
)


class TestMatchProcessor:
    """Tests para MatchProcessor."""
    
    def test_initial_state(self):
        """El procesador debe iniciar con trackers vacíos."""
        processor = MatchProcessor()
        
        assert processor.elo_tracker.get_rating("Nadal") == 1500
        assert processor.h2h_tracker.get_record("Nadal", "Federer") == (0, 0)
    
    def test_process_match_returns_features(self):
        """process_match debe retornar un diccionario con todas las features."""
        processor = MatchProcessor()
        
        row = pd.Series({
            'winner_name': 'Nadal',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': '2026-roland-garros-1',
            'score': '6-4 6-4',
            'winner_rank': 1,
            'loser_rank': 3,
            'winner_age': 37,
            'loser_age': 42,
            'winner_ht': 185,
            'loser_ht': 185,
            'winner_rank_points': 10000,
            'loser_rank_points': 5000,
            'winner_ioc': 'ESP',
            'loser_ioc': 'SUI',
            'tourney_name': 'Roland Garros',
        })
        
        features = processor.process_match(row)
        
        assert isinstance(features, dict)
        for feat in FEATURES:
            assert feat in features, f"Falta feature: {feat}"
    
    def test_elo_updates_after_match(self):
        """El ELO debe actualizarse después de un partido."""
        processor = MatchProcessor()
        
        row = pd.Series({
            'winner_name': 'Nadal',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': 'test-1',
            'score': '6-4',
            'tourney_name': 'Test',
        })
        
        processor.process_match(row)
        
        # El ganador debe tener ELO > 1500
        nadal_elo = processor.elo_tracker.get_rating('Nadal')
        federer_elo = processor.elo_tracker.get_rating('Federer')
        
        assert nadal_elo > 1500, "El ganador debe subir de ELO"
        assert federer_elo < 1500, "El perdedor debe bajar de ELO"

    def test_elo_surface_is_isolated_per_surface(self):
        """Ganar en Clay no debe afectar el ELO de Nadal en Hard/Grass."""
        processor = MatchProcessor()

        row = pd.Series({
            'winner_name': 'Nadal',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': 'test-1',
            'score': '6-4',
            'tourney_name': 'Test',
        })

        processor.process_match(row)

        assert processor.surface_elo_tracker.get_rating('Nadal', 'Clay') > 1500
        assert processor.surface_elo_tracker.get_rating('Nadal', 'Hard') == 1500

    def test_h2h_reciente_and_momentum_superficie_present_and_update(self):
        """diff_h2h_reciente y diff_momentum_superficie deben calcularse y actualizarse."""
        processor = MatchProcessor()

        row1 = pd.Series({
            'winner_name': 'Nadal',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': 'test-1',
            'score': '6-4',
            'tourney_name': 'Test',
        })
        features1 = processor.process_match(row1)

        assert features1['diff_h2h_reciente'] == 0.0  # sin historial previo
        assert features1['diff_momentum_superficie'] == 0.0  # ambos sin historial en Clay

        # Revancha en Clay: Nadal ya viene ganando el H2H y con momentum en esa superficie
        row2 = pd.Series({
            'winner_name': 'Nadal',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': 'test-2',
            'score': '6-4',
            'tourney_name': 'Test',
        })
        features2 = processor.process_match(row2)

        assert features2['diff_h2h_reciente'] > 0
        assert features2['diff_momentum_superficie'] > 0

    def test_diff_descanso_reflects_rest_days(self):
        """diff_descanso debe reflejar quién llega más descansado al partido."""
        processor = MatchProcessor()

        # Partido 1: ambos sin historial -> mismo descanso por defecto
        row1 = pd.Series({
            'winner_name': 'Nadal',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': 'test-1',
            'score': '6-4',
            'tourney_name': 'Test',
            'tourney_date': 20260101,
        })
        features1 = processor.process_match(row1)
        assert features1['diff_descanso'] == 0.0

        # Partido 2, dos semanas después: Federer (perdió el 1) juega de nuevo
        # contra un tercer jugador sin historial -> Federer descansó menos
        row2 = pd.Series({
            'winner_name': 'Djokovic',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': 'test-2',
            'score': '6-4',
            'tourney_name': 'Test',
            'tourney_date': 20260115,
        })
        features2 = processor.process_match(row2)

        # Djokovic (sin historial, default_rest) descansó "más" que Federer,
        # que jugó hace solo 14 días
        assert features2['diff_descanso'] > 0

    def test_h2h_updates_after_match(self):
        """El H2H debe actualizarse después de un partido."""
        processor = MatchProcessor()
        
        row = pd.Series({
            'winner_name': 'Nadal',
            'loser_name': 'Federer',
            'surface': 'Clay',
            'tourney_id': 'test-1',
            'score': '6-4',
            'tourney_name': 'Test',
        })
        
        processor.process_match(row)
        
        w_nadal, w_federer = processor.h2h_tracker.get_record('Nadal', 'Federer')
        assert w_nadal == 1
        assert w_federer == 0


class TestFeatures:
    """Tests para el módulo de features."""

    def test_features_list_complete(self):
        """La lista FEATURES debe contener todas las features necesarias."""
        expected = [
            'diff_elo', 'diff_rank', 'diff_points', 'diff_clutch',
            'diff_age', 'diff_ht', 'diff_skill', 'diff_fatigue',
            'diff_momentum', 'diff_h2h', 'diff_home', 'diff_elo_surface',
            'diff_descanso', 'diff_h2h_reciente', 'diff_momentum_superficie',
        ]

        assert FEATURES == expected


def _make_matches_df(n_matches: int) -> pd.DataFrame:
    """Crea un DataFrame sintético de partidos consecutivos en orden cronológico."""
    players = [f"Player{i}" for i in range(n_matches + 1)]
    rows = []
    for i in range(n_matches):
        rows.append({
            'winner_name': players[i],
            'loser_name': players[i + 1],
            'surface': 'Hard',
            'tourney_id': f'test-{i}',
            'score': '6-4 6-4',
            'winner_rank': i + 1,
            'loser_rank': i + 2,
            'winner_age': 25,
            'loser_age': 26,
            'winner_ht': 185,
            'loser_ht': 185,
            'winner_rank_points': 1000,
            'loser_rank_points': 900,
            'winner_ioc': 'ESP',
            'loser_ioc': 'USA',
            'tourney_name': 'Test Open',
        })
    return pd.DataFrame(rows)


class TestPrepareFeaturesNoLeakage:
    """
    Tests de regresión para el fix de data leakage: las dos filas espejo
    (ganador/perdedor) de un mismo partido nunca deben quedar separadas
    entre train y test.
    """

    def test_match_id_pairs_are_preserved(self):
        """Cada match_id debe aparecer en exactamente 2 filas, y el target debe estar balanceado."""
        df = _make_matches_df(10)
        X, y, groups, processor = prepare_features(df)

        assert len(X) == 20  # 2 filas por partido (ganador + perdedor)
        counts = groups.value_counts()
        assert (counts == 2).all(), "Cada match_id debe tener exactamente 2 filas"
        assert y.sum() == len(y) / 2, "El target debe estar balanceado 50/50"

    def test_temporal_split_has_no_group_overlap(self):
        """Ningún match_id debe aparecer a la vez en train y test, y test debe ser posterior a train."""
        df = _make_matches_df(20)
        X, y, groups, processor = prepare_features(df)

        X_train, X_test, y_train, y_test, groups_train, groups_test = temporal_train_test_split(
            X, y, groups, test_size=0.2
        )

        assert set(groups_train.unique()).isdisjoint(set(groups_test.unique()))
        assert groups_train.max() < groups_test.min()
        assert len(X_train) + len(X_test) == len(X)


class TestEvaluateBaselines:
    """Tests para los baselines de referencia (favorito por ranking/ELO)."""

    def test_baselines_perfect_agreement(self):
        """Si diff_rank y diff_elo coinciden siempre con el target, la accuracy debe ser 100%."""
        X = pd.DataFrame({
            'diff_rank': [5, -3, 2, -1],
            'diff_elo': [10, -20, 5, -5],
        })
        y = pd.Series([1, 0, 1, 0])

        baselines = evaluate_baselines(X, y)

        assert baselines['favorito_ranking'] == 1.0
        assert baselines['favorito_elo'] == 1.0
        assert 'clase_mayoritaria' in baselines

    def test_baselines_return_valid_accuracy_range(self):
        """Las accuracies de los baselines deben estar siempre entre 0 y 1."""
        X = pd.DataFrame({
            'diff_rank': [5, -3, 2, -1, 0],
            'diff_elo': [-10, 20, 5, -5, 0],
        })
        y = pd.Series([1, 0, 1, 0, 1])

        baselines = evaluate_baselines(X, y)

        for value in baselines.values():
            assert 0.0 <= value <= 1.0
