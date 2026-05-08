"""
Tests para el módulo de entrenamiento.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from atp_predictor.core.features import EloTracker, H2HTracker
from atp_predictor.ml.train import MatchProcessor, FEATURES


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
            'diff_momentum', 'diff_h2h', 'diff_home'
        ]
        
        assert FEATURES == expected
