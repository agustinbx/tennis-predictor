"""
Tests para el módulo de features.
"""
import pytest
from atp_predictor.core.features import EloTracker, H2HTracker, SurfaceStatsTracker


class TestEloTracker:
    """Tests para EloTracker."""
    
    def test_initial_rating(self):
        """Un jugador nuevo debe tener rating por defecto."""
        tracker = EloTracker()
        assert tracker.get_rating("Nadal") == 1500
    
    def test_update_after_match(self):
        """Después de un partido, los ratings deben cambiar."""
        tracker = EloTracker()
        r_w, r_l = tracker.update("Nadal", "Federer")
        
        # El ganador debe subir
        assert r_w > 1500
        # El perdedor debe bajar
        assert r_l < 1500
    
    def test_expected_score(self):
        """El score esperado debe estar entre 0 y 1."""
        tracker = EloTracker()
        e = tracker.expected_score(1500, 1500)
        assert 0 <= e <= 1
        assert abs(e - 0.5) < 0.01  # Jugadores iguales = 50%


class TestH2HTracker:
    """Tests para H2HTracker."""
    
    def test_initial_record(self):
        """H2H inicial debe ser 0-0."""
        tracker = H2HTracker()
        w1, w2 = tracker.get_record("Nadal", "Federer")
        assert w1 == 0 and w2 == 0
    
    def test_update_record(self):
        """Después de un partido, el H2H debe actualizarse."""
        tracker = H2HTracker()
        tracker.update("Nadal", "Federer")
        
        w_nadal, w_federer = tracker.get_record("Nadal", "Federer")
        assert w_nadal == 1 and w_federer == 0


class TestSurfaceStats:
    """Tests para SurfaceStatsTracker."""
    
    def test_get_win_rate_no_data(self):
        """Sin datos, el win rate debe ser 0.5."""
        tracker = SurfaceStatsTracker()
        rate = tracker.get_win_rate("Nadal", "Clay")
        assert rate == 0.5
    
    def test_get_win_rate_with_data(self):
        """Con datos, el win rate debe reflejar las victorias."""
        tracker = SurfaceStatsTracker()
        
        # Simular 10 partidos en Clay
        for _ in range(8):
            tracker.update("Nadal", "Federer", "Clay")
        for _ in range(2):
            tracker.update("Federer", "Nadal", "Clay")
        
        rate = tracker.get_win_rate("Nadal", "Clay", min_matches=5)
        assert abs(rate - 0.8) < 0.01
