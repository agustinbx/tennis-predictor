"""
Tests para el módulo de features.
"""
import pytest
from atp_predictor.core.features import (
    EloTracker,
    SurfaceEloTracker,
    H2HTracker,
    WeightedH2HTracker,
    SurfaceStatsTracker,
    SurfaceMomentumTracker,
    RestTracker,
)


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


class TestSurfaceEloTracker:
    """Tests para SurfaceEloTracker."""

    def test_initial_rating(self):
        """Un jugador nuevo debe tener rating por defecto en cualquier superficie."""
        tracker = SurfaceEloTracker()
        assert tracker.get_rating("Nadal", "Clay") == 1500
        assert tracker.get_rating("Nadal", "Grass") == 1500

    def test_update_only_affects_that_surface(self):
        """Ganar en Clay no debe cambiar el ELO de Hard o Grass del jugador."""
        tracker = SurfaceEloTracker()
        tracker.update("Nadal", "Federer", "Clay")

        assert tracker.get_rating("Nadal", "Clay") > 1500
        assert tracker.get_rating("Nadal", "Hard") == 1500
        assert tracker.get_rating("Nadal", "Grass") == 1500

    def test_to_dict_and_from_dict_roundtrip(self):
        """Exportar e importar el tracker debe preservar los ratings por superficie."""
        tracker = SurfaceEloTracker()
        tracker.update("Nadal", "Federer", "Clay")
        tracker.update("Federer", "Nadal", "Grass")

        data = tracker.to_dict()
        restored = SurfaceEloTracker.from_dict(data)

        assert restored.get_rating("Nadal", "Clay") == tracker.get_rating("Nadal", "Clay")
        assert restored.get_rating("Federer", "Grass") == tracker.get_rating("Federer", "Grass")


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


class TestWeightedH2HTracker:
    """Tests para WeightedH2HTracker."""

    def test_initial_diff_is_zero(self):
        """Sin historial, la diferencia ponderada debe ser 0."""
        tracker = WeightedH2HTracker()
        assert tracker.get_weighted_diff("Nadal", "Federer") == 0.0

    def test_recent_result_outweighs_old_dominance(self):
        """Un historial viejo dominante debe pesar menos que un resultado reciente."""
        tracker = WeightedH2HTracker(decay=0.5)

        # Nadal le gana a Federer muchas veces "en el pasado"
        for _ in range(5):
            tracker.update("Nadal", "Federer")

        # Con decay=0.5 agresivo, tras suficientes derrotas recientes de Nadal
        # la balanza debería inclinarse a favor de Federer
        for _ in range(5):
            tracker.update("Federer", "Nadal")

        diff = tracker.get_weighted_diff("Nadal", "Federer")
        assert diff < 0, "El resultado reciente debe pesar más que la racha vieja"

    def test_symmetry(self):
        """La diferencia debe invertirse al invertir el orden de los jugadores."""
        tracker = WeightedH2HTracker()
        tracker.update("Nadal", "Federer")

        diff_nf = tracker.get_weighted_diff("Nadal", "Federer")
        diff_fn = tracker.get_weighted_diff("Federer", "Nadal")
        assert diff_nf == -diff_fn


class TestSurfaceMomentumTracker:
    """Tests para SurfaceMomentumTracker."""

    def test_no_history_returns_neutral(self):
        """Sin historial en esa superficie, el momentum debe ser 0.5."""
        tracker = SurfaceMomentumTracker()
        assert tracker.get_momentum("Nadal", "Grass") == 0.5

    def test_momentum_isolated_per_surface(self):
        """Ganar seguido en Clay no debe afectar el momentum en Grass."""
        tracker = SurfaceMomentumTracker(window_size=5)

        for _ in range(3):
            tracker.update("Nadal", "Federer", "Clay")

        assert tracker.get_momentum("Nadal", "Clay") == 1.0
        assert tracker.get_momentum("Nadal", "Grass") == 0.5

    def test_window_keeps_only_last_n_results(self):
        """El momentum debe reflejar solo los últimos N resultados en esa superficie."""
        tracker = SurfaceMomentumTracker(window_size=3)

        tracker.update("Nadal", "Federer", "Hard")  # Nadal gana
        tracker.update("Nadal", "Federer", "Hard")  # Nadal gana
        tracker.update("Federer", "Nadal", "Hard")  # Nadal pierde
        tracker.update("Federer", "Nadal", "Hard")  # Nadal pierde (sale el 1er resultado)

        # Últimos 3: gana, pierde, pierde -> 1/3
        assert abs(tracker.get_momentum("Nadal", "Hard") - (1 / 3)) < 0.01


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


class TestRestTracker:
    """Tests para RestTracker."""

    def test_no_history_returns_default_rest(self):
        """Un jugador sin partidos previos debe devolver el descanso por defecto."""
        tracker = RestTracker(default_rest=21.0)
        assert tracker.get_rest_days("Nadal", 20260101) == 21.0

    def test_rest_days_computed_between_matches(self):
        """El descanso debe reflejar los días reales entre dos partidos."""
        tracker = RestTracker()
        tracker.update("Nadal", 20260101)

        rest = tracker.get_rest_days("Nadal", 20260111)
        assert abs(rest - 10.0) < 0.01

    def test_rest_is_capped_at_max(self):
        """Un hueco enorme (lesión larga, debut) no debe generar un valor absurdo."""
        tracker = RestTracker(max_rest=365.0)
        tracker.update("Nadal", 20200101)

        rest = tracker.get_rest_days("Nadal", 20260101)
        assert rest == 365.0

    def test_invalid_date_returns_default(self):
        """Una fecha faltante (NaN) no debe romper el cálculo."""
        import pandas as pd
        tracker = RestTracker(default_rest=21.0)
        assert tracker.get_rest_days("Nadal", pd.NA) == 21.0
