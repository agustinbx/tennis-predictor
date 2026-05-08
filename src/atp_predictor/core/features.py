"""
Feature Engineering para predicción de tenis.
Extraído de predict_xgboost.py para reutilización.
"""
from typing import Dict, Tuple, Any
import pandas as pd


def get_clutch_score(stats: list) -> float:
    """
    Calcula el score de clutch (capacidad bajo presión).
    
    Args:
        stats: Lista con [bp_saved, bp_faced, service_hold, service_games]
    
    Returns:
        Score de clutch entre 0 y 1
    """
    bp_saved, bp_faced, sv_hold, sv_games = stats
    if len(stats) != 4:
        return 0.5
    
    bp_rate = bp_saved / bp_faced if bp_faced > 0 else 0.5
    sv_rate = sv_hold / sv_games if sv_games > 0 else 0.5
    return (bp_rate + sv_rate) / 2.0


class EloTracker:
    """
    Tracker de ratings ELO para jugadores de tenis.
    
    El rating ELO se actualiza después de cada partido,
    considerando la diferencia de habilidad entre jugadores.
    """
    
    def __init__(self, k_factor: float = 32, default_rating: float = 1500):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings: Dict[str, float] = {}
    
    def get_rating(self, player: str) -> float:
        """Obtiene el rating ELO actual de un jugador."""
        return self.ratings.get(player, self.default_rating)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calcula el score esperado de A contra B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update(self, winner: str, loser: str) -> Tuple[float, float]:
        """
        Actualiza los ratings después de un partido.
        
        Returns:
            Tupla con (nuevo_rating_winner, nuevo_rating_loser)
        """
        r_w = self.get_rating(winner)
        r_l = self.get_rating(loser)
        
        e_w = self.expected_score(r_w, r_l)
        e_l = self.expected_score(r_l, r_w)
        
        new_r_w = r_w + self.k_factor * (1 - e_w)
        new_r_l = r_l + self.k_factor * (0 - e_l)
        
        self.ratings[winner] = new_r_w
        self.ratings[loser] = new_r_l
        
        return new_r_w, new_r_l
    
    def to_dict(self) -> Dict[str, float]:
        """Exporta los ratings como diccionario."""
        return self.ratings.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, float], **kwargs) -> "EloTracker":
        """Crea un tracker desde un diccionario."""
        tracker = cls(**kwargs)
        tracker.ratings = data.copy()
        return tracker


class H2HTracker:
    """
    Tracker de Head-to-Head (enfrentamientos directos).
    """
    
    def __init__(self):
        # Key: tupla ordenada (player1, player2), Value: [wins_p1, wins_p2]
        self.records: Dict[Tuple[str, str], list] = {}
    
    def _normalize_key(self, player1: str, player2: str) -> Tuple[str, str]:
        """Normaliza la clave para que siempre sea consistente."""
        return tuple(sorted([player1, player2]))
    
    def get_record(self, player1: str, player2: str) -> Tuple[int, int]:
        """
        Obtiene el récord entre dos jugadores.
        
        Returns:
            Tupla con (victorias_player1, victorias_player2)
        """
        key = self._normalize_key(player1, player2)
        record = self.records.get(key, [0, 0])
        
        if key[0] == player1:
            return record[0], record[1]
        else:
            return record[1], record[0]
    
    def update(self, winner: str, loser: str) -> None:
        """Actualiza el récord después de un partido."""
        key = self._normalize_key(winner, loser)
        
        if key not in self.records:
            self.records[key] = [0, 0]
        
        if key[0] == winner:
            self.records[key][0] += 1
        else:
            self.records[key][1] += 1
    
    def get_h2h_diff(self, player1: str, player2: str) -> int:
        """Obtiene la diferencia de H2H (player1 - player2)."""
        w1, w2 = self.get_record(player1, player2)
        return w1 - w2
    
    def to_dict(self) -> Dict[Tuple[str, str], list]:
        """Exporta los registros como diccionario."""
        return self.records.copy()


class SurfaceStatsTracker:
    """
    Tracker de estadísticas por superficie.
    """
    
    def __init__(self):
        # Key: (player, surface), Value: [wins, losses]
        self.stats: Dict[Tuple[str, str], list] = {}
    
    def update(self, winner: str, loser: str, surface: str) -> None:
        """Actualiza las estadísticas después de un partido."""
        key_w = (winner, surface)
        key_l = (loser, surface)
        
        if key_w not in self.stats:
            self.stats[key_w] = [0, 0]
        if key_l not in self.stats:
            self.stats[key_l] = [0, 0]
        
        self.stats[key_w][0] += 1  # Win
        self.stats[key_l][1] += 1  # Loss
    
    def get_win_rate(self, player: str, surface: str, min_matches: int = 5) -> float:
        """
        Obtiene el win rate de un jugador en una superficie.
        
        Args:
            player: Nombre del jugador
            surface: Superficie ('Hard', 'Clay', 'Grass')
            min_matches: Mínimo de partidos para considerar válido
        
        Returns:
            Win rate entre 0 y 1, o 0.5 si no hay suficientes datos
        """
        key = (player, surface)
        if key not in self.stats:
            return 0.5
        
        wins, losses = self.stats[key]
        total = wins + losses
        
        if total < min_matches:
            return 0.5
        
        return wins / total
    
    def to_win_rates_dict(self, min_matches: int = 5) -> Dict[Tuple[str, str], float]:
        """
        Exporta todos los win rates como diccionario.
        
        Returns:
            Dict con key=(player, surface), value=win_rate
        """
        result = {}
        for key, (wins, losses) in self.stats.items():
            total = wins + losses
            if total >= min_matches:
                result[key] = wins / total
            else:
                result[key] = 0.5
        return result


class MomentumTracker:
    """
    Tracker de momentum (racha de victorias).
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # Key: player, Value: lista de últimos N resultados (1=W, 0=L)
        self.history: Dict[str, list] = {}
    
    def update(self, winner: str, loser: str) -> None:
        """Actualiza el historial después de un partido."""
        for player, result in [(winner, 1), (loser, 0)]:
            if player not in self.history:
                self.history[player] = []
            
            self.history[player].append(result)
            
            # Mantener solo los últimos N
            if len(self.history[player]) > self.window_size:
                self.history[player].pop(0)
    
    def get_momentum(self, player: str) -> float:
        """
        Obtiene el momentum actual de un jugador.
        
        Returns:
            Momentum entre 0 y 1 (porcentaje de victorias recientes)
        """
        if player not in self.history or len(self.history[player]) == 0:
            return 0.5
        
        return sum(self.history[player]) / len(self.history[player])
