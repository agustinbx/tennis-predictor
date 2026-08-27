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


class SurfaceEloTracker:
    """
    Rating ELO independiente por superficie (Hard/Clay/Grass).

    Un ELO único global diluye la señal de jugadores que rinden muy distinto
    según la superficie (el caso clásico: un especialista de polvo de
    ladrillo que es mediocre en pasto). Reutiliza EloTracker, un tracker
    separado por cada superficie vista.
    """

    def __init__(self, k_factor: float = 32, default_rating: float = 1500):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self._trackers: Dict[str, EloTracker] = {}

    def _get_tracker(self, surface: str) -> EloTracker:
        if surface not in self._trackers:
            self._trackers[surface] = EloTracker(self.k_factor, self.default_rating)
        return self._trackers[surface]

    def get_rating(self, player: str, surface: str) -> float:
        """Obtiene el rating ELO actual de un jugador en una superficie."""
        return self._get_tracker(surface).get_rating(player)

    def update(self, winner: str, loser: str, surface: str) -> None:
        """Actualiza los ratings de esa superficie después de un partido."""
        self._get_tracker(surface).update(winner, loser)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Exporta los ratings como {superficie: {jugador: rating}}."""
        return {surface: tracker.to_dict() for surface, tracker in self._trackers.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]], **kwargs) -> "SurfaceEloTracker":
        """Crea un tracker desde un diccionario {superficie: {jugador: rating}}."""
        tracker = cls(**kwargs)
        for surface, ratings in data.items():
            tracker._trackers[surface] = EloTracker.from_dict(ratings, **kwargs)
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


class WeightedH2HTracker:
    """
    Head-to-Head ponderado por recencia (a diferencia de H2HTracker, que
    cuenta victorias históricas sin distinguir cuándo pasaron).

    Un 5-2 de hace varios años no debería pesar igual que un cruce de la
    semana pasada. Cada nuevo enfrentamiento decae el score acumulado antes
    de sumar el resultado nuevo, así el score converge al historial reciente.
    """

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        # Key: tupla ordenada (player1, player2), Value: [score_p1, score_p2]
        self.scores: Dict[Tuple[str, str], list] = {}

    def _normalize_key(self, player1: str, player2: str) -> Tuple[str, str]:
        """Normaliza la clave para que siempre sea consistente."""
        return tuple(sorted([player1, player2]))

    def get_weighted_diff(self, player1: str, player2: str) -> float:
        """Obtiene la diferencia de H2H ponderada (player1 - player2)."""
        key = self._normalize_key(player1, player2)
        score = self.scores.get(key, [0.0, 0.0])

        if key[0] == player1:
            return score[0] - score[1]
        return score[1] - score[0]

    def update(self, winner: str, loser: str) -> None:
        """Decae el score acumulado y suma el resultado de este partido."""
        key = self._normalize_key(winner, loser)

        if key not in self.scores:
            self.scores[key] = [0.0, 0.0]

        self.scores[key][0] *= self.decay
        self.scores[key][1] *= self.decay

        if key[0] == winner:
            self.scores[key][0] += 1.0
        else:
            self.scores[key][1] += 1.0

    def to_dict(self) -> Dict[Tuple[str, str], list]:
        """Exporta los scores como diccionario."""
        return self.scores.copy()

    @classmethod
    def from_dict(cls, data: Dict[Tuple[str, str], list], **kwargs) -> "WeightedH2HTracker":
        """Crea un tracker desde un diccionario {(p1,p2): [score_p1, score_p2]}."""
        tracker = cls(**kwargs)
        tracker.scores = data.copy()
        return tracker


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


class RestTracker:
    """
    Días de descanso desde el último partido de cada jugador.

    Complementa la fatiga dentro del mismo torneo (sets jugados, ver
    fatigue_tracker en MatchProcessor): esto mide el descanso ENTRE
    torneos, que también afecta el rendimiento (llegar fresco tras
    semanas libres vs. venir de una final larga la semana anterior).
    """

    def __init__(self, default_rest: float = 21.0, max_rest: float = 365.0):
        self.default_rest = default_rest
        self.max_rest = max_rest
        self.last_played: Dict[str, float] = {}  # player -> tourney_date (YYYYMMDD)

    def get_rest_days(self, player: str, current_date: float) -> float:
        """Días transcurridos desde el último partido registrado del jugador."""
        last_date = self.last_played.get(player)
        if last_date is None or pd.isna(current_date):
            return self.default_rest

        dias = self._days_between(last_date, current_date)
        if dias is None:
            return self.default_rest
        return max(0.0, min(dias, self.max_rest))

    def update(self, player: str, current_date: float) -> None:
        """Registra la fecha de este partido como "último partido jugado"."""
        if pd.notna(current_date):
            self.last_played[player] = current_date

    @staticmethod
    def _days_between(date_a: float, date_b: float) -> float:
        """Convierte dos fechas en formato YYYYMMDD a diferencia real en días."""
        try:
            da = pd.to_datetime(str(int(date_a)), format='%Y%m%d')
            db = pd.to_datetime(str(int(date_b)), format='%Y%m%d')
            return abs((db - da).days)
        except (ValueError, TypeError):
            return None

    def to_dict(self) -> Dict[str, float]:
        """Exporta la última fecha jugada de cada jugador."""
        return self.last_played.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, float], **kwargs) -> "RestTracker":
        """Crea un tracker desde un diccionario {jugador: última_fecha}."""
        tracker = cls(**kwargs)
        tracker.last_played = data.copy()
        return tracker


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


class SurfaceMomentumTracker:
    """
    Momentum (racha de los últimos N resultados) separado por superficie.

    MomentumTracker mezcla resultados de todas las superficies: un jugador
    puede venir de ganar 5 seguidos en Hard y aun así llegar "frío" a Clay.
    Este tracker captura la forma reciente específica de cada superficie.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # Key: (player, surface), Value: lista de últimos N resultados (1=W, 0=L)
        self.history: Dict[Tuple[str, str], list] = {}

    def update(self, winner: str, loser: str, surface: str) -> None:
        """Actualiza el historial de esa superficie después de un partido."""
        for player, result in [(winner, 1), (loser, 0)]:
            key = (player, surface)
            if key not in self.history:
                self.history[key] = []

            self.history[key].append(result)

            if len(self.history[key]) > self.window_size:
                self.history[key].pop(0)

    def get_momentum(self, player: str, surface: str) -> float:
        """
        Obtiene el momentum reciente de un jugador en una superficie.

        Returns:
            Momentum entre 0 y 1, o 0.5 si no hay historial en esa superficie
        """
        key = (player, surface)
        history = self.history.get(key, [])
        if not history:
            return 0.5

        return sum(history) / len(history)

    def to_momentum_dict(self) -> Dict[Tuple[str, str], float]:
        """Exporta el momentum ya calculado por (jugador, superficie)."""
        return {key: (sum(h) / len(h) if h else 0.5) for key, h in self.history.items()}
