"""Firestore manager module for DataBeach — replaces DBManager for Streamlit pages."""
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(BASE_DIR, "data-beach-fd370-firebase-adminsdk-fbsvc-322b1f0788.json")

_COLUMNS_SERVE_DF = [
    'point_id', 'player', 'grade', 'point_won',
    'game_id', 'team_a_score', 'team_a_score_diff', 'team_b', 'serie'
]


def _init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(KEY_PATH)
        firebase_admin.initialize_app(cred)


class FirestoreManager:
    """Read-only Firestore client for Streamlit pages.
    Mirrors the DBManager interface used in the Streamlit layer.
    """

    def __init__(self):
        _init_firebase()
        self.db = firestore.client()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass  # Firestore connections don't need explicit closing

    # ============================================================
    # TEAMS / PLAYERS
    # ============================================================

    def list_teams_with_players_names(self) -> list[tuple[str, str]]:
        """Returns [(paire_id, 'PlayerA - PlayerB'), ...] for all teams."""
        docs = self.db.collection('players').stream()
        result = []
        for doc in docs:
            d = doc.to_dict()
            label = f"{d.get('player_a', '')} - {d.get('player_b', '')}"
            result.append((doc.id, label))
        return result

    def get_player_names(self, paire_id: str) -> tuple[str | None, str | None]:
        """Returns (player_a, player_b) for the given paire_id, or (None, None)."""
        doc = self.db.collection('players').document(paire_id).get()
        if doc.exists:
            d = doc.to_dict()
            return d.get('player_a'), d.get('player_b')
        return None, None

    # ============================================================
    # GAMES
    # ============================================================

    def teams_names_from_game_id(self, game_id: str) -> tuple[str | None, str | None]:
        """Returns (team_a, team_b) for the given game_id, or (None, None)."""
        doc = self.db.collection('games').document(game_id).get()
        if doc.exists:
            d = doc.to_dict()
            team_a, team_b = d.get('team_a'), d.get('team_b')
            print(f"✅ game_id '{game_id}' found: teamA='{team_a}', teamB='{team_b}'")
            return team_a, team_b
        print(f"⚠️  No result found for game_id '{game_id}'.")
        return None, None

    def get_game_ids_for_team(self, paire_id: str) -> list[str]:
        """Returns all game_ids where the team appears as team_a or team_b."""
        docs_a = self.db.collection('games').where('team_a', '==', paire_id).stream()
        docs_b = self.db.collection('games').where('team_b', '==', paire_id).stream()
        return [doc.id for doc in docs_a] + [doc.id for doc in docs_b]

    # ============================================================
    # SERVE DATA (replaces the complex SQL join in serve_focus.py)
    # ============================================================

    def get_serve_data_df(self, paire_id: str, player: str | None = None) -> pd.DataFrame:
        """Returns a DataFrame equivalent to the SQL join:
        table_serve LEFT JOIN table_point LEFT JOIN table_game
        filtered on team_a == paire_id (and optionally player).

        Columns: point_id, player, grade, point_won,
                 game_id, team_a_score, team_a_score_diff, team_b, serie
        """
        # 1. Games where this team served (team_a)
        games_docs = self.db.collection('games').where('team_a', '==', paire_id).stream()
        games_dict: dict[str, dict] = {doc.id: doc.to_dict() for doc in games_docs}
        game_ids = list(games_dict.keys())

        # 2. Points for those games — Firestore 'in' supports up to 30 values
        points_dict: dict[str, dict] = {}
        for i in range(0, len(game_ids), 30):
            chunk = game_ids[i:i + 30]
            for doc in self.db.collection('points').where('game_id', 'in', chunk).stream():
                points_dict[doc.id] = doc.to_dict()

        # 3. Serves for this team (optionally filtered by player)
        serves_query = self.db.collection('serves').where('paire_id', '==', paire_id)
        if player:
            serves_query = serves_query.where('player', '==', player)

        # 4. Client-side join
        rows = []
        for doc in serves_query.stream():
            s = doc.to_dict()
            point_id = s.get('point_id')
            point_data = points_dict.get(point_id, {})
            game_id = point_data.get('game_id')
            game_data = games_dict.get(game_id, {}) if game_id else {}
            rows.append({
                'point_id': point_id,
                'player': s.get('player'),
                'grade': s.get('grade'),
                'point_won': s.get('point_won'),
                'game_id': game_id,
                'team_a_score': point_data.get('team_a_score'),
                'team_a_score_diff': point_data.get('team_a_score_diff'),
                'team_b': game_data.get('team_b'),
                'serie': game_data.get('serie'),
            })

        if not rows:
            return pd.DataFrame(columns=_COLUMNS_SERVE_DF)
        return pd.DataFrame(rows, columns=_COLUMNS_SERVE_DF)
