"""Database manager module for DataBeach SQLite operations."""
import sqlite3
import csv
import os
import pandas as pd

# Local imports
from etl_utils import extract_transform_indexed_df_points_csv

class DBManager:
    """Gère toutes les opérations liées à la base de données SQLite."""

    DB_PATH = "database/databeach_base.db"
    CSV_DIR = "prov_database"

    # ============================================================
    # CONNEXIONS TO DATABASE
    # ============================================================
    def __init__(self):
        os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(self.DB_PATH)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()
        print(f"✅ Connexion établie : {self.DB_PATH}")

    def close(self):
        """Ferme la connexion à la base."""
        self.conn.close()
        print("🔒 Connexion fermée.")

    def __enter__(self):
        """Permet l'utilisation avec 'with DBManager() as db'."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Ferme automatiquement la connexion en fin de bloc 'with'."""
        self.close()
    
    # ============================================================
    # CRÉATION DES TABLES
    # Version initiale : table_players, table_serie, table_game
    # ============================================================

    def create_initial_tables(self):
        """Crée les 3 tables si elles n'existent pas déjà."""

        # 1. table_players (pas de FK)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS table_players (
                PAIRE_ID        TEXT PRIMARY KEY,
                Name_joueurA TEXT NOT NULL,
                Name_joueurB TEXT NOT NULL,
                Genre        TEXT NOT NULL
            )
        """
        )

        # 2. table_serie (pas de FK)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS table_serie (
                SERIE_ID TEXT PRIMARY KEY,
                club  TEXT NOT NULL,
                type  TEXT NOT NULL,
                genre TEXT NOT NULL,
                date  DATE NOT NULL
            )
        """
        )

        # 3. table_game (FK vers table_players x2 et table_serie)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS table_game (
                GAME_ID        TEXT PRIMARY KEY,
                serie          TEXT NOT NULL,
                stage          TEXT,
                teamA          TEXT NOT NULL,
                teamB          TEXT,
                victoire       TEXT,
                set1_score     INT,
                set2_score     INT,
                set3_score     INT,
                set1_score_adv INT,
                set2_score_adv INT,
                set3_score_adv INT,
                FOREIGN KEY (serie) REFERENCES table_serie(SERIE_ID),
                FOREIGN KEY (teamA) REFERENCES table_players(PAIRE_ID),
                FOREIGN KEY (teamB) REFERENCES table_players(PAIRE_ID)
            )
        """
        )

        self.conn.commit()
        print("✅ Tables créées.")

    # -----------------------------------------------------------

    def load_initial_csv(self, table_name: str, filename: str, ignore_fk: bool = False):
        """Imports a CSV file into the specified table.
        This method only works for csv files with column names that exactly match the table schema.
        (initial tables : table_players.csv, table_serie.csv, table_game.csv)

        Arguments:
            table_name (str): Name of the target table in the database.
            filename (str): Name of the CSV file to import.
            ignore_fk (bool): Whether to ignore foreign key constraints during import.
        """

        # Build the full path to the CSV file
        filepath = os.path.join(self.CSV_DIR, filename)

        # Read the CSV and strip whitespace from column names
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v for k, v in row.items()} for row in reader]

        if rows:
            # Temporarily disable FK checks if requested (e.g. for out-of-order imports)
            if ignore_fk:
                self.conn.execute("PRAGMA foreign_keys = OFF")

            # Dynamically build the INSERT query from the CSV column names
            columns = ", ".join(rows[0].keys())
            placeholders = ", ".join(["?" for _ in rows[0]])
            query = f"INSERT OR IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"

            try:
                # Insert all rows in a single batch operation
                self.cursor.executemany(query, [list(row.values()) for row in rows])
                self.conn.commit()
                print(f"✅ {len(rows)} rows imported into {table_name}.")
            except sqlite3.IntegrityError as e:
                # Roll back the transaction if a FK or UNIQUE constraint is violated
                print(f"❌ Integrity error for {table_name}: {e}")
                print(
                    "   Please ensure that the parent data exists in the referenced tables."
                )
                self.conn.rollback()
            finally:
                # Always re-enable FK checks after the import, regardless of outcome
                if ignore_fk:
                    self.conn.execute("PRAGMA foreign_keys = ON")

    # -----------------------------------------------------------

    def load_all_initial_csv(self):
        """Imports the 3 CSV files in the correct order (parents before children)."""
        self.create_initial_tables()
        self.load_initial_csv("table_players", "table_players.csv")
        self.load_initial_csv("table_serie", "table_serie.csv")
        self.load_initial_csv("table_game", "table_game.csv")

    # ============================================================
    # FREQUENT QUERIES METHODS
    # ============================================================

    def teams_names_from_game_id(
        self, game_id: str
    ) -> tuple[str, str] | tuple[None, None]:
        """Récupère les noms des équipes à partir du game_id."""
        query = "SELECT teamA, teamB FROM table_game WHERE GAME_ID = ?"
        self.cursor.execute(query, (game_id,))
        result = self.cursor.fetchone()
        if result:
            print(
                f"✅ GAME_ID '{game_id}' trouvé : "
                f"teamA='{result[0]}', teamB='{result[1]}'"
            )
            return result[0], result[1]
        print(f"⚠️  Aucun résultat trouvé pour GAME_ID '{game_id}'.")
        return None, None

    # -----------------------------------------------------------------------------------

    def new_beach_serie(
        self, serie_id: str, club: str, serie_type: str, genre: str, date: str
    ) -> None:
        """Insère une nouvelle série de beach dans la table_serie."""
        # Check if the serie_id already exists
        query_check = "SELECT SERIE_ID FROM table_serie WHERE SERIE_ID = ?"
        self.cursor.execute(query_check, (serie_id,))
        result = self.cursor.fetchone()
        if result:
            print(f"⚠️  La série '{serie_id}' existe déjà.")
            return  # Ne pas insérer si déjà présente

        # Check if the date format is correct (DD-MM-YYYY)
        try:
            day, month, year = map(int, date.split("/"))
            if not (1 <= day <= 31 and 1 <= month <= 12 and year > 1900):
                raise ValueError
        except ValueError:
            print(f"❌ Format de date invalide pour '{date}'. Utilisez 'DD/MM/YYYY'.")
            return

        query = """
            INSERT OR IGNORE INTO table_serie (SERIE_ID, club, type, genre, date)
            VALUES (?, ?, ?, ?, ?)
        """
        self.execute_query(query, (serie_id, club, serie_type, genre, date))
        print(
            f"✅ Nouvelle série ajoutée : "
            f"{serie_id} - {club} - {serie_type} - {genre} - {date}"
        )

    # -----------------------------------------------------------------------------------

    def new_team(
        self, paire_id: str, name_joueur_a: str, name_joueur_b: str, genre: str
    ) -> None:
        """Insère une nouvelle équipe dans la table_players."""
        # Check if the paire_id already exists
        query_check = "SELECT PAIRE_ID FROM table_players WHERE PAIRE_ID = ?"
        self.cursor.execute(query_check, (paire_id,))
        result = self.cursor.fetchone()
        if result:
            print(f"⚠️  L'équipe '{paire_id}' existe déjà.")
            return  # Ne pas insérer si déjà présente

        # Check if there is the same team but with swapped players
        query_check_swapped = """SELECT PAIRE_ID FROM table_players
         WHERE (Name_joueurA = ? AND Name_joueurB = ?)
         OR (Name_joueurA = ? AND Name_joueurB = ?)"""
        self.cursor.execute(
            query_check_swapped,
            (name_joueur_a, name_joueur_b, name_joueur_b, name_joueur_a),
        )
        result_swapped = self.cursor.fetchone()
        if result_swapped:
            print(f"⚠️  L'équipe '{paire_id}' existe déjà (joueurs inversés).")
            return  # Ne pas insérer si déjà présente

        # If neither the exact pair nor the swapped pair exists, insert the new team
        if not result and not result_swapped:
            query = """
                INSERT OR IGNORE INTO table_players
                (PAIRE_ID, Name_joueurA, Name_joueurB, Genre)
                VALUES (?, ?, ?, ?)
            """
            self.execute_query(
                query, (paire_id, name_joueur_a, name_joueur_b, genre)
            )
            print(
                f"✅ Nouvelle équipe ajoutée : {paire_id} - "
                f"{name_joueur_a} & {name_joueur_b} - {genre}"
            )

    # -----------------------------------------------------------------------------------

    def list_all_tables(self) -> list[str]:
        """Retourne la liste de toutes les tables présentes dans la base."""
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in self.cursor.fetchall()]
        print(f"📋 Tables disponibles : {tables}")
        return tables

    # -----------------------------------------------------------------------------------

    def table_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """Exporte une table entière sous forme de DataFrame pandas."""

        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, self.conn)
        print(
            f"✅ Table '{table_name}' exportée : {len(df)} lignes, {len(df.columns)} colonnes"
        )
        return df

    # ---------------------------------------------------------------------------

    def drop_all_tables(self):
        """Drops all tables from the database."""
        # Disable FK constraints temporarily to avoid dependency errors
        self.cursor.execute("PRAGMA foreign_keys = OFF")

        # Fetch all table names
        tables = self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        for (table_name,) in tables:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        self.cursor.execute("PRAGMA foreign_keys = ON")
        self.conn.commit()

        print(f"✅ Dropped {len(tables)} table(s): {[t[0] for t in tables]}")

    # -----------------------------------------------------------------------------------

    def execute_query(self, query: str, params=None):
        """Exécute une requête SQL avec ou sans paramètres."""
        if params is None:
            params = []
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            print("✅ Requête exécutée avec succès.")
        except sqlite3.Error as e:
            print(f"❌ Erreur lors de l'exécution de la requête: {e}")
            self.conn.rollback()

    # ====================================================================
    # ETL METHODS
    # ====================================================================

    def load_indexed_df_points_csv_to_db(
            self,
            game_id: str) -> pd.DataFrame:
        """From a given game_id, extract, transform and load the corresponding 
        indexed points CSV file into the table_points in the database.

        Arguments:
            game_id (str): The unique identifier for the game,
                used to locate the corresponding CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the indexed points data
                for the specified game_id, ready for analysis or further processing.
        """
        # Retrieve team names from the game_id using the table_game
        team_a_name, team_b_name = str(), str()
        team_a_name, team_b_name = self.teams_names_from_game_id(game_id)

        # Extract and transform the indexed points data from the corresponding CSV file
        df_points_formatted = pd.DataFrame()
        df_points_formatted = extract_transform_indexed_df_points_csv(
            game_id=game_id,
            team_a_name=team_a_name,
            team_b_name=team_b_name
        )

        # Create a new table_points with the appropriate schema
        create_table_query = """
            CREATE TABLE IF NOT EXISTS table_points (
            POINT_ID TEXT PRIMARY KEY,
            Service_side TEXT,
            team_a_score INT,
            team_b_score INT,
            team_a_sets INT,
            team_b_sets INT,
            team_a_score_diff INT,
            team_b_score_diff INT,
            point_winner TEXT,
            game_id TEXT,
            FOREIGN KEY (game_id) REFERENCES table_game(GAME_ID)
            )
        """
        self.execute_query(create_table_query)

        # Insert the data from the DataFrame into the new table
        insert_query = """
            INSERT OR IGNORE INTO table_points (
            POINT_ID, Service_side, team_a_score, team_b_score,
            team_a_sets, team_b_sets, team_a_score_diff, team_b_score_diff,
            point_winner, game_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows_to_insert = [
            (
                row["POINT_ID"],
                row["Service_side"],
                row["team_a_score"],
                row["team_b_score"],
                row["team_a_sets"],
                row["team_b_sets"],
                row["team_a_score_diff"],
                row["team_b_score_diff"],
                row["point_winner"],
                row["game_id"],
            )
            for _, row in df_points_formatted.iterrows()
        ]
        try:
            with self.conn:
                self.cursor.executemany(insert_query, rows_to_insert)
            print(f"✅ {len(rows_to_insert)} points insérés pour game_id '{game_id}'.")
        except sqlite3.Error as e:
            print(f"❌ Erreur lors de l'insertion des points : {e}")

    # ---------------------------------------------------------------------------

    def load_all_indexed_df_points_csv_to_db(
            self,
            indexed_df_points_dir: str = None
    ) -> None:
        """Load the indexed points DataFrames for a list of game_ids
        and insert them into the database.

        This method iterates over the provided list of game_ids, calls the
        load_indexed_df_points_csv_to_db method for each game_id to load and insert
        the corresponding points data into the database.

        Arguments:
            game_ids (list[str]): A list of unique identifiers for the games, used to
                locate the corresponding CSV files and to name the new tables in the database.
        """
        # If no directory is provided, use the default 'indexed_df_points' directory
        if indexed_df_points_dir is None:
            indexed_df_points_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "indexed_df_points"
            )

        # Get the list of game_ids from the CSV files in the indexed_df_points directory
        game_ids = []
        for filename in os.listdir(indexed_df_points_dir):
            if filename.startswith("indexed_df_points_") and filename.endswith(".csv"):
                game_id = filename[len("indexed_df_points_") : -len(".csv")]
                game_ids.append(game_id)

        # DEV DEBUG - to be removed later
        print(
            f"✅ {len(game_ids)} fichiers CSV de points trouvés "
            f"pour les game_ids suivants : {game_ids}"
        )

        for game_id in game_ids:
            self.load_indexed_df_points_csv_to_db(game_id)

    # ============================================================
    # RÉINITIALISATION
    # ============================================================


    def check_fk_integrity(self):
        """Vérifie les FK de table_game avant import.
        Early version: only checks that teamA and teamB in table_game exist in table_players,
        and that serie in table_game exists in table_serie."""

        print("\n🔍 Vérification des Foreign Keys...\n")

        self.cursor.execute("SELECT PAIRE_ID FROM table_players")
        paires = {row[0] for row in self.cursor.fetchall()}

        self.cursor.execute("SELECT SERIE_ID FROM table_serie")
        series = {row[0] for row in self.cursor.fetchall()}

        # print(f"   PAIRE_ID disponibles  : {paires}")
        # print(f"   SERIE_ID disponibles  : {series}\n")

        filepath = os.path.join(self.CSV_DIR, "table_game.csv")
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v for k, v in row.items()} for row in reader]

        errors = []
        for row in rows:
            if row["teamA"] not in paires:
                errors.append(
                    f"  ❌ GAME_ID {row['GAME_ID']} — teamA '{row['teamA']}' introuvable"
                )
            if row["teamB"] not in paires:
                errors.append(
                    f"  ❌ GAME_ID {row['GAME_ID']} — teamB '{row['teamB']}' introuvable"
                )
            if row["serie"] not in series:
                errors.append(
                    f"  ❌ GAME_ID {row['GAME_ID']} — serie '{row['serie']}' introuvable"
                )

        if errors:
            print(f"⚠️  {len(errors)} problème(s) détecté(s) :")
            for e in errors:
                print(e)
        else:
            print("✅ Toutes les FK sont valides.")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

game_id = "JOMR_nov25_MBV_03"
team_serving = "JOMR"

if __name__ == "__main__":
    with DBManager() as db:
        # db.load_all_initial_csv()
        # # db.drop_all_tables()
        # db.list_all_tables()
        # db.check_fk_integrity()
        # db.load_indexed_df_points_csv(game_id)
        # db.load_all_indexed_df_points_csv_to_db()
        pass

        db.cursor.execute(
            """SELECT POINT_ID, Service_side 
            FROM table_points 
            WHERE game_id = ? AND Service_side = ?""",
            (game_id,team_serving)
        )
        POINTS_IDS = [row[0] for row in db.cursor.fetchall()]
        print(f"POINTS_IDS for game_id '{game_id}' and team_serving '{team_serving}': {POINTS_IDS}")