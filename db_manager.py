"""Database manager module for DataBeach SQLite operations."""
import json
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
    # Version initiale : table_player, table_serie, table_game
    # ============================================================

    def create_initial_tables(self):
        """Crée les 3 tables si elles n'existent pas déjà."""

        # 1. table_player (pas de FK)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS table_player (
                paire_id        TEXT PRIMARY KEY,
                player_a TEXT NOT NULL,
                player_b TEXT NOT NULL,
                genre        TEXT NOT NULL
            )
        """
        )

        # 2. table_serie (pas de FK)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS table_serie (
                serie_id TEXT PRIMARY KEY,
                club  TEXT NOT NULL,
                type  TEXT NOT NULL,
                genre TEXT NOT NULL,
                date  DATE NOT NULL
            )
        """
        )

        # 3. table_game (FK vers table_player x2 et table_serie)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS table_game (
            game_id        TEXT PRIMARY KEY,
            serie          TEXT NOT NULL,
            stage          TEXT,
            team_a          TEXT NOT NULL,
            team_b         TEXT,
            victory     TEXT NOT NULL,
            set1_score     INT,
            set2_score     INT,
            set3_score     INT,
            set1_score_adv INT,
            set2_score_adv INT,
            set3_score_adv INT,
            FOREIGN KEY (serie) REFERENCES table_serie(serie_id),
            FOREIGN KEY (team_a) REFERENCES table_player(paire_id),
            FOREIGN KEY (team_b) REFERENCES table_player(paire_id)
            )
        """
        )

        self.conn.commit()
        print("✅ Tables créées.")

    # -----------------------------------------------------------

    def load_initial_csv(self, table_name: str, filename: str, ignore_fk: bool = False):
        """Imports a CSV file into the specified table.
        This method only works for csv files with column names that exactly match the table schema.
        (initial tables : table_player.csv, table_serie.csv, table_game.csv)

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
        initial_table_list = ['table_game','table_player','table_serie']
        for table_name in initial_table_list:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.create_initial_tables()
        self.load_initial_csv("table_player", "table_player.csv")
        self.load_initial_csv("table_serie", "table_serie.csv")
        self.load_initial_csv("table_game", "table_game.csv")

    # -----------------------------------------------------------

    def create_simple_actions_table(self,
        action_name: str):
        """Creates a new table for the specified action (e.g. serve, pass) with the appropriate schema.
        Arguments:
            action_name (str): The name of the action for which to create the table (e.g. 'serve', 'pass')
        """  
        self.cursor.execute(f"SELECT * FROM table_{action_name}")
        if self.cursor.fetchone() is not None:
            print(f"⚠️ Table 'table_{action_name}' already exists.")
            return  # Do not recreate if already present

        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS table_{action_name} (
            {action_name}_id INTEGER PRIMARY KEY AUTOINCREMENT,
            point_id TEXT NOT NULL,
            paire_id TEXT NOT NULL,
            player TEXT NOT NULL,
            action TEXT NOT NULL,
            grade TEXT NOT NULL,
            point_won BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (point_id, player, action),
            FOREIGN KEY (point_id) REFERENCES table_point(point_id),
            FOREIGN KEY (paire_id) REFERENCES table_player(paire_id)
            )
        """
        self.execute_query(create_table_query)
        print(f"✅ Table 'table_{action_name}' has been created")
    
    # ============================================================
    # FREQUENT QUERIES METHODS
    # ============================================================

    def teams_names_from_game_id(
        self, 
        game_id : str
    ) -> tuple[str, str] | tuple[None, None]:
        """Récupère les noms des équipes à partir du game_id."""
        query = "SELECT team_a, team_b FROM table_game WHERE game_id = ?"
        self.cursor.execute(query, (game_id,))
        result = self.cursor.fetchone()
        if result:
            print(
                f"✅ game_id '{game_id}' trouvé : "
                f"teamA='{result[0]}', teamB='{result[1]}'"
            )
            return result[0], result[1]
        print(f"⚠️  Aucun résultat trouvé pour game_id '{game_id}'.")
        return None, None

    # -----------------------------------------------------------------------------------

    def new_beach_serie(
        self, serie_id: str, club: str, serie_type: str, genre: str, date: str
    ) -> None:
        """Insère une nouvelle série de beach dans la table_serie."""
        # Check if the serie_id already exists
        query_check = "SELECT serie_id FROM table_serie WHERE serie_id = ?"
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
            INSERT OR IGNORE INTO table_serie (serie_id, club, type, genre, date)
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
        """Insère une nouvelle équipe dans la table_player."""
        # Check if the paire_id already exists
        query_check = "SELECT paire_id FROM table_player WHERE paire_id = ?"
        self.cursor.execute(query_check, (paire_id,))
        result = self.cursor.fetchone()
        if result:
            print(f"⚠️  L'équipe '{paire_id}' existe déjà.")
            return  # Ne pas insérer si déjà présente

        # Check if there is the same team but with swapped players
        query_check_swapped = """SELECT paire_id FROM table_player
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
                INSERT OR IGNORE INTO table_player
                (paire_id, Name_joueurA, Name_joueurB, Genre)
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
        indexed points CSV file into the table_point in the database.

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

        # Create a new table_point with the appropriate schema
        create_table_query = """
            CREATE TABLE IF NOT EXISTS table_point (
            point_id TEXT PRIMARY KEY,
            service_side TEXT,
            team_a_score INT,
            team_b_score INT,
            team_a_sets INT,
            team_b_sets INT,
            team_a_score_diff INT,
            team_b_score_diff INT,
            point_winner TEXT,
            game_id TEXT,
            FOREIGN KEY (game_id) REFERENCES table_game(game_id)
            )
        """
        self.execute_query(create_table_query)

        # Insert the data from the DataFrame into the new table
        insert_query = """
            INSERT OR IGNORE INTO table_point (
            point_id, service_side, team_a_score, team_b_score,
            team_a_sets, team_b_sets, team_a_score_diff, team_b_score_diff,
            point_winner, game_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows_to_insert = [
            (
                row["point_id"],
                row["service_side"],
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

        # # DEV DEBUG - to be removed later
        # print(
        #     f"✅ {len(game_ids)} fichiers CSV de points trouvés "
        #     f"pour les game_ids suivants : {game_ids}"
        # )

        for game_id in game_ids:
            self.load_indexed_df_points_csv_to_db(game_id)

    # ---------------------------------------------------------------------------

    def load_json_actions(self,
        action_name: str,
        action_graded_dir: str = None,
        rewrite_db: bool = False
        ) -> None: 
        """Load the actions from the JSON files in the recap_dict_score directory
        and insert them into the corresponding action tables in the database.
        Arguments:
            action_name (str): The name of the action for which to load the data (e.g. 'serve', 'pass')
        """
        # If no directory is provided, use the default 'recap_dict_score' directory
        if action_graded_dir is None:
            action_graded_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "actions_graded"
            )

        # Get a list of files in action_graded_dir that match the pattern 
        # 'list_grades_{action_name}_{game_id}.json'
        json_files = []
        for filename in os.listdir(action_graded_dir):
            if filename.startswith(f"list_grades_{action_name}_") and filename.endswith(".json"):
                json_files.append(filename)
        # DEV DEBUG - to be removed later
        print(
            f"✅ {len(json_files)} fichiers JSON de grades trouvés "
            f"pour les game_ids suivants : {[f[len(f'list_grades_{action_name}_'):-len('.json')] for f in json_files]}"
        )

        for json_file in json_files:
            with open(os.path.join(action_graded_dir, json_file), "r") as f:
                actions_grades_list = json.load(f)

            # # Create the action table if it doesn't exist
            # self.create_simple_actions_table(action_name)

            # Insert the grades into the action table
            insert_query = f"""
                INSERT INTO table_{action_name} (
                    point_id, paire_id, player, action, grade
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(point_id, player, action) 
                DO {"UPDATE SET grade=excluded.grade" if rewrite_db else "NOTHING"}
            """
            rows_to_insert = [
                (
                    action_grade["point_id"],
                    action_grade["paire_id"],
                    action_grade["player"],
                    action_grade["action"],
                    action_grade["grade"],
                )
                for action_grade in actions_grades_list
            ]
            try:
                with self.conn:
                    self.cursor.executemany(insert_query, rows_to_insert)
                print(
                    f"✅ {len(rows_to_insert)} grades de '{action_name}' insérés "
                    f"pour le fichier '{json_file}'."

                )
            except sqlite3.Error as e:
                print(f"❌ Erreur lors de l'insertion des grades : {e}")



    # ============================================================
    # UTILS
    # ============================================================

    def false_aces_corrector(self, 
                             paire_id: str) -> None:
        """
        Corrects the false aces in the table_serve for a given paire_id.

        This method identifies serves that were graded as 'ace' in table_serve,
        but the serving team (paire_id) did not actually win the point (i.e., the point_winner in table_point is not the serving team).
        Such cases are considered "false aces" and should be corrected to 'error' in the grade column.

        Arguments:
            paire_id (str): The unique identifier for the team (paire) 
                            for which to correct the false aces.
        """
        # Find all point_ids where the serve was graded as 'ace' but the serving team did not win the point
        self.cursor.execute(
            """
            SELECT table_serve.point_id
            FROM table_serve
            INNER JOIN table_point ON table_serve.point_id = table_point.point_id
            WHERE table_point.point_winner != ? AND table_serve.grade = ?
            """,
            (paire_id, "ace")
        )
        result = self.cursor.fetchall()

        if not result:
            print(f"✅ No false aces found for '{paire_id}'.")
            return

        # Extract the point_ids that need correction
        point_ids_to_correct = [row[0] for row in result]

        # Build the update query to set grade='error' for these point_ids
        update_query = f"""
            UPDATE table_serve
            SET grade = 'error'
            WHERE point_id IN ({','.join(['?']*len(point_ids_to_correct))})
            AND grade = 'ace'
        """
        try:
            # Execute the update within a transaction
            with self.conn:
                self.cursor.execute(update_query, point_ids_to_correct)
            print(
                f"✅ {len(point_ids_to_correct)} false aces corrected for '{paire_id}'."
            )
        except sqlite3.Error as e:
            print(f"❌ Error correcting false aces for '{paire_id}': {e}")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

game_id = "JOMR_nov25_MBV_03"
team_serving = "JOMR"

if __name__ == "__main__":
    with DBManager() as db:
        # db.load_all_initial_csv()
        # db.drop_all_tables()
        # db.check_fk_integrity()
        # db.load_indexed_df_points_csv(game_id)
        # db.load_all_indexed_df_points_csv_to_db()
        # db.list_all_tables()
        db.load_json_actions(
            action_name='serve',
            action_graded_dir=r'C:\Users\habib\Documents\GitHub\DataBeach\actions_graded',
            rewrite_db=True)
        # db.create_simple_actions_table('pass')


