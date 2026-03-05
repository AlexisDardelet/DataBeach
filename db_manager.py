import sqlite3
import csv
import os
import pandas as pd


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

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ferme automatiquement la connexion en fin de bloc 'with'."""
        self.close()

    # ============================================================
    # QUERIES METHODS
    # ============================================================

    def teams_names_from_game_id(self,
                                 game_id:str
                                 ) -> tuple[str, str] | tuple[None, None]:
        """Récupère les noms des équipes à partir du game_id."""
        query = "SELECT teamA, teamB FROM table_game WHERE GAME_ID = ?"
        self.cursor.execute(query, (game_id,))
        result = self.cursor.fetchone()
        if result:
            print(f"✅ GAME_ID '{game_id}' trouvé : teamA='{result[0]}', teamB='{result[1]}'")
            return result[0], result[1]
        else:
            print(f"⚠️  Aucun résultat trouvé pour GAME_ID '{game_id}'.")
            return None, None
    
    # -----------------------------------------------------------------------------------

    def new_beach_serie(self,
                        serie_id:str,
                        club:str,
                        type:str,
                        genre:str,
                        date:str
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
        self.execute_query(query, (serie_id, club, type, genre, date))
        print(f"✅ Nouvelle série ajoutée : {serie_id} - {club} - {type} - {genre} - {date}")

    # -----------------------------------------------------------------------------------

    def new_team(self,
                 paire_id:str,
                 name_joueurA:str,
                 name_joueurB:str,
                 genre:str
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
         WHERE (Name_joueurA = ? AND Name_joueurB = ?) OR (Name_joueurA = ? AND Name_joueurB = ?)"""
        self.cursor.execute(query_check_swapped, (name_joueurA, name_joueurB, name_joueurB, name_joueurA))
        result_swapped = self.cursor.fetchone()
        if result_swapped:
            print(f"⚠️  L'équipe '{paire_id}' existe déjà (joueurs inversés).")
            return  # Ne pas insérer si déjà présente

        # If neither the exact pair nor the swapped pair exists, insert the new team
        if not result and not result_swapped:
            query = """
                INSERT OR IGNORE INTO table_players (PAIRE_ID, Name_joueurA, Name_joueurB, Genre)
                VALUES (?, ?, ?, ?)
            """
            self.execute_query(query, (paire_id, name_joueurA, name_joueurB, genre))
            print(f"✅ Nouvelle équipe ajoutée : {paire_id} - {name_joueurA} & {name_joueurB} - {genre}")
    

    # -----------------------------------------------------------------------------------

    def table_to_dataframe(
        self,
        table_name: str
        ) -> pd.DataFrame:
        """Exporte une table entière sous forme de DataFrame pandas."""
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, self.conn)
        print(f"✅ Table '{table_name}' exportée : {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    
    # -----------------------------------------------------------------------------------

    def execute_query(
        self,
        query: str,
        params=None
        ):
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
    
    # ====================================================================
    # ETL METHODS
    # ====================================================================

    def load_indexed_df_points_csv(
            self,
            game_id: str
    ) -> pd.DataFrame:
        """Load a CSV file containing the indexed points for a specific game_id and return it as a DataFrame.
        the CSV file should be located in the 'indexed_df_points' directory and named 'indexed_df_points_{game_id}.csv'.
        It requires that the CSV file has a header row with column names that match the expected schema for the points data.
        It returns a pandas DataFrame ready to be loaded into the database with a query

        Arguments:
            game_id (str): The unique identifier for the game, used to locate the corresponding CSV file. 
        Returns:
            pd.DataFrame: A DataFrame containing the indexed points data for the specified game_id, ready for analysis or further processing.
        """

        teamA_name, teamB_name = self.teams_names_from_game_id(game_id)
        # DEV DEBUG - to be removed later
        print(f"Team A: {teamA_name}, Team B: {teamB_name}")

        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indexed_df_points', f'indexed_df_points_{game_id}.csv')

        # Load the CSV file into a DataFrame
        df_points_formatted = pd.DataFrame()
        df_points_formatted = pd.read_csv(
            filepath_or_buffer=csv_path,
            keep_default_na=True,
            )
        df_points_formatted['point_index'] = df_points_formatted['point_index'].astype('Int64')

        # Create the 'POINT_ID' column based on 'game_id' and 'point_index'
        df_points_formatted['POINT_ID'] = df_points_formatted.apply(
            lambda row: f"{game_id}_p{row['point_index']}" if pd.notna(row['point_index']) else pd.NA,
            axis=1
        )
        df_points_formatted.insert(0, 'POINT_ID', df_points_formatted.pop('POINT_ID'))  # Move 'POINT_ID' to the first column

        # Columns dropped
        df_points_formatted.drop(columns=['Start_frame', 'End_frame', 'point_index'], inplace=True)

        # Columns renamed
        df_points_formatted.rename(
            columns={
                f'{teamA_name}_score': 'teamA_score',
                f'{teamB_name}_score': 'teamB_score',
                f'{teamA_name}_sets': 'teamA_sets',
                f'{teamB_name}_sets': 'teamB_sets',
                }, inplace=True)

        # Drop the 'match start - ' and 'set start - ' and 'service ' prefixes from the 'Service_side' column
        df_points_formatted['Service_side'] = df_points_formatted['Service_side'].str.replace(r'^(match start - |set start - )', '', regex=True)
        df_points_formatted['Service_side'] = df_points_formatted['Service_side'].str.replace(str('service '), '')

        # # TO BE DETERMINED IF NEEDED LATER - if we want to keep the team names in the 'Service_side' column or if we want to replace them with 'teamA' and 'teamB'
        # # Rename the strings in the 'Service_side' column to match the team names
        # df_points_formatted['Service_side'] = df_points_formatted['Service_side'].apply(
        #     lambda x: str('teamA') if x == teamA_name else (str('teamB') if x == teamB_name else x)
        # )

        # Create columns 'teamA_score_diff' and 'teamB_score_diff'
        # which are the differences between the current point's score and the previous point's score for team A and team B respectively
        df_points_formatted['teamA_score_diff'] = df_points_formatted['teamA_score'] - df_points_formatted['teamB_score']
        df_points_formatted['teamB_score_diff'] = df_points_formatted['teamB_score'] - df_points_formatted['teamA_score']

        # # Drop the rows with '*SWITCH*', 'Timeout', 'end of set' values in the 'Service_side' column
        df_points_formatted = df_points_formatted[df_points_formatted['Service_side'] != '*SWITCH*']
        df_points_formatted = df_points_formatted[df_points_formatted['Service_side'] != 'Timeout']

        # Create a column 'point_winner' which indicates the winner of the point based on the next row
        df_points_formatted['point_winner'] = df_points_formatted['Service_side'].shift(-1)
        # Specific treatment for the last point of each set, which has 'end of set' in the 'Service_side' column of the next row
        for i, row in enumerate(df_points_formatted.itertuples()):
            row_index = df_points_formatted.index[i]
            if i + 1 < len(df_points_formatted):
                next_row_index = df_points_formatted.index[i + 1]
                if df_points_formatted.loc[next_row_index, 'Service_side'] == 'end of set':
                    set_winner = str()
                    # DEV DEBUG - to be removed later
                    print(f"Row index: {row_index}, Service_side: {df_points_formatted.loc[next_row_index, 'Service_side']}")
                    if df_points_formatted.loc[row_index, 'teamA_score'] > df_points_formatted.loc[row_index, 'teamB_score']:
                        set_winner = teamA_name
                    else:
                        set_winner = teamB_name
                    # DEV DEBUG - to be removed later
                    print(f"Set winner: {set_winner}")
                    df_points_formatted.loc[row_index, 'point_winner'] = set_winner

        # Drop the rows with 'end of set' values in the 'Service_side' column
        df_points_formatted = df_points_formatted[df_points_formatted['Service_side'] != 'end of set']

        # Add a column 'game_id' with the value of the game_id for all rows
        df_points_formatted['game_id'] = game_id

        return df_points_formatted

    def load_indexed_df_points_csv_to_db(
            self,
            game_id: str
    ) -> None:
        """Load the indexed points DataFrame for a specific game_id and insert it into the database.
        This method uses the load_indexed_df_points_csv method to get the formatted DataFrame and then inserts it into a new table in the database named 'table_points_{game_id}'.
        The new table will have columns corresponding to the DataFrame's columns, and the method will handle the creation of the table and the insertion of the data.

        Arguments:
            game_id (str): The unique identifier for the game, used to locate the corresponding CSV file and to name the new table in the database.
        """

        df_points_formatted = self.load_indexed_df_points_csv(game_id)

        # Create a new table for the points of this game
        table_name = f"table_points_{game_id}"
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                POINT_ID TEXT PRIMARY KEY,
                Service_side TEXT,
                teamA_score INT,
                teamB_score INT,
                teamA_sets INT,
                teamB_sets INT,
                teamA_score_diff INT,
                teamB_score_diff INT,
                point_winner TEXT,
                game_id TEXT,
                FOREIGN KEY (game_id) REFERENCES table_game(GAME_ID)
            )
        """
        self.execute_query(create_table_query)

        # Insert the data from the DataFrame into the new table
        for _, row in df_points_formatted.iterrows():
            insert_query = f"""
                INSERT OR IGNORE INTO {table_name} (
                    POINT_ID, Service_side, teamA_score, teamB_score, 
                    teamA_sets, teamB_sets, teamA_score_diff, teamB_score_diff, point_winner, game_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.execute_query(insert_query, (
                row['POINT_ID'], row['Service_side'], row['teamA_score'], row['teamB_score'],
                row['teamA_sets'], row['teamB_sets'], row['teamA_score_diff'], row['teamB_score_diff'], row['point_winner'], row['game_id']
            ))

    def load_all_indexed_df_points_csv_to_db(
        self,
        indexed_df_points_dir: str = None
    ) -> None:
        """Load the indexed points DataFrames for a list of game_ids and insert them into the database.
        This method iterates over the provided list of game_ids, calls the load_indexed_df_points_csv_to_db method for each game_id to load and insert the corresponding points data into the database.

        Arguments:
            game_ids (list[str]): A list of unique identifiers for the games, used to locate the corresponding CSV files and to name the new tables in the database.
        """
        if indexed_df_points_dir is None:
            indexed_df_points_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indexed_df_points')
        
        # Get the list of game_ids from the CSV files in the indexed_df_points directory
        game_ids = []
        for filename in os.listdir(indexed_df_points_dir):
            if filename.startswith('indexed_df_points_') and filename.endswith('.csv'):
                game_id = filename[len('indexed_df_points_'):-len('.csv')]
                game_ids.append(game_id)

        print(f"✅ {len(game_ids)} fichiers CSV de points trouvés pour les game_ids suivants : {game_ids}")

        # for game_id in game_ids:
        #     self.load_indexed_df_points_csv_to_db(game_id)

    # ============================================================
    # IMPORT CSV - initial tables : table_players.csv, table_serie.csv, table_game.csv
    # ============================================================
    def load_initial_csv(
        self,
        table_name: str,
        filename: str,
        ignore_fk: bool = False
    ):
        """Imports a CSV file into the specified table.
        This method only works for csv files with column names that exactly match the table schema.
        (initial tables : table_players.csv, table_serie.csv, table_game.csv)

        Arguments:
            table_name (str): Name of the target table in the database.
            filename (str): Name of the CSV file to import.
            ignore_fk (bool): Whether to ignore foreign key constraints during import."""
        
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

    def load_all_csv(
        self
        ):
        """Imports the 3 CSV files in the correct order (parents before children)."""
        self.load_initial_csv("table_players", "table_players.csv")
        self.load_initial_csv("table_serie", "table_serie.csv")
        self.load_initial_csv("table_game", "table_game.csv")


    # ============================================================
    # RÉINITIALISATION
    # ============================================================
    def reset_database(self):
        """Supprime et recrée toutes les tables (utile en développement)."""
        # Disable FK checks to allow dropping tables in any order
        self.conn.execute("PRAGMA foreign_keys = OFF")
        # Suppression dans l'ordre inverse (enfants d'abord)
        self.cursor.execute("DROP TABLE IF EXISTS table_game")
        self.cursor.execute("DROP TABLE IF EXISTS table_serie")
        self.cursor.execute("DROP TABLE IF EXISTS table_players")
        self.conn.commit()
        # Re-enable FK checks after dropping
        self.conn.execute("PRAGMA foreign_keys = ON")
        print("🗑️  Tables supprimées.")
        self.create_initial_tables()
        print("✅ Base réinitialisée.")

    def check_fk_integrity(self):
        """Vérifie les FK de table_game avant import."""

        print("\n🔍 Vérification des Foreign Keys...\n")

        self.cursor.execute("SELECT PAIRE_ID FROM table_players")
        paires = {row[0] for row in self.cursor.fetchall()}

        self.cursor.execute("SELECT SERIE_ID FROM table_serie")
        series = {row[0] for row in self.cursor.fetchall()}

        print(f"   PAIRE_ID disponibles  : {paires}")
        print(f"   SERIE_ID disponibles  : {series}\n")

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
if __name__ == "__main__":
    with DBManager() as db:
        db.reset_database()
        db.create_initial_tables()
        db.load_all_csv()
        db.check_fk_integrity()
