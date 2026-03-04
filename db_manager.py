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

    def table_to_dataframe(self, 
                           table_name: str
                           ) -> pd.DataFrame:
        """Exporte une table entière sous forme de DataFrame pandas."""
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, self.conn)
        print(f"✅ Table '{table_name}' exportée : {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    
    # -----------------------------------------------------------------------------------

    def execute_query(self, query, params=None):
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
    def create_tables(self):
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

    # ============================================================
    # IMPORT CSV
    # ============================================================
    def load_csv(self, table_name, filename, ignore_fk=False):
        """Importe un fichier CSV dans la table spécifiée."""
        filepath = os.path.join(self.CSV_DIR, filename)

        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v for k, v in row.items()} for row in reader]

        if rows:
            if ignore_fk:
                self.conn.execute("PRAGMA foreign_keys = OFF")

            columns = ", ".join(rows[0].keys())
            placeholders = ", ".join(["?" for _ in rows[0]])
            query = f"INSERT OR IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
            try:
                self.cursor.executemany(query, [list(row.values()) for row in rows])
                self.conn.commit()
                print(f"✅ {len(rows)} lignes importées dans {table_name}.")
            except sqlite3.IntegrityError as e:
                print(f"❌ Erreur d'intégrité pour {table_name}: {e}")
                print(
                    "   Vérifiez que les données parentes existent dans les tables référencées."
                )
                self.conn.rollback()
            finally:
                if ignore_fk:
                    self.conn.execute("PRAGMA foreign_keys = ON")

    def load_all_csv(self):
        """Importe les 3 CSV dans l'ordre correct (parents avant enfants)."""
        self.load_csv("table_players", "table_players.csv")
        self.load_csv("table_serie", "table_serie.csv")
        self.load_csv("table_game", "table_game.csv")

    # ============================================================
    # RÉINITIALISATION
    # ============================================================
    def reset_database(self):
        """Supprime et recrée toutes les tables (utile en développement)."""
        # Suppression dans l'ordre inverse (enfants d'abord)
        self.cursor.execute("DROP TABLE IF EXISTS table_game")
        self.cursor.execute("DROP TABLE IF EXISTS table_serie")
        self.cursor.execute("DROP TABLE IF EXISTS table_players")
        self.conn.commit()
        print("🗑️  Tables supprimées.")
        self.create_tables()
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
        db.create_tables()
        db.load_all_csv()
        db.check_fk_integrity()
