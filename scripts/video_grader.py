# Packages imports
import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv
import datetime

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "db_manager"))
from db_manager import DBManager
sys.path.append(os.path.join(os.path.dirname(__file__), "video_edit_utils"))
from video_edit_utils import basic_action_grader

# Environment variables
load_dotenv()
SEGMENTED_POINTS_DIR = os.getenv("SEGMENTED_POINTS_DIR")

class VideoGrader:
    """
    Class to grade actions in beach volleyball games videos
    Its powered by cv2 package, and it uses the DBManager class to load the
    specific points for the method used, and to save the results in the database.
    """
    def __init__(
        self,
        paire_id: str
        ) -> None:
        """
        Initializes the VideoGrader class.
        Args:
            video_dir (str): The directory containing the video files.
            paire_id (str): The unique identifier for the pair.
        """
        self.segmented_points_dir = SEGMENTED_POINTS_DIR
        self.paire_id = paire_id
    
    # ==============================================================================

    def service_passing_grading(
        self,
        serve_or_pass: str = "serve" or "pass",
        game_id: str = None,
        serie_id: str = None,
        rewrite_db: bool = False,
    ) -> None:
        """
        Grades the service or passing actions in a beach volleyball game video.

        This method uses the DBManager class to load the indexed points data
        for the specified identifiers, and then processes the video to grade
        the service or passing actions. The results are saved back to the database.

        Args:
            serve_or_pass (str): A string indicating whether to grade 'serve' or 'pass'.
            game_id (str): The unique identifier for the game. Defaults to None.
            serie_id (str): The unique identifier for the series. Defaults to None.
            rewrite_db (bool): Whether to rewrite the database with new results. Defaults to False.

        Raises:
            ValueError: If none of game_id or serie_id are provided.
        """
        # Validate the input parameters
        if game_id is None and serie_id is None:
            raise ValueError(
                "At least one of game_id or serie_id must be provided."
            )
        if serve_or_pass not in ["serve", "pass"]:
            raise ValueError(
                "serve_or_pass must be either 'serve' or 'pass'."
            )
        if game_id is not None and serie_id is not None:
            raise ValueError(
                "Only one of game_id or serie_id should be provided, not both."
            )

        # Initiating variables
        points_ids = list()

        ## SERVICE GRADING ##################################################
        if serve_or_pass == "serve":
            # Loading the point_ids to grade according to the provided arguments
            with DBManager() as db:
                if game_id is not None:  # Only 1 game to grade
                    db.cursor.execute(
                        """SELECT point_id, service_side
                        FROM table_point 
                        WHERE game_id = ? AND service_side = ?""",
                        (game_id, self.paire_id),
                    )
                elif serie_id is not None:  # All games in the serie to grade
                    db.cursor.execute(
                        """SELECT point_id, service_side 
                        FROM table_point
                        WHERE game_id IN
                            (SELECT GAME_ID
                            FROM table_game
                            WHERE serie = ?)
                        AND service_side = ?""",
                        (serie_id, self.paire_id),
                    )
                # Fetching the results and storing the point_ids in a list
                points_ids = [row[0] for row in db.cursor.fetchall()]

        ## PASSING GRADING ##################################################
        elif serve_or_pass == "pass":
            # Loading the point_ids to grade according to the provided arguments
            with DBManager() as db:
                if game_id is not None:  # Only 1 game to grade
                    db.cursor.execute(
                        """SELECT point_id, service_side
                        FROM table_point 
                        WHERE game_id = ? AND service_side != ?""",
                        (game_id, self.paire_id),
                    )
                elif serie_id is not None:  # All games in the series to grade
                    db.cursor.execute(
                        """SELECT point_id, service_side 
                        FROM table_point
                        WHERE game_id IN
                            (SELECT GAME_ID
                            FROM table_game
                            WHERE serie = ?)
                        AND service_side != ?""",
                        (serie_id, self.paire_id),
                    )
                # Fetching the results and storing the point_ids in a list
                points_ids = [row[0] for row in db.cursor.fetchall()]

        # Fetching the player names in table_player for the game_id or serie_id provided
        with DBManager() as db:
            db.cursor.execute(
                """SELECT player_a, player_b 
                FROM table_player 
                WHERE PAIRE_ID = ?""",
                (self.paire_id,),
            )
            result = db.cursor.fetchone()
            if result:
                player_a, player_b = result

        # Initiate a list to store the grades for each action in each point
        actions_grades_list = list()
        quit_grading = bool(False)

        # Create the table if it doesn't exist in the database
        with DBManager() as db:
            db.create_simple_actions_table(serve_or_pass)

        # Looping through the points to grade and applying the basic_action_grader
        for point_id in points_ids:
            if quit_grading:
                break
            # Constructing the path to the segmented video for the point
            video_path = os.path.join(
                self.segmented_points_dir,
                f"{point_id}.mp4",
            )

            # Grading the action in the video and storing the results in a list
            action_grades, quit_grading = basic_action_grader(
                video_path=video_path,
                point_id=point_id,
                paire_id=self.paire_id,
                player_a=player_a,
                player_b=player_b,
                action_to_grade=serve_or_pass,
            )
            # Appending the grades for the current action to the main list
            actions_grades_list.append(action_grades)

        if not quit_grading:
            # Saving the list in a JSON file, dated with the current date and time
            # for history and traceability purposes, and to be able to reuse it later if needed
            datetime_str = str(
                datetime.datetime.now().strftime("%Y-%m-%d")
            )
            actions_graded_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "actions_graded",
            )
            # Ensure the actions_graded directory exists
            os.makedirs(actions_graded_dir, exist_ok=True)
            # Build the JSON file path
            game_or_serie_id = str(game_id if game_id is not None else serie_id)
            json_filename = (
                f"list_grades_{serve_or_pass}_{game_or_serie_id}_{self.paire_id}_graded_at_{datetime_str}.json"
            )
            json_filepath = os.path.join(actions_graded_dir, json_filename)
            # Write the grades list to the JSON file
            with open(json_filepath, "w") as f:
                json.dump(actions_grades_list, f, indent=2)

            # Entering the grades in the database
            with DBManager() as db:
                # Insert the grades into the action table
                insert_query = (
                    f"""
                    INSERT INTO table_{serve_or_pass} (
                        point_id, paire_id, player, action, grade
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(point_id, player, action) 
                    DO {"UPDATE SET action=excluded.action, grade=excluded.grade, player=excluded.player" if rewrite_db else "NOTHING"}
                    """
                )
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
                    with db.conn:
                        db.cursor.executemany(insert_query, rows_to_insert)
                    print(
                        f"""{len(actions_grades_list)} grades for {serve_or_pass} actions have been {'updated' if rewrite_db else 'inserted'} in the database."""
                    )
                except Exception as e:
                    print(f"❌ Error inserting grades into the database: {e}")

        # poin_won querry update according to the grades for serve or pass
        # and false_aces_corrector() [ONLY FOR SERVE GRADING]
        if serve_or_pass == 'serve':
            with DBManager() as db:
                point_won_query = """
                UPDATE table_serve
                SET point_won = CASE
                    WHEN paire_id = (
                        SELECT tp.point_winner
                        FROM table_point AS tp
                        WHERE tp.point_id = table_serve.point_id
                    ) THEN 1
                    ELSE 0
                END
                """
                db.execute_query(point_won_query)
                # False aces and direct serve errors correction
                db.false_aces_corrector()

    # -----------------------------------------------------------------------
    # Missing games to grade for serve and pass
    # -------------------------------------------------------------------------

    def missing_games_to_grade(self,
                               action_to_grade: str,
                               ) -> list:
        """ For a specified action, gives the games that are still to be graded
        according to the content of the database.
        [DEVELOPMENT IN PROGRESS] So far, only available for 'serve' and 'pass'

        Args:
            action_to_grade (str): The action to check for missing grades.
        Returns:
            list: A list of game_ids that are missing grades for the specified action.
        """
        if action_to_grade not in ['serve', 'pass']:
            raise ValueError("action_to_grade must be either 'serve' or 'pass'")
        

        with DBManager() as db:
            # Point_ids
            sub_subquery = str(f"SELECT point_id FROM table_{action_to_grade}")
            
            subquery = str(f"""
                SELECT tp.game_id
                FROM table_point AS tp
                LEFT JOIN table_{action_to_grade} AS ta ON tp.point_id = ta.point_id
                WHERE tp.point_id IN ({sub_subquery})
                GROUP BY tp.game_id
                """)

            db.execute_query(
                f"""SELECT game_id 
                FROM table_game

                WHERE game_id NOT IN ({subquery})
                """
            )
            result = db.cursor.fetchall()
        missing_serve_game_ids_list = [row[0] for row in result]

        print(f"""[DEV] {len(missing_serve_game_ids_list)} games missing {action_to_grade} grades: 
              {missing_serve_game_ids_list}""")

        return missing_serve_game_ids_list

#######################################################################################
# Main script for testing the VideoGrader class 

if __name__ == "__main__":
    grader = VideoGrader(paire_id='JOMR')
    # grader.service_passing_grading(
    #     serie_id='MBV_S2-500_F_nov25',
    #     # game_id='JOMR_nov24_Leuven_01',
    #     serve_or_pass='serve',
    #     rewrite_db=False,
    #     )
    grader.missing_games_to_grade(action_to_grade='serve')
    