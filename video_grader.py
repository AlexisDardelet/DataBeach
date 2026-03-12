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
        """Grades the service or passing actions in a beach volleyball game video.

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
            raise ValueError("At least one of game_id or serie_id must be provided.")
        if serve_or_pass not in ["serve", "pass"]:
            raise ValueError("serve_or_pass must be either 'serve' or 'pass'.")
        if game_id is not None and serie_id is not None:
            raise ValueError("Only one of game_id or serie_id should be provided, not both.")

        # Initiating variables
        points_ids = list()

        ## SERVICE GRADING ##################################################
        if serve_or_pass == "serve":

            # Loading the point_ids to grade according to the provided arguments
            with DBManager() as db:
                if game_id is not None: # Only 1 game to grade
                        db.cursor.execute(
                            """SELECT POINT_ID, Service_side 
                            FROM table_points 
                            WHERE game_id = ? AND Service_side = ?""",
                            (game_id,self.paire_id)
                        )
                elif serie_id is not None: # All games in the serie to grade
                        db.cursor.execute(
                            """SELECT POINT_ID, Service_side 
                            FROM table_points
                            WHERE game_id IN
                                (SELECT GAME_ID
                                FROM table_game
                                WHERE serie = ?)
                            AND Service_side = ?""",
                            (serie_id,self.paire_id)
                        )
                # Fetching the results and storing the point_ids in a list       
                points_ids = [row[0] for row in db.cursor.fetchall()]

        ## PASSING GRADING ##################################################
        elif serve_or_pass == "pass":

            # Loading the point_ids to grade according to the provided arguments
            with DBManager() as db:
                if game_id is not None: # Only 1 game to grade
                        db.cursor.execute(
                            """SELECT POINT_ID, Service_side 
                            FROM table_points 
                            WHERE game_id = ? AND Service_side != ?""",
                            (game_id,self.paire_id)
                        )
                elif serie_id is not None: # All games in the serie to grade
                        db.cursor.execute(
                            """SELECT POINT_ID, Service_side 
                            FROM table_points
                            WHERE game_id IN
                                (SELECT GAME_ID
                                FROM table_game
                                WHERE serie = ?)
                            AND Service_side != ?""",
                            (serie_id,self.paire_id)
                        )
                # Fetching the results and storing the point_ids in a list       
                points_ids = [row[0] for row in db.cursor.fetchall()]  

        # DEV DEBUG [REMOVE LATER]
        # print(f"Points to grade : {points_ids}")

        # Fetching the player names in table_players for the game_id or serie_id provided
        with DBManager() as db:
            db.cursor.execute(
                """SELECT Name_joueurA, Name_joueurB 
                FROM table_players 
                WHERE PAIRE_ID = ?""",
                (self.paire_id,)
            )
            result = db.cursor.fetchone()
            if result: 
                player_a, player_b = result
                # DEV DEBUG [REMOVE LATER]
                print(f"Player A: {player_a}, Player B: {player_b}")

        # Initiate a list to store the grades for each action in each point
        actions_grades_list = list()
        quit_grading = bool(False)

        # Looping through the points to grade and applying the basic_action_grader
        for point_id in points_ids:
            if quit_grading:
                break
            # Constructing the path to the segmented video for the point
            video_path = os.path.join(
                self.segmented_points_dir,
                f"{point_id}.mp4"
            )
            # Grading the action in the video and storing the results in a list
            action_grades, quit_grading = basic_action_grader(
                video_path=video_path,
                point_id=point_id,
                player_a=player_a,
                player_b=player_b,
                action_to_grade=serve_or_pass,
            )
            # Appending the grades for the current action to the main list
            actions_grades_list.append(action_grades)

        if not quit_grading:
            # Saving the list in a JSON file, dated with the current date and time
            # for history and traceability purposes, and to be able to reuse it later if needed
            datetime_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
            with open(f'list_grades_{serve_or_pass}_{game_id}_{datetime_str}.json', 'w') as f:
                json.dump(actions_grades_list, f, indent=2)
            
            # Entering the grades in the database
            with DBManager() as db:
                # If rewrite_db is False, using a query
                # with ON CONFLICT () DO NOTHING
                if not rewrite_db:
                    for action_graded in actions_grades_list:
                        db.cursor.execute(
                            f"""INSERT INTO table_{serve_or_pass} 
                            (point_id, paire_id, player, action, grade)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT (POINT_ID) DO NOTHING""",
                            (
                                action_graded['point_id'],
                                self.paire_id,
                                action_graded['player'],
                                action_graded['action'],
                                action_graded['grade']
                            )
                        )
                # If rewrite_db is True, using a query
                # with ON CONFLICT () DO UPDATE SET
                else:
                    for action_graded in actions_grades_list:
                        db.cursor.execute(
                            f"""INSERT INTO table_{serve_or_pass} 
                            (point_id, paire_id, player, action, grade)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT (POINT_ID) DO UPDATE SET""",
                            (
                                action_graded['point_id'],
                                self.paire_id,
                                action_graded['player'],
                                action_graded['action'],
                                action_graded['grade']
                            )
                        )


        return actions_grades_list

#######################################################################################
# Main script for testing the VideoGrader class 

if __name__ == "__main__":
    grader = VideoGrader(paire_id="JOMR")
    test_list = grader.service_passing_grading(
        serve_or_pass="pass",
        game_id="JOMR_nov25_BSD_02",
        serie_id=None,
        rewrite_db=False
    )
    print(test_list)