# Packages imports
import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv
load_dotenv()
INDEXED_DF_POINTS_DIR = os.getenv("INDEXED_DF_POINTS_DIR")
RECAP_DICT_SCORE_DIR = os.getenv("RECAP_DICT_SCORE_DIR")
SEGMENTED_POINTS_DIR = os.getenv("SEGMENTED_POINTS_DIR")
ALL_POSSESSION_DIR = os.getenv("ALL_POSSESSION_DIR")

# Local imports
from db_manager import DBManager
sys.path.append(os.path.join(os.path.dirname(__file__), "video_edit_utils"))
from video_edit_utils import (
    montage_operations,
    cut_point_gpu,
    video_rotation,
    cv2_point_segment_cut,
    point_indexeer,
    score_checker,
    extract_segments_from_df_gpu,
    all_possession_game
)


class GameEditor:
    """
    Class to edit a video of a beach volleyball game,
    from the raw video to segmented clips of each point.
    It also returns dictionaries with information about the score.
    """
    # Initialisation of the class with the path to the video(s)
    # and the output directory for the edited videos
    def __init__(
        self,
        video_dir: str = None,
        video_path: str = None,
        output_dir: str = None,
    ) -> None:
        """
        Initializes the GameEditor class.

        Args:
            video_dir (str): Path to the directory containing
                the raw game video(s).
            video_path (str): Path to a specific video file.
            output_dir (str, optional): Path to the output directory.
        """
        self.video_dir = video_dir
        self.video_path = video_path
        self.output_dir = output_dir if output_dir else video_dir

        self.indexed_df_points_dir = INDEXED_DF_POINTS_DIR
        self.recap_dict_score_dir = RECAP_DICT_SCORE_DIR
        self.segmented_point_dir = SEGMENTED_POINTS_DIR
        self.all_possession_dir = ALL_POSSESSION_DIR

    # Method for pre-match editing of the videos
    # in a directory (rotation and pre-match cutting)
    def pre_match_editing(
        self,
        play_speed: float = 1.0,
    ) -> None:
        """Performs pre-match editing of videos in a directory.
        The script uses OpenCV to display the video
        and detect key presses.

        The user must:
            - Indicate the correct video rotation
              (if necessary) using the 'r' and 'l' keys
            - Press '0' to indicate the start of the match,
              and the video is then cut from this point
              using ffmpeg.

        Args:
            video_dir (str): Directory containing
                the video(s) to cut.
            play_speed (float): Video playback speed.
            output_dir (str, optional): Output directory
                for cut videos. If None, cut videos will
                be saved in the same folder as the
                original videos.
        """
        # Initialize a pandas DataFrame to store cutting
        # information for each video
        # (start frame, last frame, rotation)
        columns = [
            "video_path",
            "starting_game_frame",
            "last_game_frame",
            "output_dir",
            "rotation_state",
        ]
        match_info_df = pd.DataFrame(columns=columns)

        # List video files in the directory
        valid_ext = (".mp4", ".avi", ".mov", ".mkv")
        video_files = [
            f for f in os.listdir(self.video_dir) if f.lower().endswith(valid_ext)
        ]
        if not video_files:
            print("No videos found in the " "specified directory.")
            return

        # Apply cv2_actions_to_operate() to each video
        # to retrieve editing actions
        # (start frame, last frame, rotation)
        for video_file in video_files:
            video_path = os.path.join(self.video_dir, video_file)
            print(f"Processing video: {video_path}")

            # Retrieve the editing actions for the video
            montage_actions = dict()
            montage_actions = montage_operations(video_path, play_speed)
            starting_game_frame = montage_actions.get("start_frame", 0)
            last_game_frame = montage_actions.get("last_frame", None)
            rotation_state = montage_actions.get("rotation_state", 0)

            # Store the match start frame and rotation
            # in a temporary pandas DataFrame
            # for subsequent pipeline steps
            out_dir = (
                self.output_dir if self.output_dir else os.path.dirname(video_path)
            )
            match_info_df.loc[len(match_info_df)] = {
                "video_path": video_path,
                "starting_game_frame": starting_game_frame,
                "last_game_frame": last_game_frame,
                "output_dir": out_dir,
                "rotation_state": rotation_state,
            }

        # Apply cut_point_gpu() to each row of match_info_df
        for _, row in match_info_df.iterrows():
            base_name = os.path.splitext(os.path.basename(row["video_path"]))[0]
            output_video = os.path.join(row["output_dir"], f"{base_name}_started.mp4")
            cut_point_gpu(
                video_path=row["video_path"],
                start_frame=int(row["starting_game_frame"]),
                end_frame=int(row["last_game_frame"]),
                output_video=output_video,
            )

        # Apply video_rotation() to each row of
        # match_info_df if rotation_state != 0
        for _, row in match_info_df.iterrows():
            if row["rotation_state"] != 0:
                base_name = os.path.splitext(os.path.basename(row["video_path"]))[0]
                started_path = os.path.join(
                    row["output_dir"], f"{base_name}_started.mp4"
                )
                video_rotation(
                    video_path=started_path,
                    rotation_state=int(row["rotation_state"]),
                    output_dir=row["output_dir"],
                )
                # Delete the intermediate unrotated video
                os.remove(started_path)



    # -----------------------------------------------------------------------------------

    def game_to_segmented_points(
        self,
        team1_name: str,
        team2_name: str,
        rewrite_videos: bool = False,
        temp_indexed_df_point : pd.DataFrame = None,
    ):
        """
        Pipeline from the preprocessed video to
        segmented points videos, with the associated
        score information.

        It creates a csv file with the segments of points extracted,
        and a json file with the suppary of the match

        Args:
            video_path (str): Absolute path to the
                source video to segment.
            team1_name (str): Name of team 1.
            team2_name (str): Name of team 2.
            output_dir (str): Absolute path to the
                output directory where the segmented
                points videos will be stored.
        """
        # If temp_indexed_df_point is provided, skip to the segment extraction step
        if temp_indexed_df_point is not None:
            print("Using provided temp_indexed_df_point for segment extraction.")
            start_again_frame = int(temp_indexed_df_point['end_frame'].iloc[-2])


        # Validate that video_path is set
        if self.video_path is None:
            raise ValueError(
                "video_path is not set. Please provide a valid video_path "
                "when initializing GameEditor."
            )

        # Check if 'indexed_df_points' and 'recap_dict_score' directories exist
        if not os.path.exists(self.indexed_df_points_dir):
            raise ValueError(
                f"Directory '{self.indexed_df_points_dir}' does not exist. "
                f"Please ensure the directory is created before running the pipeline."
            )
        if not os.path.exists(self.recap_dict_score_dir):
            raise ValueError(
                f"Directory '{self.recap_dict_score_dir}' does not exist. "
                f"Please ensure the directory is created before running the pipeline."
            )

        # Retrieve video information from the file name
        game_id = str(os.path.splitext(os.path.basename(self.video_path))[0])
        # Removing '_started' etc. suffixes from the game_id
        if "_started" in game_id:
            game_id = game_id.split("_started", 1)[0] + "_started"
            game_id = game_id.replace("_started", "")

        # Check if the video has already been processed in self.output_dir
        for file in os.listdir(self.output_dir):
            if file.startswith(f"{game_id}_p"):
                if rewrite_videos is False:
                    # Segmented points videos already exist, skip processing
                    print(
                        f"Segmented videos for {game_id} already exist in "
                        f"{self.output_dir}. Skipping processing."
                    )
                    return
                else:
                    # Remove existing segmented videos
                    os.remove(os.path.join(self.output_dir, file))

        # Read the video and extract start_frame and end_frame
        df_points = pd.DataFrame()  # Initialize an empty DataFrame
        df_points = cv2_point_segment_cut(
            video_path=self.video_path, 
            team1_name=team1_name, 
            team2_name=team2_name
        )
        # Index the points
        indexed_df_points = pd.DataFrame()  # Initialize an empty DataFrame
        indexed_df_points = point_indexeer(df_points)
        # Score checking
        recap_dict_score = dict()  # Initialize an empty dictionary
        recap_dict_score = score_checker(indexed_df_points)

        # Save the indexing and score to CSV and JSON files
        indexed_df_points.to_csv(
            path_or_buf=os.path.join(
                self.indexed_df_points_dir, f"indexed_df_points_{game_id}.csv"
            ),
            index=False,
        )
        with open(
            os.path.join(self.recap_dict_score_dir, f"recap_dict_score_{game_id}.json"),
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(recap_dict_score, json_file, indent=4)

        # Extract the segments
        extract_segments_from_df_gpu(
            video_path=self.video_path,
            actions_df=indexed_df_points,
            output_dir=self.output_dir,
        )

    def all_possession_montage(
        self,
        game_id: str,
        video_dir: str = None,
        output_dir: str = None,
    ) -> None:
        """
        Creates a montage video for all possessions of a game.
        Args:
            game_id (str): game_id of the game to create the montage for.
                The game must previously have been segmented into points
                using the game_to_segmented_points() method.
        """
        # Default video directory
        video_dir = self.segmented_point_dir if video_dir is None else video_dir

        # Default output directory
        output_dir = self.all_possession_dir if output_dir is None else output_dir

        # Validate that the game has been segmented
        video_files = [
            f for f in os.listdir(video_dir)
            if f.startswith(f"{game_id}_p") and f.endswith(".mp4")
        ]
        # If no segmented point videos are found, print a message and return
        if not video_files:
            print(
                f"No segmented point videos found for game_id '{game_id}' "
                f"in directory '{video_dir}'."
            )
            return

        ## Create the montage video with all_possession_game()
        # Fetch the information in the database and in the indexed_df_points
        indexed_df_points_csv_path = os.path.join(
            self.indexed_df_points_dir, f"indexed_df_points_{game_id}.csv"
        )
        with DBManager() as db:
            team1_name, team2_name = db.teams_names_from_game_id(
                game_id=game_id
                )
        # Montage the full game
        all_possession_game(
            game_id=game_id,
            video_dir=video_dir,
            indexed_df_points_csv_path=indexed_df_points_csv_path,
            team1_name=team1_name,
            team2_name=team2_name,
            output_dir=output_dir
        )

# -------------------------------------------------------------------

if __name__ == "__main__":
    editor = GameEditor()
    editor.all_possession_montage(
        game_id='JOMR_mar26_VSG_03'
    )
