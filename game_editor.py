# Packages imports
import os
import sys
import pandas as pd
import json

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "video_edit_utils"))
from video_edit_utils import (
    montage_operations,
    cut_point_gpu,
    video_rotation,
    cv2_point_segment_cut,
    point_indexeer,
    score_checker,
    extract_segments_from_df_gpu,
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

        # Validate that video_path is set
        if self.video_path is None:
            raise ValueError(
                "video_path is not set. Please provide a valid video_path "
                "when initializing GameEditor."
            )

        # Check if 'indexed_df_points' and 'recap_dict_score' directories exist
        if not os.path.exists("indexed_df_points"):
            os.makedirs("indexed_df_points")
        if not os.path.exists("recap_dict_score"):
            os.makedirs("recap_dict_score")

        # Retrieve video information from the file name
        game_id = os.path.splitext(os.path.basename(self.video_path))[0]

        # Check if the video has already been processed in self.output_dir
        for file in os.listdir(self.output_dir):
            if file.startswith(f"{game_id}_p"):
                if rewrite_videos == False:
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
            video_path=self.video_path, team1_name=team1_name, team2_name=team2_name
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
                "indexed_df_points", f"indexed_df_points_{game_id}.csv"
            ),
            index=False,
        )
        with open(
            os.path.join("recap_dict_score", f"recap_dict_score_{game_id}.json"),
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
