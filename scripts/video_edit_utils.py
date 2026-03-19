"""Video editing utilities for match segmentation and analysis."""
import os
import subprocess
import sys
import cv2
import pandas as pd

# Environment variables
from dotenv import load_dotenv
load_dotenv()
FFMPEG_PATH = os.getenv("FFMPEG_PATH")
SEGMENTED_POINTS_DIR = os.getenv("SEGMENTED_POINTS_DIR")

# -------------------------------------------------------------------
# Core GPU extraction for a single played point (start-end frames)
# -------------------------------------------------------------------
def cut_point_gpu(
    video_path: str,
    start_frame: int,
    end_frame: int,
    output_video: str
):
    """
    Extract a frame-accurate segment using CUDA + NVENC via
    an optimized GPU ffmpeg command. Start and end frames are
    inclusive.

    Args:
        video_path: path to the source video
        start_frame: first frame of the segment to extract
        end_frame: last frame of the segment to extract
        output_video: path for the generated segment video
    """

    # Video filter to select frames between start_frame and
    # end_frame, and reset timestamps starting from 0
    vf = (
        f"select='between(n,{start_frame},{end_frame})',"
        f"setpts=PTS-STARTPTS"
    )

    # Path to the ffmpeg build compiled with NVENC support
    # for GPU-accelerated extraction
    ffmpeg_path = FFMPEG_PATH

    # ffmpeg command to extract the segment using the GPU
    cmd = [
        ffmpeg_path, "-y",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-vf", vf,
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "constqp",
        "-qp", "18",
        output_video
    ]

    # Run the ffmpeg command to extract the segment using the GPU
    subprocess.run(cmd, check=True)

# -------------------------------------------------------------------
# Video rotation (if needed) with ffmpeg + GPU
# -------------------------------------------------------------------
def video_rotation(
    video_path: str,
    rotation_state: int = 0,
    output_dir: str = None
) -> None:
    """Apply a rotation to the video using ffmpeg.

    Args:
        video_path (str): Path to the video to rotate.
        rotation_state (int): Rotation state (0, 90, 180, 270).
        output_dir (str, optional): Output folder for the rotated
            video. If None, the rotated video is saved in the same
            folder as the original.
    """

    transpose_commands = {
        0: None,  # No rotation
        90: "transpose=1",  # Rotate right
        180: "transpose=1,transpose=1",  # 180-degree rotation
        270: "transpose=2"  # Rotate left
    }

    # Select the rotation to apply
    filter_str = transpose_commands[rotation_state]

    # Determine the output path
    if output_dir is None:
        output_path = (
            f'{os.path.splitext(video_path)[0]}'
            f'_rotated_{rotation_state}.mp4'
        )
    else:
        base_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]
        output_path = os.path.join(
            output_dir,
            f'{base_name}_rotated_{rotation_state}.mp4'
        )

    # ffmpeg command to apply the rotation
    if filter_str is not None:

        # Path to the ffmpeg build compiled with NVENC support
        # for GPU-accelerated rotation
        ffmpeg_path = FFMPEG_PATH

        command = [
            ffmpeg_path,
            '-y',
            '-i', video_path,
            '-vf', filter_str,
            '-c:a', 'copy',  # Copy the audio track without re-encoding
            output_path
        ]

        # Run the ffmpeg command to apply the rotation
        subprocess.run(command, check=True)

# -------------------------------------------------------------------
# Record montage actions for video pre-processing via cv2
# and keyboard interaction, on a single video
# -------------------------------------------------------------------
def montage_operations(
    video_path: str,
    play_speed: float = 1.0
) -> dict:
    """
    Records the montage actions for video pre-processing.

    Args:
        play_speed (float): Video playback speed. Defaults to 1.0.

    Returns:
        dict: Dictionary with keys 'start_frame',
            'last_frame', 'rotation_state'.
    """
    # Ensure a video path has been provided before proceeding
    if video_path is None:
        raise ValueError(
            "video_path must be set before calling "
            "montage_operation()."
        )

    montage_actions = {}
    starting_game_frame = 0

    # Define the help overlay text shown on each frame
    help_lines = [
        "Keys:",
        "q : quit",
        "space : pause/resume",
        "0 : start of match",
        "+ : speed up",
        "- : speed down",
        "r : rotate right",
        "l : rotate left",
    ]

    # Monkey-patch cv2.imshow to overlay help text on every
    # displayed frame
    _orig_imshow = cv2.imshow

    def _imshow_with_help(winname, frame):
        if frame is not None:
            x, y = 30, 120
            for i, line in enumerate(help_lines):
                cv2.putText(
                    frame, line,
                    (x, y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
            _orig_imshow(winname, frame)

    cv2.imshow = _imshow_with_help

    # Open the video and retrieve total frame count for
    # last-frame default
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_game_frame = (
        frame_count - 1 if frame_count > 0 else None
    )
    print(f"Last frame index: {last_game_frame}")

    # Helper to adjust waitKey delay based on current
    # playback speed
    def _wait_key_fast(ms):
        adj = max(1, int(ms / play_speed))
        return cv2.waitKey(adj)

    # Validate that the video was opened successfully
    if not cap.isOpened():
        print("Error: unable to open the video.")
        sys.exit()

    # Initialize playback state variables
    _fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    paused = False
    rotation_state = 0
    ret = False

    try:
        # Main playback loop: read, transform, display,
        # and handle input
        while cap.isOpened():
            # Read the next frame only when not paused
            if not paused:
                ret, frame = cap.read()

                if not ret:
                    print("End of video or read error.")
                    break

                # Apply rotation based on the current
                # rotation state
                if rotation_state == 90:
                    frame = cv2.rotate(
                        frame, cv2.ROTATE_90_CLOCKWISE
                    )
                elif rotation_state == 180:
                    frame = cv2.rotate(
                        frame, cv2.ROTATE_180
                    )
                elif rotation_state == 270:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_90_COUNTERCLOCKWISE,
                    )

                # Overlay playback speed indicator on the frame
                cv2.putText(
                    frame,
                    f"Playback speed: x{play_speed:.1f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA,
                )

                frame_number += 1

            # Show a pause indicator when the video is paused
            if paused and ret:
                cv2.putText(
                    frame, "|| PAUSE ||",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA,
                )

            # Display the current frame (with help overlay
            # via patch)
            if ret:
                cv2.imshow(
                    f'{video_path}', frame
                )

            # Handle keyboard input for playback control
            key = _wait_key_fast(30) & 0xFF
            if key == ord('q'):
                # Quit the playback loop
                break
            if key == ord(' '):
                # Toggle pause/resume
                paused = not paused
            elif key == ord('0'):
                # Mark the current frame as the start
                # of the match
                starting_game_frame = frame_number
                start_time = starting_game_frame / _fps
                print(
                    f"Match start marked at frame "
                    f"{starting_game_frame}, "
                    f"i.e. {start_time:.2f} seconds"
                )
                break
            elif key == ord('+'):
                # Increase playback speed
                play_speed += 0.5
            elif key == ord('-'):
                # Decrease playback speed (minimum 0.5x)
                play_speed = max(0.5, play_speed - 0.5)
            elif key == ord('r'):
                # Rotate the video 90° clockwise
                rotation_state = (
                    rotation_state + 90
                ) % 360
            elif key == ord('l'):
                # Rotate the video 90° counter-clockwise
                rotation_state = (
                    rotation_state - 90
                ) % 360

    finally:
        # Release resources and restore the original
        # cv2.imshow
        cap.release()
        cv2.destroyAllWindows()
        cv2.imshow = _orig_imshow

    # Store and return the collected montage metadata
    montage_actions = {
        'start_frame': starting_game_frame,
        'last_frame': last_game_frame,
        'rotation_state': rotation_state,
    }

    return montage_actions

# -------------------------------------------------------------------
# Cut each played point into a segmented video, based on the
# start-end frames from a pandas DataFrame
# -------------------------------------------------------------------
def extract_segments_from_df_gpu(
    video_path: str,
    actions_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Cut the source video into segments defined by the start-end
    frames of a DataFrame.

    Args:
        input_video (str): path to the source video
        actions_df (pandas.DataFrame): DataFrame containing the
            start-end frames of the segments to extract, with at
            least 'start_frame' and 'end_frame' columns
        output_dir (str): folder to store the extracted clips
    """
    # Create the output folder if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    # Insert the game ID into the name of each extracted clip
    game_id = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    # Build intervals: 1 row = time(Point) - following
    # time(Dead time)
    for row in actions_df.iterrows():
        # Only cut rows corresponding to played points
        # (not timeouts or other actions)
        if not pd.isna(row[1]['point_index']):
            print(
                f"Cutting point {row[1]['point_index']}: "
                f"frames {row[1]['start_frame']} - "
                f"{row[1]['end_frame']}"
            )

            cut_point_gpu(
                video_path=video_path,
                start_frame=int(row[1]["start_frame"]),
                end_frame=int(row[1]["end_frame"]),
                output_video=os.path.join(
                    output_dir,
                    f"{game_id}_p{row[1]['point_index']}.mp4"
                ),
            )

# -------------------------------------------------------------------
# Create a DataFrame for cutting a match into played-point
# segments (accounting for the score)
# -------------------------------------------------------------------
def cv2_point_segment_cut(
    video_path: str,
    play_speed: float = 1.0,
    team1_name: str = "JOMR",
    team2_name: str = "adversaire"
) -> pd.DataFrame:
    """
    Create a DataFrame containing the point segments extracted
    from a match video, with score and team information.
    To be used afterwards with cut_point_gpu to cut the point
    segments from this DataFrame.
    Also allows controlling video playback (pause, speed) to
    facilitate point identification.

    Args:
        video_path (str): Path to the video to process.
        play_speed (float, optional): Video playback speed
            (1=normal, 0=pause, >1=faster). Defaults to 1.0.
        team1_name (str, optional): Name of team 1.
            Defaults to "JOMR".
        team2_name (str, optional): Name of team 2.
            Defaults to "adversaire".
    Returns:
        DataFrame containing information about the extracted
        point segments, with the columns:
        'point_index', 'action', 'start_frame', 'end_frame',
        'score_team1', 'score_team2', 'set_team1', 'set_team2'
    """
    # Initialize variables
    temp_list = []
    last_action = None

    # Map keys to actions
    key_action_map = {
        ord('0'): 'set start',
        ord('1'): f'service {team1_name}',
        ord('3'): f'service {team2_name}',
        ord('2'): 'end of point',
        ord('5'): '*SWITCH*',
        ord('8'): 'Timeout',
    }

    # Display available keys as an overlay on the video
    help_lines = [
        "0 : set start",
        f"1 : service {team1_name}",
        f"3 : service {team2_name}",
        "5 : switch",
        "8 : timeout",
        "4 : back to previous *SWITCH*"
    ]

    # Initialize scores for overlay display
    score_team1 = 0
    score_team2 = 0
    switch_scores_team1 = 0
    switch_scores_team2 = 0

    # Override cv2.imshow to add help overlay and scores
    _orig_imshow = cv2.imshow

    def _imshow_with_help(winname, frame):
        if frame is not None:
            # Display help in the top-left corner
            x, y = 30, 120
            for i, line in enumerate(help_lines):
                cv2.putText(
                    frame,
                    line,
                    (x, y + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Display scores in the bottom-right corner
            h, w = frame.shape[:2]
            score_text = (
                f"{team1_name}: {score_team1}  "
                f"{team2_name}: {score_team2}"
            )
            text_size = cv2.getTextSize(
                score_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 2,
            )[0]
            score_x = w - text_size[0] - 20
            score_y = h - 20
            cv2.putText(
                frame,
                score_text,
                (score_x, score_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        _orig_imshow(winname, frame)

    cv2.imshow = _imshow_with_help

    # Open the video
    cap = cv2.VideoCapture(video_path)

    def _wait_key_fast(ms):
        # Reduce the delay proportionally to the speed
        # (at least 1 ms)
        adj = max(1, int(ms / play_speed))
        return cv2.waitKey(adj)

    if not cap.isOpened():
        print("Error: unable to open the video.")
        sys.exit()

    # Retrieve FPS to convert frames to time
    cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    # Pause and rotation state
    paused = False

    # Video playback loop
    try:
        while cap.isOpened():

            # Read a frame only if not paused
            if not paused:
                ret, frame = cap.read()

                # Stop at end of video or on read error
                if not ret:
                    print("End of video or read error.")
                    break

                # Increment the frame counter
                frame_number += 1

            # Display the last action
            if last_action and ret:
                cv2.putText(
                    frame,
                    f"Last action: {last_action}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Show pause indicator on the displayed frame
            if paused and ret:
                cv2.putText(
                    frame,
                    "|| PAUSE ||",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Display the current frame
            if ret:
                cv2.imshow(f'{video_path}', frame)

            # Handle keyboard input
            key = _wait_key_fast(30) & 0xFF
            if key == ord('q'):
                # Quit
                break
            if key == ord(' '):
                # Pause/resume
                paused = not paused
            elif key == ord('+'):
                # Increase speed
                play_speed += 0.5
                continue
            elif key == ord('-'):
                # Decrease speed
                play_speed = max(0.5, play_speed - 0.5)
                continue
            elif key == ord('4') and len(temp_list) > 0:
                # Find the last "switch" action
                last_switch_index = int(-1)
                for i in range(len(temp_list) - 1, -1, -1):
                    if temp_list[i]['Action'] == '*SWITCH*':
                        last_switch_index = i
                        break
                if last_switch_index != -1:
                    # Remove all actions after the last switch from temp_list
                    temp_list = temp_list[:last_switch_index + 1]
                    # Reset scores at last *SWITCH* and last action
                    score_team1 = switch_scores_team1
                    score_team2 = switch_scores_team2
                    last_action = (
                        temp_list[-1]['Action']
                        if temp_list else None
                    )
                    # Go back to the frame of the last switch action and pause playback
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        temp_list[-1]['Frame'],
                    )
                    frame_number = temp_list[-1]['Frame']
                    paused = True

            elif key in key_action_map:
                # Record the action associated with the key,
                # along with the frame number
                
                ## Match start and set start
                if key == ord('0'):
                    # Reset scores at set start
                    score_team1 = 0
                    score_team2 = 0
                    
                    # Pause playback at set start
                    paused = True

                    # Match start vs set start
                    if len(temp_list) == 0:
                        # "Set start" is actually "match start"
                        # if it is the first recorded action
                        action_name = str('match start')
                    else:
                        # "Set start" after the first one is a real new set
                        action_name = key_action_map[key]
                
                    # Assign the team serving based on second key pressed after the set start or match start
                    # Display overlay asking user to define the serving team
                    if ret:
                        overlay_frame = frame.copy()
                        cv2.putText(
                            overlay_frame,
                            "Define serving team:",
                            (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            overlay_frame,
                            f"1: {team1_name}  |  3: {team2_name}",
                            (30, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(f'{video_path}', overlay_frame)

                    # Wait for the user to press '1' or '3' to define the serving team
                    while True:
                        serve_key = cv2.waitKey(0) & 0xFF
                        if serve_key == ord('1'):
                            action_name = f'{action_name} - service {team1_name}'
                            break
                        elif serve_key == ord('3'):
                            action_name = f'{action_name} - service {team2_name}'
                            break

                else:
                    # For other keys, just record the associated action
                    action_name = key_action_map[key]

                if key == ord('5'):
                    # Save the current scores at the time of the *SWITCH* action
                    # to be able to restore them if we go back to this switch with key '4'
                    switch_scores_team1 = int(score_team1)
                    switch_scores_team2 = int(score_team2)

                last_action = action_name
                temp_list.append({
                    'Frame': frame_number,
                    'Action': action_name
                })

                # Refresh display to update the score overlay
                if ret:
                    cv2.imshow(f'{video_path}', frame)
                # Add a point to the scores based on the action
                if action_name == f'service {team1_name}':
                    score_team1 += 1
                elif action_name == f'service {team2_name}':
                    score_team2 += 1


    finally:
        # Release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()

    # Initialize a list to store point segments
    list_actions = []

    # Process temp_list to create point segments with score
    # and set information
    i = 0
    while i < len(temp_list):
        action = temp_list[i]

        # If the action is not 'end of point', process it
        if action['Action'] != 'end of point':
            # If it is a set start, create an 'end of set' item
            # in list_actions for later processing
            if action['Action'] == f'set start - service {team1_name}' or action['Action'] == f'set start - service {team2_name}':
                list_actions.append({
                    'service_side': 'end of set',
                    'start_frame': int(0),
                    'end_frame': int(0)
                })
            # Then process the action normally
            start_frame = action['Frame']
            service_side = action['Action']
            # Look for the next 'end of point'
            end_frame = None
            j = i + 1
            while j < len(temp_list):
                if temp_list[j]['Action'] == 'end of point':
                    end_frame = temp_list[j]['Frame']
                    break
                j += 1

            # Add to the list if an 'end of point' was found
            if end_frame is not None:
                list_actions.append({
                    'service_side': service_side,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                })

        i += 1

    # Convert list_actions to a DataFrame
    df_points = pd.DataFrame(list_actions)

    # Add columns for scores and sets, initialized to 0
    df_points[f'{team1_name}_score'] = 0
    df_points[f'{team2_name}_score'] = 0
    df_points[f'{team1_name}_sets'] = 0
    df_points[f'{team2_name}_sets'] = 0

    # Update scores based on the service side
    for idx, row in df_points.iterrows():
        if (
            df_points['service_side'].iloc[idx] == f'set start - service {team1_name}'
            or df_points['service_side'].iloc[idx] == f'set start - service {team2_name}'
            or df_points['service_side'].iloc[idx] == f'match start - service {team1_name}'
            or df_points['service_side'].iloc[idx] == f'match start - service {team2_name}'
        ):
            # First row: initialize according to the service
            if row['service_side'] == (
                f'service {team1_name}'
            ):
                df_points.at[
                    idx, f'{team1_name}_score'
                ] = 1
                df_points.at[
                    idx, f'{team2_name}_score'
                ] = 0
            elif row['service_side'] == (
                f'service {team2_name}'
            ):
                df_points.at[
                    idx, f'{team1_name}_score'
                ] = 0
                df_points.at[
                    idx, f'{team2_name}_score'
                ] = 1
        else:
            # Subsequent rows: carry forward the previous score
            # and increment based on the service
            df_points.at[idx, f'{team1_name}_score'] = (
                df_points.at[idx - 1, f'{team1_name}_score']
            )
            df_points.at[idx, f'{team2_name}_score'] = (
                df_points.at[idx - 1, f'{team2_name}_score']
            )

            if row['service_side'] == (
                f'service {team1_name}'
            ):
                df_points.at[
                    idx, f'{team1_name}_score'
                ] += 1
            elif row['service_side'] == (
                f'service {team2_name}'
            ):
                df_points.at[
                    idx, f'{team2_name}_score'
                ] += 1

    # Update set scores at the beginning of each new set
    for idx, row in df_points.iterrows():
        if (
            row['service_side'] == f'set start - service {team1_name}'
            or row['service_side'] == f'set start - service {team2_name}'
        ) and idx > 0:
            # Compare the scores of the previous set
            prev_score_team1 = df_points.at[
                idx - 1, f'{team1_name}_score'
            ]
            prev_score_team2 = df_points.at[
                idx - 1, f'{team2_name}_score'
            ]

            # Carry forward the previous set counts
            df_points.at[idx, f'{team1_name}_sets'] = (
                df_points.at[idx - 1, f'{team1_name}_sets']
            )
            df_points.at[idx, f'{team2_name}_sets'] = (
                df_points.at[idx - 1, f'{team2_name}_sets']
            )

            # Award a set to the winner
            if prev_score_team1 > prev_score_team2:
                df_points.at[
                    idx, f'{team1_name}_sets'
                ] += 1
            else:
                df_points.at[
                    idx, f'{team2_name}_sets'
                ] += 1
        elif idx > 0:
            # For other rows, keep the set count unchanged
            df_points.at[idx, f'{team1_name}_sets'] = (
                df_points.at[idx - 1, f'{team1_name}_sets']
            )
            df_points.at[idx, f'{team2_name}_sets'] = (
                df_points.at[idx - 1, f'{team2_name}_sets']
            )

    # Add an 'end of set' row at the end of df_points for
    # easier downstream processing
    last_idx = len(df_points) - 1
    df_points.loc[len(df_points)] = {
        'service_side': 'end of set',
        'start_frame': int(0),
        'end_frame': int(0),
        f'{team1_name}_score': df_points.at[
            last_idx, f'{team1_name}_score'
        ],
        f'{team2_name}_score': df_points.at[
            last_idx, f'{team2_name}_score'
        ],
        f'{team1_name}_sets': df_points.at[
            last_idx, f'{team1_name}_sets'
        ],
        f'{team2_name}_sets': df_points.at[
            last_idx, f'{team2_name}_sets'
        ],
    }

    # For 'end of set' rows, update the set scores based on
    # the previous set's score
    for idx, row in df_points.iterrows():
        if (
            row['service_side'] == 'end of set'
            and idx > 0
        ):
            # The winner of the previous point/set is the one
            # with the highest score at the previous point
            prev_score_team1 = df_points.at[
                idx - 1, f'{team1_name}_score'
            ]
            prev_score_team2 = df_points.at[
                idx - 1, f'{team2_name}_score'
            ]
            df_points.at[idx, f'{team1_name}_sets'] = (
                df_points.at[idx - 1, f'{team1_name}_sets']
            )
            df_points.at[idx, f'{team2_name}_sets'] = (
                df_points.at[idx - 1, f'{team2_name}_sets']
            )
            if prev_score_team1 > prev_score_team2:
                df_points.at[
                    idx, f'{team1_name}_sets'
                ] += 1
                df_points.at[
                    idx, f'{team1_name}_score'
                ] += 1
            elif prev_score_team2 > prev_score_team1:
                df_points.at[
                    idx, f'{team2_name}_sets'
                ] += 1
                df_points.at[
                    idx, f'{team2_name}_score'
                ] += 1

    return df_points

# -------------------------------------------------------------------
# Point indexer
# -------------------------------------------------------------------
def point_indexeer(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'point_index' column to the points DataFrame, assigning
    a unique point number to each played point (ignoring switch
    and timeout actions).

    Args:
        df (pd.DataFrame): DataFrame containing the extracted
            point segments, with the columns:
            'service_side', 'start_frame', 'end_frame',
            'score_team1', 'score_team2', 'set_team1',
            'set_team2'
    Returns:
        pd.DataFrame: DataFrame with an additional 'point_index'
            column assigning a unique point number to each
            played point.
    """
    if 'point_index' in df.columns:
        print(
            "The 'point_index' column already exists in the "
            "DataFrame. Please remove or rename the existing "
            "column before running this function."
        )
        return df

    if not all(col in df.columns for col in ['service_side']):
        print(
            "Error: the DataFrame does not contain the "
            "'service_side' column. Please check the "
            "DataFrame columns."
        )
        return df

    point_idx = int(0)
    point_indices = []
    excluded = ('*SWITCH*', 'Timeout', 'end of set')
    for _, row in df.iterrows():
        if row['service_side'] not in excluded:
            point_idx += int(1)
        point_indices.append(
            point_idx
            if row['service_side'] not in excluded
            else None
        )

    df['point_index'] = pd.array(
        point_indices, dtype=pd.Int64Dtype()
    )

    return df

# -------------------------------------------------------------------
# Score checker
# -------------------------------------------------------------------
def score_checker(
    df_points: pd.DataFrame,
) -> dict:
    """
    Check score consistency against side switches.
    After a *SWITCH*, the sum of the '_score' columns must equal
    a multiple of 5 or 7 (depending on points per set).
    Reports the match format (15 or 21 points per set) based on
    the multiple found.
    Also reports the final score, with per-set detail, for both
    teams.
    If an inconsistency is detected, returns an error message
    indicating the problem.

    Args:
        df_points: DataFrame containing the extracted point
            segments, with columns: 'point_index', 'action',
            'start_frame', 'end_frame', 'score_team1',
            'score_team2', 'set_team1', 'set_team2'
    Returns:
        dict: Dictionary containing match format info and the
            final score.
    """
    recap_dict = {
        'teams': (
            df_points.columns[3].replace('_score', ''),
            df_points.columns[4].replace('_score', ''),
        ),
        'match_format': None,
        'winner': None,
        'final_score': None,
        'score_by_set': [],
    }

    # Remove 'Timeout' rows to avoid skewing calculations
    df_points = df_points[
        df_points['service_side'] != 'Timeout'
    ].reset_index(drop=True)

    # Retrieve team names from the DataFrame columns
    team1_name = df_points.columns[3].replace('_score', '')
    team2_name = df_points.columns[4].replace('_score', '')

    # Initialize variables for score and match format
    score_switch_points = []
    for idx, row in df_points.iterrows():
        if row['service_side'] == '*SWITCH*':
            if idx + 1 < len(df_points):
                score_sum = (
                    df_points.at[
                        idx + 1, f'{team1_name}_score'
                    ]
                    + df_points.at[
                        idx + 1, f'{team2_name}_score'
                    ]
                )
            else:
                score_sum = (
                    row[f'{team1_name}_score']
                    + row[f'{team2_name}_score']
                )
            score_switch_points.append(score_sum)

    # Check for multiples of 5 or 7
    multiples_of_5 = all(
        score % 5 == 0 for score in score_switch_points
    )
    multiples_of_7 = all(
        score % 7 == 0 for score in score_switch_points
    )
    if multiples_of_5:
        match_format = "15 points per set"
    elif multiples_of_7:
        match_format = "21 points per set"
    else:
        # Report the detected inconsistency and the scores
        # at the time of switches
        print(
            "Possible inconsistency detected: scores at "
            "switch time are not multiples of 5 or 7."
        )
        print(
            "Scores at switch time:",
            score_switch_points,
        )
        match_format = "(to be checked manually)"

    # Retrieve the final score and per-set detail
    recap_dict['match_format'] = match_format
    final_score_team1 = df_points[
        f'{team1_name}_sets'
    ].iloc[-1]
    final_score_team2 = df_points[
        f'{team2_name}_sets'
    ].iloc[-1]
    recap_dict['final_score'] = (
        f"{final_score_team1} - {final_score_team2}"
    )

    # Determine the winner
    if final_score_team1 > final_score_team2:
        recap_dict['winner'] = team1_name
    else:
        recap_dict['winner'] = team2_name

    # Per-set detail
    set_count = 0
    current_set_scores = []
    for idx, row in df_points.iterrows():
        if row['service_side'] == 'end of set':
            current_set_scores.append({
                'set': set_count + 1,
                'score': (
                    f"{row[f'{team1_name}_score']}"
                    f" - "
                    f"{row[f'{team2_name}_score']}"
                ),
            })
            set_count += 1
        elif idx == len(df_points) - 1:
            # Last row of the DataFrame
            current_set_scores.append({
                'set': set_count + 1,
                'score': (
                    f"{row[f'{team1_name}_score']}"
                    f" - "
                    f"{row[f'{team2_name}_score']}"
                ),
            })

    recap_dict['score_by_set'] = current_set_scores

    return recap_dict

# -------------------------------------------------------------------
# Basic action grader
# -------------------------------------------------------------------
def basic_action_grader(
        video_path: str,
        point_id: str,
        paire_id: str,
        player_a: str,
        player_b: str,
        action_to_grade: str ="serve" or "pass",
        play_speed: float = 1.0,
) -> dict:
    """
    Grade a specific action for a given player across a points of,
    based on user input while watching the video.
    The user can assign a grade from 0 to 3.
    The function returns a dictionary containing the player's name, the action graded,
    and the assigned grade for this point.

    Args:
        video_path (str): Path to the specific point video.
        player_a (str): Name of the first player whose actions are to be graded.
        player_b (str): Name of the second player whose actions are to be graded.
        action_to_grade (str): Type of action to grade (e.g.,
            'service', 'attack').
        play_speed (float, optional): Video playback speed
    Returns:
        dict: Dictionary containing the action graded for a player
    """
    # Initialize variables
    action_graded = dict()
    player_to_grade = str()
    grade = str()
    video_loop_active = bool(True)
    quit_grading = bool(False)

    # Define the grading options for serve and pass
    dict_grades_result=dict({
        'serve': {
            0: 'undetermined',
            1: 'error',
            2: 'good pass',
            3: 'average pass',
            4: 'out-of-system pass',
            5: 'ace'
        },
        'pass': {
            0: 'undetermined',
            1: 'error',
            2: 'out-of-system pass',
            3: 'average pass',
            4: 'good pass',
            5: '<NA>'
        }
    })

    # Map keys to player selected and grade assigned
    key_player_map = dict({
        ord('7'): player_a,
        ord('9'): player_b,
    })
    key_grade_result = dict({
        ord('0'): dict_grades_result[action_to_grade][0],
        ord('1'): dict_grades_result[action_to_grade][1],
        ord('2'): dict_grades_result[action_to_grade][2],
        ord('3'): dict_grades_result[action_to_grade][3],
        ord('4'): dict_grades_result[action_to_grade][4],
        ord('5'): dict_grades_result[action_to_grade][5]
    })

    # Display available keys as an overlay on the video
    help_lines = list([
        f"Grading {action_to_grade} for :",
        f"{player_a} (7) and {player_b} (9)",
        f"0 : {dict_grades_result[action_to_grade][0]}",
        f"1 : {dict_grades_result[action_to_grade][1]}",
        f"2 : {dict_grades_result[action_to_grade][2]}",
        f"3 : {dict_grades_result[action_to_grade][3]}",
        f"4 : {dict_grades_result[action_to_grade][4]}",
        f"5 : {dict_grades_result[action_to_grade][5]}",
    ])
    
    # Override cv2.imshow to add help overlay and scores
    _orig_imshow = cv2.imshow

    def _imshow_with_help(winname, frame):
        if frame is not None:
            # Display help in the top-left corner
            x, y = 30, 120
            for i, line in enumerate(help_lines):
                cv2.putText(
                    frame,
                    line,
                    (x, y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        _orig_imshow(winname, frame)

    cv2.imshow = _imshow_with_help

    # Open the video
    cap = cv2.VideoCapture(video_path)

    def _wait_key_fast(ms):
        # Reduce the delay proportionally to the speed
        # (at least 1 ms)
        adj = max(1, int(ms / play_speed))
        return cv2.waitKey(adj)

    if not cap.isOpened():
        print("Error: unable to open the video.")
        sys.exit()

    # Pause and state
    paused = False
    frame = int(0)

    # Video playback loop
    try:
        while video_loop_active:
            # Open the video if not already open
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("Error: unable to open the video.")
                    break

            # Read a frame only if not paused
            if not paused:
                ret, frame = cap.read()

                # Loop the video when it reaches the end
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

            # Display the player being graded and the grade assigned
            if action_to_grade and ret:
                cv2.putText(
                    frame,
                    f"player: {player_to_grade}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{action_to_grade} grade: {grade}",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Display the current frame
            if ret:
                cv2.imshow(f'{video_path}', frame)

            # Keyboard player input
            key = _wait_key_fast(30) & 0xFF
            if key == ord('q'):
                # Stop the whole grading process and exit
                quit_grading = True
                break
            if key == ord('8') and player_to_grade and grade:
                # Confirm the grade and exit
                video_loop_active = False
            elif key == ord('\r'):
                # Enter key: restart playback from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
            if key == ord(' '):
                # Pause/resume
                paused = not paused
            elif key == ord('+'):
                # Increase speed
                play_speed += 0.5
                continue
            elif key == ord('-'):
                # Decrease speed
                play_speed = max(0.5, play_speed - 0.5)
                continue
            # Action grading keys
            elif key in key_player_map:
                player_to_grade = key_player_map[key]
            elif key in key_grade_result and player_to_grade:
                grade = key_grade_result[key]   

                # Pause to confirm the grade
                paused = True

                # Store the graded action in the dictionary
                action_graded = {
                    'point_id': point_id,
                    'paire_id': paire_id,
                    'player': player_to_grade,
                    'action': action_to_grade,
                    'grade': grade
                }
            
    finally:
        # Release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        cv2.imshow = _orig_imshow

    return action_graded, quit_grading

# -------------------------------------------------------------------
# All possession game
# -------------------------------------------------------------------

def all_possession_game(
        video_dir: str,
        game_id: str,
        indexed_df_points_csv_path: str,
        team1_name: str = None,
        team2_name: str = None,
        output_video_name: str = None,
        output_dir: str = None,
    ) -> None:
    """
    Do a montage of all the possessions of a game, based on the extracted
    points segments. If a CSV path is provided, it reads the indexed points 
    DataFrame from the CSV file and displays the score on the video output.
    """
    # Check if the video directory is NOT the segmented points directory
    if video_dir == SEGMENTED_POINTS_DIR:
        raise ValueError(f"""WARNING: The video directory cannot 
                         be the same as the segmented points directory.""")

    # Set the output directory to the video directory if not provided
    output_dir = video_dir if output_dir is None else output_dir

    # Get all .mp4 files in the video directory
    video_files = [
        os.path.join(video_dir, f) for f in os.listdir(video_dir)
        if f.endswith('.mp4')
    ]
    # Rename files to ensure correct order in the montage (p1, p2, ..., p10, etc.)
    replace_dict = {
        'p1.mp4': 'p01.mp4',
        'p2.mp4': 'p02.mp4',
        'p3.mp4': 'p03.mp4',
        'p4.mp4': 'p04.mp4',
        'p5.mp4': 'p05.mp4',
        'p6.mp4': 'p06.mp4',
        'p7.mp4': 'p07.mp4',
        'p8.mp4': 'p08.mp4',
        'p9.mp4': 'p09.mp4',
    }
    for filename in video_files:
        new_filename = filename
        for old, new in replace_dict.items():
            new_filename = new_filename.replace(old, new)
        os.rename(filename, new_filename)
    
    # Error message if no .mp4 files found in the directory 
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return
        # # [DEV] Get just the file names without the directory for printing
    # video_file_names = [os.path.basename(f) for f in video_files]
    # print(video_file_names)


    # Open the first video to get properties
    cap = cv2.VideoCapture(video_files[0])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()



    # Create VideoWriter for the output montage
    if output_video_name is not None:
        output_path = os.path.join(output_dir, output_video_name)
    else:
        output_path = os.path.join(output_dir, f'all_possessions_montage_{game_id}.mp4')
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    # Load the indexed points DataFrame from the CSV file
    indexed_df_points = pd.DataFrame()
    indexed_df_points = pd.read_csv(indexed_df_points_csv_path)
    # Retrieve team names from the DataFrame columns
    team1_id = indexed_df_points.columns[3].replace('_score', '')
    team2_id = indexed_df_points.columns[4].replace('_score', '')

    # Ensure the DataFrame is sorted by point_index
    indexed_df_points = indexed_df_points.sort_values(by='point_index').reset_index(drop=True)
    # Create a list of scores for each point index
    test_score_list = list(zip(
        indexed_df_points[f'{team1_id}_score'],
        indexed_df_points[f'{team2_id}_score'],
        indexed_df_points[f'{team1_id}_sets'],
        indexed_df_points[f'{team2_id}_sets']
    ))

    if game_id is not None:
        output_video_name = f'all_possessions_montage_{team1_name}_vs_{team2_name}.mp4'
    else:
        output_video_name = f'all_possessions_montage_{game_id}.mp4'

    # Concatenate all videos, with the score overlay 
    for i, video_file in enumerate(video_files):
        cap = cv2.VideoCapture(video_file)
        score = test_score_list[i] if i < len(test_score_list) else (0, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Overlay score on the frame
            score_text = f"""{team1_id}: {score[0]} - {team2_id}: {score[1]}\nSets: {team1_id} {score[2]} - {team2_id} {score[3]}"""

            cv2.putText(
                frame,
                score_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            out.write(frame)
        cap.release()

    out.release()

    # Remove the mp4 files in video_dir after montage is created
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
        if f.endswith('.mp4')]
    for video_file in video_files:
        os.remove(video_file)

    print(f"Montage saved to: {output_path}")


# -------------------------------------------------------------------
# Testing in main script
if __name__ == "__main__":
    # Example usage of the functions
    all_possession_game(
        video_dir=r'C:\Users\habib\Desktop\Montages volley et beach\Jade&Math\matchs preprocess\test_all_possessions',
        indexed_df_points_csv_path=r'C:\Users\habib\Documents\GitHub\DataBeach\indexed_df_points\indexed_df_points_JOMR_jan26_MBV_02.csv',
        team1_name='Jade et Math',
        team2_name='Alex Neel et Emeraude Panaget',
        game_id='JOMR_jan26_MBV_02',
        output_dir=r'C:\Users\habib\Desktop\Montages volley et beach\Jade&Math\matchs preprocess\test_all_possessions\all_poss'
    )