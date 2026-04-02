import cv2
import pandas as pd
import sys

def cv2_point_segment_cut(
    video_path: str,
    play_speed: float = 1.0,
    team1_name: str = "JOMR",
    team2_name: str = "adversaire",
    start_frame: int = None,
    display_size: tuple = (960, 540)
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
        start_frame (int, optional): Frame number to start processing from.
            Defaults to None.
    Returns:
        DataFrame containing information about the extracted
        point segments, with the columns:
        'point_index', 'action', 'start_frame', 'end_frame',
        'score_team1', 'score_team2', 'set_team1', 'set_team2'
    """
    # Initialize variables
    temp_list = []
    last_action = None
    color_map = dict({
        'black': (0, 0, 0),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'red': (0, 0, 255),
        'white': (255, 255, 255)
    })
    color_map_keys = list(color_map.keys())
    last_action_color_index = 0
    score_color_index = 0
    help_color_index = 0

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
                    color_map[color_map_keys[help_color_index]],
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
                color_map[color_map_keys[score_color_index]],
                2,
                cv2.LINE_AA,
            )
        _orig_imshow(winname, frame)

    cv2.imshow = _imshow_with_help

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Resize the display window to the specified size
    win_name = f'{video_path}'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, display_size[0], display_size[1])

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
    frame_number = 0 if start_frame is None else start_frame

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

                # Resize the window before display
                frame = cv2.resize(
                    frame,
                    display_size,
                    interpolation=cv2.INTER_AREA,
                )

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
                    color_map[color_map_keys[last_action_color_index]],
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
                    color_map['red'],
                    2,
                    cv2.LINE_AA,
                )

            # Display the current frame
            if ret:
                cv2.imshow(f'{video_path}', frame)

            # Handle keyboard input
            key = _wait_key_fast(30) & 0xFF
            if key == ord('h'):
                help_color_index = (help_color_index + 1) % len(color_map_keys)
                continue
            if key == ord('g'):
                last_action_color_index = (last_action_color_index + 1) % len(color_map_keys)
                continue
            if key == ord('j'):
                score_color_index = (score_color_index + 1) % len(color_map_keys)
                continue
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

# Example usage --------------------------------------------------
if __name__ == "__main__":
    video_path = r"C:\Users\habib\Desktop\Montages volley et beach\Jade&Math\matchs preprocess\JOMR_mar26_MON_02_started_rotated.mp4"
    df_points = cv2_point_segment_cut(video_path)
    print(df_points)