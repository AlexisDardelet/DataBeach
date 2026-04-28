## Main page for the Streamlit interface of DataBeach
import streamlit as st
from streamlit_option_menu import option_menu
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import cv2
import datetime
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
import subprocess

# Local imports
from streamlit_utils import (
    assign_game_id_to_video_name,
    select_folder
)
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scripts'
    )
)
from firestore_manager import FirestoreManager
from game_editor import GameEditor
from video_grader import VideoGrader
from etl_utils import video_file_renamer
from video_edit_utils import *

# Environment variables
load_dotenv()
ROOT_VIDEO_DIR = os.getenv("ROOT_VIDEO_DIR")
SEGMENTED_POINTS_DIR = os.getenv("SEGMENTED_POINTS_DIR")

######################################################################

def editor_interface():
    """Main editor interface function with two workflow modes."""
    paire_id = st.session_state.get("paire_id", None)
    game_ids = st.session_state.get("game_ids", [])

    folder = st.session_state.get("video_dir", None)
    output_folder = st.session_state.get("output_dir", None)

    ########################################################
    # Pre-processing mode (by default) #####################
    # Raw video ingestion and initial file organization   #
    ########################################################
    if st.session_state.get(
        "editor_mode", "Pre-processing"
    ) == "Pre-processing":
        
        st.title(
            body="🪨➡️🎞️ Preprocessing mode ↩️🏷️🔢",
            )

        launch_preprocessing = None

        col1, col2 = st.columns([1, 2.5])
        col3, col4 = st.columns([1, 2.5])
        col5, col6, col7 = st.columns([1, 1, 1])
        col_dict_ids, col_success = st.columns([4, 1])

        # Select raw games folder
        with col1:
            if st.button(
                label="📁➡️ Select raw games folder",
                use_container_width=True,
                type="secondary",
            ):
                folder = select_folder()
                if folder:
                    st.session_state["video_dir"] = folder

        # Display selected raw games folder path
        with col2:
            if "video_dir" in st.session_state:
                st.markdown(
                    f"""
                    <style>
                    .small-success {{
                        font-size: 15px;
                        padding: 10px;
                        background-color: #1e5631;
                        border-radius: 10px;
                        color: white;
                    }}
                    </style>
                    <div class="small-success">
                    ✅ {st.session_state["video_dir"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Select preprocessed output folder
        with col3:
            if st.button(
                label="➡️📁 Select preprocessed folder",
                use_container_width=True,
                type="secondary",
            ):
                output_folder = select_folder()
                if output_folder:
                    st.session_state["output_dir"] = output_folder

        # Display selected preprocessed folder path
        if "output_dir" in st.session_state:
            with col4:
                st.markdown(
                    f"""
                    <style>
                    .small-success {{
                        font-size: 15px;
                        padding: 10px;
                        background-color: #1e5631;
                        border-radius: 10px;
                        color: white;
                    }}
                    </style>
                    <div class="small-success">
                    ✅ {st.session_state['output_dir']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Assign game_ids to raw video files - interface for mapping videos to games
        if (
            "video_dir" in st.session_state
            and "output_dir" in st.session_state
        ):
            with col5:
                # Get list of video files from the selected directory
                video_files = [
                    video
                    for video in os.listdir(st.session_state["video_dir"])
                    if video.endswith(('.mp4', '.avi', '.mov','MP4', '.AVI', '.MOV'))
                ]

                # Initialize dictionary to track game_id assignments
                if "assign_game_ids_dict" not in st.session_state:
                    assign_game_ids_dict = {
                        video: None for video in video_files
                    }
                    st.session_state["assign_game_ids_dict"] = (
                        assign_game_ids_dict
                    )
                else:
                    assign_game_ids_dict = st.session_state[
                        "assign_game_ids_dict"
                    ]

                # Dropdown to select raw video file
                selected_raw_video = st.selectbox(
                    label='Select a raw video file to assign a game_id:',
                    options=(
                        video_files
                        if video_files
                        else ['No video files found in the selected folder']
                    ),
                )

            # Select game_id to assign to the video
            with col6:
                selected_game_id = st.selectbox(
                    label='Select a game_id to assign to the video:',
                    options=(
                        game_ids
                        if game_ids
                        else ['No game_ids found for the selected team']
                    ),
                )

            # Button to assign selected game_id to video
            with col7:
                if st.button(
                    "Assign game_id to selected video file",
                    use_container_width=True,
                    type="primary",
                ):
                    assign_game_ids_dict[
                        selected_raw_video
                    ] = selected_game_id
                    st.session_state["assign_game_ids_dict"] = (
                        assign_game_ids_dict
                    )

                # Display current assignments
                with col_dict_ids:
                    st.write(assign_game_ids_dict)

            # Initialize preprocessing only if all videos have been assigned game_ids
            with col7:
                if (
                    assign_game_ids_dict
                    and all(assign_game_ids_dict.values())
                ):
                    launch_preprocessing = st.button(
                        "Initialize GameEditor",
                        use_container_width=True,
                        type="secondary"
                    )

        # Initialize GameEditor and run pre-processing on all videos
        if launch_preprocessing:
            game_editor = GameEditor(
                video_dir=st.session_state.get("video_dir"),
                output_dir=st.session_state.get("output_dir")
            )
            game_editor.pre_match_editing(play_speed=2.0)
            st.session_state["game_editor_initialized"] = True

            with col_success:
                if st.session_state.get("game_editor_initialized"):
                    st.success(
                        "GameEditor initialized and "
                        "pre-processing completed successfully!"
                    )

        # Rename preprocessed video files using game_id assignments
        with col7:
            if st.button(
                "Rename preprocessed games",
                use_container_width=True,
                type="primary"
            ):
                video_file_renamer(
                    rename_dict=st.session_state.get(
                        "assign_game_ids_dict"
                    ),
                    output_dir=st.session_state.get("output_dir")
                )

    ########################################################
    # Game-to-points mode ##################################
    # Segment preprocessed videos into individual points   #
    ########################################################
    elif st.session_state.get(
        "editor_mode", "Game-to-points"
    ) == "Game-to-points":

        st.title(
            body="🎬➡️ Game-to-points mode 🏐✂️🏐✂️🏐",
            anchor="game_to_points_mode",
            )
            
        # Column set up for game-to-points workflow
        col1, col2 = st.columns([1, 4])
        col3, col4 = st.columns([1, 4])
        col5, col6, col7 = st.columns([1, 1, 1])


        # Select preprocessed folder containing ready-to-segment games
        with col1:
            if st.button(
                label="🎬 Preprocessed folder",
                use_container_width=True,
                type="secondary",
            ):
                preprocessed_folder = select_folder()
                if preprocessed_folder:
                    st.session_state["preprocessed_folder"] = preprocessed_folder

        # Extract and display list of available preprocessed game_ids
        if "preprocessed_folder" in st.session_state:
            preprocessed_game_ids_list = list(
                os.listdir(st.session_state["preprocessed_folder"])
            )
            preprocessed_game_ids_list = [
                filename.split('_started')[0] for filename in preprocessed_game_ids_list
            ]

            st.session_state["preprocessed_game_ids"] = preprocessed_game_ids_list

            with col2:
                st.markdown(
                    f"""
                    <style>
                    .small-success {{
                        font-size: 15px;
                        padding: 10px;
                        background-color: #1e5631;
                        border-radius: 10px;
                        color: white;
                    }}
                    </style>
                    <div class="small-success">
                    ✅ {st.session_state['preprocessed_folder']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Select output folder for segmented point videos
        with col3:
            if st.button(
                label="✂️ Segmented folder",
                use_container_width=True,
                type="secondary",
            ):
                segmented = select_folder()
                if segmented:
                    st.session_state["segmented_folder"] = segmented

        # Display selected segmented folder path
        if "segmented_folder" in st.session_state:

            with col4:
                st.markdown(
                    f"""
                    <style>
                    .small-success {{
                        font-size: 15px;
                        padding: 10px;
                        background-color: #1e5631;
                        border-radius: 10px;
                        color: white;
                    }}
                    </style>
                    <div class="small-success">
                    ✅ {st.session_state['segmented_folder']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Extract unique game_ids from existing segmented files
            segmented_points_list = list(
                os.listdir(st.session_state["segmented_folder"])
                )
            temp_list = []
            for filename in segmented_points_list:
                game_id = filename.split('_p')[0]
                temp_list.append(game_id)
            unique_game_ids_list = list(set(temp_list))
            st.session_state["segmented_game_ids"] = unique_game_ids_list

            # Toggle widget to show game_ids not already segmented
            with col7:
                st.toggle(
                    label="Only non-segmented games",
                    value=True,
                    key="toggle_show_non_segmented_only"
                )
                # Display only non-segmented game_ids in dropdown if toggle is on
                if st.session_state["toggle_show_non_segmented_only"]:
                    non_segmented_game_ids = [
                        game_id for game_id in st.session_state.get("preprocessed_game_ids", [])
                        if game_id not in st.session_state.get("segmented_game_ids", [])
                    ]
                    st.session_state["non_segmented_game_ids"] = non_segmented_game_ids

        # Dropdown to select a game for segmentation
        with col5:
            # Dropdown options depend on toggle state - show all game_ids or only non-segmented ones
            if "toggle_show_non_segmented_only" in st.session_state and st.session_state["toggle_show_non_segmented_only"]:
                selected_game_id = st.selectbox(
                    label='All preprocessed games :',
                    options=(
                        ['None'] + st.session_state.get("non_segmented_game_ids", [])
                        if st.session_state.get("non_segmented_game_ids", [])
                        else ['None']
                    ),
                )
            # Dropdown to show all game_ids regardless of segmentation status
            else:
                selected_game_id = st.selectbox(
                    label='All preprocessed games :',
                    options=(
                        ['None'] + st.session_state.get("preprocessed_game_ids", [])
                        if st.session_state.get("preprocessed_game_ids", [])
                        else ['None']
                    ),
                )

            if selected_game_id == 'None':
                st.session_state.pop("selected_game_id", None) 
            else:
                st.session_state["selected_game_id"] = selected_game_id
                # Fetch team information from database for selected game
                if st.button(
                    "Fetching datas for the selected game",
                    use_container_width=True,
                    type="secondary"
                ):
                    # Query database for team names associated with this game_id
                    with FirestoreManager() as db:
                        team_a, team_b = db.teams_names_from_game_id(
                            st.session_state["selected_game_id"])

                        st.session_state["team_a"] = team_a
                        st.session_state["team_b"] = team_b

                        # Player names for team A
                        pa_a, pb_a = db.get_player_names(team_a)
                        st.session_state["team_a_players"] = (
                            (pa_a, pb_a) if pa_a else ("(No player found)", "(No player found)")
                        )
                        # Player names for team B
                        pa_b, pb_b = db.get_player_names(team_b)
                        st.session_state["team_b_players"] = (
                            (pa_b, pb_b) if pa_b else ("(No player found)", "(No player found)")
                        )

                    st.write(f"Team A: {st.session_state['team_a_players'][0]} & {st.session_state['team_a_players'][1]}")
                    st.write(f"Team B: {st.session_state['team_b_players'][0]} & {st.session_state['team_b_players'][1]}")

                    # Warn user if game_id already has segmented files (overwrite risk)
                    if "segmented_folder" in st.session_state:
                        temp_list = list(
                            os.listdir(st.session_state["segmented_folder"]))
                        temp_list = [filename.split('_p')[0] for filename in temp_list]
                        temp_list = list(set(temp_list))
                        if st.session_state["selected_game_id"] in temp_list:
                            st.warning(
                                "This game_id is already present in the segmented folder. "
                                "Initializing GameEditor will overwrite the existing segmented files for this game_id."
                            )

        # Execute game-to-points segmentation on selected game
        with col6:
            if st.button(
                "Run game-to-points segmentation",
                use_container_width=True,
                type="primary"
            ):
                game_id_to_run = st.session_state.get("selected_game_id", None)

                if not game_id_to_run or game_id_to_run == 'None':
                    st.error("No game_id selected. Please select a valid game_id")
                else:
                    video_path = str(os.path.join(
                        st.session_state.get("preprocessed_folder", ""),
                        game_id_to_run
                    ) + '_started_rotated.mp4')

                    # Subprocess to run segmentation script with streamlit opened 
                    run_script_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "run_segmentation.py"
                    )
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            run_script_path,
                            video_path,
                            st.session_state.get("segmented_folder", ""),
                            st.session_state.get("team_a", "team_a"),
                            st.session_state.get("team_b", "team_b"),
                            str('True' if st.session_state.get("toggle_show_non_segmented_only", False) else str('False'))
                        ]
                    )
                    st.toast("Point segmentation started", icon="🎬")

                    process.wait()  # Wait for the subprocess to finish before showing success message

                    with col6:
                        st.success(
                            body='Segmentated points created successfully!')

    ########################################################
    # All possession mode ##################################
    # Individual points into condensed game video ##########
    ########################################################
    elif st.session_state.get(
        "editor_mode", "All possessions"
    ) == "All possessions":
        st.title(
            body="🏐✂️🏐➡️ All possessions mode ⚡🎞️",
            anchor="all_possessions_mode",
            )
        
        # # Default folders for all possessions workflow
        # st.session_state["segmented_folder"] = SEGMENTED_POINTS_DIR

        # Column set up for all possessions workflow
        col1, col2, col3 = st.columns([1, 1.2, 4])
        col4, col5, col6 = st.columns([1.2,1,2])

        # Toggle to use default segmented folder or select a different one
        with col1:
            st.toggle(label="Default folder",
                      value=True,
                      key="toggle_default_all_possessions_folder",
                      )
            if st.session_state.get("toggle_default_all_possessions_folder", True) is False:
                with col2:
                    if st.button(
                        label="✂️ Segmented folder",
                        use_container_width=True,
                        type="secondary",
                    ):
                        segmented_folder = select_folder()
                        if segmented_folder:
                            st.session_state["segmented_folder"] = segmented_folder
                            with col3:
                                st.markdown(
                                    f"""
                                    <style>
                                    .small-success {{
                                        font-size: 15px;
                                        padding: 10px;
                                        background-color: #1e5631;
                                        border-radius: 10px;
                                        color: white;
                                    }}
                                    </style>
                                    <div class="small-success">
                                    ✅ {st.session_state["segmented_folder"]}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
        # Display selected segmented folder path
        if st.session_state.get("toggle_default_all_possessions_folder", True) is True:
            # Default folders for all possessions workflow
            st.session_state["segmented_folder"] = SEGMENTED_POINTS_DIR

        # Dropdown to select game_id for all possessions video creation
        with col4:
            # Extract unique game_ids from segmented files in the segmented folder
            segmented_points_list = list(
                os.listdir(st.session_state.get("segmented_folder", ""))
                )
            temp_list = []
            for filename in segmented_points_list:
                game_id = filename.split('_p')[0]
                temp_list.append(game_id)
            unique_game_ids_list = sorted(list(set(temp_list)))

            selected_game_id_all_possessions = st.selectbox(
                label='Select a game to edit:',
                label_visibility="collapsed",
                options=(
                    ['None'] + unique_game_ids_list
                    if unique_game_ids_list
                    else ['None']
                ),
            )
        # Button to run all possessions montage creation for the selected game_id
        with col5:
            if st.button(
                "Create all possessions video",
                use_container_width=True,
                type="primary"
            ):
                if selected_game_id_all_possessions == 'None':
                    st.error("No game_id selected. Please select a valid game_id")
                else:
                    segmented_points_folder = st.session_state.get("segmented_folder", "")
                    # DEV
                    st.write(f'game id : {selected_game_id_all_possessions}')

                    # Subprocess to run segmentation script with streamlit opened 
                    run_script_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "run_all_possession.py"
                    )
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            run_script_path,
                            selected_game_id_all_possessions,
                        ]
                    )
                    process.wait()  # Wait for the subprocess to finish before showing success message
                    st.toast("All possessions video creation started", icon="🎬")

                    with col6:
                        st.success(
                            body='All possessions video created successfully!')


                    







