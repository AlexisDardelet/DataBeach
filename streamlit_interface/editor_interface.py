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

# Local import
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scripts'
    )
)
from db_manager import DBManager
from game_editor import GameEditor
from video_grader import VideoGrader
from etl_utils import video_file_renamer
from video_edit_utils import *

load_dotenv()
ROOT_VIDEO_DIR = os.getenv("ROOT_VIDEO_DIR")


def select_folder(default_path=ROOT_VIDEO_DIR):
    """Open a folder selection dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory(initialdir=default_path)
    root.destroy()
    return folder


def assign_game_id_to_video_name(
    df=pd.DataFrame,
    video_name=str,
    game_id=str
) -> pd.DataFrame:
    """Assign a game_id to a raw video file."""
    df.loc[df['Raw video name'] == video_name, 'game_id associated'] = (
        game_id
    )
    return df


def editor_interface():
    """Main editor interface function."""
    paire_id = st.session_state.get("paire_id", None)
    game_ids = st.session_state.get("game_ids", [])

    if 'game_editor' not in st.session_state:
        folder = st.session_state.get("video_dir", None)
        output_folder = st.session_state.get("output_dir", None)

        ########################################################
        # Pre-processing mode (by default) #####################
        ########################################################
        if st.session_state.get(
            "editor_mode", "Pre-processing"
        ) == "Pre-processing":
            
            st.title(
                body="🪨➡️🎞️ Preprocessing ↩️🏷️🔢",
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

            # Select preprocessed folder
            with col3:
                if st.button(
                    label="➡️📁 Select preprocessed folder",
                    use_container_width=True,
                    type="secondary",
                ):
                    output_folder = select_folder()
                    if output_folder:
                        st.session_state["output_dir"] = output_folder

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

            # Assign game_ids to raw video files
            if (
                "video_dir" in st.session_state
                and "output_dir" in st.session_state
            ):
                with col5:
                    video_files = [
                        video
                        for video in os.listdir(st.session_state["video_dir"])
                        if video.endswith(('.mp4', '.avi', '.mov'))
                    ]

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

                    selected_raw_video = st.selectbox(
                        label='Select a raw video file to assign a game_id:',
                        options=(
                            video_files
                            if video_files
                            else ['No video files found in the selected folder']
                        ),
                    )

                # Select game_id to assign
                with col6:
                    selected_game_id = st.selectbox(
                        label='Select a game_id to assign to the video:',
                        options=(
                            game_ids
                            if game_ids
                            else ['No game_ids found for the selected team']
                        ),
                    )

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

                    with col_dict_ids:
                        st.write(assign_game_ids_dict)

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

            # Initialize GameEditor and run pre-processing
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

            # Rename the video files in the output directory
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
        ########################################################
        if st.session_state.get(
            "editor_mode", "Game-to-points"
        ) == "Game-to-points":
            
            st.title(
                body="🎬➡️ Game-to-points mode 🏐✂️🏐✂️🏐",
                anchor="game_to_points_mode",
                )
             
             

            col1, col2 = st.columns([1, 4])
            col3, col4 = st.columns([1, 4])
            col5, col6, col7 = st.columns([1, 1, 1])


            # Listing game_ids ready for the game-to-points mode 
            with col1:
                if st.button(
                    label="🎬 Preprocessed folder",
                    use_container_width=True,
                    type="secondary",
                ):
                    preprocessed_folder = select_folder()
                    if preprocessed_folder:
                        st.session_state["preprocessed_folder"] = preprocessed_folder

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

            with col3:
                if st.button(
                    label="✂️ Segmented folder",
                    use_container_width=True,
                    type="secondary",
                ):
                    segmented = select_folder()
                    if segmented:
                        st.session_state["segmented_folder"] = segmented

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
                
                # Isolating game_ids that are present in segmented folder
                segmented_points_list = list(
                    os.listdir(st.session_state["segmented_folder"])
                    )
                temp_list = []
                for filename in segmented_points_list:
                    game_id = filename.split('_p')[0]
                    temp_list.append(game_id)
                unique_game_ids_list = list(set(temp_list))
                st.session_state["segmented_game_ids"] = unique_game_ids_list
                # st.write(st.session_state["segmented_game_ids"])

            # Select a game_id to edit (all preprocessed games availables)
            with col5:
                selected_game_id = st.selectbox(
                    label='All preprocessed games :',
                    options=(
                        ['None'] + st.session_state.get("preprocessed_game_ids", [])
                        if st.session_state.get("preprocessed_game_ids", [])
                        else ['None']
                    ),
                )
                if selected_game_id != 'None':
                    st.session_state["selected_game_id"] = selected_game_id
                    # Initialize GameEditor for the selected game_id
                    if st.button(
                        "Fetching datas for the selected game",
                        use_container_width=True,
                        type="secondary"
                    ):
                        # Fetching the data for that game_id
                        with DBManager() as db:
                            team_a, team_b = db.teams_names_from_game_id(
                                st.session_state["selected_game_id"])
                            st.session_state["team_a"] = team_a
                            st.session_state["team_b"] = team_b
                        st.write(f"Selected game: {st.session_state['selected_game_id']}")
                        st.write(f"Team A: {st.session_state['team_a']}")
                        st.write(f"Team B: {st.session_state['team_b']}")

                        # Checking if game_id is already present in segmented folder
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

            with col6:
                if st.button(
                    "Run game-to-points segmentation",
                    use_container_width=True,
                    type="primary"
                ):
                    game_editor = GameEditor(
                    video_path=str(os.path.join(
                        st.session_state.get("preprocessed_folder", ""),
                        selected_game_id
                        )+'_started_rotated.mp4'),
                    output_dir=st.session_state.get("segmented_folder", ""),
                    )
                    game_editor.game_to_segmented_points(
                        team1_name=st.session_state.get("team_a", "team_a"),
                        team2_name=st.session_state.get("team_b", "team_b"),
                    )
                    st.success(
                        "Game-to-points segmentation completed successfully!"
                    )

                
        




