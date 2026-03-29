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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from db_manager import DBManager
from game_editor import GameEditor
from video_grader import VideoGrader
from etl_utils import video_file_renamer
from video_edit_utils import *

load_dotenv()
ROOT_VIDEO_DIR = os.getenv("ROOT_VIDEO_DIR")

# Function to open a folder selection dialog and return the selected path
def select_folder(default_path=ROOT_VIDEO_DIR):
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory(initialdir=default_path)
    root.destroy()
    return folder

# Function to assign a game_id to a raw video file and rename it accordingly
def assign_game_id_to_video_name(
    df = pd.DataFrame,
    video_name = str,
    game_id = str)-> pd.DataFrame:
    df.loc[df['Raw video name'] == video_name, 'game_id associated'] = game_id
    return df


def editor_interface():
    # Fetch the paire_id from session state
    paire_id = st.session_state.get("paire_id", None)
    # Fetch the game_ids for the selected team from session state
    game_ids = st.session_state.get("game_ids", [])

    # Initialize GameEditor in Streamlit session state
    if 'game_editor' not in st.session_state:

        folder = st.session_state.get("video_dir", None)
        output_folder = st.session_state.get("output_dir", None)    
    
        ###### Pre-processing mode (by default) ####################################
        if st.session_state.get("editor_mode", "Pre-processing") == "Pre-processing":

            # Initiating variables
            launch_preprocessing = None

            # Set up the layout with two columns for folder selection
            col1, col2 = st.columns([1, 2.5]) # First row for video_dir
            col3, col4 = st.columns([1, 2.5]) # Second row for output_dir
            col5, col6, col7 = st.columns([1, 1, 1]) 
            col_dict_ids,col_success = st.columns([3,1])


            # Widgets to select the directory
            with col1: 
                if st.button(label="📁➡️ Select raw games folder",
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
                                <div class="small-success">✅ {st.session_state["video_dir"]}</div>
                                """,
                                unsafe_allow_html=True
                            )

            # Widgets to select the output directory
            with col3:
                if st.button(label="➡️📁 Select preprocessed folder",
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
                            <div class="small-success">✅ {st.session_state['output_dir']}</div>
                            """,
                            unsafe_allow_html=True
                        )

            # Widgets to assign game_ids to raw video files and initialize GameEditor
            if "video_dir" in st.session_state and "output_dir" in st.session_state:

                # Display dropdown to select a raw video file and assign a game_id
                with col5:
                    video_files = [video for video in os.listdir(st.session_state["video_dir"]) 
                                   if video.endswith(('.mp4', '.avi', '.mov'))]
                    
                    # Initialize a dictionary to store the assigned game_ids for each video file
                    if "assign_game_ids_dict" not in st.session_state:
                        assign_game_ids_dict = {video: None for video in video_files}
                        st.session_state["assign_game_ids_dict"] = assign_game_ids_dict
                    else:
                        assign_game_ids_dict = st.session_state["assign_game_ids_dict"]

                    selected_raw_video = st.selectbox(
                        label='Select a raw video file to assign a game_id:',
                        options=video_files if video_files else ['No video files found in the selected folder'],
                    )
                # Display dropdown of game_ids available for paire_id
                with col6:
                    selected_game_id = st.selectbox(
                        label='Select a game_id to assign to the selected video:',
                        options=game_ids if game_ids else ['No game_ids found for the selected team'],
                    )
                with col7:
                    if st.button("Assign game_id to selected video file",
                                        use_container_width=True,
                                        type="primary",
                                        ):                             
                        # Update the assign_game_ids_dict with the assigned game_id for the selected video file
                        assign_game_ids_dict[selected_raw_video] = selected_game_id
                        st.session_state["assign_game_ids_dict"] = assign_game_ids_dict


                    with col_dict_ids:
                        st.write(assign_game_ids_dict)
                with col7:
                    if assign_game_ids_dict and all(assign_game_ids_dict.values()):
                        launch_preprocessing = st.button("Initialize GameEditor",
                                    use_container_width=True,
                                    type="secondary"
                        )
            # Initialize GameEditor and run pre-processing if both folders are selected
            if launch_preprocessing:
                game_editor = GameEditor(
                            video_dir=st.session_state.get("video_dir"),
                            output_dir=st.session_state.get("output_dir")
                            )
                game_editor.pre_match_editing(play_speed=2.0)
                st.session_state["game_editor_initialized"] = True

                with col_success:
                    if st.session_state.get("game_editor_initialized", True):
                        st.success("GameEditor initialized and pre-processing completed successfully!")

            # Renaming the video files in the output directory based on the assigned game_ids
            with col7:
                if st.button("Rename preprocessed games",
                            use_container_width=True,
                            type="primary"):
                    video_file_renamer(
                        rename_dict=st.session_state.get("assign_game_ids_dict"),
                        output_dir=st.session_state.get("output_dir")
                    )



