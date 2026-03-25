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
from dotenv import load_dotenv

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from db_manager import DBManager
from game_editor import GameEditor
from video_grader import VideoGrader
from etl_utils import *
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

            # Set up the layout with two columns for folder selection
            col1, col2 = st.columns([1, 2.5]) # First row for video_dir
            col3, col4 = st.columns([1, 2.5]) # Second row for output_dir
            col5, col6 = st.columns([1, 1.5]) # Third row for pre-processing options and GameEditor initialization

            # Widgets to select the directory
            with col1: 
                if st.button(label="📁➡️ Select raw games folder",
                                 use_container_width=True,
                                 type="secondary",
                                 ):
                    folder = select_folder()
                    if folder:
                        st.session_state["video_dir"] = folder
                if "video_dir" in st.session_state:
                    st.session_state["video_dir_list"] = list(os.listdir(st.session_state["video_dir"]))
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
                            <div class="small-success">✅ {st.session_state["video_dir"]}</div>
                            """,
                            unsafe_allow_html=True
                        )
            # Widget to assign a game_id into the preprocessed videos
            with col5:
                if st.session_state.get("video_dir_list", []):
                    selected_raw_video = st.selectbox(
                        "Raw video file", 
                        st.session_state.video_dir_list,

                        )
                    game_id_to_assign = st.selectbox(
                        "Assign to game_id", 
                        game_ids
                        )
                    assign_button = st.button(
                        "Assign/Rename game_id to video",
                        use_container_width=True, 
                        type="primary",
                        )
            with col6:
                if st.session_state.get("video_dir_list", []):
                    df = pd.DataFrame({
                        'Raw video name': st.session_state.get("video_dir_list", []),
                        'game_id associated': [''] * len(st.session_state.get("video_dir_list", []))
                    },
                    index=None)
                    st.dataframe(df, use_container_width=True)





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
            # [DEV] Placeholders
            # with col3:
            #     st.write("Additional pre-processing options can go here.")
            # with col4:
            #     st.write("Initializing GameEditor in Pre-processing mode...")

            # Initialize GameEditor and run pre-processing if both folders are selected
            game_editor = GameEditor(
                video_dir=st.session_state.get("video_dir"),
                output_dir=st.session_state.get("output_dir")
            )
            if folder and output_folder:
                # st.button("Run pre-processing",
                #           use_container_width=True)

                if st.button("Run pre-processing"):
                    game_editor.pre_match_editing(play_speed=2.0)







        # Game-to-points mode
        elif st.session_state.get("editor_mode", "Pre-processing") == "Game-to-points":
            with col1:

                st.write("Initializing GameEditor in Game-to-points mode...")

    # with col1:
    #     paire_id = st.selectbox("Select a team:", [team[1] for team in st.session_state.teams_list])
    # with col1:
    #     with DBManager() as db:
    #         db.execute_query(f"""SELECT game_id
    #                         FROM table_game
    #                         WHERE team_a = '{paire_id}' OR team_b = '{paire_id}'
    #                         """)
    #         results = db.cursor.fetchall()
    #         st.session_state.game_ids = [result[0] for result in results]

    #     selected_game_id = st.selectbox("Select a game:", st.session_state.game_ids)
