## Action grading interface for the Streamlit interface of DataBeach
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

# Local import
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)
from db_manager import DBManager
from game_editor import GameEditor
from video_grader import VideoGrader

# Environment variables
load_dotenv()
ROOT_VIDEO_DIR = os.getenv("ROOT_VIDEO_DIR")
SEGMENTED_POINTS_DIR = os.getenv("SEGMENTED_POINTS_DIR")

# --------------------------------------------------------------------


def action_grading_interface():
    st.title("Action grading interface")
    st.info(
        "Paire name : {paire_name} - Paire ID : {paire_id}".format(
            paire_name=st.session_state.get("paire_name", "N/A"),
            paire_id=st.session_state.get("paire_id", "N/A"),
        )
    )
    # Fetching the game_ids and serie_ids for the selected team and storing them in session state
    game_ids = st.session_state.get("game_ids", [])
    serie_ids = st.session_state.get("serie_ids", [])

    # Initializing the video grader for the selected team
    video_grader = VideoGrader(paire_id=st.session_state.get("paire_id", None))

    # Setting the action grade
    selected_action = str(
        st.selectbox(label="Select an action to grade:", options=["serve", "pass"])
    )
    st.session_state.update({"selected_action": selected_action})

    # Service or passing grading interface
    if selected_action in ["serve", "pass"]:
        # Specific widgets for serve or pass grading
        col_toggle, col_recall, col_game_or_serie, col_launch_grading = st.columns(
            [1, 1, 1, 1]
        )

        # Toggle to grade a single game or a whole serie
        with col_toggle:
            st.toggle(
                label="Grading serie/game",
                key="Grading serie",
                value=True,
                help="If toggled : grading a whole serie, else grading a single game",
                on_change=lambda: st.session_state.update(
                    {"selected_game_id": None, "selected_serie_id": None}
                ),
            )
        # Recalling single game or serie
        with col_recall:
            st.markdown(
                body=f"<div style='border: 2px solid {'#e87e1a' if st.session_state.get('Grading serie', False) else '#1a73e8'}; background-color: {'#f8c88a' if st.session_state.get('Grading serie', False) else '#a8c8f8'}; border-radius: 6px; padding: 8px; text-align: center; color: {'#6e2d0d' if st.session_state.get('Grading serie', False) else '#0d2f6e'}; font-weight: bold;'>📺 {'Serie grading' if st.session_state.get('Grading serie', False) else 'Game grading'}</div>",
                unsafe_allow_html=True,
            )
        # Selectbox displaying the games or series availables
        with col_game_or_serie:
            if st.session_state.get("Grading serie", False):
                selected_serie_id = st.selectbox(
                    label="Select a serie to grade:",
                    options=serie_ids,
                    label_visibility="collapsed",
                )
                st.session_state.update({"selected_serie_id": selected_serie_id})
            else:
                selected_game_id = st.selectbox(
                    label="Select a game to grade:",
                    options=game_ids,
                    label_visibility="collapsed",
                )
                st.session_state.update({"selected_game_id": selected_game_id})
        # Button to launch the grading
        with col_launch_grading:
            # st.info(
            #     "serie_id : {serie_id} - game_id : {game_id}".format(
            #         serie_id=st.session_state.get("selected_serie_id", "N/A"),
            #         game_id=st.session_state.get("selected_game_id", "N/A"),
            #     )
            # )
            st.button(
                label="Launch grading",
                type="primary",
                on_click=lambda: video_grader.service_passing_grading(
                    serve_or_pass=selected_action,
                    game_id=st.session_state.get("selected_game_id", None),
                    serie_id=st.session_state.get("selected_serie_id", None),
                    rewrite_db=False,
                ),
                use_container_width=True,
            )
        
        # Diplaying missing games to be graded
        st.info(
        f"Games not graded yet for : {selected_action}"
        )
        missing_games_list=list(
            video_grader.missing_games_to_grade(
                action_to_grade=st.session_state.get("selected_action")
            )
        )
        st.write(missing_games_list)

