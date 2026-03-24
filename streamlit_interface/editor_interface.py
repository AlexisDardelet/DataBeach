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

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from db_manager import DBManager
from game_editor import GameEditor
from video_grader import VideoGrader
from etl_utils import *
from video_edit_utils import *

# Function to open a folder selection dialog and return the selected path
def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder

def editor_interface():
    # Fetch the paire_id from session state
    paire_id = st.session_state.get("paire_id", None)

    # Create a Streamlit column layout for the editor
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    # Initialize GameEditor in Streamlit session state
    if 'game_editor' not in st.session_state:
        # Pre-processing mode by default
        if st.session_state.get("editor_mode", "Pre-processing") == "Pre-processing":
            
            with col1:
                if st.button("📁 Choisir un dossier"):
                    folder = select_folder()
                    if folder:
                        st.session_state["folder"] = folder

                if "folder" in st.session_state:
                    st.success(f"Dossier sélectionné : `{st.session_state['folder']}`")


            with col4:
                st.write("Initializing GameEditor in Pre-processing mode...")

            game_editor = GameEditor()
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
