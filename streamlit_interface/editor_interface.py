## Main page for the Streamlit interface of DataBeach
import streamlit as st
from streamlit_option_menu import option_menu
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

def editor_interface():
    # Initialize GameEditor in Streamlit session state
    if 'game_editor' not in st.session_state:
        st.session_state.game_editor = None

    # Fetch the paire_id from session state
    paire_id = st.session_state.get("paire_id", None)

    # # Fetch the list of teams for the dropdown menu in session state
    # if 'teams_list' not in st.session_state:
    #     with DBManager() as db:
    #         st.session_state.teams_list = db.list_teams_with_players_names()

    # Create a Streamlit column layout for the editor
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # with col1:
    #     paire_id = st.selectbox("Select a team:", [team[1] for team in st.session_state.teams_list])
    with col1:
        with DBManager() as db:
            db.execute_query(f"""SELECT game_id
                            FROM table_game
                            WHERE team_a = '{paire_id}' OR team_b = '{paire_id}'
                            """)
            results = db.cursor.fetchall()
            st.session_state.game_ids = [result[0] for result in results]

        selected_game_id = st.selectbox("Select a game:", st.session_state.game_ids)

    with col2:
        editor_mode = str(st.selectbox("Select a mode :", ['Pre-processing', 'Game-to-points','All possessions']))




        

    # Display editor output in Streamlit
    if st.session_state.game_editor is not None:
        st.image(st.session_state.game_editor.get_current_frame(), use_column_width=True)