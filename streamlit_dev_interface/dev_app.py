## Main page for the Streamlit interface of DataBeach
import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import cv2
import datetime
import json
import sys

from editor_interface import editor_interface
from action_grading_interface import action_grading_interface

# Pages import

# Local import
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)
from db_manager import DBManager
from game_editor import GameEditor
from video_grader import VideoGrader
from etl_utils import *
from video_edit_utils import *

# Listing the teams with players names for the dropdown menu in the sidebar
## [FULL SCALE VERSION] Uncomment the following code to fetch the teams list from the database
# with DBManager() as db:
#     teams_list = db.list_teams_with_players_names()
#     st.session_state.update({"teams_list": teams_list})
## [TESTING VERSION] Comment the following code to fetch the teams list from the database
st.set_page_config(layout="wide")

teams_list = [
    ("JOMR", "OFFREDI Jade - RANC Mathilde"),
]

# Sidebar menu for navigation
with st.sidebar:

    # Select team dropdown menu
    with st.container(border=True):
        paire_name = st.selectbox("Select a team:", [team[1] for team in teams_list])
        paire_id = next((team[0] for team in teams_list if team[1] == paire_name), None)
        st.session_state.update({"paire_id": paire_id})
        st.session_state.update({"paire_name": paire_name})

        # Calling the db to get the game_ids for the selected team and storing them in session state
        with DBManager() as db:
            # Fetching game_ids
            db.execute_query(
                """SELECT game_id 
                FROM table_game
                WHERE team_a = ? OR team_b = ?""",
                (paire_id, paire_id),
            )
            results = db.cursor.fetchall()
            st.session_state.update({"game_ids": [result[0] for result in results]})
            # Fetching serie_ids
            series_ids = db.get_serie_ids_by_paire_name(paire_name)
            st.session_state.update({"serie_ids": series_ids})

        # Menu options for the dev view
        selected = option_menu(
            menu_title="Dev pages",
            options=[
                "Game editor",
                "Action grading",
            ],
            menu_icon="cast",
            default_index=0,
        )


if selected == "Game editor":
    editor_interface()
    # Move the selectbox to the sidebar
    with st.sidebar:
        editor_mode = st.selectbox(
            "Select the editor mode:",
            ["Pre-processing", "Game-to-points", "All possessions"],
            on_change=lambda: st.session_state.update(
                {"editor_mode": st.session_state.editor_mode_select}
            ),
            key="editor_mode_select",
        )
elif selected == "Action grading":
    action_grading_interface()
