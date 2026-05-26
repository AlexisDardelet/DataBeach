## Main page for the Streamlit interface of DataBeach
import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import cv2
import datetime
import json
import sys

from coach_overview import coach_overview
from serve_focus import serve_focus

# Pages import

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from db_manager import DBManager

# Listing the teams with players names for the dropdown menu in the sidebar
## [FULL SCALE VERSION] Uncomment the following code to fetch the teams list from the database
# with DBManager() as db:
#     teams_list = db.list_teams_with_players_names()
#     st.session_state.update({"teams_list": teams_list})
## [TESTING VERSION] Comment the following code to fetch the teams list from the database
st.set_page_config(layout="wide")

teams_list = [
    ('JOMR', 'OFFREDI Jade - RANC Mathilde'),
]

# Sidebar menu for navigation
with st.sidebar:
    
    # Select team dropdown menu
    with st.container(border=True):
        paire_name = st.selectbox("Select a team:", [team[1] for team in teams_list])
        paire_id = next((team[0] for team in teams_list if team[1] == paire_name), None)
        st.session_state.update({"paire_id": paire_id})
        # Calling the db to get the game_ids for the selected team and storing them in session state
        with DBManager() as db:
            db.execute_query(
                """SELECT game_id 
                FROM table_game
                WHERE team_a = ? OR team_b = ?""",
                (paire_id, paire_id)
                )
            results = db.cursor.fetchall()
            st.session_state.update({"game_ids": [result[0] for result in results]})

    # Menu options for the coaching view
        selected = option_menu(
            menu_title="Coaching pages",
            options=[
            "Coach overview",
            "Serve focus",
            "Side-out focus", 
            "Block-defense focus",
            "Specific plays focus",
            ],
            icons=["house","arrow-up-right", "arrow-left", "bricks", "eyeglasses"],
            menu_icon="cast",
            default_index=1,
        )


## Page content based on the selected menu option

# [PREVIOUS APP VERSION] Uncomment the following code to display the coaching view pages
# Coaching view pages
if st.session_state.get("Coaching view", True):
    if selected == "Coach overview":
        coach_overview(paire_id)
    elif selected == "Serve focus":
        serve_focus(paire_id)

# Dev view pages
# if st.session_state.get("Dev view", True):
#     if selected == "Game editor":
#         editor_interface()
#         # Move the selectbox to the sidebar
#         with st.sidebar:
#             editor_mode = st.selectbox(
#                 "Select the editor mode:", 
#                 ["Pre-processing", "Game-to-points", "All possessions"],
#                 on_change=lambda: st.session_state.update({"editor_mode": st.session_state.editor_mode_select}),
#                 key="editor_mode_select"
#             )
