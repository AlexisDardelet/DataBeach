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

# Pages import

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from db_manager import DBManager
from game_editor import GameEditor
from video_grader import VideoGrader
from etl_utils import *
from video_edit_utils import *

# Listing the teams with players names for the dropdown menu in the sidebar
with DBManager() as db:
    teams_list = db.list_teams_with_players_names()

# Sidebar menu for navigation
with st.sidebar:
    # Coaching view and Dev view buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Coaching view",
                  use_container_width=True,
                  type="primary",
                  on_click=lambda: st.session_state.update({"Coaching view": True, "Dev view": False}),
                  )
    with col2:
        st.button("Dev view", 
                  use_container_width=True,
                  type="secondary",
                  on_click=lambda: st.session_state.update({"Coaching view": False, "Dev view": True})
                  )
    
    # Select team dropdown menu
    with st.container(border=True):
        paire_id = st.selectbox("Select a team:", [team[1] for team in teams_list])

    # Menu options for the coaching view
    if st.session_state.get("Coaching view", True):
        selected = option_menu(
            menu_title="Coaching pages",
            options=[
            "Coach overview",
            "Serve focus",
            "Side-out focus", 
            "Block-defense focus",
            "Specific play focus",
            ],
            icons=["house","arrow-up-right", "arrow-left", "bricks", "eyeglasses"],
            menu_icon="cast",
            default_index=0,
        )
    elif st.session_state.get("Dev pages", True):
        selected = option_menu(
            menu_title="Main Menu",
            options=[
            "Dev overview",
            "Game editor",
            "Action grading",
            ],
            # icons=["house", "arrow-right", "arrow-left", "eyeglasses", "pencil", "check"],
            menu_icon="cast",
            default_index=0,
        )

## Page content based on the selected menu option
# Coaching view pages
if st.session_state.get("Coaching view", True):
    if selected == "Coach overview":
        coach_overview(paire_id)

        


# # Dev view pages
# if st.session_state.get("Dev view", True):
#     if selected == "Dev overview":
#         st.title("Development overview")
#         st.write("This page provides an overview of the development activities, including progress updates and upcoming milestones.")

