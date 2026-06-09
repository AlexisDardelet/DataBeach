## Main page for the Streamlit interface of DataBeach — coach view
import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)
from firestore_manager import FirestoreManager

from coach_overview import coach_overview
from serve_focus import serve_focus

st.set_page_config(layout="wide")

# Sidebar menu for navigation
with st.sidebar:

    # Select team dropdown menu
    with st.container(border=True):

        # [FULL VERSION] Fetch teams from Firestore and populate the dropdown
        # with FirestoreManager() as fm:
        #     teams_list = fm.list_teams_with_players_names()
        # [DEMO VERSION] Hardcoded teams
        teams_list = [
            ("JOMR", "Jade & Mathilde"),
            ("AleD_RonP", "Alexis & Ronan"),
        ]

        selected_paire_name = st.selectbox(
            "Select a team:", [team[1] for team in teams_list]
        )
        paire_id = next(
            (team[0] for team in teams_list if team[1] == selected_paire_name), None
        )
        st.session_state.update({"paire_id": paire_id})

        with FirestoreManager() as fm:
            game_ids = fm.get_game_ids_for_team(paire_id)
        st.session_state.update({"game_ids": game_ids})

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
        icons=["house", "arrow-up-right", "arrow-left", "bricks", "eyeglasses"],
        menu_icon="cast",
        default_index=1,
    )

## Page content based on the selected menu option
if selected == "Coach overview":
    coach_overview(paire_id)
elif selected == "Serve focus":
    serve_focus(paire_id)
