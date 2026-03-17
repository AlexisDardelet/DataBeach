import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import sys
import plotly.express as px
import plotly.graph_objects as go
import bokeh.plotting as bp

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from db_manager import DBManager

def serve_focus(
        paire_id:str) -> None:
    # Page configuration
    st.set_page_config(layout="wide")
    st.write(f"Serve focus page for team {paire_id}")

    # Fetching the data for the selected team
    with DBManager() as db:
        subquery = f"""
            SELECT tp.point_id, tg.game_id, tp.team_a_score, tp.team_a_score_diff, tg.team_a, tg.team_b
            FROM table_game AS tg
            LEFT JOIN table_point AS tp ON tg.game_id = tp.game_id
            WHERE tg.team_a = '{paire_id}'"""

        db.execute_query(f"""SELECT ts.point_id, ts.player, ts.grade, ts.point_won, sub.game_id, sub.team_a_score, sub.team_a_score_diff, sub.team_b
                        FROM table_serve AS ts
                        LEFT JOIN ({subquery}) AS sub
                        ON ts.point_id = sub.point_id
                        WHERE ts.paire_id = '{paire_id}'
                        """)

        results = db.cursor.fetchall()

    results_df = pd.DataFrame(results, columns=[desc[0] for desc in db.cursor.description])
    columns_to_display = ["point_id", "game_id", "player", "grade", "point_won", "team_a_score", "team_a_score_diff", "team_b"]
    print(f"Columns in the DataFrame: {results_df.columns.tolist()}")
    st.dataframe(results_df[columns_to_display].head())
