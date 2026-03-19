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
    """Serve focus page for the Streamlit interface of DataBeach
    Args:
        paire_id (str): The ID of the team to display the serve focus for
    """
    # Page configuration
    st.set_page_config(layout="wide")

    # Colour dictionary for the grades
    grade_color_dict = {
        'undetermined': 'gray',
        'error': 'red',
        'good pass': 'orange',
        'average pass': 'blue',
        'out-of-system pass': 'lightgreen',
        'ace': 'green'
    }

    # Filters box above the barplots
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    # Columns set up - 2 columns with a 2:1 ratio
    col1, col2 = st.columns([2, 1])
    full_width_container = st.container()

    # Fetching the player names for the selected team
    with DBManager() as db:
        db.execute_query(f"""SELECT player_a, player_b
                        FROM table_player
                        WHERE paire_id = '{paire_id}'
                        """)
        name_results = db.cursor.fetchall()
        player_a, player_b = str(name_results[0][0]), str(name_results[0][1])
    # Filter button on 'player' above the barplot
    with filter_col2:
        selected_player = st.selectbox(
            label="Sélection d'un joueur :",
            options=[player_a, player_b, 'Total paire'],
            width='stretch')

    # Fetching the data for the selected team
    with DBManager() as db:
        # Fetching the serve data
        subquery = f"""
            SELECT tp.point_id, tg.game_id, tp.team_a_score, tp.team_a_score_diff, tg.team_a, tg.team_b, tg.serie
            FROM table_game AS tg
            LEFT JOIN table_point AS tp ON tg.game_id = tp.game_id
            WHERE tg.team_a = '{paire_id}'"""
    
        if selected_player != 'Total paire':
            db.execute_query(f"""SELECT ts.point_id, ts.player, ts.grade, ts.point_won, sub.game_id, sub.team_a_score, sub.team_a_score_diff, sub.team_b, sub.serie
                            FROM table_serve AS ts
                            LEFT JOIN ({subquery}) AS sub
                            ON ts.point_id = sub.point_id
                            WHERE ts.paire_id = '{paire_id}' AND ts.player = '{selected_player}' 
                            """)
        else:
            db.execute_query(f"""SELECT ts.point_id, ts.player, ts.grade, ts.point_won, sub.game_id, sub.team_a_score, sub.team_a_score_diff, sub.team_b, sub.serie
                            FROM table_serve AS ts
                            LEFT JOIN ({subquery}) AS sub
                            ON ts.point_id = sub.point_id
                            WHERE ts.paire_id = '{paire_id}' 
                            """)
        results = db.cursor.fetchall()
    results_df = pd.DataFrame(results, columns=[desc[0] for desc in db.cursor.description])

    # Filter button on 'serie' above the barplot
    with filter_col1:
        unique_series = results_df['serie'].unique()
        selected_serie = st.selectbox("Sélection d'une série :", unique_series, width='stretch')
    filtered_results_df = results_df[results_df['serie'] == selected_serie]


    # Create the total serve per game column
    game_ids = results_df['game_id'].unique()
    total_serves_per_game = {}
    for game_id in game_ids:
        total_serves = results_df[results_df['game_id'] == game_id].shape[0]
        total_serves_per_game[game_id] = total_serves
    # Create a dictionary to map game_id, total serves, and grade ratio
    serve_grade_ratios = dict({})
    for game_id in game_ids:
        total_serves = total_serves_per_game[game_id]
        undetermined_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'undetermined')].shape[0]
        error_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'error')].shape[0]
        easy_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'good pass')].shape[0]
        average_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'average pass')].shape[0]
        oos_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'out-of-system pass')].shape[0]
        ace_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'ace')].shape[0]
        undetermined_ratio = undetermined_count / total_serves if total_serves > 0 else 0
        error_ratio = error_count / total_serves if total_serves > 0 else 0
        easy_ratio = easy_count / total_serves if total_serves > 0 else 0
        average_ratio = average_count / total_serves if total_serves > 0 else 0
        oos_ratio = oos_count / total_serves if total_serves > 0 else 0
        ace_ratio = ace_count / total_serves if total_serves > 0 else 0
        serve_grade_ratios[game_id] = {
            'total_serves': total_serves,
            'undetermined_count': undetermined_count,
            'error_count': error_count,
            'easy_count': easy_count,
            'average_count': average_count,
            'oos_count': oos_count,
            'ace_count': ace_count,
            'undetermined_ratio': undetermined_ratio,
            'error_ratio': error_ratio,
            'easy_ratio': easy_ratio,
            'average_ratio': average_ratio,
            'oos_ratio': oos_ratio,
            'ace_ratio': ace_ratio
        }
    serve_grade_ratios_df = pd.DataFrame.from_dict(serve_grade_ratios, orient='index')
    
    # # [DEV] Displaying the data
    # full_width_container.dataframe(serve_grade_ratios_df.head(3))

    # Update the serve_grade_ratios_df based on the filtered results
    filtered_serve_grade_ratios = {}
    for game_id in filtered_results_df['game_id'].unique():
        if game_id in serve_grade_ratios:
            filtered_serve_grade_ratios[game_id] = serve_grade_ratios[game_id]
    serve_grade_ratios_df = pd.DataFrame.from_dict(filtered_serve_grade_ratios, orient='index')


    # Barplot of the serve grade ratios per game
    fig = go.Figure(data=[
        go.Bar(name='Undetermined', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['undetermined_ratio'], marker_color=grade_color_dict['undetermined']),
        go.Bar(name='Error', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['error_ratio'], marker_color=grade_color_dict['error']),
        go.Bar(name='Good pass', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['easy_ratio'], marker_color=grade_color_dict['good pass']),
        go.Bar(name='Average pass', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['average_ratio'], marker_color=grade_color_dict['average pass']),
        go.Bar(name='Out-of-system pass', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['oos_ratio'], marker_color=grade_color_dict['out-of-system pass']),
        go.Bar(name='Ace', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['ace_ratio'], marker_color=grade_color_dict['ace'])
    ])
    fig.update_layout(barmode='stack', title='Serve Grade Ratios per Game', xaxis_title='Game ID', yaxis_title='Ratio')
    col1.plotly_chart(fig, use_container_width=True)




    # # [DEV] Displaying the data 
    # full_width_container.dataframe(results_df.head())
