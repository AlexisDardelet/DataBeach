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
from firestore_manager import FirestoreManager

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
        'serve error': 'darkred',
        'excellent pass (++)': 'red',
        'good pass (+)': 'darkorange',
        'average pass (0)': 'yellow',
        'bad pass (-)': 'green',
        'ace': 'darkgreen'
    }

    # Filters box above the barplots
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    # Columns set up - 2 columns with a 2:1 ratio
    col1, col2 = st.columns([1, 1])
    full_width_container = st.container()

    # Fetching the player names for the selected team
    with FirestoreManager() as db:
        player_a, player_b = db.get_player_names(paire_id)
        player_a, player_b = str(player_a), str(player_b)
    # Filter button on 'player' above the barplot
    with filter_col2:
        selected_player = st.selectbox(
            label="Sélection d'un joueur :",
            options=[player_a, player_b, 'Total paire'],
            width='stretch')

    # Fetching the data for the selected team
    with FirestoreManager() as db:
        player_filter = selected_player if selected_player != 'Total paire' else None
        results_df = db.get_serve_data_df(paire_id, player=player_filter)

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
    serve_grade_counts = dict({})

    for game_id in game_ids:
        total_serves = total_serves_per_game[game_id]
        undetermined_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'undetermined')].shape[0]
        error_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'serve error')].shape[0]
        excellent_pass_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'excellent pass (++)')].shape[0]
        good_pass_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'good pass (+)')].shape[0]
        average_pass_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'average pass (0)')].shape[0]
        bad_pass_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'bad pass (-)')].shape[0]
        ace_count = results_df[(results_df['game_id'] == game_id) & (results_df['grade'] == 'ace')].shape[0]
        serve_grade_counts[game_id] = {
            'total_serves': total_serves,
            'undetermined_count': undetermined_count,
            'error_count': error_count,
            'excellent_pass_count': excellent_pass_count,
            'good_pass_count': good_pass_count,
            'average_pass_count': average_pass_count,
            'bad_pass_count': bad_pass_count,
            'ace_count': ace_count
        }
        serve_grade_counts_df = pd.DataFrame.from_dict(serve_grade_counts, orient='index')

        undetermined_ratio = undetermined_count / total_serves if total_serves > 0 else 0
        error_ratio = error_count / total_serves if total_serves > 0 else 0
        excellent_pass_ratio = excellent_pass_count / total_serves if total_serves > 0 else 0
        good_pass_ratio = good_pass_count / total_serves if total_serves > 0 else 0
        average_pass_ratio = average_pass_count / total_serves if total_serves > 0 else 0
        bad_pass_ratio = bad_pass_count / total_serves if total_serves > 0 else 0
        ace_ratio = ace_count / total_serves if total_serves > 0 else 0
        serve_grade_ratios[game_id] = {
            'total_serves': total_serves,
            'undetermined_count': undetermined_count,
            'error_count': error_count,
            'exc_count': excellent_pass_count,
            'good_pass_count': good_pass_count,
            'average_count': average_pass_count,
            'bad_pass_count': bad_pass_count,
            'ace_count': ace_count,
            'undetermined_ratio': undetermined_ratio,
            'error_ratio': error_ratio,
            'excellent_pass_ratio': excellent_pass_ratio,
            'good_pass_ratio': good_pass_ratio,
            'average_ratio': average_pass_ratio,
            'bad_pass_ratio': bad_pass_ratio,
            'ace_ratio': ace_ratio
        }
    serve_grade_ratios_df = pd.DataFrame.from_dict(serve_grade_ratios, orient='index')
    
    # # [DEV] Displaying the data
    # full_width_container.dataframe(serve_grade_ratios_df.head(3))

    # Update the serve_grade_ratios_df based on the filtered results
    filtered_serve_grade_ratios = {}
    filtered_serve_grade_counts = {}
    for game_id in filtered_results_df['game_id'].unique():
        if game_id in serve_grade_ratios:
            filtered_serve_grade_ratios[game_id] = serve_grade_ratios[game_id]
        if game_id in serve_grade_counts:
            filtered_serve_grade_counts[game_id] = serve_grade_counts[game_id]
    serve_grade_counts_df = pd.DataFrame.from_dict(filtered_serve_grade_counts, orient='index')
    serve_grade_ratios_df = pd.DataFrame.from_dict(filtered_serve_grade_ratios, orient='index')

    # Barplot of the serve grade counts per game
    fig_1 = go.Figure(data=[
        go.Bar(name='Indéterminé', x=serve_grade_counts_df.index, y=serve_grade_counts_df['undetermined_count'], marker_color=grade_color_dict['undetermined']),
        go.Bar(name='Faute service', x=serve_grade_counts_df.index, y=serve_grade_counts_df['error_count'], marker_color=grade_color_dict['serve error']),
        go.Bar(name='Réception parfaite (++)', x=serve_grade_counts_df.index, y=serve_grade_counts_df['excellent_pass_count'], marker_color=grade_color_dict['excellent pass (++)']),
        go.Bar(name='Bonne réception (+)', x=serve_grade_counts_df.index, y=serve_grade_counts_df['good_pass_count'], marker_color=grade_color_dict['good pass (+)']),
        go.Bar(name='Réception moyenne (0)', x=serve_grade_counts_df.index, y=serve_grade_counts_df['average_pass_count'], marker_color=grade_color_dict['average pass (0)']),
        go.Bar(name='Mauvaise réception (-)', x=serve_grade_counts_df.index, y=serve_grade_counts_df['bad_pass_count'], marker_color=grade_color_dict['bad pass (-)']),
        go.Bar(name='Ace', x=serve_grade_counts_df.index, y=serve_grade_counts_df['ace_count'], marker_color=grade_color_dict['ace'])
    ])
    fig_1.update_layout(
        barmode='stack',
        title='Services par match',
        xaxis_title='Matchs', 
        yaxis_title='Nombre de services'
    )
    col1.plotly_chart(
        fig_1,
        use_container_width=True
    )

    # Barplot of the serve grade ratios per game
    fig_2 = go.Figure(data=[
        go.Bar(name='Indéterminé', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['undetermined_ratio'], marker_color=grade_color_dict['undetermined']),
        go.Bar(name='Faute service', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['error_ratio'], marker_color=grade_color_dict['serve error']),
        go.Bar(name='Réception parfaite (++)', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['excellent_pass_ratio'], marker_color=grade_color_dict['excellent pass (++)']),
        go.Bar(name='Bonne réception (+)', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['good_pass_ratio'], marker_color=grade_color_dict['good pass (+)']),
        go.Bar(name='Réception moyenne (0)', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['average_ratio'], marker_color=grade_color_dict['average pass (0)']),
        go.Bar(name='Mauvaise réception (-)', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['bad_pass_ratio'], marker_color=grade_color_dict['bad pass (-)']),
        go.Bar(name='Ace', x=serve_grade_ratios_df.index, y=serve_grade_ratios_df['ace_ratio'], marker_color=grade_color_dict['ace'])
    ])
    fig_2.update_layout(
        barmode='stack', 
        title='Ratio de services par match', 
        xaxis_title='Matchs', 
        yaxis_title='Ratio'
        )
    col2.plotly_chart(
        fig_2, 
        use_container_width=True
        )




    # # [DEV] Displaying the data 
    # full_width_container.dataframe(results_df.head())
