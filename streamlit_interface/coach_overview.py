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

def coach_overview(paire_id):
    # Page configuration
    st.set_page_config(layout="wide")

    # # Fetching the data for the selected team
    # with DBManager() as db:
        

    