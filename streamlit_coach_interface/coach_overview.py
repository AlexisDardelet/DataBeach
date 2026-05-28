import streamlit as st
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from firestore_manager import FirestoreManager


def coach_overview(paire_id):
    st.title("Coach overview")
    st.info("Page en cours de construction.")
