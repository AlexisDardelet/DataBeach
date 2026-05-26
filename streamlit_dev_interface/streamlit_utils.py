## Utils functions for the Streamlit interface

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv

# Environment variables
load_dotenv()
ROOT_VIDEO_DIR = os.getenv("ROOT_VIDEO_DIR")

# -------------------------------------------------------------------
def assign_game_id_to_video_name(
    df=pd.DataFrame,
    video_name=str,
    game_id=str
) -> pd.DataFrame:
    """Assign a game_id to a raw video file.
    Args:
        df (pd.DataFrame): DataFrame containing the raw video names and associated game_ids.
        video_name (str): Name of the raw video file to which the game_id will be assigned.
        game_id (str): Game ID to assign to the specified video name.
    Returns:
        pd.DataFrame: Updated DataFrame with the assigned game_id for the specified video name.
    """
    df.loc[df['Raw video name'] == video_name, 'game_id associated'] = (
        game_id
    )
    return df
# -------------------------------------------------------------------
def select_folder(default_path=ROOT_VIDEO_DIR):
    """Open a folder selection dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory(initialdir=default_path)
    root.destroy()
    return folder
# -------------------------------------------------------------------