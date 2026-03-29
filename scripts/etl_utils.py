# This module contains utility functions for the ETL process of loading
# volleyball match data into a database. It includes functions for loading
# indexed points data from CSV files, which are expected to be preprocessed

import os
import pandas as pd
import dotenv

dotenv.load_dotenv()
INDEXED_DF_POINTS_DIR = os.getenv("INDEXED_DF_POINTS_DIR")

# =============================================================

def extract_transform_indexed_df_points_csv(
        game_id: str,
        team_a_name: str,
        team_b_name: str,
        ) -> pd.DataFrame:
    """Extract and transform the indexed points data from a CSV file for a specific game_id.

    The CSV file should be located in the 'indexed_df_points' directory
    and named 'indexed_df_points_{game_id}.csv'.
    It requires that the CSV file has a header row with column names
    that match the expected schema for the points data.
    It returns a pandas DataFrame ready to be loaded into the database.

    Arguments:
        game_id (str): The unique identifier for the game,
            used to locate the corresponding CSV file.
        team_a_name (str): The name of the first team.
        team_b_name (str): The name of the second team.
    Returns:
        pd.DataFrame: A DataFrame containing the indexed points data
            for the specified game_id, ready for analysis or further processing.
    """

    # Path to the CSV file containing the indexed points df for the specified game_id
    csv_path = os.path.join(
        INDEXED_DF_POINTS_DIR,
        f"indexed_df_points_{game_id}.csv",
    )

    # Load the CSV file into a DataFrame 
    # (with 'point_index' column as string to preserve leading zeros)
    df_points_formatted = pd.DataFrame()
    df_points_formatted = pd.read_csv(
        filepath_or_buffer=csv_path,
        keep_default_na=True,
        dtype={"point_index": "str"}, 
    )

    # Create the 'point_id' column based on 'game_id' and 'point_index'
    df_points_formatted["point_id"] = df_points_formatted.apply(
        lambda row: (
            f"{game_id}_p{row['point_index']}"
            if pd.notna(row["point_index"])
            else pd.NA
        ),
        axis=1,
    )
    # Move 'point_id' to the first column
    df_points_formatted.insert(
        0, "point_id", df_points_formatted.pop("point_id")
    )  

    # Columns dropped
    df_points_formatted.drop(
        columns=["start_frame", "end_frame", "point_index"], inplace=True
    )

    # Columns renamed
    df_points_formatted.rename(
        columns={
            f"{team_a_name}_score": "team_a_score",
            f"{team_b_name}_score": "team_b_score",
            f"{team_a_name}_sets": "team_a_sets",
            f"{team_b_name}_sets": "team_b_sets",
        },
        inplace=True,
    )

    # Drop the 'match start - ', 'set start - ' and 'service '
    # prefixes from the 'service_side' column
    df_points_formatted["service_side"] = df_points_formatted[
        "service_side"
    ].str.replace(r"^(match start - |set start - )", "", regex=True)
    df_points_formatted["service_side"] = df_points_formatted[
        "service_side"
    ].str.replace(str("service "), "")

    # # TO BE DETERMINED IF NEEDED LATER
    # # if we want to keep the team names in the 'service_side' column
    # # or if we want to replace them with 'teamA' and 'teamB'
    # # Rename the strings in the 'service_side' column to match the team names
    # df_points_formatted['service_side'] = df_points_formatted[
    #     'service_side'
    # ].apply(
    #     lambda x: str('teamA') if x == team_a_name
    #     else (str('teamB') if x == team_b_name else x)
    # )

    # Create columns 'teamA_score_diff' and 'teamB_score_diff'
    # which are the differences between the current point's score
    # and the previous point's score for team A and team B respectively
    df_points_formatted["team_a_score_diff"] = (
        df_points_formatted["team_a_score"] - df_points_formatted["team_b_score"]
    )
    df_points_formatted["team_b_score_diff"] = (
        df_points_formatted["team_b_score"] - df_points_formatted["team_a_score"]
    )

    # # Drop the rows with '*SWITCH*', 'Timeout', 'end of set'
    # # values in the 'service_side' column
    df_points_formatted = df_points_formatted[
        df_points_formatted["service_side"] != "*SWITCH*"
    ]
    df_points_formatted = df_points_formatted[
        df_points_formatted["service_side"] != "Timeout"
    ]

    # Create a column 'point_winner' which indicates the winner
    # of the point based on the next row
    df_points_formatted["point_winner"] = df_points_formatted["service_side"].shift(
        -1
    )
    # Specific treatment for the last point of each set,
    # which has 'end of set' in the 'service_side' column of the next row
    for i, _ in enumerate(df_points_formatted.itertuples()):
        row_index = df_points_formatted.index[i]
        if i + 1 < len(df_points_formatted):
            next_row_index = df_points_formatted.index[i + 1]
            if (
                df_points_formatted.loc[next_row_index, "service_side"]
                == "end of set"
            ):
                set_winner = str()
                # # DEV DEBUG - to be removed later
                # service_side_val = df_points_formatted.loc[
                #     next_row_index, "service_side"
                # ]
                # print(
                #     f"Row index: {row_index}, "
                #     f"service_side: {service_side_val}"
                # )
                
                if (
                    df_points_formatted.loc[row_index, "team_a_score"]
                    > df_points_formatted.loc[row_index, "team_b_score"]
                ):
                    set_winner = team_a_name
                else:
                    set_winner = team_b_name
                # # DEV DEBUG - to be removed later
                # print(f"Set winner: {set_winner}")
                df_points_formatted.loc[row_index, "point_winner"] = set_winner

    # Drop the rows with 'end of set' values in the 'service_side' column
    df_points_formatted = df_points_formatted[
        df_points_formatted["service_side"] != "end of set"
    ]

    # Add a column 'game_id' with the value of the game_id for all rows
    df_points_formatted["game_id"] = game_id

    return df_points_formatted

# =============================================================

def video_file_renamer(
        rename_dict: dict,
        output_dir: str,
    ) -> None:
    """Rename video files based on a provided mapping of old names to new names.
    The dict should have the format {old_name: game_id}"""

    # # Warning if any key in the rename_dict does not end with '.mp4' or '.mov'
    # if not all(key.endswith(('.mp4', '.mov')) for key in rename_dict.keys()):
    #     raise ValueError("All keys in the rename_dict should end with '.mp4' or '.mov'")
    
    # Temp dict matching the format strings of GameEditor.preprocess()
    temp_rename_dict = {key.replace('.mp4','_started_rotated.mp4'): value for key, value in rename_dict.items()}
    
    # Renaming the video files in output_dir based on the temp_rename_dict mapping
    for new_name, game_id in temp_rename_dict.items():
        if new_name in os.listdir(output_dir):
            old_path = os.path.join(output_dir, new_name)
            new_path = os.path.join(output_dir, f"{game_id}_started_rotated.mp4")
            # print(f"Renaming from {old_path} to {new_path}")
            os.rename(old_path, new_path)

# =============================================================

test_dict = {
    'Alex poule match 1.mp4': 'game_id_1',
    'AleD-RonP_mar26_session_01.mp4': 'game_id_2',
}

if __name__ == "__main__":
    video_file_renamer(
        rename_dict=test_dict,
        output_dir=r'C:\Users\habib\Desktop\Montages volley et beach\Jade&Math\matchs preprocess'
    )
