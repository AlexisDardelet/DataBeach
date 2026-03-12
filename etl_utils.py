# This module contains utility functions for the ETL process of loading
# volleyball match data into a database. It includes functions for loading
# indexed points data from CSV files, which are expected to be preprocessed

import os
import pandas as pd


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
        os.path.dirname(os.path.abspath(__file__)),
        "indexed_df_points",
        f"indexed_df_points_{game_id}.csv",
    )

    # Load the CSV file into a DataFrame
    df_points_formatted = pd.DataFrame()
    df_points_formatted = pd.read_csv(
        filepath_or_buffer=csv_path,
        keep_default_na=True,
    )
    df_points_formatted["point_index"] = df_points_formatted["point_index"].astype(
        "Int64"
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
    df_points_formatted.insert(
        0, "point_id", df_points_formatted.pop("point_id")
    )  # Move 'point_id' to the first column

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

