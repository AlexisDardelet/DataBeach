# run_segmentation.py
import sys
import os
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
)
from game_editor import GameEditor

if __name__ == "__main__":
    video_path  = sys.argv[1]
    output_dir  = sys.argv[2]
    team1_name  = sys.argv[3]
    team2_name  = sys.argv[4]

    game_editor = GameEditor(
        video_path=video_path,
        output_dir=output_dir,
    )
    game_editor.game_to_segmented_points(
        team1_name=team1_name,
        team2_name=team2_name,
    )