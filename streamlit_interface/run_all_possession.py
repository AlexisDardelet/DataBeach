# run_all_possession.py
import sys
import os
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
)
from game_editor import GameEditor

if __name__ == "__main__":
    game_id  = sys.argv[1]

    game_editor = GameEditor()
    
    game_editor.all_possession_montage(
        game_id=game_id)
    