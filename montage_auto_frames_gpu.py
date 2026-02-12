"""
##### (à reprendre une fois toutes les fonctions MAJ) #####

Modules pour l’extraction de segments vidéo via GPU NVIDIA NVENC.
Adapté pour un GPU GTX 1060

Fonctions exposées :
    - extract_segments_gpu(input_video, ranges_file, output_dir)
    - pregame_cutting(video_path, play_speed=1.0)

Format du fichier ranges :
    startFrame-endFrame
    Exemple :
        242-473
        701-996
"""

import os
import subprocess
import sys
import cv2
import pandas as pd
import sys

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

def ensure_dir(path: str):
    """Create directory if missing."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_ranges(path: str):
    """Load start-end frame ranges from file."""
    ranges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "-" not in line:
                continue
            s, e = line.split("-")
            ranges.append((int(s), int(e)))
    return ranges


# -------------------------------------------------------------------
# Core GPU extraction pour un point unitaire joué (start-end frames)
# -------------------------------------------------------------------

def cut_point_gpu(
    video_path: str,
    start_frame: int,
    end_frame: int,
    output_video: str
):
    """
    Extrait un segment frame-accurate en utilisant CUDA + NVENC via une commande ffmpeg optimisée pour le GPU. Les frames de début et de fin sont inclusifs, c'est-à-dire que
    start_frame et end_frame sont inclusifs.

    Args:
        video_path : chemin de la vidéo source
        start_frame : frame de début du segment à extraire
        end_frame : frame de fin du segment à extraire
        output_video : chemin de la vidéo segmentée à générer
    """

    vf = f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS"

    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    cmd = [
        ffmpeg_path, "-y",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-vf", vf,
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "constqp",
        "-qp", "18",
        output_video
    ]

    print(f"[GPU] Extraction frames {start_frame} → {end_frame}")
    print("      Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

# -------------------------------------------------------------------
# Rotation de la vidéo (si nécessaire) avec ffmpeg + GPU
# -------------------------------------------------------------------

def video_rotation (video_path: str,
                   rotation_state: int = 0,
                   output_dir: str = None) -> None:
    
    """ Applique une rotation à la vidéo en utilisant ffmpeg.

    Args:
        video_path (str): Chemin de la vidéo à faire pivoter.
        rotation_state (int): État de rotation (0, 90, 180, 270).
        output_dir (str, optional): Dossier de sortie pour la vidéo pivotée. Si None, la vidéo pivotée sera enregistrée dans le même dossier que la vidéo d'origine.
    """

    transpose_commands = {
        0: None,  # Pas de rotation
        90: "transpose=1",  # Rotation à droite
        180: "transpose=1,transpose=1",  # Rotation à 180 degrés
        270: "transpose=2"  # Rotation à gauche
    }

    # Sélection de la rotation à appliquer
    filter_str = transpose_commands[rotation_state]

    # Déterminer le chemin de sortie
    if output_dir is None:
        output_path = f'{os.path.splitext(video_path)[0]}_rotated_{rotation_state}.mp4'
    else:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_rotated_{rotation_state}.mp4')
    print(f"Vidéo pivotée enregistrée : {output_path}")

    # Commande ffmpeg pour appliquer la rotation
    if filter_str is None:
        command = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-c:v', 'copy',
            '-c:a', 'copy',
            output_path
        ]
    else:
        command = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-vf', filter_str,
            '-c:a', 'copy',  # Copier la piste audio sans ré-encoder
            output_path
        ]

    # print(f"Appliquer la rotation : {rotation_state} degrés")
    subprocess.run(command, check=True)


# -------------------------------------------------------------------
#  Enregistrement des actions de montage à effectuer pour un pre process de video, via cv2 et interaction clavier, sur 1 vidéo
# -------------------------------------------------------------------

def cv2_actions_to_operate(
        video_path : str,
        play_speed : float = 1.0,
    ) -> dict:
    
    """
    Enregistre les actions de montage à effectuer pour un pre process de video
    
    Args:
        video_path (str): Chemin vers la vidéo à traiter.
        play_speed (float, optional): Vitesse de lecture de la vidéo (1=normale, 0=pause, >1=plus rapide). Par défaut à 1.0.
    Returns:
        montage_actions(dict) : Dictionnaire d'actions taguées à effectuer sur la vidéo avec les éléments 'Start_frame', 'Last_frame', 'Rotation_state'
    """
    montage_actions = dict()

    # Afficher les touches disponibles en overlay sur la vidéo
    help_lines = [
        "Touches :",
        "q : quitter",
        "espace : pause/reprise",
        "0 : debut du match",
        "+ : vitesse +",
        "- : vitesse -",
        "r : rotation droite",
        "l : rotation gauche",
    ]

    _orig_imshow = cv2.imshow

    def _imshow_with_help(winname, frame):
        if frame is not None:
            x, y = 30, 120
            for i, line in enumerate(help_lines):
                cv2.putText(frame,
                            line,
                            (x, y + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA)
        _orig_imshow(winname, frame)

    cv2.imshow = _imshow_with_help


    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    # Récupérer le nombre total de frames pour calculer le dernier frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_game_frame = frame_count - 1 if frame_count > 0 else None
    print(f"Index du dernier frame : {last_game_frame}")

    def _waitKey_fast(ms):
        # Réduire le délai proportionnellement à la vitesse (au moins 1 ms)
        adj = max(1, int(ms / play_speed))
        return cv2.waitKey(adj)

    # Vérifier que la vidéo est bien ouverte
    if not cap.isOpened():
        print("Erreur : impossible d’ouvrir la vidéo.")
        sys.exit()

    # Récupérer les FPS pour convertir les frames en temps
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    # État de pause et rotation
    paused = False
    rotation_state = 0  # 0, 90, 180, 270

    try:
        while cap.isOpened():

            # Lire une frame seulement si on n'est pas en pause
            if not paused:
                ret, frame = cap.read()

                # Appliquer la rotation si nécessaire
                if rotation_state == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_state == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation_state == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Afficher la vitesse de lecture en overlay
                cv2.putText(frame,
                            f"Vitesse de lecture : x{play_speed:.1f}",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA)

                # Arrêter si fin de vidéo ou erreur
                if not ret:
                    print("Fin de la vidéo ou erreur de lecture.")
                    break

                # Incrémenter le compteur de frames
                frame_number += 1

            # Indiquer le mode pause sur l'image affichée
            if paused and ret:
                cv2.putText(frame,
                            "|| PAUSE ||",
                            (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA)

            # Afficher la frame courante
            if ret:
                cv2.imshow(f'{video_path}', frame)

            # Gestion des entrées clavier
            key = _waitKey_fast(30) & 0xFF
            if key == ord('q'):  # quitter
                break
            elif key == ord(' '):  # pause/reprise
                paused = not paused
            elif key == ord('0'):  # marquer le début du match
                starting_game_frame = frame_number
                start_time = starting_game_frame / fps
                print(f"Début du match marqué au frame {starting_game_frame}, soit {start_time:.2f} secondes")
                break
            elif key == ord('+'):  # augmenter la vitesse
                play_speed += 0.5
                continue
            elif key == ord('-'):  # diminuer la vitesse
                play_speed = max(0.5, play_speed - 0.5)
                continue
            elif key == ord('r'):  # rotation à droite
                rotation_state = (rotation_state + 90) % 360
                continue
            elif key == ord('l'):  # rotation à gauche
                rotation_state = (rotation_state - 90) % 360
                continue


    finally:
        # Libérer les ressources OpenCV
        cap.release()
        cv2.destroyAllWindows()

    montage_actions = {
        'start_frame': starting_game_frame,
        'last_frame': last_game_frame,
        'rotation_state': rotation_state
    }

    return montage_actions






# -------------------------------------------------------------------
# Découpage pré-match (pour éviter les longues vidéos)
# -------------------------------------------------------------------

def pre_match_editing(video_dir: str,
                      play_speed: float = 1.0,
                      output_dir: str = None) -> None:
    """ Réalise le découpage pré-match des vidéos d'un dossier.
    Le script utilise OpenCV pour afficher la vidéo et détecter la touche pressée.

    L'utilisateur doit :
        - Indiquer la bonne rotation de la vidéo (si nécessaire) via les touches 'r' et 'l'
        - Appuyer sur '0' pour indiquer le début du match, et la vidéo est ensuite découpée à partir de ce point en utilisant ffmpeg.

    Args:
        video_dir (str): dossier contenant la/les vidéo(s) à découper.
        play_speed (float): Vitesse de lecture de la vidéo.
        output_dir (str, optional): Dossier de sortie pour les vidéos découpées. Si None, les vidéos découpées seront enregistrées dans le même dossier que les vidéos d'origine.
    """
    # Initialiser un df pandas pour stocker les informations de découpage de chaque vidéo (start frame, last frame, rotation)
    match_info_df = pd.DataFrame(columns=["video_path", "starting_game_frame", "last_game_frame", "output_dir", "rotation_state"])

    # Lister les fichiers vidéo dans le dossier
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print("Aucune vidéo trouvée dans le dossier spécifié.")
        return
    
    # Appliquer cv2_actions_to_operate() à chaque vidéo pour récupérer les actions de montage à effectuer (start frame, last frame, rotation)
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Traitement de la vidéo : {video_path}")

        # Récupérer les actions de montage à effectuer pour la vidéo
        montage_actions = cv2_actions_to_operate(video_path, play_speed)
        starting_game_frame = montage_actions.get('start_frame', 0)
        last_game_frame = montage_actions.get('last_frame', None)
        rotation_state = montage_actions.get('rotation_state', 0)


        # Stocker le frame de début du match et la rotation dans un df pandas temporaire pour les étapes suivantes du pipeline
        match_info_df.loc[len(match_info_df)] = {
            "video_path": video_path,
            "starting_game_frame": starting_game_frame,
            "last_game_frame": last_game_frame,
            "output_dir": output_dir if output_dir else os.path.dirname(video_path),
            "rotation_state": rotation_state,
        }
    
    # Appliquer cut_point_gpu() à chaque ligne de match_info_df
    for _, row in match_info_df.iterrows():
        cut_point_gpu(
            video_path=row["video_path"],
            start_frame=int(row["starting_game_frame"]),
            end_frame=int(row["last_game_frame"]),
            output_video=os.path.join(row["output_dir"], f'{os.path.splitext(os.path.basename(row["video_path"]))[0]}_started.mp4')
        )

    # Appliquer video_rotation() à chaque ligne de match_info_df si rotation_state != 0
    for _, row in match_info_df.iterrows():
        if row["rotation_state"] != 0:
            video_rotation(
                video_path=os.path.join(row["output_dir"], f'{os.path.splitext(os.path.basename(row["video_path"]))[0]}_started.mp4'),
                rotation_state=int(row["rotation_state"]),
                output_dir=row["output_dir"]
            )
            # Supprimer la vidéo intermédiaire non-rotatée
            os.remove(os.path.join(row["output_dir"], f'{os.path.splitext(os.path.basename(row["video_path"]))[0]}_started.mp4'))


    return match_info_df


# -------------------------------------------------------------------
# Découpage Core GPU, à partir d'un dataframe contenant les start-end frames
# -------------------------------------------------------------------

def extract_segments_from_df_gpu(
    input_video: str,
    actions_df: pd.DataFrame,
    output_dir: str
    ) -> None:

    # Construire les intervalles : 1 ligne = time(Point) - time(Temps hors-jeu) suivant
    for _, row in actions_df.iterrows():
        cut_point_gpu(
            video_path=input_video,
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            output_video=os.path.join(output_dir, f"extrait_{_+1:03d}.mp4")
        )
    

# -------------------------------------------------------------------
# OLD PACKAGE : TO BE REPLACED BY extract_segments_from_df_gpu() which takes a DataFrame as input
# Script pour extraction de segments via GPU, à partir d'un fichier .txt de ranges et du chemin de la vidéo source
# Utilisé initialement pour l'entrainement du modèle de reconnaissances des 'Points' vs. 'Temps hors-jeu'
# Pour le pipeline manuel de montage, on utilisera plutôt la fonction extract_segments_from_df_gpu() qui prend en entrée un DataFrame 
# -------------------------------------------------------------------

# def extract_segments_gpu(
#     input_video: str,
#     ranges_file: str,
#     output_dir: str
# ):
#     """
#     Script pour extraction de segments via GPU, à partir d'un fichier .txt de ranges et du chemin de la vidéo source
#     Utilisé initialement pour l'entrainement du modèle de reconnaissances des 'Points' vs. 'Temps hors-jeu'
#     Pour le pipeline manuel de montage, on utilisera plutôt la fonction extract_segments_from_df_gpu() qui prend en entrée un DataFrame 

#     Découpe la vidéo source en segments définis par les start-end frames du fichier .txt de ranges, en utilisant le GPU pour accélérer l'extraction.

#     Args:
#         input_video : chemin de la vidéo source
#         ranges_file : fichier .txt contenant des start-end frames
#         output_dir  : dossier où stocker les extraits

#     """

#     ensure_dir(output_dir)
#     ranges = load_ranges(ranges_file)

#     print(f"[INFO] Nombre de segments à extraire : {len(ranges)}")

#     for idx, (start, end) in enumerate(ranges, start=1):
#         output_path = os.path.join(output_dir, f"extrait_{idx:03d}.mp4")
#         cut_point_gpu(input_video, start, end, output_path)

#     print("[FIN] Tous les extraits GPU sont générés.")
#     return True