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

    # Filtre vidéo pour sélectionner les frames entre start_frame et end_frame, et réinitialiser les timestamps à partir de 0
    vf = f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS"

    # Chemin vers la version de ffmpeg compilée avec support NVENC pour accélérer l'extraction via GPU
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    # Commande ffmpeg pour extraire le segment en utilisant le GPU
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
    # print("      Command:", " ".join(cmd))

    # Exécuter la commande ffmpeg pour extraire le segment en utilisant le GPU
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
    # print(f"Vidéo pivotée enregistrée : {output_path}")

    # Commande ffmpeg pour appliquer la rotation
    if filter_str is not None:

        # Path pour diriger vers la version de ffmpeg compilée avec support NVENC pour accélérer la rotation via GPU
        ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

        command = [
            ffmpeg_path,
            '-y',
            '-i', video_path,
            '-vf', filter_str,
            '-c:a', 'copy',  # Copier la piste audio sans ré-encoder
            output_path
        ]

        # print(f"Appliquer la rotation : {rotation_state} degrés")

        # Exécuter la commande ffmpeg pour appliquer la rotation
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
# Découpage pré-match (rotation + retirer le temps d'avant match)
# -------------------------------------------------------------------

def pre_match_editing(
    video_dir: str,
    play_speed: float = 1.0,
    output_dir: str = None
    ) -> None:

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


    # return match_info_df


# -------------------------------------------------------------------
# Découpage de chaque point joué en 1 vidéo segmentée, à partir des start-end frames d'un DataFrame
# -------------------------------------------------------------------

def extract_segments_from_df_gpu(
    video_path: str,
    actions_df: pd.DataFrame,
    output_dir: str
    ) -> None:

    """
    Découpe la vidéo source en segments définis par les start-end frames d'un DataFrame
    
    Args:
        input_video(str) : chemin de la vidéo source
        actions_df(pandas.DataFrame) : DataFrame contenant les start-end frames des segments à extraire, avec au moins les colonnes 'start_frame' et 'end_frame'
        output_dir(str)  : dossier où stocker les extraits

    """

    # Construire les intervalles : 1 ligne = time(Point) - time(Temps hors-jeu) suivant
    for _, row in actions_df.iterrows():
        cut_point_gpu(
            video_path=video_path,
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            output_video=os.path.join(output_dir, f"extrait_{_+1:03d}.mp4")
        )

# -------------------------------------------------------------------------------------------------
# Création du Dataframe pour découpage d'un match en segments de points joués (prise en compte du score)
# -------------------------------------------------------------------------------------------------

def cv2_point_segment_cut(
        video_path : str,
        play_speed : float = 1.0,
        team1_name: str = "JOMR",
        team2_name: str = "adversaire"
    ) -> pd.DataFrame:

    """
    Création d'un dataframe contenant les segments de points extraits d'une vidéo de match, avec les informations sur le score et les équipes.
    A utiliser ensuite avec cut_point_gpu pour découper les segments de points à partir de ce dataframe. 
    Permet également de contrôler la lecture de la vidéo (pause, vitesse) pour faciliter l'identification des points.
    
    Args:
        video_path (str): Chemin vers la vidéo à traiter.
        play_speed (float, optional): Vitesse de lecture de la vidéo (1=normale, 0=pause, >1=plus rapide). Par défaut à 1.0.
        team1_name (str, optional): Nom de l'équipe 1. Par défaut à "JOMR".
        team2_name (str, optional): Nom de l'équipe 2. Par défaut à "adversaire".
        output_dir (str, optional): Dossier de sortie pour les segments extraits. Si None, les segments seront extraits dans le même dossier que la vidéo source.
    Returns:
        DataFrame contenant les informations sur les segments de points extraits, avec les colonnes : 
        'action',
        'start_frame',
        'end_frame',
        'score_team1',
        'score_team2',
        'set_team1',
        'set_team2'
    """
    # Initialisation des variables
    temp_list = list()
    last_action = None

    # Mapping des touches aux temp_list
    key_action_map = {
        ord('0'): 'debut du set',
        ord('1'): f'service {team1_name}',
        ord('3'): f'service {team2_name}',
        ord('2'): 'fin point',
        ord('5'): '*SWITCH*',
        ord('8'): 'Temps mort',
        }


    # Afficher les touches disponibles en overlay sur la vidéo
    help_lines = [
        "0 : debut du set",
        f"1 : service {team1_name}",
        f"3 : service {team2_name}",
        "2 : fin du point",
        "5 : switch",
        "8 : temps mort"
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


    try:
        while cap.isOpened():

            # Lire une frame seulement si on n'est pas en pause
            if not paused:
                ret, frame = cap.read()

                # Arrêter si fin de vidéo ou erreur
                if not ret:
                    print("Fin de la vidéo ou erreur de lecture.")
                    break

                # Incrémenter le compteur de frames
                frame_number += 1

            # Affiche la dernière action
            if last_action and ret:
                cv2.putText(frame,
                            f"Derniere action : {last_action}",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA)

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
            elif key == ord('+'):  # augmenter la vitesse
                play_speed += 0.5
                continue
            elif key == ord('-'):  # diminuer la vitesse
                play_speed = max(0.5, play_speed - 0.5)
                continue
            elif key in key_action_map: # enregistrer l'action associée à la touche, avec le numéro de frame
                action_name = key_action_map[key]
                last_action = action_name
                temp_list.append({
                    'Frame': frame_number,
                    'Action': action_name
                })
            elif key == ord('7'): # erreur de codage, revenir en arrière
                if temp_list:
                    removed_action = temp_list.pop()
                    print(f"Action supprimée : {removed_action}")
                    last_action = temp_list[-1]['Action'] if temp_list else None
                else:
                    print("Aucune action à supprimer.")
                # Revenir à la frame de l'action supprimée et mettre la lecture en pause
                if temp_list:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, temp_list[-1]['Frame'])
                    frame_number = temp_list[-1]['Frame']
                    paused = True
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_number = 0



    finally:
        # Libérer les ressources OpenCV
        cap.release()
        cv2.destroyAllWindows()


    # print(temp_list)

    # Initialiser une liste pour stocker les segments de points
    list_actions = list()

    i = 0
    while i < len(temp_list):
        action = temp_list[i]
        

        # Si l'action n'est pas 'fin point', on la traite
        if action['Action'] != 'fin point':
            start_frame = action['Frame']
            service_side = action['Action']
            # Chercher le prochain 'fin point'
            end_frame = None
            j = i + 1
            while j < len(temp_list):
                if temp_list[j]['Action'] == 'fin point':
                    end_frame = temp_list[j]['Frame']
                    break
                j += 1
            
            # Ajouter à la liste si on a trouvé un 'fin point'
            if end_frame is not None:
                list_actions.append({
                    'Service_side': service_side,
                    'Start_frame': start_frame,
                    'End_frame': end_frame
                })
        
        i += 1

    # Convertir en DataFrame pandas
    df_points = pd.DataFrame(list_actions)

    # Ajouter les colonnes de score
    df_points[f'{team1_name}_score'] = 0
    df_points[f'{team2_name}_score'] = 0
    df_points[f'{team1_name}_sets'] = 0
    df_points[f'{team2_name}_sets'] = 0


    # Mettre à jour les scores en fonction du service_side
    for idx, row in df_points.iterrows():
        if df_points['Service_side'].iloc[idx] == 'debut du set':
            # Première ligne : initialiser selon le service
            if row['Service_side'] == f'service {team1_name}':
                df_points.at[idx, f'{team1_name}_score'] = 1
                df_points.at[idx, f'{team2_name}_score'] = 0
            elif row['Service_side'] == f'service {team2_name}':
                df_points.at[idx, f'{team1_name}_score'] = 0
                df_points.at[idx, f'{team2_name}_score'] = 1
        else:
            # Lignes suivantes : reprendre le score précédent et incrémenter selon le service
            df_points.at[idx, f'{team1_name}_score'] = df_points.at[idx - 1, f'{team1_name}_score']
            df_points.at[idx, f'{team2_name}_score'] = df_points.at[idx - 1, f'{team2_name}_score']
            
            if row['Service_side'] == f'service {team1_name}':
                df_points.at[idx, f'{team1_name}_score'] += 1
            elif row['Service_side'] == f'service {team2_name}':
                df_points.at[idx, f'{team2_name}_score'] += 1

    # Mettre à jour les scores de sets au début de chaque nouveau set
    for idx, row in df_points.iterrows():
        if row['Service_side'] == 'debut du set' and idx > 0:
            # Comparer les scores du set précédent
            prev_score_team1 = df_points.at[idx - 1, f'{team1_name}_score']
            prev_score_team2 = df_points.at[idx - 1, f'{team2_name}_score']
            
            # Récupérer les sets précédents
            df_points.at[idx, f'{team1_name}_sets'] = df_points.at[idx - 1, f'{team1_name}_sets']
            df_points.at[idx, f'{team2_name}_sets'] = df_points.at[idx - 1, f'{team2_name}_sets']
            
            # Ajouter un set au gagnant
            if prev_score_team1 > prev_score_team2:
                df_points.at[idx, f'{team1_name}_sets'] += 1
            else:
                df_points.at[idx, f'{team2_name}_sets'] += 1
        elif idx > 0:
            # Pour les autres lignes, conserver le nombre de sets
            df_points.at[idx, f'{team1_name}_sets'] = df_points.at[idx - 1, f'{team1_name}_sets']
            df_points.at[idx, f'{team2_name}_sets'] = df_points.at[idx - 1, f'{team2_name}_sets']

    return list_actions, df_points

# -----------------------------------------------------------------------------------------------
# Point indexeer
# -----------------------------------------------------------------------------------------------
def point_indexeer(df: pd.DataFrame
                   ) -> pd.DataFrame:
    """ 
    Ajoute une colonne 'point_index' au dataframe des points, qui attribue un numéro de point unique à chaque point joué 
    (en ignorant les actions de switch et temps mort).
    
    Arg :
        df (pd.DataFrame) : DataFrame contenant les segments de points extraits, avec les colonnes : 
            'Service_side',
            'Start_frame',
            'End_frame',
            'score_team1',
            'score_team2',
            'set_team1',
            'set_team2'
    Returns :
        pd.DataFrame : DataFrame avec une colonne supplémentaire 'point_index' qui attribue un numéro de point unique à chaque point joué
    """
    if 'point_index' in df.columns:
        print("La colonne 'point_index' existe déjà dans le DataFrame. Veuillez supprimer ou renommer la colonne existante avant d'exécuter cette fonction.")
        return df
    
    elif not all(col in df.columns for col in ['Service_side']):
        print("Erreur : le DataFrame ne contient pas la colonne 'Service_side'. Veuillez vérifier les colonnes du DataFrame.")
        return df

    point_idx = 0
    point_indices = []
    for _, row in df.iterrows():
        if row['Service_side'] not in ('*SWITCH*', 'Temps mort'):
            point_idx += 1
        point_indices.append(point_idx if row['Service_side'] not in ('*SWITCH*', 'Temps mort') else None)
    df['point_index'] = pd.array(point_indices, dtype=pd.Int64Dtype())
    
    return df

# -----------------------------------------------------------------------------------------------
# Score checker
# -----------------------------------------------------------------------------------------------
def score_checker(df_points:pd.DataFrame) -> dict:
    
    """
    Fonction pour vérifier la cohérence des score par rapport aux side switch.
    Après un *SWITCH*, la somme des colonnes '_score' doit être égale à un multiple de 5 ou 7 (selon le nombre de points par set)
    Elle indique le format du match (15 ou 21 points par set) selon le multiple trouvé.
    Elle indique également le score final, avec le détail par set, pour les deux équipes.
    Si une incohérence est détectée, elle retourne un message d'erreur indiquant le problème.

    Arg :
        df_points: DataFrame contenant les segments de points extraits, avec les colonnes : 'point_index','action','start_frame','end_frame','score_team1','score_team2','set_team1','set_team2'
    Returns:
        dict : Dictionnaire contenant les informations sur le format du match et le score final
    """
    recap_dict = dict({
        'match_format': None,
        'victoire': None,
        'final_score': None,
        'score_by_set': [],
    })
    
    # Retirer les lignes correspondant aux temps morts
    temp_df = df_points[df_points['Service_side'] != 'Temps mort'].reset_index(drop=True)

    # Récupérer les noms des équipes à partir des colonnes du DataFrame
    team1_name = temp_df.columns[3].replace('_score', '')
    team2_name = temp_df.columns[4].replace('_score', '')
    # Initialiser les variables pour le score et le format du match
    score_switch_points = []
    for idx, row in temp_df.iterrows():
        if row['Service_side'] == '*SWITCH*':
            if idx + 1 < len(temp_df):
                score_sum = temp_df.at[idx + 1, f'{team1_name}_score'] + temp_df.at[idx + 1, f'{team2_name}_score']
            else:
                score_sum = row[f'{team1_name}_score'] + row[f'{team2_name}_score']
            score_switch_points.append(score_sum)

    # print("Scores au moment des switchs :", score_switch_points)

    # Vérifier les multiples de 5 ou 7
    multiples_of_5 = all(score % 5 == 0 for score in score_switch_points)
    multiples_of_7 = all(score % 7 == 0 for score in score_switch_points)
    if multiples_of_5:
        match_format = "15 points par set"
    elif multiples_of_7:
        match_format = "21 points par set"
    else:
        return {"message": "Incohérence détectée : les scores au moment des switch ne sont pas des multiples de 5 ou 7."}
    
    # Récupérer le score final et le détail par set
    recap_dict['match_format'] = match_format
    final_score_team1 = temp_df[f'{team1_name}_sets'].iloc[-1]
    final_score_team2 = temp_df[f'{team2_name}_sets'].iloc[-1]
    recap_dict['final_score'] = f"{final_score_team1} - {final_score_team2}"
    
    # Déterminer le gagnant
    if final_score_team1 > final_score_team2:
        recap_dict['victoire'] = team1_name
    elif final_score_team2 > final_score_team1:
        recap_dict['victoire'] = team2_name
    else:
        recap_dict['victoire'] = "Égalité"

    # Détail par set
    set_count = 0
    current_set_scores = []
    for idx, row in temp_df.iterrows():
        if row['Service_side'] == 'debut du set' and idx > 0:
            current_set_scores.append({
                'set': set_count + 1,
                'score': f"{row[f'{team1_name}_score']} - {row[f'{team2_name}_score']}"
            })
            set_count += 1
        elif idx == len(temp_df) - 1:  # Dernière ligne du DataFrame
            current_set_scores.append({
                'set': set_count + 1,
                'score': f"{row[f'{team1_name}_score']} - {row[f'{team2_name}_score']}"
            })

    recap_dict['score_by_set'] = current_set_scores

    return recap_dict


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