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
import cv2

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
# Core GPU extraction
# -------------------------------------------------------------------

def extract_gpu_segment(
    input_video: str,
    start_frame: int,
    end_frame: int,
    output_video: str
):
    """
    Extrait un segment frame-accurate en utilisant CUDA + NVENC.
    start_frame et end_frame sont inclusifs.
    """

    vf = f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS"

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", input_video,
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
# Script pour extraction de segments via GPU, à partir d'un fichier de ranges
# -------------------------------------------------------------------

def extract_segments_gpu(
    input_video: str,
    ranges_file: str,
    output_dir: str
):
    """
    Fonction principale à utiliser dans ton autre script Python.

    input_video : chemin de la vidéo source
    ranges_file : fichier contenant des start-end frames
    output_dir  : dossier où stocker les extraits

    Exemple usage dans un autre fichier :
        import montage_auto_gpu_gtx1060 as m

        m.extract_segments_gpu("video.mp4", "ranges.txt", "extraits")
    """

    ensure_dir(output_dir)
    ranges = load_ranges(ranges_file)

    print(f"[INFO] Nombre de segments à extraire : {len(ranges)}")

    for idx, (start, end) in enumerate(ranges, start=1):
        output_path = os.path.join(output_dir, f"extrait_{idx:03d}.mp4")
        extract_gpu_segment(input_video, start, end, output_path)

    print("[FIN] Tous les extraits GPU sont générés.")
    return True

# -------------------------------------------------------------------
# Découpage pré-match (pour éviter les longues vidéos)
# -------------------------------------------------------------------

def pregame_cutting(video_path:str,
                    play_speed:float=1.0):
    """ Découpe le début de la vidéo avant le début du match, en utilisant cv2 et ffmpeg.

    Le script utilise OpenCV pour afficher la vidéo et détecter la touche pressée.
    L'utilisateur appuie sur '0' pour indiquer le début du match, et la vidéo est ensuite découpée à partir de ce point en utilisant ffmpeg.


    Args:
        video_path (str): Chemin de la vidéo à découper.
        play_speed (float): Vitesse de lecture de la vidéo.
    """


    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_game_frame = frame_count - 1 if frame_count > 0 else None
    print(f"Index du dernier frame : {last_game_frame}")

    def _waitKey_fast(ms):
        # réduire le délai proportionnellement à la vitesse (au moins 1 ms)
        adj = max(1, int(ms / play_speed))
        return cv2.waitKey(adj)

    if not cap.isOpened():
        print("Erreur : impossible d’ouvrir la vidéo.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    paused = False

    try:
        while cap.isOpened():
            
            if not paused:
                ret, frame = cap.read()
                cv2.putText(frame,
                        f"Vitesse de lecture : x{play_speed:.1f}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)
            
                if not ret:
                    print("Fin de la vidéo ou erreur de lecture.")
                    break
                frame_number += 1


            # Indiquer mode pause
            if paused and ret:
                cv2.putText(frame,
                            "|| PAUSE ||",
                            (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA)

            if ret:
                cv2.imshow(f'{video_path}', frame)

            key = _waitKey_fast(30) & 0xFF
            if key == ord('q'):  # touche 'q' pour quitter
                break
            elif key == ord(' '):  # barre espace
                paused = not paused
            elif key == ord('0'):  # touche '0' pour marquer le début du match
                starting_game_frame = frame_number
                start_time = starting_game_frame / fps # peut être supprimé une fois débuggé
                print(f"Début du match marqué au frame {starting_game_frame}, soit {start_time:.2f} secondes")
                break
            elif key == ord('+'):
                play_speed += 0.5
                # print(f"Vitesse de lecture augmentée à x{play_speed:.1f}")
                continue
            elif key == ord('-'):
                play_speed = max(0.5, play_speed - 0.5)
                # print(f"Vitesse de lecture diminuée à x{play_speed:.1f}")
                continue

    finally:
        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()

    extract_gpu_segment(
        input_video=video_path,
        start_frame=starting_game_frame,
        end_frame=last_game_frame,
        output_video=f'{os.path.splitext(video_path)[0]}_started.mp4'
        )
    