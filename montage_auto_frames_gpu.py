"""
Extraction frame-accurate optimisée NVIDIA NVENC (GTX 1060+)

Fonctions exposées :
    - extract_segments_gpu(input_video, ranges_file, output_dir)

Format du fichier ranges :
    startFrame-endFrame
    Exemple :
        242-473
        701-996
"""

import os
import subprocess


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
# Script principal exportable
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