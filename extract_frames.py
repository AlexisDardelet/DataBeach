import os
import subprocess
import shutil

# Dossiers source
SRC_POINTS = "segments_points"
SRC_NONPOINTS = "segments_temps_hors_jeu"

# Dossiers de sortie
DST_POINTS = "frames_points"
DST_NONPOINTS = "frames_nonpoints"

def extract_all(src_dir, dst_dir):
    print(f"--- Extraction depuis {src_dir} vers {dst_dir} ---")
    os.makedirs(dst_dir, exist_ok=True)

    for fn in os.listdir(src_dir):
        if not fn.lower().endswith(".mp4"):
            continue

        src = os.path.join(src_dir, fn)

        clip_name = os.path.splitext(fn)[0]
        out_folder = os.path.join(dst_dir, clip_name)

        # Supprime le dossier si il existe déjà (pour éviter le mélange)
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)

        os.makedirs(out_folder, exist_ok=True)

        # Commande ffmpeg
        cmd = [
            "ffmpeg",
            "-i", src,
            "-qscale:v", "2",  # bonne qualité, compression raisonnable
            os.path.join(out_folder, "frame_%05d.jpg")
        ]

        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("Extraction terminée.")


if __name__ == "__main__":
    extract_all(SRC_POINTS, DST_POINTS)
    extract_all(SRC_NONPOINTS, DST_NONPOINTS)
