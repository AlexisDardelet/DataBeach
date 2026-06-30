# DataBeach

![Licence](https://img.shields.io/badge/licence-PolyForm_Noncommercial_1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.39-red)
![Backend](https://img.shields.io/badge/backend-Firestore-orange)
![Deploy](https://img.shields.io/badge/deploy-Cloud_Run-green)

Plateforme data & BI d'analyse de la performance en beach-volley. DataBeach couvre la chaîne complète : génération du dataset à partir de vidéos de matchs jusqu'à l'exploitation BI par les entraîneurs et joueurs.

Projet développé en collaboration avec [MyVolley](https://www.my-volley.com/app).

---

## Sommaire

1. [Vue d'ensemble](#vue-densemble)
2. [Les 4 axes du projet](#les-4-axes-du-projet)
3. [Modèle de données](#modèle-de-données)
4. [Stack technique](#stack-technique)
5. [Structure du repo](#structure-du-repo)
6. [Installation & configuration](#installation--configuration)
7. [Feuille de route](#feuille-de-route)
8. [Licence](#licence)

---

## Vue d'ensemble

DataBeach permet d'analyser finement la performance des paires de beach-volley à partir de vidéos de matchs. Le pipeline complet transforme une vidéo brute en données structurées consultables par les coaches et les joueurs via une interface web.

**Statut** : opérationnel de bout en bout — en amélioration continue.

---

## Les 4 axes du projet

### 1. Capture & segmentation vidéo

Pipeline d'analyse vidéo propriétaire basé sur **OpenCV** pour la segmentation et la notation interactives, et **FFmpeg GPU** (CUDA/NVENC) pour la découpe image-précise et la génération de montages.

- Segmentation point par point avec contrôle de score en temps réel
- Découpe précise à la frame via accélération GPU
- Annulation/correction des segments via `segment_undo`

### 2. Notation des actions

Interface de notation (`streamlit_dev_interface/`) permettant à l'analyste de grader chaque action (service, réception, passe, attaque…) point par point.

- Notation interactive synchronisée avec la vidéo segmentée
- Export des grades en JSON par match et par type d'action

### 3. Stockage & ETL

Chaîne ETL complète :

```
vidéo brute → segmentation → indexation des points (CSV) → notation (JSON) → SQLite → Firestore
```

- **SQLite** : base locale pour la production et le contrôle qualité de la donnée
- **Firestore** : backend cloud pour l'interface coach (lecture seule)
- Migration automatisée SQLite → Firestore via `migration_firestore.py`

### 4. Exploitation BI

Interface Streamlit de consultation pour coachs et joueurs, déployée sur **Google Cloud Run** via Docker.

- **Serve Focus** *(opérationnel)* : analyse détaillée des services — rendement, type, placement
- **Vue d'ensemble coach** *(en cours d'enrichissement)* : synthèse des performances par paire et par tournoi

---

## Modèle de données

| Table | Description |
|---|---|
| `table_player` | Paires de joueurs (`paire_id`, `player_a`, `player_b`, `genre`) |
| `table_serie` | Tournois (`serie_id`, `club`, `type`, `genre`, `date`) |
| `table_game` | Matchs (`game_id`, `serie`, `stage`, `team_a`, `team_b`, `victory`, scores) |
| `table_point` | Points indexés par match (`point_id`, `game_id`, frames, scores) |
| `actions_*` | Tables d'actions notées par type (service, réception…) |

Fichiers de travail organisés par dossier :

- `indexed_df_points/` — points indexés par match (CSV)
- `recap_dict_score/` — récapitulatifs de score par match (JSON)
- `actions_graded/` — notations exportées par match et action (JSON)
- `root_tables/` — tables de référence (joueurs, séries, matchs)

---

## Stack technique

| Couche | Techno |
|---|---|
| Analyse vidéo | OpenCV, FFmpeg (CUDA/NVENC) |
| Interfaces | Streamlit 1.39, Plotly |
| Stockage local | SQLite |
| Backend cloud | Google Firestore |
| Déploiement | Docker, Google Cloud Run |
| Configuration | python-dotenv (`.env`) |
| Langage | Python 3.11 |

---

## Structure du repo

```
DataBeach/
├── scripts/                        # Pipeline ETL & analyse vidéo
│   ├── video_edit_utils.py         # Segmentation & montage (OpenCV + FFmpeg GPU)
│   ├── video_grader.py             # Notation interactive des actions
│   ├── game_editor.py              # Éditeur de match (score, segments)
│   ├── db_manager.py               # Gestion SQLite
│   ├── etl_utils.py                # Utilitaires ETL (chargement CSV → SQLite)
│   ├── firestore_manager.py        # Lecture Firestore (interface coach)
│   ├── migration_firestore.py      # Migration SQLite → Firestore
│   └── patch_serie_name.py         # Correction one-shot Firestore
│
├── streamlit_coach_interface/      # Interface BI coachs & joueurs (Cloud Run)
│   ├── coach_app.py
│   ├── coach_overview.py
│   └── serve_focus.py
│
├── streamlit_dev_interface/        # Atelier de production de la donnée
│   ├── dev_app.py
│   ├── action_grading_interface.py
│   ├── editor_interface.py
│   └── …
│
├── root_tables/                    # Tables de référence CSV
├── indexed_df_points/              # Points indexés par match
├── recap_dict_score/               # Récapitulatifs de score
├── actions_graded/                 # Actions notées (JSON)
├── database/                       # Base SQLite locale (non versionnée)
│
├── Dockerfile                      # Image Docker de l'interface coach
├── requirements.txt                # Dépendances interface coach (Cloud Run)
├── environment.yml                 # Environnement complet (dev local)
└── .env                            # Secrets & chemins locaux (non versionné)
```

---

## Installation & configuration

### Prérequis

- Python 3.11
- FFmpeg compilé avec support NVENC (GPU NVIDIA requis pour l'encodage)
- Compte Google Cloud avec Firestore activé

### Mise en place locale

```bash
# 1. Cloner le repo
git clone https://github.com/<compte>/DataBeach.git
cd DataBeach

# 2. Créer l'environnement Python
conda env create -f environment.yml
conda activate databeach
# ou
pip install -r requirements.txt

# 3. Configurer les variables d'environnement
# Créer un fichier .env à la racine avec les chemins vidéos et la clé Firebase
cp .env.example .env

# 4. Lancer l'interface de développement
streamlit run streamlit_dev_interface/dev_app.py

# 5. (Optionnel) Lancer l'interface coach en local
streamlit run streamlit_coach_interface/coach_app.py
```

### Déploiement Cloud Run

```bash
docker build -t databeach-coach .
docker tag databeach-coach gcr.io/<projet-gcp>/databeach-coach
docker push gcr.io/<projet-gcp>/databeach-coach
gcloud run deploy databeach-coach \
  --image gcr.io/<projet-gcp>/databeach-coach \
  --platform managed
```

---

## Feuille de route

- [x] Pipeline ETL complet (segmentation → SQLite → Firestore)
- [x] Interface coach — page Serve Focus
- [x] Déploiement Docker / Cloud Run
- [ ] Interface coach — vue d'ensemble (en cours d'enrichissement)
- [ ] Script d'anonymisation du dataset pour publication open data
- [ ] Extension de la couverture des actions notées (attaque, bloc, défense)
- [ ] Tableau de bord de progression longitudinale par paire

---

## Licence

Ce projet est distribué sous licence **PolyForm Noncommercial 1.0.0**.  
Usage non commercial libre — exploitation commerciale réservée.

> Copyright © 2025 Alexis Dardelet  
> Voir le fichier [LICENSE](LICENSE) pour le texte complet.

[![Licence PolyForm Noncommercial](https://img.shields.io/badge/licence-PolyForm_Noncommercial_1.0.0-blue)](https://polyformproject.org/licenses/noncommercial/1.0.0)
