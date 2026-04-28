# migrate_to_firestore.py
import os
import sqlite3
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ── Chemins ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = os.path.join(BASE_DIR, "..", ".json")
DB_PATH = os.path.join(BASE_DIR, "..", "database", "databeach_base.db")

# ── Init Firebase ──────────────────────────────────────────────
cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Init SQLite ────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# ── Helpers ────────────────────────────────────────────────────
def clean_int(val):
    if val == '' or val is None:
        return None
    return int(val)

def migrate_batch(collection_name, rows_with_id):
    """rows_with_id : liste de (doc_id, dict_data)"""
    col = db.collection(collection_name)
    batch = db.batch()
    for i, (doc_id, data) in enumerate(rows_with_id):
        batch.set(col.document(doc_id), data)
        if (i + 1) % 500 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()
    print(f"  ✅ {len(rows_with_id)} documents → '{collection_name}'")

# ── 1. table_player ───────────────────────────────────────────
print("Migration table_player...")
cursor.execute("SELECT * FROM table_player")
rows = [
    (r["paire_id"], {
        "player_a": r["player_a"],
        "player_b": r["player_b"],
        "genre":    r["genre"],
    })
    for r in cursor.fetchall()
]
migrate_batch("players", rows)

# ── 2. table_serie ────────────────────────────────────────────
print("Migration table_serie...")
cursor.execute("SELECT * FROM table_serie")
rows = []
for r in cursor.fetchall():
    try:
        date_obj = datetime.strptime(r["date"], "%d/%m/%Y")
    except Exception:
        date_obj = None
    rows.append((r["serie_id"], {
        "club":  r["club"],
        "type":  r["type"],
        "genre": r["genre"],
        "date":  date_obj,
    }))
migrate_batch("series", rows)

# ── 3. table_game ─────────────────────────────────────────────
print("Migration table_game...")
cursor.execute("SELECT * FROM table_game")
rows = [
    (r["game_id"], {
        "serie":          r["serie"],
        "stage":          r["stage"],
        "team_a":         r["team_a"],
        "team_b":         r["team_b"],
        "victory":        r["victory"],
        "set1_score":     clean_int(r["set1_score"]),
        "set2_score":     clean_int(r["set2_score"]),
        "set3_score":     clean_int(r["set3_score"]),
        "set1_score_adv": clean_int(r["set1_score_adv"]),
        "set2_score_adv": clean_int(r["set2_score_adv"]),
        "set3_score_adv": clean_int(r["set3_score_adv"]),
    })
    for r in cursor.fetchall()
]
migrate_batch("games", rows)

# ── 4. table_point ────────────────────────────────────────────
print("Migration table_point...")
cursor.execute("SELECT * FROM table_point")
rows = [
    (r["point_id"], {
        "game_id":           r["game_id"],
        "service_side":      r["service_side"],
        "team_a_score":      r["team_a_score"],
        "team_b_score":      r["team_b_score"],
        "team_a_sets":       r["team_a_sets"],
        "team_b_sets":       r["team_b_sets"],
        "team_a_score_diff": r["team_a_score_diff"],
        "team_b_score_diff": r["team_b_score_diff"],
        "point_winner":      r["point_winner"],
    })
    for r in cursor.fetchall()
]
migrate_batch("points", rows)

# ── 5. table_serve ────────────────────────────────────────────
print("Migration table_serve...")
cursor.execute("SELECT * FROM table_serve")
rows = [
    (f"{r['point_id']}_s{r['serve_id']}", {
        "point_id":       r["point_id"],
        "paire_id":       r["paire_id"],
        "player":         r["player"],
        "action":         r["action"],
        "grade":          r["grade"],
        "previous_grade": r["previous_grade"],
        "point_won":      bool(r["point_won"]) if r["point_won"] is not None else None,
    })
    for r in cursor.fetchall()
]
migrate_batch("serves", rows)

# ── 6. table_pass ─────────────────────────────────────────────
print("Migration table_pass...")
cursor.execute("SELECT * FROM table_pass")
rows = [
    (f"{r['point_id']}_p{r['pass_id']}", {
        "point_id":       r["point_id"],
        "paire_id":       r["paire_id"],
        "player":         r["player"],
        "action":         r["action"],
        "grade":          r["grade"],
        "previous_grade": r["previous_grade"],
        "point_won":      bool(r["point_won"]) if r["point_won"] is not None else None,
    })
    for r in cursor.fetchall()
]
migrate_batch("passes", rows)

# ── Fin ───────────────────────────────────────────────────────
conn.close()
print("\n🏁 Migration terminée !")