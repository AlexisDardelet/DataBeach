# DataBeach — © 2025 Alexis Dardelet
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0

"""One-time patch: update Firestore 'series' documents with serie_name from SQLite."""

import os
import sqlite3
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
KEY_PATH = os.getenv("FIRESTORE_KEY_PATH")
DB_PATH = os.path.join(BASE_DIR, "..", "database", "databeach_base.db")

cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT serie_id, serie_name FROM table_serie")
rows = cursor.fetchall()
conn.close()

col = db.collection("series")
updated = 0
for r in rows:
    serie_id = r["serie_id"]
    serie_name = r["serie_name"]
    col.document(serie_id).set({"serie_name": serie_name}, merge=True)
    print(f"  ✅ {serie_id} → serie_name='{serie_name}'")
    updated += 1

print(f"\n🏁 {updated} documents mis à jour.")
