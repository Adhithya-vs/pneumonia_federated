"""
database.py — SQLite persistence layer for DiagnosAI
Handles: patients, chat messages
Auth (doctors) stays in doctors_auth.json as before.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "diagnosai.db"


def get_connection():
    """Get a thread-safe SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    cur  = conn.cursor()

    # ----------------------------------------------------------
    # PATIENTS
    # ----------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            age         TEXT,
            gender      TEXT,
            phone       TEXT,
            height      TEXT,
            weight      TEXT,
            prediction  TEXT,
            confidence  REAL,
            filename    TEXT,
            heatmap     TEXT,
            report_url  TEXT,
            feedback    TEXT DEFAULT 'Pending',
            consultant  TEXT,
            date        TEXT
        )
    """)

    # ----------------------------------------------------------
    # CHAT MESSAGES
    # ----------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id           TEXT PRIMARY KEY,
            sender_email TEXT NOT NULL,
            sender_name  TEXT NOT NULL,
            text         TEXT NOT NULL,
            timestamp    TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialised at", DB_PATH)


# ==============================================================
# PATIENT HELPERS
# ==============================================================

def add_patient(patient: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO patients
            (id, name, age, gender, phone, height, weight,
             prediction, confidence, filename, heatmap,
             report_url, feedback, consultant, date)
        VALUES
            (:id, :name, :age, :gender, :phone, :height, :weight,
             :prediction, :confidence, :filename, :heatmap,
             :report_url, :feedback, :consultant, :date)
    """, patient)
    conn.commit()
    conn.close()


def get_all_patients() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM patients ORDER BY date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(patient_id: str) -> dict | None:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM patients WHERE id = ?", (patient_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_patient_feedback(patient_id: str, feedback: str):
    conn = get_connection()
    conn.execute(
        "UPDATE patients SET feedback = ? WHERE id = ?",
        (feedback, patient_id)
    )
    conn.commit()
    conn.close()


def update_patient_report(patient_id: str, report_url: str):
    conn = get_connection()
    conn.execute(
        "UPDATE patients SET report_url = ? WHERE id = ?",
        (report_url, patient_id)
    )
    conn.commit()
    conn.close()


def delete_patient(patient_id: str) -> dict | None:
    """Delete patient and return their record (for file cleanup)."""
    conn   = get_connection()
    row    = conn.execute(
        "SELECT * FROM patients WHERE id = ?", (patient_id,)
    ).fetchone()
    if row:
        conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        conn.commit()
    conn.close()
    return dict(row) if row else None


def get_model_stats() -> dict:
    conn      = get_connection()
    total     = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    correct   = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE feedback = 'Correct'"
    ).fetchone()[0]
    incorrect = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE feedback = 'Incorrect'"
    ).fetchone()[0]
    conn.close()

    pending  = total - correct - incorrect
    accuracy = round(correct / (correct + incorrect) * 100, 2) \
               if (correct + incorrect) > 0 else 0

    return {
        "total_predictions": total,
        "correct":           correct,
        "incorrect":         incorrect,
        "pending":           pending,
        "accuracy":          accuracy,
    }


# ==============================================================
# CHAT HELPERS
# ==============================================================

def add_chat_message(msg: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO chat_messages (id, sender_email, sender_name, text, timestamp)
        VALUES (:id, :sender_email, :sender_name, :text, :timestamp)
    """, msg)
    conn.commit()
    conn.close()


def get_all_chat_messages() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chat_messages ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==============================================================
# MIGRATION — import existing JSON data into SQLite (run once)
# ==============================================================

def migrate_from_json():
    """
    One-time migration: reads patients_db.json and chat_db.json
    and inserts records into SQLite.  Safe to call multiple times
    (duplicate IDs are ignored).
    """
    import json

    # --- patients ---
    if os.path.exists("patients_db.json"):
        with open("patients_db.json") as f:
            patients = json.load(f)
        conn = get_connection()
        for p in patients:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO patients
                        (id, name, age, gender, phone, height, weight,
                         prediction, confidence, filename, heatmap,
                         report_url, feedback, consultant, date)
                    VALUES
                        (:id, :name, :age, :gender, :phone, :height, :weight,
                         :prediction, :confidence, :filename, :heatmap,
                         :report_url, :feedback, :consultant, :date)
                """, p)
            except Exception as e:
                print(f"  Skipping patient {p.get('id')}: {e}")
        conn.commit()
        conn.close()
        print(f"✅ Migrated {len(patients)} patients from JSON")

    # --- chat ---
    if os.path.exists("chat_db.json"):
        with open("chat_db.json") as f:
            msgs = json.load(f)
        conn = get_connection()
        for m in msgs:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO chat_messages
                        (id, sender_email, sender_name, text, timestamp)
                    VALUES
                        (:id, :sender_email, :sender_name, :text, :timestamp)
                """, m)
            except Exception as e:
                print(f"  Skipping message {m.get('id')}: {e}")
        conn.commit()
        conn.close()
        print(f"✅ Migrated {len(msgs)} chat messages from JSON")


# Auto-init when imported
init_db()