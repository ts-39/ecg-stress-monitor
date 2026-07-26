import sqlite3
import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "stress_log.db"


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stress_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                stress_prob REAL    NOT NULL,
                is_stress   INTEGER NOT NULL,
                hr_bpm      REAL,
                rri_ms      INTEGER,
                rmssd_ms    REAL,
                event_id    TEXT,
                event_title TEXT
            )
        """)
        conn.commit()


def insert_log(
    timestamp: float,
    stress_prob: float,
    is_stress: bool,
    hr_bpm: float | None = None,
    rri_ms: int | None = None,
    rmssd_ms: float | None = None,
    event_id: str | None = None,
    event_title: str | None = None,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO stress_logs
               (timestamp, stress_prob, is_stress, hr_bpm, rri_ms, rmssd_ms, event_id, event_title)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, stress_prob, int(is_stress), hr_bpm, rri_ms, rmssd_ms, event_id, event_title),
        )
        conn.commit()


def get_today_logs() -> list[dict]:
    today_start = datetime.datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM stress_logs WHERE timestamp >= ? ORDER BY timestamp",
            (today_start,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_logs_in_range(start_ts: float, end_ts: float) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM stress_logs WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start_ts, end_ts),
        ).fetchall()
    return [dict(r) for r in rows]


def get_event_summary(event_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """SELECT
                COUNT(*)          AS sample_count,
                AVG(stress_prob)  AS avg_stress,
                MAX(stress_prob)  AS peak_stress,
                AVG(hr_bpm)       AS avg_hr,
                AVG(rmssd_ms)     AS avg_rmssd,
                SUM(is_stress)    AS stress_samples
               FROM stress_logs WHERE event_id = ?""",
            (event_id,),
        ).fetchone()
    if row is None or row[0] == 0:
        return None
    return {
        "sample_count":      row[0],
        "avg_stress":        round(row[1], 3) if row[1] is not None else None,
        "peak_stress":       round(row[2], 3) if row[2] is not None else None,
        "avg_hr":            round(row[3], 1) if row[3] is not None else None,
        "avg_rmssd":         round(row[4], 1) if row[4] is not None else None,
        "stress_seconds":    int(row[5]) * 5 if row[5] is not None else 0,
    }
