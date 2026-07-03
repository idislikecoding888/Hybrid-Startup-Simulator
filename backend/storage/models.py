from backend.storage.db import get_connection


def create_tables():
    """
    Initialize all required tables.
    Call this once at startup.
    """

    conn = get_connection()

    # ---------- LOGS TABLE ---------- #
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        type TEXT,
        data TEXT
    )
    """)

    # ---------- METRICS TABLE ---------- #
    conn.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        step INTEGER,
        timestamp TEXT,
        revenue REAL,
        customers INTEGER,
        cac REAL,
        conversion_rate REAL
    )
    """)

    # ---------- STATE TABLE (OPTIONAL BUT USEFUL) ---------- #
    conn.execute("""
    CREATE TABLE IF NOT EXISTS state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        step INTEGER,
        timestamp TEXT,
        state_json TEXT
    )
    """)

    conn.commit()