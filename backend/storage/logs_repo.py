from typing import List, Dict
import json
from datetime import datetime
from backend.storage.db import get_connection


class LogsRepository:
    """
    Handles storage and retrieval of deliberation logs.
    """

    def __init__(self):
        self.conn = get_connection()
        self._create_table()

    def _create_table(self):
        """
        Creates logs table if not exists.
        """
        query = """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            type TEXT,
            data TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    # ---------- INSERT ---------- #

    def save_log(self, log: Dict):
        """
        Save a single log entry.
        """

        timestamp = log.get("timestamp", datetime.utcnow().isoformat())
        data = log.get("data", {})
        log_type = data.get("type", "unknown")

        query = """
        INSERT INTO logs (timestamp, type, data)
        VALUES (?, ?, ?)
        """

        self.conn.execute(
            query,
            (timestamp, log_type, json.dumps(data))
        )
        self.conn.commit()

    def save_bulk(self, logs: List[Dict]):
        """
        Save multiple logs efficiently.
        """

        query = """
        INSERT INTO logs (timestamp, type, data)
        VALUES (?, ?, ?)
        """

        payload = []
        for log in logs:
            timestamp = log.get("timestamp", datetime.utcnow().isoformat())
            data = log.get("data", {})
            log_type = data.get("type", "unknown")

            payload.append(
                (timestamp, log_type, json.dumps(data))
            )

        self.conn.executemany(query, payload)
        self.conn.commit()

    # ---------- FETCH ---------- #

    def get_all_logs(self) -> List[Dict]:
        query = "SELECT * FROM logs ORDER BY id ASC"
        cursor = self.conn.execute(query)

        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_latest_log(self) -> Dict:
        query = "SELECT * FROM logs ORDER BY id DESC LIMIT 1"
        cursor = self.conn.execute(query)
        row = cursor.fetchone()

        return self._row_to_dict(row) if row else {}

    def get_logs_by_type(self, log_type: str) -> List[Dict]:
        query = "SELECT * FROM logs WHERE type = ? ORDER BY id ASC"
        cursor = self.conn.execute(query, (log_type,))

        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_last_n_logs(self, n: int) -> List[Dict]:
        query = "SELECT * FROM logs ORDER BY id DESC LIMIT ?"
        cursor = self.conn.execute(query, (n,))

        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in reversed(rows)]

    # ---------- DELETE ---------- #

    def clear_logs(self):
        query = "DELETE FROM logs"
        self.conn.execute(query)
        self.conn.commit()

    # ---------- INTERNAL ---------- #

    def _row_to_dict(self, row) -> Dict:
        """
        Convert DB row → Python dict
        """
        if not row:
            return {}

        return {
            "id": row[0],
            "timestamp": row[1],
            "type": row[2],
            "data": json.loads(row[3])
        }
