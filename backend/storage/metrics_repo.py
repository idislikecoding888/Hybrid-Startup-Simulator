from typing import List, Dict
from datetime import datetime
from backend.storage.db import get_connection


class MetricsRepository:
    """
    Handles storage and retrieval of simulation metrics.
    """

    def __init__(self):
        self.conn = get_connection()
        self._create_table()

    def _create_table(self):
        """
        Creates metrics table if not exists.
        """
        query = """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER,
            timestamp TEXT,
            revenue REAL,
            customers INTEGER,
            cac REAL,
            conversion_rate REAL
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    # ---------- INSERT ---------- #

    def save_metrics(self, step: int, metrics: Dict):
        """
        Save metrics for a simulation step.
        """

        query = """
        INSERT INTO metrics (
            step, timestamp, revenue, customers, cac, conversion_rate
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """

        self.conn.execute(
            query,
            (
                step,
                datetime.utcnow().isoformat(),
                metrics.get("revenue", 0.0),
                metrics.get("customers", 0),
                metrics.get("cac", 0.0),
                metrics.get("conversion_rate", 0.0)
            )
        )

        self.conn.commit()

    def save_bulk(self, metrics_list: List[Dict]):
        """
        Save multiple steps of metrics.
        """

        query = """
        INSERT INTO metrics (
            step, timestamp, revenue, customers, cac, conversion_rate
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """

        payload = []
        for item in metrics_list:
            payload.append((
                item.get("step"),
                item.get("timestamp", datetime.utcnow().isoformat()),
                item.get("revenue", 0.0),
                item.get("customers", 0),
                item.get("cac", 0.0),
                item.get("conversion_rate", 0.0)
            ))

        self.conn.executemany(query, payload)
        self.conn.commit()

    # ---------- FETCH ---------- #

    def get_all_metrics(self) -> List[Dict]:
        query = "SELECT * FROM metrics ORDER BY step ASC"
        cursor = self.conn.execute(query)

        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_latest_metrics(self) -> Dict:
        query = "SELECT * FROM metrics ORDER BY step DESC LIMIT 1"
        cursor = self.conn.execute(query)
        row = cursor.fetchone()

        return self._row_to_dict(row) if row else {}

    def get_metrics_by_step(self, step: int) -> Dict:
        query = "SELECT * FROM metrics WHERE step = ?"
        cursor = self.conn.execute(query, (step,))
        row = cursor.fetchone()

        return self._row_to_dict(row) if row else {}

    def get_last_n_metrics(self, n: int) -> List[Dict]:
        query = "SELECT * FROM metrics ORDER BY step DESC LIMIT ?"
        cursor = self.conn.execute(query, (n,))

        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in reversed(rows)]

    # ---------- DELETE ---------- #

    def clear_metrics(self):
        query = "DELETE FROM metrics"
        self.conn.execute(query)
        self.conn.commit()

    # ---------- INTERNAL ---------- #

    def _row_to_dict(self, row) -> Dict:
        if not row:
            return {}

        return {
            "id": row[0],
            "step": row[1],
            "timestamp": row[2],
            "revenue": row[3],
            "customers": row[4],
            "cac": row[5],
            "conversion_rate": row[6]
        }