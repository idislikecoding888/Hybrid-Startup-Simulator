import sqlite3
from threading import Lock

_DB_PATH = "simulation.db"
_connection = None
_lock = Lock()


def get_connection():
    """
    Returns a singleton SQLite connection.
    Thread-safe.
    """
    global _connection

    if _connection is None:
        with _lock:
            if _connection is None:
                _connection = sqlite3.connect(
                    _DB_PATH,
                    check_same_thread=False
                )
                _connection.row_factory = sqlite3.Row

    return _connection


def close_connection():
    """
    Close DB connection (optional cleanup).
    """
    global _connection

    if _connection:
        _connection.close()
        _connection = None