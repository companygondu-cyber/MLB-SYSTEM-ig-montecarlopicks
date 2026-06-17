"""Database connection manager for OmegaFinal-MLB. Single source of DB access."""

import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from config import CONFIG


class DBConnection:
    """SQLite connection manager. Uses WAL mode for concurrent reads, single writer."""

    def __init__(self, db_path: Optional[str] = None, timeout: float = 30.0):
        self._path = str(db_path or CONFIG.paths.db)
        self._timeout = timeout

    @property
    def path(self) -> str:
        return self._path

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=self._timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self._new_connection()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur
        finally:
            conn.close()

    def query(self, sql: str, params: tuple = ()) -> list:
        conn = self._new_connection()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def query_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        conn = self._new_connection()
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    def read_df(self, sql: str, params: tuple = ()):
        """Read SQL into a pandas DataFrame."""
        import pandas as pd
        conn = self._new_connection()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    def backup(self, dest: Optional[str] = None) -> str:
        """Copy current DB to backup path. Returns destination path."""
        import shutil
        dest = dest or str(CONFIG.paths.db_backup)
        shutil.copy2(self._path, dest)
        return dest

    def restore(self, source: Optional[str] = None) -> str:
        """Restore DB from backup. Returns source path used."""
        import shutil
        source = source or str(CONFIG.paths.db_backup)
        shutil.copy2(source, self._path)
        return source


@contextmanager
def db_cursor(db_path: Optional[str] = None) -> Iterator[sqlite3.Cursor]:
    """Context manager: yields a cursor, auto-commits on success, rolls back on error."""
    conn = sqlite3.connect(str(db_path or CONFIG.paths.db), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_default_db: Optional[DBConnection] = None


def default_db() -> DBConnection:
    """Process-wide default DB connection manager."""
    global _default_db
    if _default_db is None:
        _default_db = DBConnection()
    return _default_db
