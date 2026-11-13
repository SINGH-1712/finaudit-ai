"""Database engine, session factory and declarative base for FinAudit-AI.

SQLite is used for zero-config local development. Override with the
FINAUDIT_DATABASE_URL environment variable to point at PostgreSQL later.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("FINAUDIT_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DEBUG_DIR = DATA_DIR / "debug"
EXPORT_DIR = DATA_DIR / "exports"

for _d in (DATA_DIR, UPLOAD_DIR, DEBUG_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("FINAUDIT_DATABASE_URL", f"sqlite:///{DATA_DIR / 'finaudit.db'}")
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    pool_pre_ping=True,
    echo=os.getenv("FINAUDIT_SQL_ECHO", "0") == "1",
)

if _IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")  # enforce FK + cascades
        cursor.execute("PRAGMA journal_mode=WAL")  # concurrent readers while writing
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts / seeders: commit on success, rollback on error."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent). Models are imported here to register them."""
    import models  # noqa: F401  (import side effect: registers tables on Base.metadata)

    Base.metadata.create_all(bind=engine)
    _ensure_added_columns()


def _ensure_added_columns() -> None:
    """Add columns introduced after the first SQLite file was created.

    ``create_all`` will not ALTER an existing table. Phase 1 rows have no
    ``entry_type``; Phase 2 needs it to split Sales and Purchase vouchers.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "invoices" not in insp.get_table_names():
        return
    present = {col["name"] for col in insp.get_columns("invoices")}
    if "entry_type" in present:
        return
    ddl = "ALTER TABLE invoices ADD COLUMN entry_type VARCHAR(16) DEFAULT 'EXPENSE'"
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text("UPDATE invoices SET entry_type = 'EXPENSE' WHERE entry_type IS NULL"))


if __name__ == "__main__":  # quick smoke test: `python database.py`
    init_db()
    print(f"Database ready at {DATABASE_URL}")
