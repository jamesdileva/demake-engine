"""
Database connection and session management.
Uses SQLite with WAL mode enabled to prevent lock contention
when multiple async workers access the DB simultaneously.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from .models import Base
import os

DB_PATH = os.environ.get("DEMAKE_DB_PATH", "demake.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Required for SQLite + FastAPI
        "timeout": 30,               # Wait up to 30s if DB is locked
    },
    echo=False,  # Set True to log all SQL (useful for debugging)
)


@event.listens_for(engine, "connect")
def set_wal_mode(dbapi_conn, connection_record):
    """
    Enable WAL (Write-Ahead Logging) mode on every new connection.
    Prevents 'database is locked' errors when async workers write simultaneously.
    Architecture doc requirement: SQLite WAL mode.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Initialized at {DB_PATH} (WAL mode)")


def get_db() -> Session:
    """
    FastAPI dependency — yields a DB session and closes it after the request.

    Usage in a route:
        @router.get("/something")
        def my_route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()