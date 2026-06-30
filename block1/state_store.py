import sqlite3
from contextlib import closing


def _connect(db_path):
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path):
    """Create the Block 1 SQLite schema if it does not already exist."""

    with closing(_connect(db_path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tool_registry (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                reference_embedding BLOB NOT NULL,
                home_location TEXT,
                registered_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tool_state (
                tool_id TEXT PRIMARY KEY,
                last_seen_at TIMESTAMP,
                last_seen_location TEXT,
                status TEXT CHECK(status IN ('in_place', 'moved', 'missing', 'unseen')),
                FOREIGN KEY (tool_id) REFERENCES tool_registry(id)
            );

            CREATE TABLE IF NOT EXISTS raw_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TIMESTAMP,
                matched_tool_id TEXT,
                similarity_score REAL,
                bounding_box TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                alerted_at TIMESTAMP NOT NULL,
                message TEXT NOT NULL
            );
            """
        )


def count_registered_tools(db_path):
    """Return the number of tools currently in the registry."""

    with closing(_connect(db_path)) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM tool_registry").fetchone()
    return int(row["count"])


def get_registered_tool_names(db_path):
    """Return unique registered tool names for the detector vocabulary."""

    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT name FROM tool_registry ORDER BY id"
        ).fetchall()
    return list(dict.fromkeys(row["name"] for row in rows))


def upsert_tool_state(db_path, tool_id, location, status):
    """Record a matched tool's latest location and status."""

    with closing(_connect(db_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO tool_state (
                tool_id,
                last_seen_at,
                last_seen_location,
                status
            )
            VALUES (?, CURRENT_TIMESTAMP, ?, ?)
            ON CONFLICT(tool_id) DO UPDATE SET
                last_seen_at = CURRENT_TIMESTAMP,
                last_seen_location = excluded.last_seen_location,
                status = excluded.status
            """,
            (tool_id, location, status),
        )


def get_all_tool_states(db_path):
    """Return all tool state rows as dictionaries."""

    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT tool_id, last_seen_at, last_seen_location, status
            FROM tool_state
            ORDER BY tool_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def log_raw_detection(db_path, tool_id, similarity_score, bbox):
    """Persist a raw detection and its optional registry match."""

    bounding_box = ",".join(str(int(value)) for value in bbox)
    with closing(_connect(db_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO raw_detections (
                detected_at,
                matched_tool_id,
                similarity_score,
                bounding_box
            )
            VALUES (CURRENT_TIMESTAMP, ?, ?, ?)
            """,
            (tool_id, similarity_score, bounding_box),
        )
