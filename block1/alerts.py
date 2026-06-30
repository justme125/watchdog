import sqlite3

from config import MISSING_THRESHOLD_MINUTES


def check_and_fire_alerts(db_path):
    """Atomically mark overdue tools missing and emit one alert per absence."""

    threshold_modifier = f"-{MISSING_THRESHOLD_MINUTES} minutes"
    connection = sqlite3.connect(db_path, timeout=30)

    try:
        connection.execute("BEGIN IMMEDIATE")

        connection.execute(
            """
            UPDATE tool_state
            SET status = 'in_place'
            WHERE status = 'missing'
              AND last_seen_at IS NOT NULL
              AND datetime(last_seen_at) >= datetime('now', ?)
            """,
            (threshold_modifier,),
        )

        overdue_tools = connection.execute(
            """
            SELECT state.tool_id, registry.name
            FROM tool_state AS state
            JOIN tool_registry AS registry ON registry.id = state.tool_id
            WHERE state.last_seen_at IS NOT NULL
              AND datetime(state.last_seen_at) < datetime('now', ?)
              AND state.status != 'missing'
            """,
            (threshold_modifier,),
        ).fetchall()

        alert_messages = []
        for tool_id, tool_name in overdue_tools:
            updated = connection.execute(
                """
                UPDATE tool_state
                SET status = 'missing'
                WHERE tool_id = ? AND status != 'missing'
                """,
                (tool_id,),
            )
            if updated.rowcount != 1:
                continue

            message = (
                f"Tool {tool_name} has not been seen for over "
                f"{MISSING_THRESHOLD_MINUTES} minutes"
            )
            connection.execute(
                """
                INSERT INTO alerts (
                    tool_id,
                    tool_name,
                    alerted_at,
                    message
                )
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (tool_id, tool_name, message),
            )
            alert_messages.append(message)

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    for message in alert_messages:
        print(message)
