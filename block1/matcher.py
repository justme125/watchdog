import sqlite3
from contextlib import closing

import numpy as np

from config import SIMILARITY_THRESHOLD


def load_registry_embeddings(db_path):
    """Load float32 reference embeddings from the tool registry."""

    with closing(sqlite3.connect(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT id, name, reference_embedding
            FROM tool_registry
            ORDER BY id
            """
        ).fetchall()

    registry = []
    for tool_id, name, embedding_blob in rows:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).copy()
        registry.append(
            {
                "tool_id": tool_id,
                "name": name,
                "embedding": embedding,
            }
        )
    return registry


def match_against_registry(candidate_embedding, registry):
    """Return the thresholded best tool ID and the best similarity score."""

    candidate = np.asarray(candidate_embedding, dtype=np.float32).reshape(-1)
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate.size == 0 or candidate_norm == 0.0:
        return None, None

    best_tool_id = None
    best_score = None

    for entry in registry:
        reference = np.asarray(entry["embedding"], dtype=np.float32).reshape(-1)
        if reference.shape != candidate.shape:
            continue

        reference_norm = float(np.linalg.norm(reference))
        if reference_norm == 0.0:
            continue

        score = float(np.dot(candidate, reference) / (candidate_norm * reference_norm))
        if not np.isfinite(score):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_tool_id = entry["tool_id"]

    if best_score is None or best_score <= SIMILARITY_THRESHOLD:
        return None, best_score
    return best_tool_id, best_score
