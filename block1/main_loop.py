import time

from alerts import check_and_fire_alerts
from capture import get_latest_frame, start_capture_thread
from change_detection import has_settled_change
from config import (
    CAMERA_SOURCE,
    DB_PATH,
    DEBOUNCE_FRAMES,
    DETECTION_CONF_THRESHOLD,
)
from detector import detect_objects, load_detector
from embedder import embed_crop, load_embedder
from matcher import load_registry_embeddings, match_against_registry
from state_store import (
    count_registered_tools,
    get_registered_tool_names,
    init_db,
    log_raw_detection,
    upsert_tool_state,
)


REGISTRY_REFRESH_SECONDS = 60.0
ALERT_CHECK_SECONDS = 60.0
IDLE_SLEEP_SECONDS = 0.05


def _refresh_registry(db_path):
    registry = load_registry_embeddings(db_path)
    vocabulary = get_registered_tool_names(db_path)
    return registry, vocabulary


def _process_detections(
    frame,
    detector_model,
    embedder_model,
    transform,
    registry,
    vocabulary,
):
    detections = detect_objects(
        detector_model,
        frame,
        vocabulary,
        DETECTION_CONF_THRESHOLD,
    )
    frame_height, frame_width = frame.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))
        bbox = (x1, y1, x2, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        embedding = embed_crop(embedder_model, transform, crop)
        tool_id, similarity_score = match_against_registry(embedding, registry)

        log_raw_detection(DB_PATH, tool_id, similarity_score, bbox)
        if tool_id is not None:
            upsert_tool_state(DB_PATH, tool_id, "frame", "in_place")


def run():
    init_db(DB_PATH)
    if count_registered_tools(DB_PATH) == 0:
        print(
            "WARNING: tool_registry is empty. Monitoring will continue, "
            "but no detections can be matched."
        )

    buffer = start_capture_thread(CAMERA_SOURCE)
    detector_model = load_detector()
    embedder_model, transform = load_embedder()
    registry, vocabulary = _refresh_registry(DB_PATH)

    reference_frame = None
    motion_observed = False
    stable_frame_count = 0
    last_processed_timestamp = None
    last_registry_refresh = time.monotonic()
    last_alert_check = float("-inf")

    while True:
        try:
            now = time.monotonic()

            if now - last_alert_check >= ALERT_CHECK_SECONDS:
                last_alert_check = now
                check_and_fire_alerts(DB_PATH)

            if now - last_registry_refresh >= REGISTRY_REFRESH_SECONDS:
                last_registry_refresh = now
                refreshed_registry, refreshed_vocabulary = _refresh_registry(DB_PATH)
                registry = refreshed_registry
                vocabulary = refreshed_vocabulary

            current_frame, captured_at = get_latest_frame(buffer)
            if (
                current_frame is None
                or captured_at is None
                or captured_at == last_processed_timestamp
            ):
                time.sleep(IDLE_SLEEP_SECONDS)
                continue
            last_processed_timestamp = captured_at

            if reference_frame is None:
                reference_frame = current_frame.copy()
                continue

            changed = has_settled_change(reference_frame, current_frame)
            if changed:
                motion_observed = True
                stable_frame_count = 0
                continue

            if not motion_observed:
                continue

            stable_frame_count += 1
            if stable_frame_count < DEBOUNCE_FRAMES:
                continue

            reference_frame = current_frame.copy()
            motion_observed = False
            stable_frame_count = 0

            _process_detections(
                current_frame,
                detector_model,
                embedder_model,
                transform,
                registry,
                vocabulary,
            )
        except Exception as exc:
            print(f"Monitoring loop error: {exc}")
            time.sleep(IDLE_SLEEP_SECONDS)


if __name__ == "__main__":
    run()
