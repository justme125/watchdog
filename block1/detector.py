from ultralytics import YOLOWorld


def load_detector():
    """Load and return the YOLO-World detector."""

    return YOLOWorld("yolov8s-world.pt")


def detect_objects(model, frame, vocabulary, conf_threshold):
    """Detect vocabulary objects and return confidence-filtered boxes."""

    if not vocabulary:
        return []

    model.set_classes(list(vocabulary))
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            confidence = float(box.conf[0].item())
            if confidence < conf_threshold:
                continue
            coordinates = box.xyxy[0].tolist()
            bbox = tuple(int(value) for value in coordinates)
            detections.append({"bbox": bbox, "confidence": confidence})

    return detections
