import cv2

from config import CHANGE_THRESHOLD_PERCENT


def has_settled_change(reference_frame, new_frame):
    """Return whether changed pixels exceed the configured percentage."""

    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    reference_blurred = cv2.GaussianBlur(reference_gray, (5, 5), 0)
    new_blurred = cv2.GaussianBlur(new_gray, (5, 5), 0)

    difference = cv2.absdiff(reference_blurred, new_blurred)
    _, thresholded = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    changed_pixels = cv2.countNonZero(thresholded)
    total_pixels = thresholded.shape[0] * thresholded.shape[1]
    changed_percent = (changed_pixels / total_pixels) * 100.0

    return changed_percent > CHANGE_THRESHOLD_PERCENT
