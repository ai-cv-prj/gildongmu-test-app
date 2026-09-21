"""Conservative image motion compensation; never predicts a detection or color."""
import math

import numpy as np


def motion_gray(frame, cv2):
    height, width = frame.shape[:2]
    scale = min(1.0, 480 / max(height, width))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (round(width * scale), round(height * scale)))


def estimate_camera_motion(previous, current, shape, cv2):
    if previous is None or previous.shape != current.shape:
        return None, {"reason": "no_previous_image"}
    points = cv2.goodFeaturesToTrack(previous, 300, 0.01, 10)
    if points is None or len(points) < 16:
        return None, {"reason": "insufficient_features"}
    moved, forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if moved is None:
        return None, {"reason": "flow_failed"}
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None)
    if returned is None:
        return None, {"reason": "flow_failed"}
    a, b = points.reshape(-1, 2), moved.reshape(-1, 2)
    height, width = current.shape
    valid = (forward.reshape(-1).astype(bool) & backward.reshape(-1).astype(bool)
             & (np.linalg.norm(returned.reshape(-1, 2) - a, axis=1) <= 1.5)
             & np.isfinite(b).all(axis=1)
             & (b[:, 0] >= 0) & (b[:, 0] < width)
             & (b[:, 1] >= 0) & (b[:, 1] < height))
    a, b = a[valid], b[valid]
    if len(a) < 16:
        return None, {"reason": "insufficient_matches"}
    matrix, inliers = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=1000,
        confidence=0.99, refineIters=10,
    )
    if matrix is None or not np.isfinite(matrix).all():
        return None, {"reason": "motion_fit_failed"}
    mask = inliers.reshape(-1).astype(bool)
    matched = a[mask]
    ratio = float(mask.mean())
    # Features must cover multiple areas of the image, not just one moving car.
    cells = {(min(2, int(x * 3 / width)), min(2, int(y * 3 / height))) for x, y in matched}
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    angle = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    reliable = (len(matched) >= 16 and ratio >= 0.65 and len(cells) >= 4
                and np.ptp(matched[:, 0]) >= 0.4 * width
                and np.ptp(matched[:, 1]) >= 0.4 * height
                and 0.85 <= scale <= 1.18 and angle <= 15
                and abs(matrix[0, 2]) <= 0.3 * width
                and abs(matrix[1, 2]) <= 0.3 * height)
    diagnostic = {"reason": "compensated" if reliable else "unreliable_motion",
                  "matches": len(a), "inliers": int(mask.sum()), "inlier_ratio": round(ratio, 4)}
    if not reliable:
        return None, diagnostic
    # Account for rounded resize dimensions independently along x and y.
    scaling = np.diag([shape[1] / width, shape[0] / height, 1.0])
    affine = np.eye(3)
    affine[:2] = matrix
    matrix = (scaling @ affine @ np.linalg.inv(scaling))[:2]
    diagnostic["matrix"] = matrix.tolist()
    return matrix, diagnostic


def transform_box(box, matrix):
    if box is None or matrix is None:
        return box
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ matrix.T
    return [float(corners[:, 0].min()), float(corners[:, 1].min()),
            float(corners[:, 0].max()), float(corners[:, 1].max())]
