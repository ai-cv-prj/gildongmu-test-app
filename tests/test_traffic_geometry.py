"""Image-level regression tests for paint boundaries, clipping and missing evidence."""
import cv2
import numpy as np
import pytest

from backend.app.inference.traffic import estimate_vanishing_point


def painted_crossing(count=6, parallel=False):
    frame = np.full((640, 480, 3), 60, dtype=np.uint8)
    for y in [230, 280, 340, 415, 505, 605][:count]:
        corners = [[240 + side * (100 if parallel else 0.35 * (yy - 100)), yy]
                   for yy, side in [(y - 8, -1), (y - 8, 1), (y + 8, 1), (y + 8, -1)]]
        cv2.fillConvexPoly(frame, np.array(corners, dtype=np.int32), (240, 240, 240))
    return frame


def test_repeated_paint_boundaries_converge_at_expected_point():
    point = estimate_vanishing_point(painted_crossing(), [0, 180, 480, 640], cv2)
    assert point == pytest.approx([240, 100], abs=6)


@pytest.mark.parametrize('count', [0, 1, 2])
def test_insufficient_stripes_do_not_invent_direction(count):
    assert estimate_vanishing_point(painted_crossing(count), [0, 180, 480, 640], cv2) is None


def test_parallel_stripes_cannot_establish_finite_vanishing_point():
    assert estimate_vanishing_point(painted_crossing(parallel=True), [0, 180, 480, 640], cv2) is None


def test_cropped_stripe_ends_are_not_used_as_road_boundaries():
    assert estimate_vanishing_point(painted_crossing(), [200, 180, 300, 640], cv2) is None


def test_clutter_without_white_stripes_stays_unknown():
    frame = np.random.default_rng(27).integers(0, 170, (640, 480, 3), dtype=np.uint8)
    assert estimate_vanishing_point(frame, [0, 180, 480, 640], cv2) is None
