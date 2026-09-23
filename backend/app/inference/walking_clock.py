"""Client capture clock. Nominal FPS is never used as motion evidence."""
from dataclasses import dataclass
import math
import time

@dataclass(frozen=True)
class ClockReading:
    timestamp_s: float
    valid: bool
    gap_s: float | None
    reset_reason: str | None

class FrameClock:
    def __init__(self, reset_gap_s):
        self.reset_gap_s = reset_gap_s
        self.previous_ms = None
        self.elapsed = 0.0
        self.arrival = None

    def read(self, captured_at_ms):
        now = time.monotonic()
        fallback = max(1e-6, now - self.arrival) if self.arrival is not None else 0.0
        self.arrival = now
        numeric = (not isinstance(captured_at_ms, bool) and
                   isinstance(captured_at_ms, (int, float)) and
                   math.isfinite(captured_at_ms) and captured_at_ms >= 0)
        if not numeric:
            self.previous_ms = None
            self.elapsed += fallback
            return ClockReading(self.elapsed, False, None, "invalid_timestamp")
        delta = None if self.previous_ms is None else (captured_at_ms-self.previous_ms)/1000
        self.previous_ms = captured_at_ms
        if delta is not None and delta <= 0:
            self.elapsed += fallback
            return ClockReading(self.elapsed, False, delta, "timestamp_not_increasing")
        self.elapsed += delta if delta is not None else fallback
        gap = delta is not None and delta > self.reset_gap_s
        return ClockReading(self.elapsed, not gap, delta, "capture_gap" if gap else None)
