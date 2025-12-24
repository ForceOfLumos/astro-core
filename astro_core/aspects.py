from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Tuple

from astronomy import Body  # only for type hints

from .astrocalc import ecliptic_lon_geocentric, BodyKey


def angle_norm(deg: float) -> float:
    return deg % 360.0


def angle_diff_signed(a: float, b: float) -> float:
    """
    Signed shortest difference a-b in degrees in range (-180, +180].
    """
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def default_search_params(transit_body: BodyKey) -> tuple[int, int]:
    """
    Returns (days_ahead, step_hours) tuned by transit body speed.
    """
    # Extras are slow
    if isinstance(transit_body, str):
        return (365 * 90, 48)

    # Astronomy Engine bodies
    if transit_body == Body.Moon:
        return (365 * 5, 1)
    if transit_body in (Body.Mercury, Body.Venus):
        return (365 * 10, 3)
    if transit_body == Body.Mars:
        return (365 * 15, 6)
    if transit_body == Body.Jupiter:
        return (365 * 60, 24)
    if transit_body == Body.Saturn:
        return (365 * 90, 24)
    if transit_body in (Body.Uranus, Body.Neptune, Body.Pluto):
        return (365 * 90, 48)

    return (365 * 90, 24)


def aspect_delta_deg(body_transit: BodyKey, natal_lon: float, aspect_angle: float, at_utc: datetime) -> float:
    """
    Signed delta between transit body longitude and target longitude (natal_lon + aspect_angle).
    Exact when delta -> 0.
    """
    transit_lon = ecliptic_lon_geocentric(body_transit, at_utc)
    target = (natal_lon + aspect_angle) % 360.0
    return angle_diff_signed(transit_lon, target)


def find_next_aspect_times(
    body_transit: BodyKey,
    natal_lon: float,
    aspect_angle: float,
    start_utc: datetime,
    days_ahead: int = 365 * 2,
    step_hours: int = 12,
    max_hits: int = 3,
) -> List[datetime]:
    """
    Returns up to max_hits timestamps (UTC) where aspect is exact.
    Coarse scan + refinement by local search.
    """
    start_utc = start_utc.astimezone(timezone.utc)
    end_utc = start_utc + timedelta(days=days_ahead)
    step = timedelta(hours=step_hours)

    hits: List[datetime] = []

    prev_t = start_utc
    prev_d = aspect_delta_deg(body_transit, natal_lon, aspect_angle, prev_t)

    t = prev_t + step
    while t <= end_utc and len(hits) < max_hits:
        d = aspect_delta_deg(body_transit, natal_lon, aspect_angle, t)

        # Detect sign change around 0 (crossing exact)
        if (prev_d <= 0.0 <= d) or (d <= 0.0 <= prev_d):
            # refine between prev_t and t by bisection
            lo, hi = prev_t, t
            for _ in range(40):
                mid = lo + (hi - lo) / 2
                dm = aspect_delta_deg(body_transit, natal_lon, aspect_angle, mid)
                if (prev_d <= 0.0 <= dm) or (dm <= 0.0 <= prev_d):
                    hi = mid
                    d = dm
                else:
                    lo = mid
                    prev_d = dm
            hits.append(hi)

            # advance a bit to avoid duplicate hits
            prev_t = hi + step
            prev_d = aspect_delta_deg(body_transit, natal_lon, aspect_angle, prev_t)
            t = prev_t + step
            continue

        prev_t, prev_d = t, d
        t = t + step

    return hits


def find_orb_window(
    transit_body: BodyKey,
    natal_lon: float,
    aspect_angle: float,
    start_utc: datetime,
    days_ahead: int,
    step_hours: int,
    orb_deg: float,
) -> tuple[datetime, datetime, datetime] | None:
    """
    Finds first interval [enter, exact, exit] where abs(delta) <= orb_deg.
    Returns None if not found in days_ahead.
    """
    start_utc = start_utc.astimezone(timezone.utc)
    end_utc = start_utc + timedelta(days=days_ahead)
    step = timedelta(hours=step_hours)

    def in_orb(t: datetime) -> bool:
        return abs(aspect_delta_deg(transit_body, natal_lon, aspect_angle, t)) <= orb_deg

    # scan for entry
    t = start_utc
    prev = t
    prev_in = in_orb(prev)

    while t <= end_utc:
        t = t + step
        now_in = in_orb(t)

        if not prev_in and now_in:
            enter = _bisect_boundary(lambda x: in_orb(x), prev, t, want_true=True)
            break

        prev, prev_in = t, now_in
    else:
        return None

    # find exact near the window: use find_next_aspect_times starting from enter
    exact_hits = find_next_aspect_times(
        body_transit=transit_body,
        natal_lon=natal_lon,
        aspect_angle=aspect_angle,
        start_utc=enter,
        days_ahead=min(days_ahead, 365 * 2),
        step_hours=max(1, step_hours // 4),
        max_hits=1,
    )
    if not exact_hits:
        exact = enter
    else:
        exact = exact_hits[0]

    # scan for exit starting from enter
    t = enter
    prev = t
    prev_in = True
    while t <= end_utc:
        t = t + step
        now_in = in_orb(t)

        if prev_in and not now_in:
            exit_t = _bisect_boundary(lambda x: in_orb(x), prev, t, want_true=False)
            return (enter, exact, exit_t)

        prev, prev_in = t, now_in

    return None


def _bisect_boundary(pred: Callable[[datetime], bool], lo: datetime, hi: datetime, want_true: bool) -> datetime:
    """
    Find boundary where pred switches. If want_true=True, finds earliest time where pred becomes True.
    If want_true=False, finds earliest time where pred becomes False.
    """
    for _ in range(45):
        mid = lo + (hi - lo) / 2
        if pred(mid) == want_true:
            hi = mid
        else:
            lo = mid
    return hi