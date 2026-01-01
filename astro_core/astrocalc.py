from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Union, Tuple, List

import swisseph as swe
from astronomy import Time, Body, GeoVector, Ecliptic
from typing import Dict


# --- Swiss Ephemeris init ---
SWEPH_PATH = os.getenv("SWEPH_PATH", "/app/ephe")
swe.set_ephe_path(SWEPH_PATH)


# --- Zodiac helpers ---
SIGNS_DE = [
    "Widder", "Stier", "Zwillinge", "Krebs", "Löwe", "Jungfrau",
    "Waage", "Skorpion", "Schütze", "Steinbock", "Wassermann", "Fische"
]
SIGNS_SYM = ["♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓"]

EXTRA_BODIES = {
    "chiron": "chiron",
    "lilith": "lilith_mean",
    "lilith_mean": "lilith_mean",
    "lilith_true": "lilith_true",
}

EXTRA_SYMBOL = {
    "chiron": "⚷",
    "lilith_mean": "⚸",
    "lilith_true": "⚸",
}

# --- House system symbols / labels ---
HOUSE_SYS_LABEL = {
    "P": "Placidus",
    "W": "Whole Sign",
    "K": "Koch",
    "R": "Regiomontanus",
    "C": "Campanus",
    "E": "Equal",
}

ANGLE_SYMBOL = {
    "ASC": "ASC",
    "MC": "MC",
    "DC": "DC",
    "IC": "IC",
    "VX": "VX",     # Vertex
}

# Map Astronomy Engine bodies to symbols (adjust if you already have a mapping you like)
PLANET_SYMBOL = {
    Body.Sun: "☉",
    Body.Moon: "☽",
    Body.Mercury: "☿",
    Body.Venus: "♀",
    Body.Mars: "♂",
    Body.Jupiter: "♃",
    Body.Saturn: "♄",
    Body.Uranus: "♅",
    Body.Neptune: "♆",
    Body.Pluto: "♇",
}


BodyKey = Union[Body, str]

def wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0

def _swe_lon(body_ipl: int, dt_utc: datetime) -> float:
    jd = _jd_ut(dt_utc)
    xx, _ret = swe.calc_ut(jd, body_ipl, swe.FLG_SWIEPH)
    return float(xx[0]) % 360.0

def moon_phase_func(dt_utc: datetime, target_deg: float) -> float:
    """
    f(t) = wrap180( (lon_moon - lon_sun) - target_deg )
    Root bei f(t)=0.
    """
    moon = _swe_lon(swe.MOON, dt_utc)
    sun  = _swe_lon(swe.SUN, dt_utc)
    return wrap180((moon - sun) - target_deg)

def bisect_time_for_moon_phase(
    t0: datetime,
    t1: datetime,
    target_deg: float,
    tol_seconds: int = 60,
) -> datetime:
    f0 = moon_phase_func(t0, target_deg)
    f1 = moon_phase_func(t1, target_deg)

    if abs(f0) < 1e-12:
        return t0
    if abs(f1) < 1e-12:
        return t1

    # muss ein Vorzeichenwechsel sein, sonst keine sichere Bisection
    if f0 * f1 > 0:
        raise ValueError("No sign change for moon phase bisection.")

    lo, hi = t0, t1
    while (hi - lo).total_seconds() > tol_seconds:
        mid = lo + (hi - lo) / 2
        fm = moon_phase_func(mid, target_deg)
        if abs(fm) < 1e-10:
            return mid
        if f0 * fm <= 0:
            hi = mid
            f1 = fm
        else:
            lo = mid
            f0 = fm

    return lo + (hi - lo) / 2

def find_new_full_moons(
    start_utc: datetime,
    end_utc: datetime,
    step_hours: int = 6,
    tol_seconds: int = 60,
) -> list[tuple[str, datetime]]:
    """
    Returns list of ("Neumond"/"Vollmond", exact_dt_utc) within [start_utc, end_utc].
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start_utc/end_utc must be timezone-aware (UTC).")

    events: list[tuple[str, datetime]] = []
    step = timedelta(hours=step_hours)

    targets = [("Neumond", 0.0), ("Vollmond", 180.0)]

    t0 = start_utc
    while t0 < end_utc:
        t1 = min(t0 + step, end_utc)

        for label, target in targets:
            f0 = moon_phase_func(t0, target)
            f1 = moon_phase_func(t1, target)

            # sign change or endpoint near zero
            if (f0 == 0.0) or (f1 == 0.0) or (f0 * f1 < 0):
                try:
                    exact = bisect_time_for_moon_phase(t0, t1, target, tol_seconds=tol_seconds)

                    # Duplikate vermeiden (kann bei step klein passieren)
                    if not events or abs((events[-1][1] - exact).total_seconds()) > 3600:
                        events.append((label, exact))
                except ValueError:
                    pass

        t0 = t1

    events.sort(key=lambda x: x[1])
    return events


def parse_body_key(s: str) -> BodyKey | None:
    s = (s or "").lower().strip()

    # Common planet names used in your bot
    planet_map = {
        "sun": Body.Sun,
        "moon": Body.Moon,
        "mercury": Body.Mercury,
        "venus": Body.Venus,
        "mars": Body.Mars,
        "jupiter": Body.Jupiter,
        "saturn": Body.Saturn,
        "uranus": Body.Uranus,
        "neptune": Body.Neptune,
        "pluto": Body.Pluto,
    }
    if s in planet_map:
        return planet_map[s]

    if s in EXTRA_BODIES:
        return EXTRA_BODIES[s]

    return None
    
def aspect_delta_deg(transit_body: Body, natal_lon: float, aspect_angle: float, at_utc: datetime) -> float:
    """
    Signed delta in degrees between transit longitude and (natal_lon + aspect_angle).
    Returns value in (-180, 180].
    """
    transit_lon = ecliptic_lon_geocentric(transit_body, at_utc)
    target = (natal_lon + aspect_angle) % 360.0
    return angle_diff_signed(transit_lon, target)


def ps_any(body_or_key: BodyKey) -> str:
    if isinstance(body_or_key, Body):
        return PLANET_SYMBOL.get(body_or_key, "")
    return EXTRA_SYMBOL.get(str(body_or_key), "")


def name_any(body_or_key: BodyKey) -> str:
    if isinstance(body_or_key, Body):
        return body_or_key.name
    key = str(body_or_key)
    if key == "chiron":
        return "Chiron"
    if key in ("lilith", "lilith_mean"):
        return "Lilith"
    if key == "lilith_true":
        return "Lilith (true)"
    return key


def lon_to_sign(lon: float) -> Tuple[int, float]:
    lon = lon % 360.0
    idx = int(lon // 30)
    deg = lon - idx * 30
    return idx, deg


def fmt_lon_sign(lon: float) -> str:
    idx, deg = lon_to_sign(lon)
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        d += 1
        m = 0
    return f"{SIGNS_SYM[idx]} {SIGNS_DE[idx]} {d:02d}°{m:02d}′ ({lon%360:06.2f}°)"


def _jd_ut(dt_utc: datetime) -> float:
    dt_utc = dt_utc.astimezone(timezone.utc)
    return swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3_600_000_000.0,
    )


def ecliptic_lon_geocentric(body_or_key: BodyKey, dt_utc: datetime) -> float:
    """
    Geocentric ecliptic longitude in degrees [0, 360).

    Supports:
      - Astronomy Engine Body (Sun..Pluto, Moon)
      - strings: "chiron", "lilith_mean", "lilith_true"
    """
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware (UTC).")
    dt_utc = dt_utc.astimezone(timezone.utc)

    # Astronomy Engine bodies
    if isinstance(body_or_key, Body):
        t = Time.Make(
            dt_utc.year,
            dt_utc.month,
            dt_utc.day,
            dt_utc.hour,
            dt_utc.minute,
            float(dt_utc.second) + dt_utc.microsecond / 1_000_000.0,
        )
        gv = GeoVector(body_or_key, t, aberration=False)
        ecl = Ecliptic(gv)
        return float(ecl.elon) % 360.0

    # Swiss Ephemeris bodies
    key = str(body_or_key).lower().strip()
    jd = _jd_ut(dt_utc)

    if key == "chiron":
        ipl = swe.CHIRON
    elif key in ("lilith", "lilith_mean"):
        ipl = swe.MEAN_APOG
    elif key == "lilith_true":
        ipl = swe.OSCU_APOG
    else:
        raise ValueError(f"Unknown extra body key: {key}")

    xx, _ret = swe.calc_ut(jd, ipl, swe.FLG_SWIEPH)
    return float(xx[0]) % 360.0
    
    
def find_next_aspect_times(
    body_transit: Body,
    natal_lon: float,
    aspect_angle: float,
    start_utc: datetime,
    days_ahead: int = 365 * 2,
    step_hours: int = 12,
    max_hits: int = 3
) -> list[datetime]:
    """
    Finds next times where transit body forms aspect_angle to natal_lon.
    For conjunction: aspect_angle=0.

    Method:
    - scan forward in fixed steps
    - find sign changes of delta(t) around 0
    - refine each bracket by bisection
    """
    def delta(dt: datetime) -> float:
        lon_t = ecliptic_lon_geocentric(body_transit, dt)
        # We want (lon_t - natal_lon) == aspect_angle (mod 360)
        # So compare lon_t to natal_lon+aspect_angle
        target = (natal_lon + aspect_angle) % 360.0
        return angle_diff_signed(lon_t, target)

    hits: list[datetime] = []
    dt1 = start_utc
    d1 = delta(dt1)

    steps = int((days_ahead * 24) / step_hours)
    for _ in range(steps):
        dt2 = dt1 + timedelta(hours=step_hours)
        d2 = delta(dt2)

        # bracket condition: sign flip OR exactly 0 at an endpoint
        wrap_jump = abs(d2 - d1) > 180.0
        crosses_zero = (d1 == 0.0 or d2 == 0.0 or (d1 < 0 < d2) or (d2 < 0 < d1))

        if crosses_zero or wrap_jump:

            # bisection refine
            lo, hi = dt1, dt2
            dlo, dhi = d1, d2

            # ensure we have opposite signs; if one is 0, accept it directly
            if abs(dlo) < 1e-9:
                t_hit = lo
            elif abs(dhi) < 1e-9:
                t_hit = hi
            else:
                for _ in range(50):  # enough for < 1 minute precision
                    mid = lo + (hi - lo) / 2
                    dm = delta(mid)
                    if abs(dm) < 1e-6:
                        lo = hi = mid
                        break
                    # keep the sub-interval that contains sign change
                    if (dlo < 0 < dm) or (dm < 0 < dlo):
                        hi, dhi = mid, dm
                    else:
                        lo, dlo = mid, dm
                t_hit = lo + (hi - lo) / 2

            # de-dup (retro loops can give close hits if step is coarse)
            if not hits or abs((t_hit - hits[-1]).total_seconds()) > step_hours * 3600:
                hits.append(t_hit)

            if len(hits) >= max_hits:
                break

        dt1, d1 = dt2, d2

    return hits

def find_orb_window(
    transit_body: Body,
    natal_lon: float,
    aspect_angle: float,
    start_utc: datetime,
    days_ahead: int,
    step_hours: int,
    orb_deg: float,
) -> tuple[datetime, datetime, datetime] | None:
    """
    Returns (enter_time, exact_time, exit_time) within search horizon.
    """
    end_utc = start_utc + timedelta(days=days_ahead)

    # 1) Find first entry into orb
    t_prev = start_utc
    prev_in = abs(aspect_delta_deg(transit_body, natal_lon, aspect_angle, t_prev)) <= orb_deg

    t = t_prev
    enter = None

    while t < end_utc:
        t = t + timedelta(hours=step_hours)
        now_in = abs(aspect_delta_deg(transit_body, natal_lon, aspect_angle, t)) <= orb_deg

        if (not prev_in) and now_in:
            # bracket [t_prev, t]
            enter = _bisect_boundary(transit_body, natal_lon, aspect_angle, orb_deg, t_prev, t, want_enter=True)
            break

        t_prev, prev_in = t, now_in

    if enter is None:
        return None

    # 2) Find exact inside the orb window (use your existing exact finder with a finer step)
    exact_hits = find_next_aspect_times(
        body_transit=transit_body,
        natal_lon=natal_lon,
        aspect_angle=aspect_angle,
        start_utc=start_utc,
        days_ahead=days_ahead,
        step_hours=step_hours,
        max_hits=3,
    )
    if not exact_hits:
        # fallback: pick minimum abs(delta) by scanning, if no root found (rare)
        best_t = enter
        best_v = abs(aspect_delta_deg(transit_body, natal_lon, aspect_angle, best_t))
        scan_step = timedelta(hours=max(1, step_hours // 3))
        tscan = enter
        for _ in range(int((days_ahead * 24) / max(1, step_hours // 3))):
            v = abs(aspect_delta_deg(transit_body, natal_lon, aspect_angle, tscan))
            if v < best_v:
                best_v, best_t = v, tscan
            if v > orb_deg and tscan > enter + timedelta(days=10):
                break
            tscan += scan_step
        exact = best_t
    else:
        exact = exact_hits[0]

    # 3) Find exit from orb (start scanning from exact)
    t_prev = exact
    prev_in = abs(aspect_delta_deg(transit_body, natal_lon, aspect_angle, t_prev)) <= orb_deg

    t = t_prev
    exit_t = None

    while t < end_utc:
        t = t + timedelta(hours=step_hours)
        now_in = abs(aspect_delta_deg(transit_body, natal_lon, aspect_angle, t)) <= orb_deg

        if prev_in and (not now_in):
            exit_t = _bisect_boundary(transit_body, natal_lon, aspect_angle, orb_deg, t_prev, t, want_enter=False)
            break

        t_prev, prev_in = t, now_in

    if exit_t is None:
        exit_t = end_utc

    return enter, exact, exit_t

def default_search_params(transit_body) -> tuple[int, int]:
    """
    Returns (days_ahead, step_hours) tuned by transit body speed.
    days_ahead in days, step_hours in hours.
    Supports Astronomy Engine Body and also string keys like 'chiron', 'lilith_mean'.
    """
    # Extras (Swiss) sind i.d.R. langsam -> große Horizonte, grober Schritt
    if isinstance(transit_body, str):
        return (365 * 90, 48)

    # Astronomy Engine bodies
    if transit_body == Body.Moon:
        return (365 * 5, 1)          # Mond: schnell, aber braucht keine 90 Jahre
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

    # Default
    return (365 * 90, 24)

def angle_diff_signed(a: float, b: float) -> float:
    """
    Signed smallest difference a-b in degrees, range (-180, 180].
    """
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def compute_houses(dt_utc: datetime, lat: float, lon: float, hsys: str = "P") -> tuple[list[float], dict[str, float]]:
    """
    Returns:
      houses: list of 12 house cusps in degrees (1..12) as list[0..11]
      angles: dict with ASC/MC/DC/IC (+ optional others)
    Notes:
      - dt_utc must be timezone-aware (UTC)
      - lon: East positive (Swiss Ephemeris expects geographic longitude, east positive)
    """
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware (UTC).")
    dt_utc = dt_utc.astimezone(timezone.utc)

    jd_ut = swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3_600_000_000.0,
    )

    # houses_ex: returns (cusps[1..12], ascmc[0..]) in many wrappers.
    hs = (hsys or "P")
    hs = hs[0].upper() if isinstance(hs, str) else hs
    if isinstance(hs, str):
        hs = hs.encode("ascii")
    cusps, ascmc = swe.houses_ex(jd_ut, lat, lon, hs)

    # pyswisseph can return cusps indexed 1..12 (length 13) or 0..11 (length 12)
    cusps_list = list(cusps)

    if len(cusps_list) >= 13:
        # 1..12 valid
        houses = [float(cusps_list[i]) % 360.0 for i in range(1, 13)]
    elif len(cusps_list) == 12:
        # 0..11 valid
        houses = [float(c) % 360.0 for c in cusps_list]
    else:
        raise RuntimeError(f"Unexpected cusps length from swe.houses_ex: {len(cusps_list)}")


    asc = float(ascmc[0]) % 360.0
    mc  = float(ascmc[1]) % 360.0
    # Derive DC/IC (opposition)
    dc  = (asc + 180.0) % 360.0
    ic  = (mc  + 180.0) % 360.0

    angles = {
        "ASC": asc,
        "MC": mc,
        "DC": dc,
        "IC": ic,
    }

    # Optional: Vertex is often at index 3 in SwissEph ascmc array (depends on wrapper),
    # but we keep it safe:
    try:
        vx = float(ascmc[3]) % 360.0
        angles["VX"] = vx
    except Exception:
        pass

    return houses, angles


def house_of_lon(lon: float, houses: list[float]) -> int:
    """
    Determine house number (1..12) for a given ecliptic longitude, based on cusps.
    Works with cusps crossing 0°.
    """
    lon = lon % 360.0
    for i in range(12):
        start = houses[i]
        end = houses[(i + 1) % 12]
        if start <= end:
            if start <= lon < end:
                return i + 1
        else:
            # wrap around 360->0
            if lon >= start or lon < end:
                return i + 1
    return 12


def fmt_cusp(i: int, lon: float) -> str:
    # i: 1..12
    return f"H{i:02d}: {fmt_lon_sign(lon)}"


def fmt_angles(angles: dict[str, float]) -> str:
    parts = []
    for k in ("ASC", "MC", "DC", "IC", "VX"):
        if k in angles:
            parts.append(f"{ANGLE_SYMBOL.get(k,k)}: {fmt_lon_sign(angles[k])}")
    return "\n".join(parts)