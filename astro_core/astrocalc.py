from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Union, Tuple, List

import swisseph as swe
from astronomy import Time, Body, GeoVector, Ecliptic


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