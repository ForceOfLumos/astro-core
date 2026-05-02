# astro-core

Astrological calculation core used by Force of Lumos projects.

This package provides:
- Tropical, geocentric ecliptic longitudes for Sun/Moon/planets (Astronomy Engine)
- Chiron and Lilith (Swiss Ephemeris via `pyswisseph`)
- Aspect math helpers and search utilities (exact hits, orb windows)

It is designed to be imported by a separate "app" layer (e.g., a Telegram bot, web app, etc.).
Keep your content/business logic in the app layer; keep ephemeris calculations here.

## Features

- `ecliptic_lon_geocentric(body_or_key, dt_utc)`  
  Returns ecliptic longitude in degrees `[0, 360)`.

  Supported bodies:
  - Astronomy Engine: `Body.Sun .. Body.Pluto`, `Body.Moon`
  - Extras (strings): `"chiron"`, `"lilith_mean"`, `"lilith_true"`

- `find_orb_window(...)`  
  Finds first window where an aspect is within orb: `(enter, exact, exit)`.

- `find_next_aspect_times(...)`  
  Finds timestamps where an aspect is exact (crossing zero delta).

- Helpers:
  - `parse_body_key("mars")` → Body.Mars
  - `fmt_lon_sign(197.8)` → `♎ Waage 17°48′ (197.80°)`

## Requirements

- Python 3.11+
- `astronomy-engine`
- `pyswisseph`

### Swiss Ephemeris data files

Swiss Ephemeris requires ephemeris data files (e.g. `sepl_18.se1`, `semo_18.se1`, `seas_18.se1`).
The application layer should mount/provide these files and set:

- `SWEPH_PATH=/app/ephe` (or any directory containing the files)

Example (Docker):
- host: `./ephe`
- container: `/app/ephe`
- env: `SWEPH_PATH=/app/ephe`

## Installation

```
pip install "git+https://github.com/<ORG>/astro-core.git@main#egg=astro-core"
```

## Quick example

```python
from datetime import datetime, timezone
from astronomy import Body
from astro_core import ecliptic_lon_geocentric, fmt_lon_sign

dt = datetime(1991, 10, 31, 4, 0, tzinfo=timezone.utc)
lon = ecliptic_lon_geocentric(Body.Sun, dt)
print(fmt_lon_sign(lon))
```

## AGPL Section 13 — Source code for network users

This software is used in a networked service (Telegram bot). In accordance with
GNU Affero General Public License v3.0 Section 13, users interacting with the
service over a network can obtain the corresponding source code via the
`/source` command in the bot, or directly at:

> https://github.com/<ORG>/astro-core

## License

This project is licensed under the **GNU Affero General Public License v3.0 or later**
(AGPL-3.0-or-later). See [LICENSE](LICENSE) for the full license text.

### Third-party dependencies

| Package | License | Notes |
|---|---|---|
| [pyswisseph](https://github.com/astrorigin/pyswisseph) | AGPL-3.0 | Python binding for Swiss Ephemeris by Astrodienst AG |
| [astronomy-engine](https://github.com/cosinekitty/astronomy) | MIT | Compatible with AGPL-3.0 |

Swiss Ephemeris is © Astrodienst AG. `pyswisseph` is itself licensed under AGPL-3.0,
which is the primary reason this project adopts the same license.
