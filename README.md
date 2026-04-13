# Mission Veridian

Project-ready structure for trajectory optimization + animation.

## Folder Structure

```text
final_aero/
├── README.md
├── data/
│   └── veridian_ephemeris.csv
├── docs/
│   └── Veridian.pdf
├── src/
│   ├── veridian_structured.py
│   └── veridian_animation.py
├── outputs/
│   ├── optimal_trajectory.csv
│   ├── spacecraft_ephemeris_5day.csv
│   └── animations/
└── legacy/
    └── main.py
```

## Requirements

Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

For MP4 output, install FFmpeg (otherwise GIF fallback is used).

## Run (from project root)

### 1) Optimize trajectory + export spacecraft ephemeris

```bash
python src/veridian_structured.py --workers 8
```

Quick debug mode:

```bash
python src/veridian_structured.py --quick --workers 8
```

### 2) Generate animations from optimal trajectory

```bash
python src/veridian_animation.py --fps 20 --dpi 140
```

## Outputs

### `outputs/optimal_trajectory.csv`

Best solution row:

- `departure_mjd`
- `tof_ventus`
- `altitude`
- `tof_glacia`
- `deltaV_total`

### `outputs/spacecraft_ephemeris_5day.csv`

Spacecraft heliocentric position every 5 days from mission start to rendezvous:

- `MJD`
- `spacecraft_x` (km)
- `spacecraft_y` (km)
- `spacecraft_z` (km, planar model so currently 0)

### `outputs/animations/01_heliocentric_trajectory.mp4`

Veridian-centered trajectory animation.

### `outputs/animations/02_ventus_flyby_stationary.mp4`

Ventus-centered flyby animation.

### `outputs/animations/03_glacia_arrival_stationary.mp4`

Glacia-centered arrival animation.

## Notes

- Scoring is rendezvous-focused (coordinator clarification), not circular capture at Glacia.
- Thermal and max-duration constraints are enforced in `src/veridian_structured.py`.
