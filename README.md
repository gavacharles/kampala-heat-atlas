# Kampala Heat Atlas

Open-data research on how Kampala grew between 2017 and 2025, where it is hottest, who lives in the
heat, and what trees and rooftops could do about it, with a 3D model of sun, shade and the Lake
Victoria breeze.

**Website:** https://gavacharles.github.io/kampala-heat-atlas/
- [Atlas](https://gavacharles.github.io/kampala-heat-atlas/): 11 map layers (heat, growth, settlements, trees, solar)
- [3D model](https://gavacharles.github.io/kampala-heat-atlas/3d.html): three districts with real-time sun shadows and breeze arrows
- [Figures & animations](https://gavacharles.github.io/kampala-heat-atlas/gallery.html)

Status: first full run of PhD research (Gava Charles). Several layers carry open validation items,
listed on the site next to each result. Please don't cite numbers without checking the caveats.

## What is in it

| Module | Question | Method |
|---|---|---|
| M0 Cube | What terrain, building heights, sun and wind define Kampala? | FABDEM, Open Buildings 2.5D Temporal, pvlib, ERA5-Land |
| M1 Growth | How did the city grow, 2017–2025? | Three-date Sentinel-2 random forest, Dynamic World weak labels, building-evidence constraint |
| M2 Heat | Which thermal zones exist? | Gap-free Landsat LST (TOA + NDVI emissivity, calibrated to C2 ST), LCZ |
| M3 Settlements | Are informal settlements heating faster? | Building-morphology typology, thermal trajectories |
| M4 Trees | Where should trees go? | Heat × people × canopy deficit where space exists, cooling model |
| M5 Solar | How far can rooftop solar scale? | 2.0 M Overture roofs, ERA5-Land irradiance, adoption scenarios |
| M6 3D model | Why: shade, sky view, ventilation, orientation | 2 m ray-traced shadows, sky view factor, potential flow, frontal area index |

## Reproduce

```bash
python3.12 -m venv ~/.venvs/kampala && source ~/.venvs/kampala/bin/activate
pip install -e . && pip install geedim duckdb pyvista imageio statsmodels netcdf4
earthengine authenticate          # then set earth_engine.project in config/config.yaml
python scripts/00_check_setup.py
```

| Step | Script |
|---|---|
| Sun geometry, terrain + buildings, wind | `01`, `02`, `03` |
| Footprints (Overture) | `04` |
| LST, MODIS, LCZ | `05` |
| Carry over earlier work (author's own repos) | `06` |
| Land cover, change, building constraint | `07`, `07b`, `07c` |
| Thermal classes · settlements · trees · solar | `08`, `09`/`09b`, `10`, `11` |
| Figures · animations · atlas | `12`, `13`, `14` |
| 3D model · 3D visuals · website | `15`, `16`, `17` |

Large data go to `~/kampala-thesis-data` (set in `config/config.yaml`) and are not in this repository.

## Licences

Code: MIT (`LICENSE`). Maps and derived layers inherit their input licences, and anything using
FABDEM terrain is non-commercial (`DATA_LICENSE.md`).
© OpenStreetMap contributors · Overture Maps Foundation · Google Open Buildings · Copernicus · USGS · ECMWF.
