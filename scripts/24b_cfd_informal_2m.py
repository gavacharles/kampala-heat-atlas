"""Resolution check for the viscous breeze model in the packed low-rise district (Kawempe).

At 4 m, alleys of 1-3 m between buildings close completely, which can overstate blocking. This
re-runs the district on the 2 m building grid of the 3D model (1000 x 1000 cells; 50,000 steps,
mean over the last 20,000) and compares with the 4 m result. Output: DATA/processed/m8/lbm_informal_2m.npz,
outputs/tables/m8_resolution_check.csv
"""
import numpy as np
import pandas as pd

from kampala import lbm
from kampala.config import DATA, OUTPUTS

M6, M8 = DATA / "processed" / "m6", DATA / "processed" / "m8"
d = np.load(M6 / "informal.npz")
sol = d["bh"] > 0                                   # 2 m building mask from the 3D model
spd = lbm.run(sol, steps=50000, avg_from=30000, log=lambda s: print("2 m", s, flush=True))
np.savez_compressed(M8 / "lbm_informal_2m.npz", speed=spd, solid=sol)
rows = []
for res, f in (("4 m", M8 / "lbm_informal.npz"), ("2 m", M8 / "lbm_informal_2m.npz")):
    z = np.load(f); s, o = z["speed"], ~z["solid"]
    b = 6 if res == "4 m" else 12
    o[:b] = False; o[-b:] = False
    rows.append(dict(resolution=res, plan_area_fraction=float(z["solid"].mean()), open_median_speed=float(np.median(s[o])),
                     weak_breeze_pct=100 * float((s[o] < 0.5).mean())))
out = pd.DataFrame(rows); out.to_csv(OUTPUTS / "tables" / "m8_resolution_check.csv", index=False)
print(out.round(3).to_string(index=False))
