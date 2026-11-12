"""Paper 3: from technical to integrable rooftop PV, and its economics.

Hourly PV per kWp: flat panels, ERA5-scaled clear sky for the 21st of every month (sun positions as
in the 3D model), performance ratio 0.80, 1.5 % shading (3D model districts: 0.8-2.3 %).
Demand (ASSUMPTION, no public hourly load data for Uganda): national distribution sales of 4,674 GWh
(Umeme 2024) spread over three daily shapes normalised to mean 1:
  flat        constant
  evening     trough 0.70 x mean at 04:00, plateau ~1.0 by day, peak 1.35 x mean at 20:00
  daytime     commercial-led: day plateau 1.20, evening 1.05, night 0.70
Study-area share of demand: 25 % and 50 % of national (and 100 % = national).
Grid absorption: PV energy used = sum over hours of min(PV, demand); the rest is curtailed.
Battery: 1 kWh per kWp (4 h at 0.25 C), 95 % one-way efficiency, charged from surplus, discharged
into deficits. Integrable capacity = largest capacity with curtailment <= 10 %.
Economics (ASSUMPTIONS for sensitivity): capex US$1,100/kWp (small, < 5 kWp), US$750/kWp
(commercial); O&M 1.5 % of capex a year; 25 years; 0.5 %/yr degradation; real discount rate 12 %;
batteries US$350/kWh replaced at year 10. Tariffs (ERA, Q4 2026, from the author's UEDCL study):
domestic 779.4, commercial 562.1, medium industrial 363.8, large industrial 308.1 UGX/kWh at
3,700 UGX/US$; diesel back-up US$0.35-0.47/kWh. Only self-consumed energy is valued.
"""
import numpy as np
import pandas as pd

from kampala import model3d as M
from kampala.config import OUTPUTS

TAB = OUTPUTS / "tables"
PR, SHADE = 0.80, 0.985
SUN = M.sun_positions(hours=range(0, 24), elev_min=0.0)

pv = np.zeros((12, 24)); days = np.zeros(12)
for t, r in SUN.iterrows():
    ghi = r.k * (r.dni * np.sin(np.radians(r.elevation)) + r.dhi)
    pv[t.month - 1, t.hour] = max(ghi, 0) / 1000 * PR * SHADE
    days[t.month - 1] = r.weight
annual_yield = float((pv.sum(axis=1) * days).sum())
print(f"annual yield {annual_yield:.0f} kWh/kWp (PVGIS-ERA5 with 14 % losses: 1,446)")

h = np.arange(24)
shapes = {
    "flat": np.ones(24),
    "evening": 0.70 + 0.30 * np.clip((h - 5) / 4, 0, 1) - 0.05 * np.exp(-((h - 14) / 3) ** 2) + 0.38 * np.exp(-((h - 20) / 1.6) ** 2),
    "daytime": 0.70 + 0.50 * np.exp(-((h - 13) / 3.6) ** 2) + 0.30 * np.exp(-((h - 20) / 2.0) ** 2),
}
shapes = {k: v / v.mean() for k, v in shapes.items()}
MEAN_GW = 4674 / 8760


def absorb(cap_gw, shape, share, battery=False):
    used = curt = 0.0
    for m in range(12):
        load = shape * MEAN_GW * share
        gen = pv[m] * cap_gw
        soc, cap_e = 0.0, cap_gw * 1.0
        for _ in range(2):                        # second pass starts from the first day's end state
            soc_day, u, c = soc, 0.0, 0.0
            for hr in range(24):
                direct = min(gen[hr], load[hr]); surplus = gen[hr] - direct; deficit = load[hr] - direct
                if battery:
                    charge = min(surplus, cap_gw * 0.25, (cap_e - soc_day) / 0.95)
                    soc_day += charge * 0.95; surplus -= charge
                    dis = min(deficit, cap_gw * 0.25, soc_day * 0.95)
                    soc_day -= dis / 0.95; direct += dis
                u += direct; c += surplus
            soc = soc_day
        used += u * days[m]; curt += c * days[m]
    return used, curt


rows = []
for shape in shapes:
    for share in (0.25, 0.50, 1.00):
        for bat in (False, True):
            for cap in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 10.0, 22.0):
                used, curt = absorb(cap, shapes[shape], share, bat)
                rows.append(dict(shape=shape, demand_share=share, battery=bat, pv_gwp=cap, gen_twh=annual_yield * cap / 1000,
                                 used_twh=used / 1000, curtailed_pct=100 * curt / max(used + curt, 1e-9),
                                 share_of_demand_pct=100 * used / (4674 * share)))
grid = pd.DataFrame(rows); grid.to_csv(TAB / "m5_grid_absorption.csv", index=False)


def integrable(d, limit=10):
    ok = d[d.curtailed_pct <= limit]
    return (ok.pv_gwp.max(), ok.loc[ok.pv_gwp.idxmax(), "share_of_demand_pct"]) if len(ok) else (0.0, 0.0)


summ = pd.DataFrame([dict(shape=s, demand_share=sh, battery=b, integrable_gwp=integrable(d)[0], demand_met_pct=integrable(d)[1])
                     for (s, sh, b), d in grid.groupby(["shape", "demand_share", "battery"])])
summ.to_csv(TAB / "m5_integrable_capacity.csv", index=False)
print(summ.round(2).to_string(index=False))


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


R, N, OM, DEG = 0.12, 25, 0.015, 0.005
deg = np.mean([(1 - DEG) ** t for t in range(N)])
tariffs = {"domestic": 779.4 / 3700, "commercial": 562.1 / 3700, "medium industrial": 363.8 / 3700,
           "large industrial": 308.1 / 3700, "diesel back-up (low)": 0.35, "diesel back-up (high)": 0.47}
econ, pb = [], []
for seg, capex, selfc, bat in (("household, small roof", 1100, 0.35, 0), ("household with 1 kWh/kWp battery", 1450, 0.75, 350),
                               ("commercial, daytime load", 750, 0.85, 0)):
    for yld in (1446, round(annual_yield)):
        annual_cost = (capex - bat) * crf(R, N) + bat * crf(R, 10) + capex * OM
        econ.append(dict(segment=seg, capex_usd_per_kwp=capex, yield_kwh_kwp=yld, self_consumed=selfc,
                         lcoe_usd_kwh=annual_cost / (yld * deg)))
        for k, v in tariffs.items():
            pb.append(dict(segment=seg, yield_kwh_kwp=yld, tariff=k, payback_years=capex / (yld * selfc * v - capex * OM)))
econ = pd.DataFrame(econ); pb = pd.DataFrame(pb)
econ.to_csv(TAB / "m5_lcoe.csv", index=False); pb.to_csv(TAB / "m5_payback.csv", index=False)
print(econ.round(3).to_string(index=False))
print(pb[pb.yield_kwh_kwp == 1446].pivot(index="segment", columns="tariff", values="payback_years").round(1))
