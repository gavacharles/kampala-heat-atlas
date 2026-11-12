"""2D lattice-Boltzmann (D2Q9, BGK + Smagorinsky) flow around building footprints, on the GPU (MPS).

A viscous, unsteady solver: unlike the potential flow it produces wakes, recirculation behind
buildings and channelling along streets. Still 2D (a horizontal slice at pedestrian height), so
flow over roofs and vertical mixing are absent. Inflow: uniform wind from the south on the bottom
edge, zero-gradient outflow on the top edge, periodic sides. Buildings within `buffer` cells of the
inflow and outflow edges are removed (boundary stability), and density is clamped to a physical
range each step. Output: time-mean speed ratio (local / inflow) after spin-up.
"""
import numpy as np
import torch

C = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]


def run(solid, u_in=0.03, tau0=0.65, cs=0.2, steps=16000, avg_from=8000, buffer=6, device=None, log=None):
    """solid: (ny, nx) bool, row 0 = north. Wind from the south: inflow at the bottom row."""
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.float32
    solid = solid.copy()
    solid[:buffer] = False; solid[-buffer:] = False
    s = torch.as_tensor(np.ascontiguousarray(solid), device=dev)
    ny, nx = s.shape
    c = C.to(dev); w = W.to(dev, dt)
    cx, cy = c[:, 0].to(dt).view(9, 1, 1), c[:, 1].to(dt).view(9, 1, 1)   # cy > 0 = northward
    ux0, uy0 = 0.0, u_in

    def feq(rho, ux, uy):
        cu = 3 * (cx * ux + cy * uy)
        return w.view(9, 1, 1) * rho * (1 + cu + 0.5 * cu ** 2 - 1.5 * (ux ** 2 + uy ** 2))

    rho = torch.ones(ny, nx, device=dev, dtype=dt)
    ux = torch.zeros(ny, nx, device=dev, dtype=dt); uy = torch.full((ny, nx), uy0, device=dev, dtype=dt)
    uy[s] = 0
    f = feq(rho, ux, uy)
    acc = torch.zeros(ny, nx, device=dev, dtype=dt); n_acc = 0
    for it in range(steps):
        rho = f.sum(0).clamp(0.5, 2.0)
        ux = (f * cx).sum(0) / rho; uy = (f * cy).sum(0) / rho
        rho_in = rho[-2:-1]                                              # (1, nx)
        f[:, -1:, :] = feq(rho_in, torch.full_like(rho_in, ux0), torch.full_like(rho_in, uy0))
        f[:, 0, :] = f[:, 1, :]
        rho = f.sum(0).clamp(0.5, 2.0); ux = (f * cx).sum(0) / rho; uy = (f * cy).sum(0) / rho
        fe = feq(rho, ux, uy)
        fneq = f - fe                                                    # Smagorinsky effective relaxation
        pxx = (fneq * cx * cx).sum(0); pyy = (fneq * cy * cy).sum(0); pxy = (fneq * cx * cy).sum(0)
        q = torch.sqrt(2 * (pxx ** 2 + pyy ** 2 + 2 * pxy ** 2))
        tau = 0.5 * (tau0 + torch.sqrt(tau0 ** 2 + 18 * 1.4142 * cs ** 2 * q / rho))
        fpost = f - fneq / tau
        fpost = torch.where(s.unsqueeze(0), f[OPP], fpost)               # bounce-back on buildings
        for i in range(9):                                               # streaming (+cy = smaller row)
            f[i] = torch.roll(fpost[i], shifts=(-int(C[i, 1]), int(C[i, 0])), dims=(0, 1))
        if it >= avg_from:
            acc += torch.sqrt(ux ** 2 + uy ** 2); n_acc += 1
        if log and it % 4000 == 0:
            log(f"    step {it}, max |u| {float(torch.sqrt(ux ** 2 + uy ** 2).max()):.3f}")
    spd = (acc / max(n_acc, 1) / u_in).cpu().numpy()
    spd[solid] = 0
    return spd
