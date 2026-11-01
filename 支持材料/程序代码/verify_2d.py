# -*- coding: utf-8 -*-
# 一维无限长圆柱假设的二维端部效应检验

import os
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from common import (Chamber, props_q23, FVSolver, make_grid, R0, L0, H_HEAT, H_MASS, T_INIT,
                    C_INIT, C_TARGET, save_json, ensure_dirs, setup_plot, FIG_DIR)


def z_grid(H, h_s=2e-5, q=1.3, h_max=2.5e-3):
    # 端面加密，中截面附近步长 h_max
    sizes = []
    s = h_s
    while s < h_max and sum(sizes) + s < H:
        sizes.append(s)
        s *= q
    rest = H - sum(sizes)
    n_u = int(np.ceil(rest / h_max))
    sizes += [rest / n_u] * n_u
    edges = H - np.concatenate(([0.0], np.cumsum(sizes)))
    z = np.sort(edges)
    z[0] = 0.0
    z[-1] = H
    return z


class Solver2D:
    def __init__(self, props, ch, r_nodes, z_nodes, h=H_HEAT, hm=H_MASS):
        self.p, self.ch, self.h, self.hm = props, ch, h, hm
        r = r_nodes * R0
        z = z_nodes
        nr, nz = r.size, z.size
        self.nr, self.nz, self.r, self.z = nr, nz, r, z
        rb = np.concatenate(([0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]))
        zb = np.concatenate(([0.0], 0.5 * (z[:-1] + z[1:]), [z[-1]]))
        Vr = 0.5 * (rb[1:] ** 2 - rb[:-1] ** 2)
        Vz = np.diff(zb)
        self.V = np.outer(Vz, Vr).ravel()                   # 索引 p = j*nr + i
        idx = np.arange(nr * nz).reshape(nz, nr)
        # 径向界面
        Pr = idx[:, :-1].ravel()
        Qr = idx[:, 1:].ravel()
        gr = (np.outer(Vz, 0.5 * (r[:-1] + r[1:]) / np.diff(r))).ravel()
        # 轴向界面
        Pz = idx[:-1, :].ravel()
        Qz = idx[1:, :].ravel()
        gz = (np.outer(1.0 / np.diff(z), Vr)).ravel()
        self.P = np.concatenate((Pr, Pz))
        self.Q = np.concatenate((Qr, Qz))
        self.geo = np.concatenate((gr, gz))
        # 对流边界面积（单位弧度）
        Ab = np.zeros((nz, nr))
        Ab[:, -1] += r[-1] * Vz                            # 侧面 r=R
        Ab[-1, :] += Vr                                    # 端面 z=L/2
        self.Ab = Ab.ravel()
        n = nr * nz
        self.n = n
        self.T = np.full(n, T_INIT)
        self.C = np.full(n, C_INIT)
        self.T_old = None
        self.C_old = None
        self.dt_old = None
        self.t = 0.0
        self.rows = np.concatenate((self.P, self.Q, self.P, self.Q, np.arange(n)))
        self.cols = np.concatenate((self.P, self.Q, self.Q, self.P, np.arange(n)))

    def _mat(self, cf_p, cf_q, cf_off_pq, cf_off_qp, diag):
        vals = np.concatenate((cf_p, cf_q, cf_off_pq, cf_off_qp, diag))
        return sp.csc_matrix((vals, (self.rows, self.cols)), shape=(self.n, self.n))

    def step(self, dt):
        p, P, Q = self.p, self.P, self.Q
        t_new = self.t + dt
        Ta, Ca = float(self.ch.Ta(t_new)), float(self.ch.Ca(t_new))
        if self.T_old is None:
            a0, a1, a2 = 1.0, -1.0, 0.0
            Tg, Cg = self.T.copy(), self.C.copy()
        else:
            om = dt / self.dt_old
            a0, a1, a2 = (1 + 2 * om) / (1 + om), -(1 + om), om * om / (1 + om)
            Tg = self.T + om * (self.T - self.T_old)
            Cg = self.C + om * (self.C - self.C_old)
        hT = a1 * self.T + (a2 * self.T_old if a2 else 0.0)
        hC = a1 * self.C + (a2 * self.C_old if a2 else 0.0)
        Ti, Ci = Tg, np.maximum(Cg, 1e-6)
        V, Ab = self.V, self.Ab
        for _ in range(30):
            kn = p.k(Ci)
            cf = self.geo * 0.5 * (kn[P] + kn[Q])
            rc = p.rho_cp(Ci)
            diag = rc * V * a0 / dt + self.h * Ab
            A = self._mat(cf, cf, -cf, -cf, diag)
            rhs = -rc * V * hT / dt + self.h * Ab * Ta
            T_new = splu(A).solve(rhs)
            wf = p.w(0.5 * (T_new[P] + T_new[Q]))
            cw = self.geo * wf
            Cn = Ci.copy()
            for _ in range(30):
                G, g = p.G(Cn), p.g(Cn)
                flux = cw * (G[Q] - G[P])
                F = V * (a0 * Cn + hC) / dt + self.hm * Ab * (Cn - Ca)
                np.add.at(F, P, -flux)
                np.add.at(F, Q, flux)
                J = self._mat(cw * g[P], cw * g[Q], -cw * g[Q], -cw * g[P],
                              V * a0 / dt + self.hm * Ab)
                dC = splu(J).solve(-F)
                Cn = Cn + dC
                if np.max(np.abs(dC)) < 1e-11:
                    break
            eT, eC = np.max(np.abs(T_new - Ti)), np.max(np.abs(Cn - Ci))
            Ti, Ci = T_new, Cn
            if eT < 1e-9 and eC < 1e-11:
                break
        self.T_old, self.C_old = self.T, self.C
        self.T, self.C = Ti, Ci
        self.t, self.dt_old = t_new, dt

    def advance_to(self, t_target, dt_cap, kappa=0.02, dt_min=1e-3):
        knots = self.ch.knots()
        while self.t < t_target - 1e-9:
            k_in = knots[(knots > self.t + 1e-9) & (knots < t_target - 1e-9)]
            t_next = float(k_in[0]) if k_in.size else float(t_target)
            while self.t < t_next - 1e-9:
                dt = min(dt_cap, max(dt_min, kappa * self.t))
                n_rem = int(np.ceil((t_next - self.t) / dt - 1e-9))
                self.step((t_next - self.t) / max(n_rem, 1))
            self.t = t_next


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_dirs()
    t0 = time.time()
    ch = Chamber()
    xr = make_grid(h_in=0.025, h_s=1e-3, seg=0.1, n_marks=10)
    zz = z_grid(L0 / 2.0)
    s2 = Solver2D(props_q23(), ch, xr, zz)
    s1 = FVSolver(props_q23(), ch, grid=xr)
    print('二维网格：径向 %d × 轴向 %d = %d 个节点' % (xr.size, zz.size, s2.n), flush=True)
    rec = []
    td = {}
    c_prev = {'1D': C_INIT, '2D': C_INIT}
    t_prev = 0.0
    snap = {}
    k = 0
    while True:
        k += 1
        tk = 60.0 * k
        cap = 10.0 if tk <= 14460 else 60.0
        s2.advance_to(tk, cap)
        s1.advance_to(tk, cap)
        c1, c2 = float(s1.C[0]), float(s2.C[0])        # s2.C[0] 即 (r=0, z=0) 中截面中心
        cmax2 = float(np.max(s2.C))
        rec.append((tk, c1, c2, cmax2))
        for key, cv in (('1D', c1), ('2D', c2)):
            if key not in td and cv < C_TARGET:
                td[key] = t_prev + (c_prev[key] - C_TARGET) / (c_prev[key] - cv) * (tk - t_prev)
        c_prev = {'1D': c1, '2D': c2}
        t_prev = tk
        if abs(tk - 12 * 3600) < 1e-6:
            snap['12h'] = s2.C.reshape(s2.nz, s2.nr).copy()
        if k % 360 == 0:
            print('  t=%4.1f h  1D:%.6f  2D:%.6f  diff=%.2e  2D最大值=%.6f  用时 %.0f s' % (
                tk / 3600, c1, c2, c2 - c1, cmax2, time.time() - t0), flush=True)
        if '1D' in td and '2D' in td:
            snap['end'] = s2.C.reshape(s2.nz, s2.nr).copy()
            break
    rec = np.array(rec)
    diff = rec[:, 2] - rec[:, 1]
    out = dict(nr=int(xr.size), nz=int(zz.size), td_1D_h=td['1D'] / 3600.0, td_2D_h=td['2D'] / 3600.0,
               td_diff_s=td['2D'] - td['1D'], max_abs_diff_center=float(np.abs(diff).max()),
               center_is_max=bool(np.all(rec[:, 3] <= rec[:, 2] + 1e-12)),
               elapsed=time.time() - t0)
    save_json(out, 'verify_2d.json')
    print(out)
    np.savez_compressed(os.path.join('results', 'verify_2d.npz'), rec=rec, r=s2.r, z=s2.z,
                        C12=snap['12h'], Cend=snap['end'])
    make_figure()
    return out


def make_figure():
    # 由 results/verify_2d.npz 绘图
    from matplotlib.ticker import FuncFormatter
    d = np.load(os.path.join('results', 'verify_2d.npz'))
    rec, r, z = d['rec'], d['r'], d['z']
    diff = np.abs(rec[:, 2] - rec[:, 1])
    plt = setup_plot()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    Z, Rr = np.meshgrid(z * 100, r * 100, indexing='ij')
    im = ax[0].contourf(Z, Rr, d['C12'], levels=30, cmap='viridis_r')
    fig.colorbar(im, ax=ax[0], label='C / (kg/kg)')
    ax[0].set_xlabel('距中截面的轴向距离 z / cm')
    ax[0].set_ylabel('径向距离 r / cm')
    ax[0].set_title('(a) t = 12 h 二维水分浓度场（1/4 剖面）')
    th = rec[:, 0] / 3600.0
    m = th >= 0.5
    ax[1].semilogy(th[m], diff[m] + 1e-16, color='#1f4e79', lw=1.2)
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, p: '1e%d' % int(round(np.log10(v)))))
    ax[1].set_xlabel('时间 t / h')
    ax[1].set_ylabel(r'$|C_{\mathrm{2D}}-C_{\mathrm{1D}}|$（中截面中心）')
    ax[1].set_title('(b) 端部效应对中心水分浓度的影响')
    ax[1].grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'v_2d_field.png'))
    plt.close(fig)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--plot':
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        make_figure()
    else:
        main()
