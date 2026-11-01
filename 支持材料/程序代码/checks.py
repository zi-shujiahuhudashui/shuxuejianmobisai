# -*- coding: utf-8 -*-
# 数值检验：格式、收敛、噪声、灵敏度
import os
import sys
import time
import json
import numpy as np
from common import (Chamber, FVSolver, props_q1, props_q23, analytic_T_q1, make_grid,
                    uniform_grid, simulate, find_drying_time, save_json, ensure_dirs,
                    setup_plot, FIG_DIR, T_INIT, T_SET, C_SET, H_HEAT, H_MASS, C_INIT)

R_M = np.arange(0, 21) * 0.001


def solve_q1(ch, grid, cap, kappa, tmax=1800):
    s = FVSolver(props_q1(), ch, grid=grid, coupled=False)
    C = np.empty((tmax, 21))
    T = np.empty((tmax, 21))
    t0 = time.time()
    for k in range(1, tmax + 1):
        s.advance_to(float(k), cap, kappa)
        C[k - 1] = s.at_r(R_M, 'C')
        T[k - 1] = s.at_r(R_M, 'T')
    return C, T, time.time() - t0, s.N + 1


def check_q1():
    ch = Chamber()
    TA = analytic_T_q1(R_M, np.arange(1, 1801), ch, n_terms=400)
    refC, refT, el, nref = solve_q1(ch, make_grid(h_in=2.5e-4, h_s=6.25e-6), 0.1, 0.005)
    rows = [dict(name='参考解：内部 5 μm / 表面 0.125 μm，Δt≤0.1 s', nodes=nref,
                 errC=0.0, errT=float(np.abs(refT - TA).max()))]
    cases = [('内部 20 μm / 表面 1 μm，Δt≤1 s', make_grid(h_in=1e-3, h_s=5e-5), 1.0, 0.02),
             ('内部 10 μm / 表面 0.5 μm，Δt≤1 s', make_grid(), 1.0, 0.02),
             ('内部 10 μm / 表面 0.5 μm，Δt≤0.5 s（采用）', make_grid(), 0.5, 0.02),
             ('均匀网格 6.25 μm，Δt≤1 s', uniform_grid(3200), 1.0, 0.02)]
    for name, g, cap, kap in cases:
        C, T, el, n = solve_q1(ch, g, cap, kap)
        rows.append(dict(name=name, nodes=n, errC=float(np.abs(C - refC).max()),
                         errT=float(np.abs(T - TA).max()), time=el))
        print(name, n, rows[-1]['errC'], rows[-1]['errT'], flush=True)
    save_json(dict(rows=rows), 'verify_q1.json')


def run_conv(ch, grid, cap1, cap2, label):
    s = FVSolver(props_q23(), ch, grid=grid)
    t0 = time.time(); rec = {}
    k = 0
    while True:
        k += 1; tk = 60.0*k
        prev = s.state()
        s.advance_to(tk, cap1 if tk <= 14460 else cap2)
        if k % 360 == 0: rec[k//60] = s.at_r(R_M[::5], 'C')
        if tk == 3*3600: T3 = s.at_r(R_M[::5], 'T'); C3 = s.at_r(R_M[::5], 'C')
        if s.C[0] < 0.15:
            s.set_state(prev); td = find_drying_time(s, tk, dt_fine=0.05)[0]; break
    mb = (s.mean_C() - 2.55) + 2*s.flux_int
    print(f'{label:40s} nodes={len(grid)} td={td/3600:.6f} h  C@3h={np.round(C3,6)} T@3h={np.round(T3,5)} mass_err={mb:.1e} time={time.time()-t0:.0f}s', flush=True)
    return td, rec, len(grid), mb


def check_conv():
    ch = Chamber()
    res = {}
    res['A'] = run_conv(ch, make_grid(h_in=1e-3, h_s=5e-5), 1.0, 10.0, 'grid 20um/1um cap 1/10')
    res['B'] = run_conv(ch, make_grid(), 1.0, 10.0, 'grid 10um/0.5um cap 1/10')
    res['C'] = run_conv(ch, make_grid(), 0.5, 5.0, 'grid 10um/0.5um cap 0.5/5')
    res['D'] = run_conv(ch, make_grid(), 1.0, 30.0, 'grid 10um/0.5um cap 1/30')
    out = []
    for key, lab in zip(['A', 'B', 'C', 'D'], ['内部20μm/表面1μm, Δt≤1 s/10 s', '内部10μm/表面0.5μm, Δt≤1 s/10 s',
                                              '内部10μm/表面0.5μm, Δt≤0.5 s/5 s', '内部10μm/表面0.5μm, Δt≤1 s/30 s']):
        out.append(dict(case=key, label=lab, td_h=round(res[key][0] / 3600.0, 6), nodes=res[key][2],
                        mass_err=float('%.1e' % res[key][3])))
    save_json(dict(rows=out), 'conv_q3.json')


def check_noise():
    ch = Chamber(smooth=True)
    q2 = np.load(os.path.join('results', 'q2_fields.npz'))
    T = np.empty((10800, 21))
    C = np.empty((10800, 21))

    def on_1s(s):
        i = int(round(s.t)) - 1
        T[i] = s.at_r(R_M, 'T')
        C[i] = s.at_r(R_M, 'C')

    simulate(props_q23(), ch, t_1s_until=10800.0, until_dry=False, t_max=10800.0, on_1s=on_1s,
             verbose=False)
    idx = [int(h * 3600) - 1 for h in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    cols = [0, 5, 10, 15, 20]
    dT = np.abs(T[idx][:, cols] - q2['T'][idx][:, cols])
    dC = np.abs(C[idx][:, cols] - q2['C'][idx][:, cols])
    t_, Ta, Ca = ch.t_data, ch.Ta_data, ch.Ca_data
    out = dict(pT=[float(v) for v in ch.pT], pC=[float(v) for v in ch.pC],
               rmse_T=float(np.sqrt(np.mean((ch.Ta(t_) - Ta) ** 2))),
               rmse_C=float(np.sqrt(np.mean((ch.Ca(t_) - Ca) ** 2))),
               dT_table_max=float(dT.max()), dC_table_max=float(dC.max()),
               dT_all_max=float(np.abs(T - q2['T']).max()), dC_all_max=float(np.abs(C - q2['C']).max()))
    save_json(out, 'verify_noise.json')
    # 仅新增一张稳健性对比图：不改变任何求解流程
    plt = setup_plot()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    hours = np.arange(1, 10801) / 3600.0
    ax[0].plot(hours, q2['T'][:, 0], lw=1.2, label='实测数据插值')
    ax[0].plot(hours, T[:, 0], '--', lw=1.2, label='Weibull 光滑拟合')
    ax[0].set_xlabel('时间 / h'); ax[0].set_ylabel('中心温度 / °C')
    ax[0].set_title('(a) 温度响应'); ax[0].grid(alpha=0.3); ax[0].legend(frameon=False)
    ax[1].plot(hours, q2['C'][:, 0], lw=1.2, label='实测数据插值')
    ax[1].plot(hours, C[:, 0], '--', lw=1.2, label='Weibull 光滑拟合')
    ax[1].set_xlabel('时间 / h'); ax[1].set_ylabel('中心水分浓度 / (kg/kg)')
    ax[1].set_title('(b) 水分响应'); ax[1].grid(alpha=0.3); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'verify_weibull.png')); plt.close(fig)
    print(out)


class ScaledChamber(Chamber):
    def __init__(self, T_new=T_SET, C_new=C_SET):
        super().__init__()
        self.Tk = T_INIT + (self.Tk - T_INIT) * (T_new - T_INIT) / (T_SET - T_INIT)
        self.Ck = self.Ck * (C_new / C_SET)


SENS_GRID = make_grid(h_in=2e-3, h_s=1e-4)
SENS_CACHE = os.path.join('results', 'sens_cache.json')


def cached(key, fn):
    # JSON 缓存，中断后可续算
    data = {}
    if os.path.exists(SENS_CACHE):
        with open(SENS_CACHE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    if key not in data:
        data[key] = fn()
        with open(SENS_CACHE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    return data[key]


def dry_time(props=None, ch=None, solver_kw=None):
    props = props if props is not None else props_q23()
    ch = ch if ch is not None else Chamber()
    _, dry = simulate(props, ch, grid=SENS_GRID, t_1s_until=0.0, cap_fine=2.0, cap_coarse=30.0,
                      until_dry=True, verbose=False, solver_kw=solver_kw)
    return dry[0] / 3600.0


def run_sens():
    t0 = time.time()
    base = cached('base', dry_time)
    print('基准烘干时间（粗网格）%.4f h' % base, flush=True)
    specs = [
        ('对流换热系数 h', lambda f: dry_time(solver_kw=dict(h=H_HEAT * f))),
        ('对流传质系数 h_m', lambda f: dry_time(solver_kw=dict(hm=H_MASS * f))),
        ('水分扩散系数 D', lambda f: dry_time(props=props_q23(D_scale=f))),
        ('烘房温度 T_set', lambda f: dry_time(ch=ScaledChamber(T_new=T_SET * f))),
        ('烘房水分浓度 C_set', lambda f: dry_time(ch=ScaledChamber(C_new=C_SET * f))),
        ('初始水分浓度 C_0', lambda f: dry_time(solver_kw=dict(C0=C_INIT * f))),
    ]
    rows = []
    for name, fn in specs:
        lo = cached(name + '-10%', lambda: fn(0.9))
        hi = cached(name + '+10%', lambda: fn(1.1))
        S = ((hi - lo) / base) / 0.2
        rows.append(dict(name=name, t_minus=lo, t_plus=hi, rel_minus=(lo - base) / base,
                         rel_plus=(hi - base) / base, elasticity=S))
        print('%-16s -10%%: %.4f h (%+.2f%%)   +10%%: %.4f h (%+.2f%%)   S=%.3f' % (
            name, lo, 100 * (lo - base) / base, hi, 100 * (hi - base) / base, S), flush=True)
    temps = [40.0, 45.0, 50.0, 55.0, 60.0]
    tt = []
    for Tn in temps:
        if Tn == T_SET:
            v = base
        elif abs(Tn - 0.9 * T_SET) < 1e-9:
            v = cached('烘房温度 T_set-10%', lambda: dry_time(ch=ScaledChamber(T_new=Tn)))
        elif abs(Tn - 1.1 * T_SET) < 1e-9:
            v = cached('烘房温度 T_set+10%', lambda: dry_time(ch=ScaledChamber(T_new=Tn)))
        else:
            v = cached('T=%.0f' % Tn, lambda: dry_time(ch=ScaledChamber(T_new=Tn)))
        tt.append(v)
        print('烘房温度 %.0f °C：烘干时间 %.4f h' % (Tn, v), flush=True)
    out = dict(base_h=base, rows=rows, temps=temps, t_by_temp=tt, elapsed=time.time() - t0)
    save_json(out, 'sensitivity.json')

    plt = setup_plot()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    order = np.argsort([abs(r['rel_plus'] - r['rel_minus']) for r in rows])
    disp = {'对流换热系数 h': '对流换热系数 $h$', '对流传质系数 h_m': '对流传质系数 $h_m$',
            '水分扩散系数 D': '水分扩散系数 $D$', '烘房温度 T_set': '烘房温度 $T_{\\rm set}$',
            '烘房水分浓度 C_set': '烘房水分浓度 $C_{\\rm set}$', '初始水分浓度 C_0': '初始水分浓度 $C_0$'}
    names = [disp.get(rows[i]['name'], rows[i]['name']) for i in order]
    lo = np.array([rows[i]['rel_minus'] for i in order]) * 100
    hi = np.array([rows[i]['rel_plus'] for i in order]) * 100
    y = np.arange(len(names))
    ax[0].barh(y, lo, color='#2e75b6', label='参数 -10%')
    ax[0].barh(y, hi, color='#ed7d31', label='参数 +10%')
    ax[0].axvline(0, color='k', lw=0.8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names)
    ax[0].set_xlabel('烘干时间相对变化 / %')
    ax[0].set_title('(a) 烘干时间对各参数的灵敏度')
    ax[0].legend(frameon=False, fontsize=8, loc='lower right')
    ax[0].grid(alpha=0.3, axis='x')
    ax[1].plot(temps, tt, 'o-', color='#1f4e79', lw=1.4)
    for x, v in zip(temps, tt):
        ax[1].annotate('%.1f h' % v, (x, v), textcoords='offset points', xytext=(0, 6),
                       ha='center', fontsize=8)
    ax[1].set_ylim(min(tt) - 4, max(tt) + 6)
    ax[1].set_xticks(temps)
    ax[1].set_xlabel('恒温干燥阶段烘房温度 / °C')
    ax[1].set_ylabel('烘干时间 / h')
    ax[1].set_title('(b) 烘房温度对烘干时间的影响')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'sens_tornado.png'))
    plt.close(fig)
    print('灵敏度分析完成，用时 %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_dirs()
    tasks = {'q1': check_q1, 'conv': check_conv, 'noise': check_noise, 'sens': run_sens}
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            tasks[name]()
    else:
        for f in tasks.values():
            f()