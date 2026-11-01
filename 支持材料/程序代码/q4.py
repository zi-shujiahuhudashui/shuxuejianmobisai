# -*- coding: utf-8 -*-
# 问题四：考虑收缩的烘干时长
import os
import time
import numpy as np
from common import (Chamber, props_q4, simulate, Radius, write_result_workbook, save_json,
                    ensure_dirs, setup_plot, FIG_DIR, r4, profile_at_r, load_radius_data, R0,
                    C_TARGET)

R_CM = np.round(np.arange(0, 20) * 0.1, 1)          # 0, 0.1, ..., 1.9 cm（t≥60 s 时 2.0 cm 已在药材外）
R_M = R_CM / 100.0


def run_q4(props, ch, R_func, label, record=True):
    rec = dict(t=[], C=[], Cs=[], R=[], meanC=[], Tc=[], Ts=[], Cc=[])

    def on_60s(s):
        if record:
            rec['t'].append(int(round(s.t)))
            rec['C'].append(s.at_r(R_M, 'C'))
            rec['Cs'].append(float(s.C[-1]))
            rec['R'].append(s.R)
            rec['meanC'].append(s.mean_C())
            rec['Tc'].append(float(s.T[0]))
            rec['Ts'].append(float(s.T[-1]))
            rec['Cc'].append(float(s.C[0]))

    print('%s：开始计算……' % label, flush=True)
    s, dry = simulate(props, ch, R_func=R_func, t_1s_until=0.0, until_dry=True, on_60s=on_60s)
    out = dict((k, np.array(v)) for k, v in rec.items())
    out['s'], out['dry'] = s, dry
    print('%s：烘干时间 %.4f h' % (label, dry[0] / 3600.0), flush=True)
    return out


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_dirs()
    t0 = time.time()
    ch = Chamber()
    Rp = Radius('pchip')
    main_res = run_q4(props_q4(), ch, Rp, '问题4（收缩，PCHIP 半径）')
    s = main_res['s']
    t_star, C_star, T_star, R_star = main_res['dry']
    #  result4.xlsx 
    header = [int(v) if abs(v - round(v)) < 1e-12 else float(v) for v in R_CM] + ['药材表面']
    data = np.column_stack([main_res['C'], main_res['Cs']])
    path = write_result_workbook('result4', 'result4.xlsx', [(None, header, main_res['t'], data)])
    print('已写出', path)
    #  表6 
    rows = []
    hh = 6
    while hh * 3600 < t_star:
        i = int(np.where(main_res['t'] == hh * 3600)[0][0])
        Rt = main_res['R'][i]
        vals = [float(main_res['C'][i, int(round(r * 10))]) for r in (0.0, 0.5, 1.0)]
        rows.append(['%d' % hh] + [r4(v) for v in vals] + [r4(main_res['Cs'][i]), r4(Rt * 100.0)])
        hh += 6
    C_end = profile_at_r(s.xi, C_star, R_star, np.array([0.0, 0.005, 0.010]))
    rows.append(['%.4f' % (t_star / 3600.0)] + [r4(v) for v in C_end] +
                [r4(C_star[-1]), r4(R_star * 100.0)])

    #  对比算例 
    cases = {}
    for kind in ('linear', 'fit'):
        r = run_q4(props_q4(), ch, Radius(kind), '灵敏度：半径函数 %s' % kind, record=False)
        cases['R_' + kind] = r['dry'][0] / 3600.0
    nos = run_q4(props_q4(), ch, None, '对比：不计收缩（R=2 cm，附录4物性）')
    cases['no_shrink'] = nos['dry'][0] / 3600.0
    # 收缩与失水的关系：(R/R0)^2（体积比）与平均水分浓度的线性相关
    vol = (main_res['R'] / R0) ** 2
    X = main_res['meanC']
    A = np.vstack([X, np.ones_like(X)]).T
    coef, *_ = np.linalg.lstsq(A, vol, rcond=None)
    pred = A @ coef
    r2 = 1.0 - np.sum((vol - pred) ** 2) / np.sum((vol - vol.mean()) ** 2)
    tR, R_cm = load_radius_data()
    summary = dict(
        t_star_s=float(t_star), t_star_h=float(t_star / 3600.0), table6=rows,
        R_end_cm=float(R_star * 100.0), C_end=[float(v) for v in C_end], Csurf_end=float(C_star[-1]),
        meanC_end=float(2.0 * np.sum(s.V * C_star)),
        mass_err=float((s.mean_C() - 2.55) + 2 * s.flux_int),
        sens=cases, shrink_effect_h=float(cases['no_shrink'] - t_star / 3600.0),
        vol_vs_moisture=dict(slope=float(coef[0]), intercept=float(coef[1]), r2=float(r2)),
        R_fit_params=[float(v) for v in Radius('fit').p_fit],
        n_rows=int(main_res['t'].size), last_row_t=int(main_res['t'][-1]),
        R_60s_cm=float(Rp(60.0) * 100.0), R_at_3h_cm=float(Rp(10800.0) * 100.0),
        elapsed=time.time() - t0)
    save_json(summary, 'q4_summary.json')
    np.savez_compressed(os.path.join('results', 'q4_fields.npz'), t=main_res['t'], r=R_CM,
                        C=main_res['C'], Cs=main_res['Cs'], R=main_res['R'], meanC=main_res['meanC'],
                        Cc=main_res['Cc'], Tc=main_res['Tc'], Ts=main_res['Ts'],
                        t_ns=nos['t'], Cc_ns=nos['Cc'], meanC_ns=nos['meanC'],
                        xi=s.xi, C_star=C_star, R_star=R_star, t_star=t_star)

    #  图 
    plt = setup_plot()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    tt = np.linspace(0, tR[-1], 2000)
    ax[0].plot(tR / 3600.0, R_cm, 'o', ms=3, mfc='none', color='#1f4e79', label='附件2数据')
    ax[0].plot(tt / 3600.0, Rp(tt) * 100.0, '-', color='#c00000', lw=1.2, label='PCHIP 插值（主模型）')
    ax[0].plot(tt / 3600.0, Radius('fit')(tt) * 100.0, '--', color='#70ad47', lw=1.0,
               label='双指数回归（灵敏度检验）')
    ax[0].axvline(t_star / 3600.0, color='gray', lw=0.8, ls=':')
    ax[0].set_xlabel('时间 t / h')
    ax[0].set_ylabel('药材半径 R / cm')
    ax[0].set_title('(a) 药材半径变化')
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(alpha=0.3)
    th = np.concatenate(([0.0], main_res['t'] / 3600.0))
    ax[1].plot(th, np.concatenate(([2.55], main_res['Cc'])), color='#1f4e79', lw=1.4, label='中心（收缩）')
    ax[1].plot(th, np.concatenate(([2.55], main_res['meanC'])), color='#2e75b6', lw=1.2, ls='--',
               label='平均（收缩）')
    ax[1].plot(th, np.concatenate(([2.55], main_res['Cs'])), color='#c00000', lw=1.2, label='表面（收缩）')
    thn = np.concatenate(([0.0], nos['t'] / 3600.0))
    ax[1].plot(thn, np.concatenate(([2.55], nos['Cc'])), color='gray', lw=1.2, ls='-.',
               label='中心（不计收缩）')
    ax[1].axhline(C_TARGET, color='k', lw=0.7, ls=':')
    ax[1].set_xlabel('时间 t / h')
    ax[1].set_ylabel('水分浓度 C / (kg/kg)')
    ax[1].set_title('(b) 水分浓度变化及收缩影响')
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, thn[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q4_curves.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    cm = plt.get_cmap('viridis')
    marks = list(range(6, int(t_star / 3600.0) + 1, 6))
    for n, hh in enumerate(marks):
        i = int(np.where(main_res['t'] == hh * 3600)[0][0])
        Rt = main_res['R'][i]
        ok = np.isfinite(main_res['C'][i])
        xs = np.append(R_CM[ok], Rt * 100.0)
        ys = np.append(main_res['C'][i][ok], main_res['Cs'][i])
        ax.plot(xs, ys, '-', color=cm(n / max(len(marks), 1)), lw=1.2, label='%d h' % hh)
        ax.plot(Rt * 100.0, main_res['Cs'][i], 'o', ms=3, color=cm(n / max(len(marks), 1)))
    rr = np.linspace(0, R_star, 100)
    ax.plot(rr * 100.0, profile_at_r(s.xi, C_star, R_star, rr), 'k--', lw=1.4,
            label='t* = %.2f h' % (t_star / 3600.0))
    ax.axhline(C_TARGET, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('到药材中心的距离 r / cm')
    ax.set_ylabel('水分浓度 C / (kg/kg)')
    ax.legend(ncol=2, frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q4_profiles.png'))
    plt.close(fig)
    print('问题4 完成，用时 %.1f s' % (time.time() - t0))
    return summary


if __name__ == '__main__':
    main()
