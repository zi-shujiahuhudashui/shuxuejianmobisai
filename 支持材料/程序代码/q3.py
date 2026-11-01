# -*- coding: utf-8 -*-
# 问题三：烘干时间
import os
import time
import numpy as np
from common import (Chamber, props_q23, simulate, write_result_workbook, r_header, save_json,
                    ensure_dirs, setup_plot, FIG_DIR, r4, profile_at_r, C_TARGET)

R_CM = np.round(np.arange(0, 21) * 0.1, 1)
R_M = R_CM / 100.0
TAB_R = [0.0, 0.5, 1.0, 1.5, 2.0]


def run_case(props, ch, R_func=None, label='问题3', **kw):
    rec_t, rec_C, rec_T, rec_mean, rec_R, rec_surf = [], [], [], [], [], []

    def on_60s(s):
        rec_t.append(int(round(s.t)))
        rec_C.append(s.at_r(R_M, 'C'))
        rec_T.append(s.at_r(R_M, 'T'))
        rec_mean.append(s.mean_C())
        rec_R.append(s.R)
        rec_surf.append(float(s.C[-1]))

    print('%s：开始计算……' % label, flush=True)
    s, dry = simulate(props, ch, R_func=R_func, t_1s_until=10800.0, until_dry=True,
                      on_60s=on_60s, **kw)
    return dict(s=s, dry=dry, t=np.array(rec_t), C=np.array(rec_C), T=np.array(rec_T),
                meanC=np.array(rec_mean), R=np.array(rec_R), Csurf=np.array(rec_surf))


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_dirs()
    t0 = time.time()
    ch = Chamber()
    res = run_case(props_q23(), ch)
    s, (t_star, C_star, T_star, R_star) = res['s'], res['dry']
    print('问题3：烘干所需时间 t* = %.1f s = %.4f h' % (t_star, t_star / 3600.0))
    hdr = r_header(R_CM)
    path = write_result_workbook('result3', 'result3.xlsx', [(None, hdr, res['t'], res['C'])])
    print('已写出', path)

    # 表5：每 6 h 与烘干结束时刻
    idx_r = [int(round(r * 10)) for r in TAB_R]
    rows = []
    hh = 6
    while hh * 3600 < t_star:
        i = int(np.where(res['t'] == hh * 3600)[0][0])
        rows.append(['%d' % hh] + [r4(res['C'][i, j]) for j in idx_r])
        hh += 6
    C_end = profile_at_r(s.xi, C_star, R_star, np.array(TAB_R) / 100.0)
    rows.append(['%.4f' % (t_star / 3600.0)] + [r4(v) for v in C_end])
    # 与问题2结果一致性检查（共同时刻）
    consist = None
    f2 = os.path.join('results', 'q2_fields.npz')
    if os.path.exists(f2):
        q2 = np.load(f2)
        common_t = res['t'][res['t'] <= 10800]
        d = [np.abs(res['C'][k] - q2['C'][tt - 1]).max() for k, tt in enumerate(common_t)]
        consist = float(np.max(d))
    # 过程特征量
    tt_h = np.concatenate(([0.0], res['t'] / 3600.0))
    Xm = np.concatenate(([2.55], res['meanC']))
    rate = -np.gradient(Xm, tt_h)                    # 平均干燥速率 (kg/kg)/h
    i_max = int(np.argmax(rate))
    t_mean015 = float(np.interp(-C_TARGET, -Xm, tt_h))   # 平均水分浓度降至 0.15 的时刻
    t_surf015 = float(np.interp(-C_TARGET, -np.concatenate(([2.55], res['Csurf'])), tt_h))
    frac_removed = (2.55 - res['meanC']) / (2.55 - 0.05)
    t_half = float(np.interp(0.5, frac_removed, res['t'] / 3600.0))
    summary = dict(t_star_s=float(t_star), t_star_h=float(t_star / 3600.0), table5=rows,
                   tab_r=TAB_R, C_end=[float(v) for v in C_end], T_end_center=float(T_star[0]),
                   meanC_end=float(2.0 * np.sum(s.V * C_star)), Csurf_end=float(C_star[-1]),
                   last_row_t=int(res['t'][-1]), n_rows=int(res['t'].size),
                   max_rate=float(rate[i_max]), t_max_rate=float(tt_h[i_max]),
                   t_mean015_h=t_mean015, t_surf015_h=t_surf015, t_half_removed_h=t_half,
                   consistency_with_q2=consist, mass_err=float((s.mean_C() - 2.55) + 2 * s.flux_int),
                   elapsed=time.time() - t0)
    save_json(summary, 'q3_summary.json')
    np.savez_compressed(os.path.join('results', 'q3_fields.npz'), t=res['t'], r=R_CM, C=res['C'],
                        T=res['T'], meanC=res['meanC'], Csurf=res['Csurf'], t_star=t_star,
                        C_star=C_star, xi=s.xi)

    #  图
    plt = setup_plot()
    cols = ['#1f4e79', '#2e75b6', '#70ad47', '#ed7d31', '#c00000']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, rr in zip(cols, TAB_R):
        j = int(round(rr * 10))
        ax[0].plot(tt_h, np.concatenate(([2.55], res['C'][:, j])), color=c, lw=1.3,
                   label='r = %.1f cm' % rr)
    ax[0].plot(tt_h, Xm, 'k:', lw=1.3, label='平均水分浓度')
    ax[0].axhline(C_TARGET, color='gray', lw=0.8, ls='--')
    ax[0].axvline(t_star / 3600.0, color='gray', lw=0.8, ls='--')
    ax[0].annotate('t* = %.2f h' % (t_star / 3600.0), xy=(t_star / 3600.0, C_TARGET),
                   xytext=(t_star / 3600.0 - 22, 0.9), fontsize=9,
                   arrowprops=dict(arrowstyle='->', lw=0.8))
    ax[0].set_xlabel('时间 t / h')
    ax[0].set_ylabel('水分浓度 C / (kg/kg)')
    ax[0].set_title('(a) 各位置水分浓度随时间变化')
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_xlim(0, tt_h[-1])
    ax[0].grid(alpha=0.3)
    ax1 = ax[1]
    ax1.plot(Xm, rate, color='#1f4e79', lw=1.4)
    ax1.set_xlabel('平均水分浓度 $\\bar{C}$ / (kg/kg)')
    ax1.set_ylabel('干燥速率 $-\\mathrm{d}\\bar{C}/\\mathrm{d}t$ / (kg/kg·h$^{-1}$)')
    ax1.set_title('(b) 干燥速率曲线')
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q3_curves.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    cm = plt.get_cmap('viridis')
    marks = list(range(6, int(t_star / 3600.0) + 1, 6))
    for n, hh in enumerate(marks):
        i = int(np.where(res['t'] == hh * 3600)[0][0])
        ax.plot(R_CM, res['C'][i], '-', color=cm(n / max(len(marks), 1)), lw=1.2, label='%d h' % hh)
    ax.plot(R_CM, profile_at_r(s.xi, C_star, R_star, R_M), 'k--', lw=1.4,
            label='t* = %.2f h' % (t_star / 3600.0))
    ax.axhline(C_TARGET, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('到药材中心的距离 r / cm')
    ax.set_ylabel('水分浓度 C / (kg/kg)')
    ax.set_xlim(0, 2)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q3_profiles.png'))
    plt.close(fig)
    print('问题3 完成，用时 %.1f s' % (time.time() - t0))
    return summary


if __name__ == '__main__':
    main()
