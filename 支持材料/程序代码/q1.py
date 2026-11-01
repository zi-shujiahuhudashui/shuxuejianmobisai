# -*- coding: utf-8 -*-
# 问题一：0~1800 s 温度与水分浓度
import os
import time
import numpy as np
from common import (Chamber, FVSolver, props_q1, analytic_T_q1, write_result_workbook, r_header,
                    save_json, ensure_dirs, setup_plot, FIG_DIR, r4)

R_CM = np.round(np.arange(0, 21) * 0.1, 1)          # 0, 0.1, ..., 2.0 cm
R_M = R_CM / 100.0
TIMES = np.arange(1, 1801)                           # 1, 2, ..., 1800 s
TAB_T = [100, 300, 600, 900, 1200, 1500, 1800]
TAB_R = [0.0, 0.5, 1.0, 1.5, 2.0]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_dirs()
    ch = Chamber()
    t0 = time.time()
    # 1) 温度解析解
    T_an = analytic_T_q1(R_M, TIMES, ch, n_terms=400)
    T_an100 = analytic_T_q1(R_M, TIMES, ch, n_terms=100)
    trunc = float(np.abs(T_an - T_an100).max())
    # 2) 数值解（水分场 + 温度校核）
    s = FVSolver(props_q1(), ch, coupled=False)
    T_num = np.empty((TIMES.size, R_M.size))
    C_num = np.empty((TIMES.size, R_M.size))
    mean_C = np.empty(TIMES.size)
    for i, tk in enumerate(TIMES):
        s.advance_to(float(tk), 0.5)
        T_num[i] = s.at_r(R_M, 'T')
        C_num[i] = s.at_r(R_M, 'C')
        mean_C[i] = s.mean_C()
    err_T = float(np.abs(T_num - T_an).max())
    mass_err = float((s.mean_C() - 2.55) + 2.0 * s.flux_int)
    rc = 820.0 * 2600.0
    energy_err = float(rc * (s.mean_T() - 28.0) + 2.0 * s.heat_int * 1.0)
    print('问题1：解析解级数截断误差 %.2e °C；数值解与解析解最大偏差 %.2e °C' % (trunc, err_T))
    print('       水分守恒误差 %.2e (kg/kg)；能量守恒相对误差 %.2e' % (
        mass_err, energy_err / (rc * (s.mean_T() - 28.0))))

    # 3) 写 result1.xlsx
    hdr = r_header(R_CM)
    path = write_result_workbook('result1', 'result1.xlsx', [
        ('温度', hdr, TIMES, T_an),
        ('水分浓度', hdr, TIMES, C_num)])
    print('已写出', path)

    # 4) 论文表格数据
    idx_t = [int(t) - 1 for t in TAB_T]
    idx_r = [int(round(r * 10)) for r in TAB_R]
    table1 = [[r4(T_an[i, j]) for j in idx_r] for i in idx_t]
    table2 = [[r4(C_num[i, j]) for j in idx_r] for i in idx_t]
    # 附加分析量
    lag = T_an[:, -1] - T_an[:, 0]
    i_lag = int(np.argmax(lag))
    Ta_1800 = float(ch.Ta(1800.0))
    # 水分变化深度：|C-2.55| > 1e-4 的最小半径
    dep = []
    for i in idx_t:
        ch_idx = np.where(np.abs(C_num[i] - 2.55) >= 1e-4)[0]
        dep.append(float(R_CM[ch_idx[0]]) if ch_idx.size else None)
    summary = dict(
        table1=table1, table2=table2, tab_t=TAB_T, tab_r=TAB_R,
        series_trunc=trunc, err_T_num_vs_an=err_T, mass_err=mass_err,
        energy_rel_err=float(energy_err / (rc * (s.mean_T() - 28.0))),
        Ta_1800=Ta_1800, max_lag=float(lag[i_lag]), t_max_lag=int(TIMES[i_lag]),
        lag_1800=float(lag[-1]), meanC_1800=float(mean_C[-1]),
        moisture_front_r=dep, nodes=int(s.N + 1), elapsed=time.time() - t0,
        Bi=25.0 * 0.02 / 0.36, alpha=0.36 / (820 * 2600),
        Tsurf_1800=float(T_an[-1, -1]), Tcen_1800=float(T_an[-1, 0]),
        Csurf_1800=float(C_num[-1, -1]))
    save_json(summary, 'q1_summary.json')
    np.savez_compressed(os.path.join('results', 'q1_fields.npz'), t=TIMES, r=R_CM, T=T_an, C=C_num,
                        T_num=T_num, meanC=mean_C)

    # 5) 图
    plt = setup_plot()
    cols = ['#1f4e79', '#2e75b6', '#70ad47', '#ed7d31', '#c00000']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    tt = np.concatenate(([0], TIMES))
    for c, rr in zip(cols, TAB_R):
        j = int(round(rr * 10))
        ax[0].plot(tt, np.concatenate(([28.0], T_an[:, j])), color=c, lw=1.4, label='r = %.1f cm' % rr)
        ax[1].plot(tt, np.concatenate(([2.55], C_num[:, j])), color=c, lw=1.4, label='r = %.1f cm' % rr)
    ax[0].plot(tt, ch.Ta(tt), 'k--', lw=1.0, label='烘房温度')
    ax[0].set_xlabel('时间 t / s')
    ax[0].set_ylabel('温度 T / °C')
    ax[0].set_title('(a) 药材温度')
    ax[0].legend(loc='upper left', ncol=2, frameon=False)
    ax[1].set_xlabel('时间 t / s')
    ax[1].set_ylabel('水分浓度 C / (kg/kg)')
    ax[1].set_title('(b) 药材水分浓度')
    ax[1].legend(loc='lower left', frameon=False)
    for a in ax:
        a.set_xlim(0, 1800)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q1_curves.png'))
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cm = plt.get_cmap('viridis')
    for n, tk in enumerate(TAB_T):
        c = cm(n / (len(TAB_T) - 1))
        ax[0].plot(R_CM, T_an[tk - 1], '-o', ms=2.5, color=c, lw=1.2, label='%d s' % tk)
        ax[1].plot(R_CM, C_num[tk - 1], '-o', ms=2.5, color=c, lw=1.2, label='%d s' % tk)
    ax[0].set_xlabel('到药材中心的距离 r / cm')
    ax[0].set_ylabel('温度 T / °C')
    ax[0].set_title('(a) 温度径向分布')
    ax[1].set_xlabel('到药材中心的距离 r / cm')
    ax[1].set_ylabel('水分浓度 C / (kg/kg)')
    ax[1].set_title('(b) 水分浓度径向分布')
    ax[0].legend(ncol=2, frameon=False, fontsize=8)
    ax[1].legend(ncol=2, frameon=False, fontsize=8, loc='lower left')
    for a in ax:
        a.set_xlim(0, 2)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q1_profiles.png'))
    plt.close(fig)
    print('问题1 完成，用时 %.1f s' % (time.time() - t0))
    return summary


if __name__ == '__main__':
    main()
