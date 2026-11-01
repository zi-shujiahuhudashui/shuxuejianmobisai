# -*- coding: utf-8 -*-
# 问题二：0~3 h 热-质耦合变物性模型
import os
import time
import numpy as np
from common import (Chamber, props_q23, simulate, write_result_workbook, r_header, save_json,
                    ensure_dirs, setup_plot, FIG_DIR, r4, stage_analysis)

R_CM = np.round(np.arange(0, 21) * 0.1, 1)
R_M = R_CM / 100.0
T_END = 10800                                  # 3 h
TAB_H = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TAB_R = [0.0, 0.5, 1.0, 1.5, 2.0]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_dirs()
    t0 = time.time()
    ch = Chamber()
    st = stage_analysis()
    T_rec = np.empty((T_END, R_M.size))
    C_rec = np.empty((T_END, R_M.size))
    meanC = np.empty(T_END)
    meanT = np.empty(T_END)

    def on_1s(s):
        i = int(round(s.t)) - 1
        T_rec[i] = s.at_r(R_M, 'T')
        C_rec[i] = s.at_r(R_M, 'C')
        meanC[i] = s.mean_C()
        meanT[i] = s.mean_T()

    s, _ = simulate(props_q23(), ch, t_1s_until=T_END, until_dry=False, t_max=T_END, on_1s=on_1s)
    times = np.arange(1, T_END + 1)
    hdr = r_header(R_CM)
    path = write_result_workbook('result2', 'result2.xlsx', [
        ('温度', hdr, times, T_rec), ('水分浓度', hdr, times, C_rec)])
    print('已写出', path)

    idx_t = [int(round(hh * 3600)) - 1 for hh in TAB_H]
    idx_r = [int(round(r * 10)) for r in TAB_R]
    table3 = [[r4(T_rec[i, j]) for j in idx_r] for i in idx_t]
    table4 = [[r4(C_rec[i, j]) for j in idx_r] for i in idx_t]
    # 与问题1模型（附录2常物性）比较：前 1800 s
    q1 = np.load(os.path.join('results', 'q1_fields.npz')) if os.path.exists(
        os.path.join('results', 'q1_fields.npz')) else None
    cmp = None
    if q1 is not None:
        dT = T_rec[:1800] - q1['T']
        dC = C_rec[:1800] - q1['C']
        cmp = dict(dT_max=float(np.abs(dT).max()), dC_max=float(np.abs(dC).max()),
                   T1800_q2=[r4(T_rec[1799, j]) for j in idx_r],
                   C1800_q2=[r4(C_rec[1799, j]) for j in idx_r])
    # 物性取值范围（初始时刻与 3 h 时）
    summary = dict(table3=table3, table4=table4, tab_h=TAB_H, tab_r=TAB_R,
                   meanC_3h=float(meanC[-1]), meanT_3h=float(meanT[-1]),
                   dT_center_surface_3h=float(T_rec[-1, -1] - T_rec[-1, 0]),
                   stage=dict(t_switch=st['t_switch'], T=dict((k, v) for k, v in st['T'].items()
                                                              if k != 'table'),
                              C=dict((k, v) for k, v in st['C'].items() if k != 'table')),
                   compare_q1=cmp, mean_iters=float(np.mean(s.iters)), elapsed=time.time() - t0)
    save_json(summary, 'q2_summary.json')
    np.savez_compressed(os.path.join('results', 'q2_fields.npz'), t=times, r=R_CM, T=T_rec, C=C_rec,
                        meanC=meanC, meanT=meanT)

    #  图
    plt = setup_plot()
    cols = ['#1f4e79', '#2e75b6', '#70ad47', '#ed7d31', '#c00000']
    th = np.concatenate(([0.0], times / 3600.0))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, rr in zip(cols, TAB_R):
        j = int(round(rr * 10))
        ax[0].plot(th, np.concatenate(([28.0], T_rec[:, j])), color=c, lw=1.3, label='r = %.1f cm' % rr)
        ax[1].plot(th, np.concatenate(([2.55], C_rec[:, j])), color=c, lw=1.3, label='r = %.1f cm' % rr)
    ax[0].plot(th, ch.Ta(th * 3600.0), 'k--', lw=0.8, label='烘房温度')
    ax[1].plot(th, np.concatenate(([2.55], meanC)), 'k:', lw=1.2, label='平均水分浓度')
    ax[0].axvline(st['t_switch'] / 3600.0, color='gray', lw=0.8, ls='-.')
    ax[1].axvline(st['t_switch'] / 3600.0, color='gray', lw=0.8, ls='-.')
    ax[0].text(st['t_switch'] / 3600.0 + 0.05, 30, '预热平衡阶段结束', fontsize=8, color='gray')
    ax[0].set_xlabel('时间 t / h')
    ax[0].set_ylabel('温度 T / °C')
    ax[0].set_title('(a) 药材温度')
    ax[0].legend(loc='lower right', ncol=2, frameon=False, fontsize=8)
    ax[1].set_xlabel('时间 t / h')
    ax[1].set_ylabel('水分浓度 C / (kg/kg)')
    ax[1].set_title('(b) 药材水分浓度')
    ax[1].legend(loc='lower left', frameon=False, fontsize=8)
    for a in ax:
        a.set_xlim(0, 3)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q2_curves.png'))
    plt.close(fig)

    # 时空云图
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    tg = times[::20] / 3600.0
    im0 = ax[0].pcolormesh(R_CM, tg, T_rec[::20], shading='gouraud', cmap='inferno')
    im1 = ax[1].pcolormesh(R_CM, tg, C_rec[::20], shading='gouraud', cmap='viridis_r')
    fig.colorbar(im0, ax=ax[0], label='T / °C')
    fig.colorbar(im1, ax=ax[1], label='C / (kg/kg)')
    for a, ttl in zip(ax, ['(a) 温度时空分布', '(b) 水分浓度时空分布']):
        a.set_xlabel('到药材中心的距离 r / cm')
        a.set_ylabel('时间 t / h')
        a.set_title(ttl)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q2_maps.png'))
    plt.close(fig)
    print('问题2 完成，用时 %.1f s' % (time.time() - t0))
    return summary


if __name__ == '__main__':
    main()
