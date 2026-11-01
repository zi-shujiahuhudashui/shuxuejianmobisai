# -*- coding: utf-8 -*-
# 图：本构三联图、问题3/4 三维与圆盘、四问看板
import os
import sys
import json
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from common import setup_plot, load_radius_data, save_json, FIG_DIR, R0, C_INIT, C_TARGET

RES = 'results'
C_DEEP = '#1f4e79'
C_RED = '#c0504d'
C_OLIVE = '#6a8a5d'
C_ORANGE = '#e8975a'
C_GOLD = '#c9a227'

# 数据
# 本构标定
with open(os.path.join(RES, 'innov_const_cache.json'), 'r', encoding='utf-8') as f:
    cache = json.load(f)
with open(os.path.join(RES, 'innov_constitutive.json'), 'r', encoding='utf-8') as f:
    out = json.load(f)
rec = dict((k, np.asarray(v, float)) for k, v in cache['baseline']['rec'].items())
recp = dict((k, np.asarray(v, float)) for k, v in cache['predict']['rec'].items())
t_meas, R_meas_cm = load_radius_data()
b, R2, res_rms = out['b'], out['R2'], out['res_rms']
t_meas_all = np.asarray(t_meas, float)

# q3 场
q3 = np.load(os.path.join(RES, 'q3_fields.npz'))
t3 = q3['t'].astype(float)
r3 = q3['r'].astype(float)
C3 = q3['C'].astype(float)
T3 = q3['T'].astype(float)
t3_h = t3 / 3600.0
t3_star_h = float(q3['t_star']) / 3600.0
Csurf3 = C3[:, -1]
i_surf_dry = int(np.argmax(Csurf3 < C_TARGET))
t_surf_dry_h = t3_h[i_surf_dry]
front_r = np.full(t3.size, np.nan)
for i in range(t3.size):
    col = C3[i]
    if col[-1] >= C_TARGET:
        front_r[i] = r3[-1]
    elif col[0] <= C_TARGET:
        front_r[i] = 0.0
    else:
        front_r[i] = np.interp(C_TARGET, col[::-1], r3[::-1])

# q4 场
q4f = np.load(os.path.join(RES, 'innov_q4_fields.npz'))
t4 = q4f['t'].astype(float)
t4_h = t4 / 3600.0
r4cm = q4f['r_cm'].astype(float)
C4 = q4f['C'].astype(float)
T4 = q4f['T'].astype(float)
R4 = q4f['R'].astype(float)
Cs4 = q4f['Csurf'].astype(float)
Cc4 = q4f['Cc'].astype(float)
meanC4 = q4f['meanC'].astype(float)
t4_star_h = float(q4f['t_star']) / 3600.0

q4n = np.load(os.path.join(RES, 'innov_q4noshrink.npz'))
t4n_h = q4n['t'].astype(float) / 3600.0
Cc4n = q4n['Cc'].astype(float)
t4n_star_h = float(q4n['t_star']) / 3600.0
reduct_pct = (t4n_star_h - t4_star_h) / t4n_star_h * 100.0

# q1/q2 场（看板用）
q1 = np.load(os.path.join(RES, 'q1_fields.npz'))
q2 = np.load(os.path.join(RES, 'q2_fields.npz'))

fj = {}
fj['q3'] = dict(t_star_h=round(t3_star_h, 4), t_surf_dry_h=round(t_surf_dry_h, 4))
fj['q4'] = dict(t_star_h=round(t4_star_h, 4), t_noshrink_h=round(t4n_star_h, 4),
                R_final_cm=round(R4[-1] * 100.0, 4), reduct_pct=round(reduct_pct, 2))

plt = setup_plot()
from matplotlib import cm as mcm
from matplotlib.colors import Normalize


def plot_const():
    fig = plt.figure(figsize=(13.8, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.15], hspace=0.08, wspace=0.34)

    # (a) 标定
    ax = fig.add_subplot(gs[0, 0])
    Cs = rec['Cs']
    y = (rec['R'] / R0) ** 2
    ax.scatter(Cs, y, s=9, color=C_DEEP, alpha=0.55, edgecolors='none',
               label='基准模拟 60 s 配对 $(C_s,\\,(R/R_0)^2)$', zorder=3)
    xx = np.linspace(0.0, C_INIT, 200)
    yy = 1.0 - b * (C_INIT - xx)
    n = Cs.size
    dC = C_INIT - Cs
    sigma2 = np.sum((y - (1.0 - b * dC)) ** 2) / (n - 1)
    from scipy import stats as _st
    tcrit = _st.t.ppf(0.975, n - 1)
    band = tcrit * np.sqrt(sigma2) * (C_INIT - xx) / np.sqrt(np.sum(dC ** 2))
    ax.fill_between(xx, yy - band, yy + band, color=C_RED, alpha=0.15, lw=0,
                    label='95% 置信带')
    ax.plot(xx, yy, color=C_RED, lw=1.8,
            label='拟合 $(R/R_0)^2=1-b\\,(C_0-C_s)$，$b=%.4f$' % b, zorder=4)
    ax.set_xlim(0.0, C_INIT)
    ax.set_ylim(0.25, 1.05)
    ax.set_ylabel('体积比 $(R/R_0)^2$')
    ax.set_title('(a) 收缩本构标定：表面驱动显著更优\n'
                 '$b=%.4f$，$R^2=%.4f$（平均驱动仅 %.4f）'
                 % (b, R2, out['R2_mean_driven']), fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)
    ax.tick_params(labelbottom=False)
    ax.annotate('', xy=(2.1, 0.62), xytext=(1.55, 0.80),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.9))
    ax.text(1.62, 0.83, '脱水进程', fontsize=8, color='gray')

    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    resid = y - (1.0 - b * dC)
    axr.scatter(Cs, resid, s=8, color=C_OLIVE, alpha=0.6, edgecolors='none')
    axr.axhline(0.0, color='k', lw=0.8)
    axr.set_xlabel('表面水分浓度 $C_s$ / (kg/kg)')
    axr.set_ylabel('残差', fontsize=9)
    axr.grid(alpha=0.3)
    axr.tick_params(labelsize=8.5)

    # (b) 自洽预测
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(t_meas_all / 3600.0, R_meas_cm, s=10, color=C_DEEP, alpha=0.6,
                edgecolors='none', label='附件2实测半径', zorder=3)
    ax2.plot(recp['t'] / 3600.0, recp['R'] * 100.0, color=C_RED, lw=1.8,
             label='本构自洽预测（不用附件2）', zorder=4)
    t_end_h = recp['t'][-1] / 3600.0
    ax2.set_xlim(0, t_end_h * 1.02)
    ax2.set_ylim(1.0, 2.1)
    ax2.set_xlabel('时间 $t$ / h')
    ax2.set_ylabel('药材半径 $R$ / cm')
    rmse = out['predict']['R_rmse_cm']
    ax2.set_title('(b) 本构自洽预测复现收缩历程\n'
                  'RMSE=%.3f cm；终点 %.3f cm（实测 1.198）'
                  % (rmse, out['predict']['R_final_cm']), fontsize=9.5)
    ax2.legend(frameon=False, fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.annotate('预测 $t^*=%.2f$ h\n（基准 %.2f h，偏差 %+.2f%%）'
                 % (out['predict']['t_star'], out['base_t'], out['predict']['rel_err_pct']),
                 xy=(out['predict']['t_star'], 1.30), xytext=(t_end_h * 0.45, 1.75),
                 fontsize=8.5, color=C_RED,
                 arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.9))
    ax2.axvline(out['predict']['t_star'], color=C_RED, lw=0.8, ls=':')
    ax2r = fig.add_subplot(gs[1, 1], sharex=ax2)
    Rp_at_meas = np.interp(np.minimum(t_meas_all, recp['t'][-1]), recp['t'], recp['R'])
    ax2r.scatter(t_meas_all / 3600.0, Rp_at_meas * 100.0 - R_meas_cm, s=8,
                 color=C_OLIVE, alpha=0.6, edgecolors='none')
    ax2r.axhline(0.0, color='k', lw=0.8)
    ax2r.set_xlabel('时间 $t$ / h')
    ax2r.set_ylabel('预测$-$实测 / cm', fontsize=8.5)
    ax2r.grid(alpha=0.3)
    ax2r.tick_params(labelsize=8.5)

    # (c) 路径族
    ax3 = fig.add_subplot(gs[0, 2])
    u_t = np.asarray(cache['u_grid']['t'], float)
    u = np.asarray(cache['u_grid']['u'], float)
    beta = out['beta']
    styles = {'p05': (C_GOLD, '-', '$p=0.5$（前期快收缩）'),
              'p10': (C_DEEP, '--', '$p=1$（线性）'),
              'p20': (C_OLIVE, '-.', '$p=2$（前期慢收缩）')}
    for key in ('p05', 'p10', 'p20'):
        pw = {'p05': 0.5, 'p10': 1.0, 'p20': 2.0}[key]
        Rp = R0 * np.sqrt(1.0 - (1.0 - beta ** 2) * u ** pw) * 100.0
        colr, ls, lab = styles[key]
        ts = out['path_sens'][key]['t_star']
        ax3.plot(u_t / 3600.0, Rp, ls, color=colr, lw=1.7,
                 label='%s：$t^*=%.2f$ h（%+.2f h）'
                       % (lab, ts, out['path_sens'][key]['diff_h']))
        ax3.axvline(ts, color=colr, lw=0.7, ls=':')
    ax3.plot(t_meas_all / 3600.0, R_meas_cm, 'o', ms=2.5, mfc='none', mec='gray',
             alpha=0.55, label='附件2实测')
    ax3.set_xlim(0, rec['t'][-1] / 3600.0)
    ax3.set_ylim(1.0, 2.1)
    ax3.set_xlabel('时间 $t$ / h')
    ax3.set_ylabel('药材半径 $R$ / cm')
    ax3.set_title('(c) 收缩路径形状为二阶效应，前期快收缩更省时\n'
                  '$R_p=R_0\\sqrt{1-(1-\\beta^2)\\,u^p}$，$\\beta=%.3f$；三条路径 $t^*$ 跨度 %.2f h'
                  % (beta, abs(out['path_sens']['p20']['t_star']
                               - out['path_sens']['p05']['t_star'])), fontsize=9.5)
    ax3.legend(frameon=False, fontsize=7.6, loc='upper right')
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    for a in (ax, ax2, ax3):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.savefig(os.path.join(FIG_DIR, 'innov_constitutive.png'), bbox_inches='tight')
    plt.close(fig)
    print('saved innov_constitutive.png')


# 图1：问题3 三维
def plot_q3_3d():
    stride = 8
    Tg, Rg = np.meshgrid(t3_h[::stride], r3, indexing='ij')
    Zg = C3[::stride, :]
    fig = plt.figure(figsize=(10.5, 7.0))
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(0.0, C_INIT)
    surf = ax.plot_surface(Tg, Rg, Zg, cmap='viridis', norm=norm, rstride=1,
                           cstride=1, linewidth=0, antialiased=True, alpha=0.96)
    zmin, zmax = 0.0, C_INIT
    ax.contourf(Tg, Rg, Zg, zdir='z', offset=zmin - 0.35, cmap='viridis',
                norm=norm, levels=18, alpha=0.85)
    ok = np.isfinite(front_r[::stride])
    ax.plot(t3_h[::stride][ok], front_r[::stride][ok], np.full(ok.sum(), C_TARGET + 0.02),
            color=C_ORANGE, lw=3.2, zorder=10)
    ax.plot([t3_h[0], t_surf_dry_h], [r3[-1], r3[-1]], [C_TARGET + 0.02] * 2,
            color=C_ORANGE, lw=3.2, alpha=0.9)
    ax.scatter([t_surf_dry_h], [r3[-1]], [C_TARGET + 0.02], color=C_ORANGE, s=40, zorder=11)
    ax.scatter([t3_star_h], [0.0], [C_TARGET + 0.02], color=C_RED, s=45, zorder=11)
    ax.text(30.0, 2.08, 1.42, '表面约 %.1f h 即达标' % t_surf_dry_h,
            color=C_ORANGE, fontsize=10)
    ax.plot([13.4, 28.5], [2.0, 2.02], [C_TARGET + 0.10, 1.30], color=C_ORANGE, lw=0.9)
    ax.text(t3_star_h - 13.0, 0.6, C_TARGET + 1.15,
            '中心最后达标\n$t^*$ = %.2f h' % t3_star_h, color=C_RED, fontsize=10)
    ax.set_xlabel('时间 $t$ / h', labelpad=8)
    ax.set_ylabel('半径 $r$ / cm', labelpad=8)
    ax.set_zlabel('水分浓度 $C$ / (kg/kg)', labelpad=6)
    ax.set_zlim(zmin - 0.35, zmax)
    ax.view_init(elev=28, azim=-128)
    ax.set_title('干燥前锋由表及里推进：表面 %.1f h 达标而中心迟至 %.2f h\n'
                 '问题3含水率时空曲面（固定半径 2 cm），橙色线为 $C=0.15$ 干燥前锋轨迹'
                 % (t_surf_dry_h, t3_star_h), fontsize=11.5)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.10, aspect=22)
    cb.set_label('水分浓度 $C$ / (kg/kg)')
    fig.savefig(os.path.join(FIG_DIR, 'innov_3d_q3.png'), bbox_inches='tight')
    plt.close(fig)
    print('saved innov_3d_q3.png')


def draw_disk(ax, r_prof_cm, c_prof, R_show_cm, cmap='viridis', vmin=0.0, vmax=C_INIT,
              ngrid=181):
    lin = np.linspace(-R_show_cm, R_show_cm, ngrid)
    X, Y = np.meshgrid(lin, lin)
    RR = np.sqrt(X ** 2 + Y ** 2)
    Z = np.interp(RR, r_prof_cm, c_prof)
    Z = np.ma.masked_where(RR > R_show_cm, Z)
    ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def q3_profile_at(th):
    if th <= 0:
        return r3, np.full(r3.size, C_INIT)
    i = int(np.argmin(np.abs(t3_h - th)))
    return r3, C3[i]


def q4_profile_at(th):
    if th <= 0:
        return np.append(r4cm, 2.0), np.full(r4cm.size + 1, C_INIT), 2.0
    i = int(np.argmin(np.abs(t4_h - th)))
    Rcm = R4[i] * 100.0
    ok = np.isfinite(C4[i])
    r_pts = np.append(r4cm[ok], Rcm)
    c_pts = np.append(C4[i][ok], Cs4[i])
    return r_pts, c_pts, Rcm


# 图2：问题3 圆盘
def plot_q3_disks():
    times3 = [0, 2, 4, 8, 16, 32, 48, t3_star_h]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 8.4))
    for ax, th in zip(axes.ravel(), times3):
        rp, cp_ = q3_profile_at(th)
        draw_disk(ax, rp, cp_, 2.0)
        lab = ('$t^*$=%.2f h' % th) if abs(th - t3_star_h) < 0.2 else ('%g h' % th)
        ax.set_title('t = %s' % lab, fontsize=10)
        ax.set_xlabel('中心 %.3f / 表面 %.3f' % (cp_[0], cp_[-1]),
                      fontsize=8.6, color='#444444')
    fig.suptitle('干燥前锋由表及里：内部扩散控制主导全程\n'
                 '问题3（固定半径 2 cm）横截面含水率分布演化，统一色标 0~2.55 kg/kg',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 0.90, 0.90])
    fig.subplots_adjust(hspace=0.42, wspace=0.12)
    sm = mcm.ScalarMappable(norm=Normalize(0, C_INIT), cmap='viridis')
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('水分浓度 $C$ / (kg/kg)')
    fig.savefig(os.path.join(FIG_DIR, 'innov_disks_q3.png'), bbox_inches='tight')
    plt.close(fig)
    print('saved innov_disks_q3.png')
    fj['q3_disks'] = dict(times_h=[float(x) for x in times3])


# 图3：问题4 圆盘
def plot_q4_disks():
    times4 = [0, 6, 12, 24, 40, t4_star_h]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.6))
    for ax, th in zip(axes.ravel(), times4):
        rp, cp_, Rcm = q4_profile_at(th)
        draw_disk(ax, rp, cp_, Rcm)
        lab = ('$t^*$=%.2f h' % th) if abs(th - t4_star_h) < 0.2 else ('%g h' % th)
        ax.set_title('t = %s，R = %.3f cm' % (lab, Rcm), fontsize=9.5)
        ax.set_xlabel('中心 %.3f / 表面 %.3f' % (cp_[0], cp_[-1]),
                      fontsize=8.6, color='#444444')
    fig.suptitle('收缩使传热传质路径同步缩短：半径 2.000→1.198 cm，烘干较不收缩缩短 %.1f%%'
                 '（%.2f→%.2f h）\n问题4横截面含水率分布演化（坐标轴固定 ±2.1 cm，圆盘半径按 $R(t)$ 真实缩小）'
                 % (reduct_pct, t4n_star_h, t4_star_h), fontsize=11.5)
    fig.tight_layout(rect=[0, 0.02, 0.90, 0.88])
    fig.subplots_adjust(hspace=0.42, wspace=0.10)
    sm = mcm.ScalarMappable(norm=Normalize(0, C_INIT), cmap='viridis')
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('水分浓度 $C$ / (kg/kg)')
    fig.savefig(os.path.join(FIG_DIR, 'innov_disks_q4.png'), bbox_inches='tight')
    plt.close(fig)
    print('saved innov_disks_q4.png')
    fj['q4_disks'] = dict(times_h=[float(x) for x in times4],
                          R_cm=[round(q4_profile_at(x)[2], 4) for x in times4])


# 图4：问题4 三维
def plot_q4_3d():
    stride4 = 10
    r_grid = np.round(np.arange(0, 2.001, 0.05), 3)
    Zq = np.full((t4[::stride4].size, r_grid.size), np.nan)
    for ii, i in enumerate(range(0, t4.size, stride4)):
        Rcm = R4[i] * 100.0
        ok = np.isfinite(C4[i])
        r_pts = np.append(r4cm[ok], Rcm)
        c_pts = np.append(C4[i][ok], Cs4[i])
        inside = r_grid <= Rcm + 1e-9
        Zq[ii][inside] = np.interp(r_grid[inside], r_pts, c_pts)
    Tgq, Rgq = np.meshgrid(t4_h[::stride4], r_grid, indexing='ij')
    Zqm = np.ma.masked_invalid(Zq)
    fig = plt.figure(figsize=(10.5, 7.0))
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(0.0, C_INIT)
    surf = ax.plot_surface(Tgq, Rgq, Zqm, cmap='viridis', norm=norm, rstride=1,
                           cstride=1, linewidth=0, antialiased=True, alpha=0.96)
    ax.plot(t4_h[::stride4], R4[::stride4] * 100.0, Cs4[::stride4] + 0.02,
            color=C_RED, lw=3.0, zorder=10)
    ax.plot(t4_h[::stride4], R4[::stride4] * 100.0,
            np.full(t4_h[::stride4].size, -0.02), color=C_RED, lw=1.2, ls='--',
            alpha=0.7, zorder=9)
    ax.text(t4_h[-1] * 0.42, 2.30, 2.15, '红色曲线：收缩边界 $r=R(t)$\n'
            '$R$: 2.000 → %.3f cm' % (R4[-1] * 100.0), color=C_RED, fontsize=9.5)
    ax.set_xlabel('时间 $t$ / h', labelpad=8)
    ax.set_ylabel('物理半径 $r$ / cm', labelpad=8)
    ax.set_zlabel('水分浓度 $C$ / (kg/kg)', labelpad=6)
    ax.set_zlim(-0.02, C_INIT)
    ax.set_ylim(0, 2.1)
    ax.view_init(elev=27, azim=-126)
    ax.set_title('移动边界问题：收缩边界推动干燥全程加速至 %.2f h\n'
                 '问题4物理坐标含水率时空曲面（附录4物性 + 附件2收缩），较不收缩缩短 %.1f%%'
                 % (t4_star_h, reduct_pct), fontsize=11.5)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.10, aspect=22)
    cb.set_label('水分浓度 $C$ / (kg/kg)')
    fig.savefig(os.path.join(FIG_DIR, 'innov_3d_q4.png'), bbox_inches='tight')
    plt.close(fig)
    print('saved innov_3d_q4.png')


# 图5：看板
def plot_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6))

    # (a) 问题1
    ax = axes[0, 0]
    r1 = q1['r'].astype(float)
    i1 = int(np.argmin(np.abs(q1['t'].astype(float) - 1800.0)))
    ax.plot(r1, q1['C'][i1], color=C_DEEP, lw=1.8, label='水分浓度 $C$（左轴）')
    ax.set_xlabel('到中心距离 $r$ / cm')
    ax.set_ylabel('水分浓度 $C$ / (kg/kg)', color=C_DEEP)
    ax.tick_params(axis='y', labelcolor=C_DEEP)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(r1, q1['T_num'][i1], color=C_RED, lw=1.8, ls='--',
             label='温度 $T$（数值解，右轴）')
    ax2.plot(r1, q1['T'][i1], color=C_ORANGE, lw=1.2, ls=':',
             label='温度 $T$（解析解，右轴）')
    ax2.set_ylabel('温度 $T$ / °C', color=C_RED)
    ax2.tick_params(axis='y', labelcolor=C_RED)
    ln = ax.get_legend_handles_labels()
    ln2 = ax2.get_legend_handles_labels()
    ax.legend(ln[0] + ln2[0], ln[1] + ln2[1], frameon=False, fontsize=8, loc='upper right')
    ax.set_title('(a) 问题1：1800 s 时刻温度/水分径向剖面\n'
                 '数值解与解析解（分离变量+Duhamel）逐点吻合，验证格式可靠', fontsize=10)

    # (b) 问题2
    ax = axes[0, 1]
    t2_h = q2['t'].astype(float) / 3600.0
    C2, T2 = q2['C'].astype(float), q2['T'].astype(float)
    st = 10
    ax.plot(t2_h[::st], C2[::st, 0], color=C_DEEP, lw=1.6, label='$C$ 中心')
    ax.plot(t2_h[::st], C2[::st, -1], color=C_OLIVE, lw=1.6, ls='--', label='$C$ 表面')
    ax.plot(t2_h[::st], q2['meanC'][::st], color=C_DEEP, lw=1.0, ls=':', label='$C$ 平均')
    ax.set_xlabel('时间 $t$ / h')
    ax.set_ylabel('水分浓度 $C$ / (kg/kg)')
    ax.set_ylim(0, 2.7)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t2_h[::st], T2[::st, 0], color=C_RED, lw=1.5, label='$T$ 中心')
    ax2.plot(t2_h[::st], T2[::st, -1], color=C_ORANGE, lw=1.5, ls='--', label='$T$ 表面')
    ax2.set_ylabel('温度 $T$ / °C')
    ax2.set_ylim(20, 60)
    ln = ax.get_legend_handles_labels()
    ln2 = ax2.get_legend_handles_labels()
    ax.legend(ln[0] + ln2[0], ln[1] + ln2[1], frameon=False, fontsize=7.5, ncol=2, loc='center right')
    ax.set_title('(b) 问题2：前 3 h 中心/表面温度与含水率\n'
                 '预热平衡阶段温度快速趋稳 50 °C，表面含水率先行下降', fontsize=10)

    # (c) 问题3
    ax = axes[1, 0]
    ax.plot(t3_h, C3[:, 0], color=C_DEEP, lw=1.7, label='中心 $C$')
    ax.plot(t3_h, q3['meanC'], color=C_OLIVE, lw=1.3, ls='--', label='平均 $C$')
    ax.axhline(C_TARGET, color='k', lw=0.9, ls=':')
    ax.text(2.0, C_TARGET + 0.06, '烘干要求 $C=0.15$', fontsize=8.5)
    ax.axvline(t3_star_h, color=C_RED, lw=1.2, ls='--')
    ax.annotate('$t^*$ = %.2f h' % t3_star_h, xy=(t3_star_h, 0.15),
                xytext=(t3_star_h - 18, 0.85), fontsize=10, color=C_RED,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.0))
    ax.set_xlabel('时间 $t$ / h')
    ax.set_ylabel('水分浓度 $C$ / (kg/kg)')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title('(c) 问题3：中心含水率最后达标，$t^*$ = %.2f h\n'
                 '内部扩散控制：中心与平均含水率差距全程显著' % t3_star_h, fontsize=10)

    # (d) 问题4
    ax = axes[1, 1]
    ax.plot(np.concatenate(([0.0], t4n_h)), np.concatenate(([C_INIT], Cc4n)),
            color='gray', lw=1.5, ls='-.', label='中心 $C$（不收缩）')
    ax.plot(np.concatenate(([0.0], t4_h)), np.concatenate(([C_INIT], Cc4)),
            color=C_DEEP, lw=1.7, label='中心 $C$（收缩）')
    ax.axhline(C_TARGET, color='k', lw=0.9, ls=':')
    ax.axvline(t4_star_h, color=C_RED, lw=1.0, ls=':')
    ax.axvline(t4n_star_h, color='gray', lw=1.0, ls=':')
    ax.annotate('%.2f h' % t4_star_h, xy=(t4_star_h, 0.18), xytext=(t4_star_h - 14, 0.9),
                fontsize=9, color=C_RED, arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.9))
    ax.annotate('%.2f h' % t4n_star_h, xy=(t4n_star_h, 0.18), xytext=(t4n_star_h - 20, 1.35),
                fontsize=9, color='gray', arrowprops=dict(arrowstyle='->', color='gray', lw=0.9))
    ax.set_xlabel('时间 $t$ / h')
    ax.set_ylabel('水分浓度 $C$ / (kg/kg)')
    ax.set_xlim(0, t4n_h[-1])
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(np.concatenate(([0.0], t4_h)), np.concatenate(([2.0], R4 * 100.0)),
             color=C_GOLD, lw=1.6, label='半径 $R(t)$（右轴）')
    ax2.set_ylabel('半径 $R$ / cm', color=C_GOLD)
    ax2.tick_params(axis='y', labelcolor=C_GOLD)
    ax2.set_ylim(0.9, 2.15)
    ln = ax.get_legend_handles_labels()
    ln2 = ax2.get_legend_handles_labels()
    ax.legend(ln[0] + ln2[0], ln[1] + ln2[1], frameon=False, fontsize=8, loc='upper right')
    ax.set_title('(d) 问题4：收缩加速干燥 %.1f%%（%.2f→%.2f h）\n'
                 '半径 2.000→%.3f cm 同步缩短热质传递路径'
                 % (reduct_pct, t4n_star_h, t4_star_h, R4[-1] * 100.0), fontsize=10)

    fig.suptitle('四问结果总览：热质耦合建模 → %.2f h 烘干 → 收缩加速至 %.2f h\n'
                 '柱坐标有限体积 + BDF2 + Kirchhoff 变换；收缩由附件2实测半径驱动'
                 % (t3_star_h, t4_star_h), fontsize=12.5)
    fig.tight_layout(rect=[0, 0.0, 1.0, 0.945])
    fig.savefig(os.path.join(FIG_DIR, 'innov_dashboard.png'), bbox_inches='tight')
    plt.close(fig)
    print('saved innov_dashboard.png')

    fj['q1'] = dict(profile_t_s=1800, C_center=float(q1['C'][i1, 0]),
                    C_surf=float(q1['C'][i1, -1]), T_center=float(q1['T_num'][i1, 0]),
                    T_surf=float(q1['T_num'][i1, -1]))


if __name__ == '__main__':
    plot_const()
    plot_q3_3d()
    plot_q3_disks()
    plot_q4_disks()
    plot_q4_3d()
    plot_dashboard()
    save_json(fj, 'innov_figures.json')
    print('saved results/innov_figures.json')