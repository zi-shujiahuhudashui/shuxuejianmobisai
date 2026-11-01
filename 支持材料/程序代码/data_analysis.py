# -*- coding: utf-8 -*-
# 烘房工况阶段划分与统计检验
import os
import time
import numpy as np
from common import (Chamber, stage_analysis, load_chamber_data, save_json, ensure_dirs, setup_plot,
                    FIG_DIR, T_SET, C_SET)


def adf_report(y):
    from statsmodels.tsa.stattools import adfuller
    stat, pval, nlag, nobs, crit, _ = adfuller(y, autolag='AIC')
    return dict(stat=float(stat), pvalue=float(pval), nlag=int(nlag), nobs=int(nobs),
                crit_1pct=float(crit['1%']), crit_5pct=float(crit['5%']),
                stationary_5pct=bool(stat < crit['5%']))


def series_report(t, y, name, unit):
    n = y.size
    h = n // 2
    y1, y2 = y[:h], y[h:]
    rep = dict(name=name, unit=unit, n=int(n),
               t_range=[float(t[0]), float(t[-1])],
               mean=float(y.mean()), std=float(y.std(ddof=1)),
               half1=dict(n=int(h), mean=float(y1.mean()), std=float(y1.std(ddof=1))),
               half2=dict(n=int(n - h), mean=float(y2.mean()), std=float(y2.std(ddof=1))),
               half_mean_diff=float(y2.mean() - y1.mean()),
               adf=adf_report(y))
    print('%s：n=%d, 均值=%.6g %s, 标准差=%.6g' % (name, n, rep['mean'], unit, rep['std']),
          flush=True)
    print('  前半段均值=%.6g, 后半段均值=%.6g, 均值差=%.6g %s'
          % (rep['half1']['mean'], rep['half2']['mean'], rep['half_mean_diff'], unit), flush=True)
    a = rep['adf']
    print('  ADF: stat=%.4f, p=%.4g, crit1%%=%.4f, crit5%%=%.4f -> %s'
          % (a['stat'], a['pvalue'], a['crit_1pct'], a['crit_5pct'],
             '平稳(5%)' if a['stationary_5pct'] else '不平稳(5%)'), flush=True)
    return rep


def stationarity():
    T_SPLIT = 9000.0
    t0 = time.time()
    t, Ta, Ca = load_chamber_data()
    m = t >= T_SPLIT
    ts, Ta_s, Ca_s = t[m], Ta[m], Ca[m]
    out = dict(t_split=T_SPLIT,
               temperature=series_report(ts, Ta_s, '烘房温度', '°C'),
               moisture=series_report(ts, Ca_s, '烘房水分浓度', 'kg/kg'),
               elapsed=time.time() - t0)
    save_json(out, 'innov_stationarity.json')
    print('已写出 results/innov_stationarity.json，用时 %.1f s' % (time.time() - t0), flush=True)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ensure_dirs()
    st = stage_analysis()
    t, Ta, Ca = load_chamber_data()
    ch = Chamber()
    ts = st['t_switch']
    summary = dict(t_switch=ts, T=dict((k, v) for k, v in st['T'].items() if k != 'table'),
                   C=dict((k, v) for k, v in st['C'].items() if k != 'table'),
                   T_table=st['T']['table'], C_table=st['C']['table'],
                   n_points=int(t.size), dt_data=float(t[1] - t[0]), t_end=float(t[-1]),
                   T_last=float(Ta[-1]), C_last=float(Ca[-1]))
    save_json(summary, 'data_summary.json')
    plt = setup_plot()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    tt = np.linspace(0, 6 * 3600, 3000)
    for a, y, ys, lab, key in ((ax[0], Ta, T_SET, '温度 $T_a$ / °C', 'T'),
                               (ax[1], Ca, C_SET, '水分浓度 $C_a$ / (kg/kg)', 'C')):
        a.plot(t / 3600.0, y, 'o', ms=2.2, color='#1f4e79', alpha=0.8, label='附件1实测数据')
        f = ch.Ta if key == 'T' else ch.Ca
        a.plot(tt / 3600.0, f(tt), '-', color='#c00000', lw=1.0, label='工况函数（插值+设定值）')
        sd = st[key]['sd']
        a.axhspan(ys - 2 * sd, ys + 2 * sd, color='#70ad47', alpha=0.18, label='设定值 ±2σ')
        a.axvline(ts / 3600.0, color='gray', ls='--', lw=0.9)
        a.set_xlabel('时间 t / h')
        a.set_ylabel(lab)
        a.set_xlim(0, 6)
        a.grid(alpha=0.3)
    ax[0].text(ts / 3600.0 + 0.08, 42.5, '预热平衡 | 恒温干燥\n(t = %.1f h)' % (ts / 3600.0), fontsize=8)
    ax[0].set_title('(a) 烘房温度')
    ax[1].set_title('(b) 烘房水分浓度')
    ax[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'data_chamber.png'))
    plt.close(fig)
    stationarity()
    print('数据分析完成：预热平衡阶段结束时刻 %.0f s' % ts)
    return summary


if __name__ == '__main__':
    main()
