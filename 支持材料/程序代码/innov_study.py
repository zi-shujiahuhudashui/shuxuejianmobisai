# -*- coding: utf-8 -*-
# MC + 扩展灵敏度
import os
import sys
import time
import json
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from common import (Chamber, Props, props_q23, simulate, make_grid, save_json, ensure_dirs,
                    setup_plot, FIG_DIR, T_SET, C_SET, H_HEAT, H_MASS)

BASE_EXPECT = 57.4723
TOL = 0.005

# MC

SIGMA_T = 0.1564
SIGMA_C = 1.543e-4
N_MC = 60
SEED = 2026

MC_GRID = make_grid(h_in=2e-3, h_s=1e-4)
MC_CACHE = os.path.join('results', 'innov_mc_cache.json')


class MCChamber(Chamber):
    def __init__(self, Ts, Cs):
        super().__init__(T_set=Ts, C_set=Cs)


def mc_dry_time(Ts, Cs):
    ch = MCChamber(Ts, Cs)
    _, dry = simulate(props_q23(), ch, grid=MC_GRID, t_1s_until=0.0, cap_fine=2.0,
                      cap_coarse=30.0, until_dry=True, verbose=False)
    return dry[0] / 3600.0


def load_mc_cache():
    if os.path.exists(MC_CACHE):
        with open(MC_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_mc_cache(data):
    # 原子写入，失败重试
    ensure_dirs()
    import tempfile
    for attempt in range(10):
        try:
            fd, tmp = tempfile.mkstemp(dir='results', prefix='.mc_', suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, MC_CACHE)
            return
        except Exception as e:
            print('MC 缓存写入重试 %d：%r' % (attempt + 1, e), flush=True)
            time.sleep(0.5)
    raise IOError('MC 缓存写入失败')


def run_mc():
    ensure_dirs()
    t0 = time.time()
    cache = load_mc_cache()
    # 基准验证
    if 'base' not in cache:
        base = mc_dry_time(T_SET, C_SET)
        cache['base'] = base
        save_mc_cache(cache)
        print('基准验证：t* = %.6f h（期望 %.4f ± %.3f）' % (base, BASE_EXPECT, TOL), flush=True)
        if abs(base - BASE_EXPECT) > TOL:
            print('基准验证失败，跳过 MC。', flush=True)
            return
    base = cache['base']
    if abs(base - BASE_EXPECT) > TOL:
        print('缓存基准 %.6f h 与期望不符，跳过 MC。' % base, flush=True)
        return
    print('基准验证通过：t* = %.6f h' % base, flush=True)
    # 蒙特卡洛采样
    rng = np.random.default_rng(SEED)
    Ts_all = T_SET + rng.normal(0.0, SIGMA_T, N_MC)
    Cs_all = np.maximum(C_SET + rng.normal(0.0, SIGMA_C, N_MC), 0.0)
    samples = cache.get('samples', {})
    for i in range(N_MC):
        key = '%03d' % i
        if key in samples:
            continue
        ti = time.time()
        ts_i, cs_i = float(Ts_all[i]), float(Cs_all[i])
        try:
            tstar = mc_dry_time(ts_i, cs_i)
            samples[key] = dict(T_set=ts_i, C_set=cs_i, t_star=tstar)
            print('样本 %2d/%d：T_set=%.4f, C_set=%.6f -> t*=%.4f h（%.0f s）'
                  % (i + 1, N_MC, ts_i, cs_i, tstar, time.time() - ti), flush=True)
        except Exception as e:
            samples[key] = dict(T_set=ts_i, C_set=cs_i, error=repr(e))
            print('样本 %2d/%d 失败：%r' % (i + 1, N_MC, e), flush=True)
        cache['samples'] = samples
        save_mc_cache(cache)

    vals = np.array([samples['%03d' % i]['t_star'] for i in range(N_MC)
                     if 't_star' in samples.get('%03d' % i, {})])
    errs = {k: v for k, v in samples.items() if 'error' in v}
    n = vals.size
    mean, std = float(vals.mean()), float(vals.std(ddof=1))
    ci95 = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    out = dict(base=base, samples=[float(v) for v in vals], mean=mean, std=std,
               ci95=ci95, n=int(n), seed=SEED, sigma_T=SIGMA_T, sigma_C=SIGMA_C,
               n_error=len(errs), errors=errs, elapsed=time.time() - t0)
    save_json(out, 'innov_mc.json')
    print('MC 完成：n=%d, mean=%.4f h, std=%.4f h, CI95=[%.4f, %.4f] h'
          % (n, mean, std, ci95[0], ci95[1]), flush=True)
    #  图
    from scipy.stats import gaussian_kde
    plt = setup_plot()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.hist(vals, bins='auto', density=True, color='#9dc3e6', edgecolor='#1f4e79',
            alpha=0.75, label='样本直方图（N=%d）' % n, zorder=2)
    xs = np.linspace(min(vals.min(), base) - 0.15, max(vals.max(), base) + 0.15, 400)
    kde = gaussian_kde(vals)
    ax.plot(xs, kde(xs), color='#c00000', lw=1.6, label='KDE 密度曲线', zorder=4)
    ax.axvspan(ci95[0], ci95[1], color='#ffd966', alpha=0.35, zorder=1,
               label='95%% 经验不确定性区间 [%.3f, %.3f] h' % (ci95[0], ci95[1]))
    ax.axvline(mean, color='#1f4e79', lw=1.6, label='均值 %.4f h' % mean, zorder=5)
    for v, lab in ((mean - std, '均值±标准差'), (mean + std, None)):
        ax.axvline(v, color='#1f4e79', lw=1.1, ls=':', label=lab, zorder=5)
    ax.axvline(base, color='#548235', lw=1.4, ls='--', label='基准值 %.4f h' % base, zorder=5)
    ax.set_xlabel('烘干时间 t* (h)')
    ax.set_ylabel('概率密度')
    ax.set_title('蒙特卡洛不确定性量化：t* = %.4f ± %.4f h（N=%d）' % (mean, std, n))
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'innov_mc.png'))
    plt.close(fig)
    print('图已保存 figures/innov_mc.png；MC 用时 %.0f s' % (time.time() - t0), flush=True)



# 扩展灵敏度

SENS_GRID = make_grid(h_in=2e-3, h_s=1e-4)
SENS_CACHE = os.path.join('results', 'innov_sens_ext_cache.json')


def props_ext(D0=2.4e-3, a=0.45, E=3850.0):
    return Props('Q23ext', D0, a, E,
                 rho=lambda C: 650.0 + 128.0 * C,
                 cp=lambda C: 1450.0 + 2736.0 * C / (C + 1.0),
                 k=lambda C: 0.21 + 0.38 * C / (C + 1.0))


def sens_dry_time(props=None, solver_kw=None):
    props = props if props is not None else props_ext()
    _, dry = simulate(props, Chamber(), grid=SENS_GRID, t_1s_until=0.0, cap_fine=2.0,
                      cap_coarse=30.0, until_dry=True, verbose=False, solver_kw=solver_kw)
    return dry[0] / 3600.0


def load_sens_cache():
    if os.path.exists(SENS_CACHE):
        with open(SENS_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_sens_cache(data):
    # 原子写入，失败重试
    ensure_dirs()
    import tempfile
    for attempt in range(10):
        try:
            fd, tmp = tempfile.mkstemp(dir='results', prefix='.sens_', suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, SENS_CACHE)
            return
        except Exception as e:
            print('sens 缓存写入重试 %d：%r' % (attempt + 1, e), flush=True)
            time.sleep(0.5)
    raise IOError('sens 缓存写入失败')


def run_sens_ext():
    ensure_dirs()
    t0 = time.time()
    cache = load_sens_cache()

    def cached(key, fn):
        if key not in cache:
            ti = time.time()
            try:
                cache[key] = fn()
                print('%-26s -> %.4f h（%.0f s）' % (key, cache[key], time.time() - ti),
                      flush=True)
            except Exception as e:
                cache[key] = dict(error=repr(e))
                print('%-26s 失败：%r' % (key, e), flush=True)
            save_sens_cache(cache)
        elif isinstance(cache[key], dict):
            print('%-26s 此前失败：%r' % (key, cache[key].get('error')), flush=True)
        return cache[key]

    base = cached('base', sens_dry_time)
    print('基准 t* = %.6f h（期望 %.4f ± %.3f）' % (base, BASE_EXPECT, TOL), flush=True)
    if abs(base - BASE_EXPECT) > TOL:
        print('警告：基准与已验证值偏差超过容差。', flush=True)

    specs = [
        # (显示名, 参数名, lo值, hi值, lo算例函数, hi算例函数)
        ('Arrhenius指数 E', 'E', 3850.0 * 0.9, 3850.0 * 1.1,
         lambda: sens_dry_time(props=props_ext(E=3850.0 * 0.9)),
         lambda: sens_dry_time(props=props_ext(E=3850.0 * 1.1))),
        ('浓度指数 a', 'a', 0.45 * 0.9, 0.45 * 1.1,
         lambda: sens_dry_time(props=props_ext(a=0.45 * 0.9)),
         lambda: sens_dry_time(props=props_ext(a=0.45 * 1.1))),
        ('指前因子 D0', 'D0', 2.4e-3 * 0.9, 2.4e-3 * 1.1,
         lambda: sens_dry_time(props=props_ext(D0=2.4e-3 * 0.9)),
         lambda: sens_dry_time(props=props_ext(D0=2.4e-3 * 1.1))),
        ('对流传质系数 h_m', 'hm', H_MASS * 0.8, H_MASS * 1.2,
         lambda: sens_dry_time(solver_kw=dict(hm=H_MASS * 0.8)),
         lambda: sens_dry_time(solver_kw=dict(hm=H_MASS * 1.2))),
        ('对流换热系数 h', 'h', H_HEAT * 0.8, H_HEAT * 1.2,
         lambda: sens_dry_time(solver_kw=dict(h=H_HEAT * 0.8)),
         lambda: sens_dry_time(solver_kw=dict(h=H_HEAT * 1.2))),
    ]
    cases, failed = [], []
    for name, param, plo, phi, flo, fhi in specs:
        t_lo = cached('%s lo' % param, flo)
        t_hi = cached('%s hi' % param, fhi)
        if isinstance(t_lo, dict) or isinstance(t_hi, dict):
            failed.append(name)
            continue
        cases.append(dict(name=name, param=param, lo=float(plo), hi=float(phi),
                          t_lo=float(t_lo), t_hi=float(t_hi),
                          rel_lo=float((t_lo - base) / base),
                          rel_hi=float((t_hi - base) / base)))
        print('%s：lo=%.4f h (%+.2f%%)  hi=%.4f h (%+.2f%%)'
              % (name, t_lo, 100 * (t_lo - base) / base, t_hi, 100 * (t_hi - base) / base),
              flush=True)

    out = dict(base=float(base), cases=cases, failed=failed, elapsed=time.time() - t0)
    save_json(out, 'innov_sens_ext.json')
    #  tornado 图
    plt = setup_plot()
    order = np.argsort([abs(c['rel_hi'] - c['rel_lo']) for c in cases])[::-1]
    cs = [cases[i] for i in order]
    names = [c['name'] for c in cs]
    y = np.arange(len(cs))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for yi, c in zip(y, cs):
        ax.barh(yi, c['t_lo'] - base, left=base, color='#2e75b6', height=0.62, zorder=3,
                label='负扰动' if yi == y[0] else None)
        ax.barh(yi, c['t_hi'] - base, left=base, color='#c00000', height=0.62, zorder=3,
                label='正扰动' if yi == y[0] else None)
        ax.annotate('%.2f' % c['t_lo'], (c['t_lo'], yi), textcoords='offset points',
                    xytext=(-4 if c['t_lo'] < base else 4, 0),
                    ha='right' if c['t_lo'] < base else 'left', va='center', fontsize=8.5)
        ax.annotate('%.2f' % c['t_hi'], (c['t_hi'], yi), textcoords='offset points',
                    xytext=(-4 if c['t_hi'] < base else 4, 0),
                    ha='right' if c['t_hi'] < base else 'left', va='center', fontsize=8.5)
    ax.axvline(base, color='k', lw=1.1, ls='--', zorder=4)
    ax.annotate('基准 %.2f h' % base, xy=(base, 0.0), xycoords=('data', 'axes fraction'),
                textcoords='offset points', xytext=(5, 3), fontsize=9, ha='left',
                va='bottom', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel('烘干时间 t* (h)')
    ax.set_title('扩散系数结构参数与对流系数灵敏度（tornado 图）')
    ax.legend(frameon=False, loc='lower right')
    ax.grid(alpha=0.3, axis='x', zorder=0)
    allt = [v for c in cs for v in (c['t_lo'], c['t_hi'])] + [base]
    ax.set_xlim(min(allt) - 0.11 * (max(allt) - min(allt)),
                max(allt) + 0.07 * (max(allt) - min(allt)))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'innov_tornado_ext.png'))
    plt.close(fig)
    print('图已保存 figures/innov_tornado_ext.png；sens 用时 %.0f s' % (time.time() - t0), flush=True)
    if failed:
        print('失败算例：%s' % failed, flush=True)


if __name__ == '__main__':
    ensure_dirs()
    run_mc()
    run_sens_ext()