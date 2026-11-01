# -*- coding: utf-8 -*-
# 收缩本构反演标定与自洽预测
import os
import sys
import time
import json
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (Chamber, props_q4, simulate, Radius, FVSolver, make_grid,
                    load_radius_data, save_json, ensure_dirs,
                    R0, C_INIT, C_TARGET)
ch = Chamber()

# 收缩本构

GRID = make_grid(h_in=2e-3, h_s=1e-4)
R_CM = np.round(np.arange(0, 20) * 0.1, 1)
R_M = R_CM / 100.0
CACHE = os.path.join('results', 'innov_const_cache.json')
FIELDS = os.path.join('results', 'innov_q4_fields.npz')
NOSHRINK = os.path.join('results', 'innov_q4noshrink.npz')


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


def rec_to_json(rec):
    return dict((k, np.asarray(v, float).tolist()) for k, v in rec.items())


# 快速配置 simulate 运行

def rec_from_json(d):
    return dict((k, np.asarray(v, float)) for k, v in d.items())


def fast_run(R_func):
    rec = dict(t=[], Cs=[], Cc=[], meanC=[], R=[], C=[], T=[], Ts=[], Tc=[])

    def on60(s):
        rec['t'].append(int(round(s.t)))
        rec['Cs'].append(float(s.C[-1]))
        rec['Cc'].append(float(s.C[0]))
        rec['meanC'].append(s.mean_C())
        rec['R'].append(float(s.R))
        rec['Ts'].append(float(s.T[-1]))
        rec['Tc'].append(float(s.T[0]))
        rec['C'].append([float(v) for v in s.at_r(R_M, 'C')])
        rec['T'].append([float(v) for v in s.at_r(R_M, 'T')])

    s, dry = simulate(props_q4(), ch, R_func=R_func, grid=GRID, t_1s_until=0.0,
                      cap_fine=2.0, cap_coarse=30.0, until_dry=True,
                      on_60s=on60, verbose=False)
    rec = dict((k, np.asarray(v, float)) for k, v in rec.items())
    return rec, s, dry


class RC:
    """可变半径闭包：求解器每步从 R_func(t) 读取当前半径"""
    def __init__(self):
        self.R = R0

    def __call__(self, t):
        return self.R


def predict_run(b):
    rc = RC()
    s = FVSolver(props_q4(), ch, grid=GRID, R_func=rc)
    knots = ch.knots()
    rec = dict(t=[0.0], Cs=[float(s.C[-1])], Cc=[float(s.C[0])],
               meanC=[s.mean_C()], R=[rc.R])
    c_prev, t_prev, R_prev = float(s.C[0]), 0.0, rc.R
    while float(s.C[0]) >= C_TARGET:
        cap = 1.0 if s.t < 10800.0 else 30.0
        next60 = (np.floor(s.t / 60.0) + 1.0) * 60.0
        kk = knots[knots > s.t + 1e-9]
        next_knot = float(kk[0]) if kk.size else np.inf
        dt = min(cap, next60 - s.t, max(next_knot - s.t, 1e-12))
        s.step(dt)
        # 本构更新：由新表面含水率驱动半径
        rc.R = R0 * np.sqrt(max(1.0 - b * (C_INIT - float(s.C[-1])), 0.05))
        if abs(s.t - next60) < 1e-9:
            rec['t'].append(s.t)
            rec['Cs'].append(float(s.C[-1]))
            rec['Cc'].append(float(s.C[0]))
            rec['meanC'].append(s.mean_C())
            rec['R'].append(rc.R)
        c_now = float(s.C[0])
        if c_now < C_TARGET:
            w = (c_prev - C_TARGET) / (c_prev - c_now)
            t_star = t_prev + w * (s.t - t_prev)
            R_star = R_prev + w * (rc.R - R_prev)
            rec = dict((k, np.asarray(v, float)) for k, v in rec.items())
            return rec, t_star, R_star
        c_prev, t_prev, R_prev = c_now, s.t, rc.R
        if s.t > 200 * 3600.0:
            raise RuntimeError('自洽预测超过 200 h 未烘干')
    raise RuntimeError('不应到达')


def run_constitutive():
    T0 = time.time()

    def log(msg):
        print('[%6.0f s] %s' % (time.time() - T0, msg), flush=True)

    t_meas, R_meas_cm = load_radius_data()
    cache = load_cache()
    out = {}

    if 'baseline' not in cache:
        log('(a) 快速配置基准（PCHIP 半径）开始……')
        rec, s, dry = fast_run(Radius('pchip'))
        t_star_base = float(dry[0])
        log('    基准 t* = %.4f h（期望 51.0870 ± 0.05）' % (t_star_base / 3600.0))
        cache['baseline'] = dict(rec=rec_to_json(rec), t_star=t_star_base,
                                 R_star=float(dry[3]))
        save_cache(cache)
    else:
        rec = rec_from_json(cache['baseline']['rec'])
        t_star_base = cache['baseline']['t_star']
        log('(a) 基准读取缓存：t* = %.4f h' % (t_star_base / 3600.0))
    base_ok = abs(t_star_base / 3600.0 - 51.0870) <= 0.05
    log('    基准验证%s' % ('通过' if base_ok else '失败（超差！）'))

    np.savez_compressed(FIELDS, t=rec['t'], r_cm=R_CM, C=np.asarray(rec['C']),
                        T=np.asarray(rec['T']), meanC=rec['meanC'], R=rec['R'],
                        Csurf=rec['Cs'], Cc=rec['Cc'], t_star=t_star_base)

    xs = rec['Cs']
    ys = (rec['R'] / R0) ** 2 - 1.0
    dC = C_INIT - xs
    b_hat = float(np.sum(dC * (-ys)) / np.sum(dC ** 2))
    resid = ys - (-b_hat * dC)
    res_rms = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    R2 = float(1.0 - np.sum(resid ** 2) / ss_tot)
    dCm = C_INIT - rec['meanC']
    bm = float(np.sum(dCm * (-ys)) / np.sum(dCm ** 2))
    resid_m = ys - (-bm * dCm)
    R2m = float(1.0 - np.sum(resid_m ** 2) / ss_tot)
    res_rms_m = float(np.sqrt(np.mean(resid_m ** 2)))
    out.update(b=b_hat, R2=R2, res_rms=res_rms, R2_mean_driven=R2m,
               b_mean_driven=bm, res_rms_mean_driven=res_rms_m,
               base_t=round(t_star_base / 3600.0, 4))
    log('(a) 标定：b = %.4f, R^2 = %.5f, 残差RMS = %.5f；平均驱动 R^2 = %.5f'
        % (b_hat, R2, res_rms, R2m))

    if 'predict' not in cache:
        log('(b) 自洽预测开始……')
        recp, t_star_p, R_star_p = predict_run(b_hat)
        cache['predict'] = dict(rec=rec_to_json(recp), t_star=t_star_p, R_star=R_star_p)
        save_cache(cache)
    else:
        recp = rec_from_json(cache['predict']['rec'])
        t_star_p, R_star_p = cache['predict']['t_star'], cache['predict']['R_star']
    Rp_at_meas = np.interp(np.minimum(t_meas, recp['t'][-1]), recp['t'], recp['R'])
    R_rmse_cm = float(np.sqrt(np.mean((Rp_at_meas * 100.0 - R_meas_cm) ** 2)))
    rel_err = (t_star_p - t_star_base) / t_star_base * 100.0
    out['predict'] = dict(t_star=round(t_star_p / 3600.0, 4),
                          rel_err_pct=round(rel_err, 3),
                          R_rmse_cm=round(R_rmse_cm, 4),
                          R_final_cm=round(R_star_p * 100.0, 4),
                          R_final_meas_cm=1.198,
                          note='t* 由逐 60 s 步（步长上限 1 s/30 s）间线性插值，分辨率 <= 30 s')
    log('(b) 自洽预测 t* = %.4f h（偏差 %+.3f%%），R(t) RMSE = %.4f cm，终点 R = %.4f cm'
        % (t_star_p / 3600.0, rel_err, R_rmse_cm, R_star_p * 100.0))

    meanC_end = float(rec['meanC'][-1])
    u = np.clip((C_INIT - rec['meanC']) / (C_INIT - meanC_end), 0.0, 1.0)
    beta = 1.198 / 2.000
    path_sens = {}
    for key, pw in (('p05', 0.5), ('p10', 1.0), ('p20', 2.0)):
        ck = 'path_' + key
        if ck not in cache:
            log('(c) 路径 p=%.1f 开始……' % pw)
            Rp = R0 * np.sqrt(1.0 - (1.0 - beta ** 2) * u ** pw)
            tgrid = rec['t'].copy()
            Rf = lambda tt, _t=tgrid, _R=Rp: np.interp(tt, _t, _R,
                                                       left=_R[0], right=_R[-1])
            _, _, dryp = fast_run(Rf)
            cache[ck] = float(dryp[0])
            save_cache(cache)
        tp = cache[ck]
        path_sens[key] = dict(t_star=round(tp / 3600.0, 4),
                              diff_h=round((tp - t_star_base) / 3600.0, 4),
                              diff_pct=round((tp - t_star_base) / t_star_base * 100.0, 3))
        log('(c) p=%.1f: t* = %.4f h（相对基准 %+.4f h）' % (pw, tp / 3600.0,
                                                            (tp - t_star_base) / 3600.0))
    out['path_sens'] = path_sens
    out['beta'] = beta
    cache['u_grid'] = dict(t=rec['t'].tolist(), u=u.tolist())
    save_cache(cache)

    if not os.path.exists(NOSHRINK):
        log('不收缩对照（R=2 cm 固定，附录4物性）开始……')
        recn, _, dryn = fast_run(lambda t: R0)
        np.savez_compressed(NOSHRINK, t=recn['t'], Cc=recn['Cc'], meanC=recn['meanC'],
                            Cs=recn['Cs'], t_star=float(dryn[0]))
        log('    不收缩 t* = %.4f h' % (dryn[0] / 3600.0))
    else:
        d = np.load(NOSHRINK)
        log('不收缩对照已存在：t* = %.4f h' % (float(d['t_star']) / 3600.0))

    save_json(out, 'innov_constitutive.json')
    log('本构完成，结果写入 results/innov_constitutive.json')



if __name__ == '__main__':
    ensure_dirs()
    run_constitutive()