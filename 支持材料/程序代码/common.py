# -*- coding: utf-8 -*-
# 公共模块：数据读取、工况、物性、求解器、解析解、出图
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.special import exp1, j0, j1, jn_zeros
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

# 0. 常数与目录（相对路径）
 
R0 = 0.02          # 初始半径 / m
L0 = 0.25          # 药材长度 / m
T_INIT = 28.0      # 初始温度 / °C
C_INIT = 2.55      # 初始水分浓度（干基含水率）/ (kg/kg)
H_HEAT = 25.0      # 对流换热系数 / (W/(m^2·K))
H_MASS = 8e-7      # 对流传质系数 / (m/s)
KELVIN = 273.15
C_TARGET = 0.15    # 烘干要求：各处水分浓度低于 0.15 kg/kg
T_SET = 50.0       # 恒温干燥阶段烘房温度设定值 / °C（由附件1检验得到）
C_SET = 0.05       # 恒温干燥阶段烘房水分浓度设定值 / (kg/kg)

OUT_DIR = 'results'
FIG_DIR = 'figures'


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def find_file(keyword, ext='.xlsx'):
    """
    在项目目录下找含 keyword 的 Excel；跳过 results/ 和临时文件。
    """
    cands = []
    for pat in ('*%s*%s' % (keyword, ext), '*/*%s*%s' % (keyword, ext),
                '*/*/*%s*%s' % (keyword, ext)):
        for p in glob.glob(pat):
            parts = os.path.normpath(p).split(os.sep)
            if parts[0] == OUT_DIR or os.path.basename(p).startswith('~$'):
                continue
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda s: (s.count(os.sep), len(s)))
    return cands[0]


def save_json(obj, name):
    ensure_dirs()
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(name):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 1. 数据读取

def load_chamber_data():
    """
    附件1：时间(s)、烘房温度(°C)、烘房水分浓度(kg/kg)
    """
    path = find_file('附件1')
    if path is None:
        raise FileNotFoundError('未找到“附件1.xlsx”，请将其与程序放在同一目录')
    df = pd.read_excel(path, engine='openpyxl')
    a = df.iloc[:, :3].to_numpy(dtype=float)
    return a[:, 0], a[:, 1], a[:, 2]


def load_radius_data():
    """附件2：时间(s)、药材半径(cm)"""
    path = find_file('附件2')
    if path is None:
        raise FileNotFoundError('未找到“附件2.xlsx”，请将其与程序放在同一目录')
    df = pd.read_excel(path, engine='openpyxl')
    a = df.iloc[:, :2].to_numpy(dtype=float)
    return a[:, 0], a[:, 1]


# 2. 烘房边界工况
 
class Chamber:
    """
    烘房温湿度：0~14400 s 用附件1插值，14460 s 后取设定值，中间线性过渡。
    smooth=True 时用 Weibull 拟合曲线（只给噪声检验用）。
    """

    def __init__(self, T_set=T_SET, C_set=C_SET, smooth=False):
        t, Ta, Ca = load_chamber_data()
        self.t_data, self.Ta_data, self.Ca_data = t, Ta, Ca
        self.t_end = float(t[-1])
        self.T_set, self.C_set = float(T_set), float(C_set)
        self.smooth = smooth
        if smooth:
            from scipy.optimize import curve_fit
            f = lambda tt, yinf, tau, n, y0: yinf - (yinf - y0) * np.exp(-(tt / tau) ** n)
            self.pT, _ = curve_fit(lambda tt, a, b, c: f(tt, a, b, c, Ta[0]), t, Ta,
                                   p0=[50.0, 2000.0, 1.0], maxfev=20000)
            self.pC, _ = curve_fit(lambda tt, a, b, c: f(tt, a, b, c, Ca[0]), t, Ca,
                                   p0=[0.05, 2500.0, 1.0], maxfev=20000)
            self._f = f
        self.tk = np.append(t, self.t_end + 60.0)
        self.Tk = np.append(Ta, self.T_set)
        self.Ck = np.append(Ca, self.C_set)

    def Ta(self, tt):
        if self.smooth:
            return self._f(np.asarray(tt, float), *self.pT, self.Ta_data[0])
        return np.interp(tt, self.tk, self.Tk)

    def Ca(self, tt):
        if self.smooth:
            return self._f(np.asarray(tt, float), *self.pC, self.Ca_data[0])
        return np.interp(tt, self.tk, self.Ck)

    def knots(self):
        """工况函数的折点（数值积分须落在这些时刻上）"""
        return self.tk.copy()


def stage_analysis(alpha=0.05):
    """
    找最早时刻 ts 使 [ts, 14400] 内斜率不显著、均值与设定值无显著差异；
    温度和浓度分别做，取大者为预热平衡结束时刻。
    """
    from scipy import stats
    t, Ta, Ca = load_chamber_data()
    out = {}
    for name, y, ys in (('T', Ta, T_SET), ('C', Ca, C_SET)):
        ts_found = None
        rows = []
        for ts in np.arange(600.0, t[-1] - 3600.0 + 1.0, 600.0):
            m = t >= ts
            lr = stats.linregress(t[m], y[m])
            pm = stats.ttest_1samp(y[m], ys).pvalue
            rows.append((float(ts), float(lr.pvalue), float(pm)))
            if ts_found is None and lr.pvalue > alpha and pm > alpha:
                # 要求此后所有候选时刻均满足（稳健性）
                ok_after = True
                for ts2 in np.arange(ts, t[-1] - 3600.0 + 1.0, 600.0):
                    m2 = t >= ts2
                    if (stats.linregress(t[m2], y[m2]).pvalue <= alpha or
                            stats.ttest_1samp(y[m2], ys).pvalue <= alpha):
                        ok_after = False
                        break
                if ok_after:
                    ts_found = float(ts)
        m = t >= ts_found
        mean = float(y[m].mean())
        sd = float(y[m].std(ddof=1))
        n = int(m.sum())
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
        out[name] = dict(ts=ts_found, mean=mean, sd=sd, n=n, ci95=[float(ci[0]), float(ci[1])],
                         p_mean=float(stats.ttest_1samp(y[m], ys).pvalue),
                         p_slope=float(stats.linregress(t[m], y[m]).pvalue), table=rows)
    out['t_switch'] = max(out['T']['ts'], out['C']['ts'])
    return out


# 3. 物性模型
 
class Props:
    """
    D(C,T) = D0·exp(-a/C)·exp(-E/T_K)；E=0 时与温度无关。
    Kirchhoff 变换 G(C) = C·exp(-a/C) - a·E1(a/C)，G'(C) = exp(-a/C)。
    """

    def __init__(self, name, D0, a, E, rho, cp, k):
        self.name, self.D0, self.a, self.E = name, D0, a, E
        self._rho, self._cp, self._k = rho, cp, k

    def rho(self, C):
        return self._rho(np.asarray(C, float))

    def cp(self, C):
        return self._cp(np.asarray(C, float))

    def rho_cp(self, C):
        C = np.asarray(C, float)
        return self._rho(C) * self._cp(C)

    def k(self, C):
        return self._k(np.asarray(C, float))

    def w(self, T_C):
        """温度因子 D0·exp(-E/T_K)"""
        if self.E == 0.0:
            return self.D0 * np.ones_like(np.asarray(T_C, float))
        return self.D0 * np.exp(-self.E / (np.asarray(T_C, float) + KELVIN))

    def g(self, C):
        return np.exp(-self.a / C)

    def G(self, C):
        a = self.a
        return C * np.exp(-a / C) - a * exp1(a / C)

    def D(self, C, T_C):
        return self.w(T_C) * self.g(np.asarray(C, float))


def props_q1(D_scale=1.0):
    """附录2：ρ=820，cp=2600，k=0.36，D=7e-9·exp(-0.89/C)"""
    return Props('Q1', 7e-9 * D_scale, 0.89, 0.0,
                 rho=lambda C: np.full_like(C, 820.0),
                 cp=lambda C: np.full_like(C, 2600.0),
                 k=lambda C: np.full_like(C, 0.36))


def props_q23(D_scale=1.0):
    """附录3：问题2、问题3"""
    return Props('Q23', 2.4e-3 * D_scale, 0.45, 3850.0,
                 rho=lambda C: 650.0 + 128.0 * C,
                 cp=lambda C: 1450.0 + 2736.0 * C / (C + 1.0),
                 k=lambda C: 0.21 + 0.38 * C / (C + 1.0))


def props_q4(D_scale=1.0):
    """附录4：问题4"""
    return Props('Q4', 4.2e-4 * D_scale, 0.30, 3850.0,
                 rho=lambda C: 760.0 + 90.0 * C,
                 cp=lambda C: 1850.0 + 2150.0 * C / (C + 1.0),
                 k=lambda C: 0.12 + 0.20 * C / (C + 1.0))


# 4. 药材半径函数（问题4）
 
class Radius:
    """
    附件2 构造 R(t)（m）。pchip=主模型，linear/fit 仅灵敏度用。
    t 超出数据范围取末值。
    """

    def __init__(self, kind='pchip'):
        t, R_cm = load_radius_data()
        self.t_data, self.R_data = t, R_cm / 100.0
        self.kind = kind
        self.t_last = float(t[-1])
        if kind == 'pchip':
            self._f = PchipInterpolator(t, self.R_data)
        elif kind == 'fit':
            from scipy.optimize import curve_fit
            f2 = lambda tt, Ri, A1, t1, A2, t2: Ri + A1 * np.exp(-tt / t1) + A2 * np.exp(-tt / t2)
            p, _ = curve_fit(f2, t, R_cm, p0=[1.2, 0.2, 4000.0, 0.6, 17000.0], maxfev=50000)
            self.p_fit = p
            self._f = lambda tt: f2(np.asarray(tt, float), *p) / 100.0
        elif kind != 'linear':
            raise ValueError(kind)

    def __call__(self, tt):
        tt = np.asarray(tt, float)
        tc = np.minimum(tt, self.t_last)
        if self.kind == 'linear':
            v = np.interp(tc, self.t_data, self.R_data)
        else:
            v = self._f(tc)
        if self.kind == 'fit':
            v = v * (R0 / (self._f(0.0)))   # 保证 R(0)=2 cm
        return v


# 5. 数值求解器
 
def make_grid(h_in=5e-4, h_s=2.5e-5, seg=0.05, n_marks=20):
    """
    ξ∈[0,1]：内部均匀 h_in，表面段几何加密到 h_s，段长 seg。
    保证 ξ=j/n_marks 都在节点上。默认 R=2 cm 时 Δr 内部 10 μm、表面 0.5 μm。
    """
    n_in = int(round((1.0 - seg) / h_in))
    xi_in = np.linspace(0.0, 1.0 - seg, n_in + 1)
    # 末段：自表面向内，单元尺寸 h_s·q^k (k=0..m-1)，要求末个单元约等于 h_in
    q_guess = 1.0 + (h_in - h_s) / (seg - h_in)
    m = int(round(1 + np.log(h_in / h_s) / np.log(q_guess)))
    f = lambda q: h_s * (q ** m - 1.0) / (q - 1.0) - seg
    q = brentq(f, 1.0 + 1e-12, 2.0)
    sizes = h_s * q ** np.arange(m)          # 表面→内部
    edges = 1.0 - np.concatenate(([0.0], np.cumsum(sizes)))[::-1]
    edges[0] = 1.0 - seg
    xi = np.concatenate((xi_in[:-1], edges))
    xi[-1] = 1.0
    marks = np.arange(n_marks + 1) / n_marks if n_marks else np.array([0.0, 1.0])
    for mk in marks:
        assert np.min(np.abs(xi - mk)) < 1e-12, '输出位置必须为网格节点'
        xi[np.argmin(np.abs(xi - mk))] = mk
    return xi


def uniform_grid(N):
    return np.linspace(0.0, 1.0, int(N) + 1)


class FVSolver:
    """
    ξ = r/R(t) ∈ [0,1] 上的顶点型有限体积。
    控制体 V_i = ξ_i·Δξ（端点 1/4、1/2 修正），界面系数 A = ξ_{i+1/2}/Δξ。
    半离散：
      (ρc_p)_i V_i T_i' = [A k ΔT]/R$^2$ − δ_iN (h/R)(T_N − T_a)
      V_i C_i'          = [A w ΔG]/R$^2$ − δ_iN (h_m/R)(C_N − C_a)
    w = D0 exp(−E/T_f)，G 为 Kirchhoff 变换。
    时间：变步长 BDF2（首步 BDF1）；传质 Newton，与传热 Picard 耦合。
    """
    def __init__(self, props, chamber, grid=None, R_func=None, h=H_HEAT, hm=H_MASS,
                 T0=T_INIT, C0=C_INIT, coupled=True, tol_T=1e-10, tol_C=1e-12):
        self.p, self.ch = props, chamber
        self.R_func = R_func if R_func is not None else (lambda t: R0)
        self.h, self.hm = float(h), float(hm)
        self.coupled = coupled
        self.tol_T, self.tol_C = tol_T, tol_C
        xi = make_grid() if grid is None else np.asarray(grid, float)
        N = xi.size - 1
        self.N = N
        xf = 0.5 * (xi[:-1] + xi[1:])                       # 界面位置 ξ_{i+1/2}
        xb = np.concatenate(([0.0], xf, [1.0]))             # 控制体边界
        self.V = 0.5 * (xb[1:] ** 2 - xb[:-1] ** 2)          # 控制体“体积”（Σ V_i = 1/2）
        self.A = xf / np.diff(xi)                            # 界面系数 ξ_{i+1/2}/(ξ_{i+1}-ξ_i)
        self.xi = xi
        self.t = 0.0
        self.T = np.full(N + 1, float(T0))
        self.C = np.full(N + 1, float(C0))
        self.T_old = None
        self.C_old = None
        self.dt_old = None
        self.R = float(self.R_func(0.0))
        self.flux_int = 0.0        # ∫ (h_m/R)(C_N − C_a) dt，用于水分守恒检验
        self.heat_int = 0.0        # ∫ (h/R)(T_N − T_a) dt，用于能量守恒检验（常物性）
        self.iters = []
        self._ab = np.zeros((3, N + 1))


    #  三对角求解（对称：上、下对角均为 -g）

    def _solve_sym(self, diag, off, rhs):
        ab = self._ab
        ab[0, 0] = 0.0
        ab[0, 1:] = off
        ab[1] = diag
        ab[2, :-1] = off
        ab[2, -1] = 0.0
        return solve_banded((1, 1), ab, rhs, check_finite=False)

    def _solve_gen(self, lower, diag, upper, rhs):
        ab = self._ab
        ab[0, 0] = 0.0
        ab[0, 1:] = upper
        ab[1] = diag
        ab[2, :-1] = lower
        ab[2, -1] = 0.0
        return solve_banded((1, 1), ab, rhs, check_finite=False)

    def step(self, dt, first_order=False):
        """
        推进一个时间步 dt（BDF2；首步或 first_order=True 时为 BDF1）
        """
        t_new = self.t + dt
        R = float(self.R_func(t_new))
        Ta = float(self.ch.Ta(t_new))
        Ca = float(self.ch.Ca(t_new))
        if self.T_old is None or first_order:
            a0, a1, a2 = 1.0, -1.0, 0.0
            Tg, Cg = self.T.copy(), self.C.copy()
        else:
            om = dt / self.dt_old
            a0 = (1.0 + 2.0 * om) / (1.0 + om)
            a1 = -(1.0 + om)
            a2 = om * om / (1.0 + om)
            Tg = self.T + om * (self.T - self.T_old)
            Cg = self.C + om * (self.C - self.C_old)
        histT = a1 * self.T + (a2 * self.T_old if a2 != 0.0 else 0.0)
        histC = a1 * self.C + (a2 * self.C_old if a2 != 0.0 else 0.0)
        V, A, iR2 = self.V, self.A, 1.0 / (R * R)
        p = self.p
        Ti = Tg
        Ci = np.maximum(Cg, 1e-6)
        n_outer = 0
        for n_outer in range(1, 51):
            #  传热方程（给定 C 时为线性） ------
            rc = p.rho_cp(Ci)
            kn = p.k(Ci)
            gT = A * 0.5 * (kn[:-1] + kn[1:]) * iR2
            diag = rc * V * (a0 / dt)
            diag[:-1] += gT
            diag[1:] += gT
            diag[-1] += self.h / R
            rhs = -rc * V * histT / dt
            rhs[-1] += self.h / R * Ta
            T_new = self._solve_sym(diag, -gT, rhs)
            #  传质方程（Newton 迭代） ------
            wf = p.w(0.5 * (T_new[:-1] + T_new[1:])) if p.E != 0.0 else p.D0 * np.ones(self.N)
            cf = A * wf * iR2
            Cn = Ci.copy()
            for _ in range(30):
                Gn = p.G(Cn)
                gn = p.g(Cn)
                flux = cf * (Gn[1:] - Gn[:-1])          # 第 i+1/2 界面向内（+ξ→-ξ）的通量
                F = V * (a0 * Cn + histC) / dt
                F[:-1] -= flux
                F[1:] += flux
                F[-1] += self.hm / R * (Cn[-1] - Ca)
                diag = V * (a0 / dt)
                diag[:-1] += cf * gn[:-1]
                diag[1:] += cf * gn[1:]
                diag[-1] += self.hm / R
                upper = -cf * gn[1:]
                lower = -cf * gn[:-1]
                dC = self._solve_gen(lower, diag, upper, -F)
                Cn = Cn + dC
                if np.max(np.abs(dC)) < self.tol_C:
                    break
            eT = float(np.max(np.abs(T_new - Ti)))
            eC = float(np.max(np.abs(Cn - Ci)))
            Ti, Ci = T_new, Cn
            if not self.coupled:
                break
            if eT < self.tol_T and eC < self.tol_C:
                break
        # 守恒量累计（梯形公式）
        qm_old = self.hm / self.R * (self.C[-1] - float(self.ch.Ca(self.t)))
        qm_new = self.hm / R * (Ci[-1] - Ca)
        self.flux_int += 0.5 * (qm_old + qm_new) * dt
        qh_old = self.h / self.R * (self.T[-1] - float(self.ch.Ta(self.t)))
        qh_new = self.h / R * (Ti[-1] - Ta)
        self.heat_int += 0.5 * (qh_old + qh_new) * dt
        self.iters.append(n_outer)
        self.T_old, self.C_old = self.T, self.C
        self.T, self.C = Ti, Ci
        self.t, self.dt_old, self.R = t_new, dt, R

    #  按输出时刻推进 
    def advance_to(self, t_target, dt_cap, kappa=0.02, dt_min=1e-3):
        """
        推进到 t_target。dt = min(dt_cap, max(dt_min, κ·t))，
        步长对齐工况折点和输出时刻。
        """
        knots = self.ch.knots()
        while self.t < t_target - 1e-9:
            k_in = knots[(knots > self.t + 1e-9) & (knots < t_target - 1e-9)]
            t_next = float(k_in[0]) if k_in.size else float(t_target)
            while self.t < t_next - 1e-9:
                dt = min(dt_cap, max(dt_min, kappa * self.t))
                n_rem = int(np.ceil((t_next - self.t) / dt - 1e-9))
                dt = (t_next - self.t) / max(n_rem, 1)
                self.step(dt)
            self.t = t_next      # 消除舍入误差累积

    #  取值 
    def mean_C(self):
        """
        干物质质量加权平均
        """
        return 2.0 * float(np.sum(self.V * self.C))

    def mean_T(self):
        return 2.0 * float(np.sum(self.V * self.T))

    def at_r(self, r_m, field='C'):
        """
        物理坐标 r（m）取值，r > R 返回 nan；非节点用三次样条
        """
        y = self.C if field == 'C' else self.T
        r_m = np.atleast_1d(np.asarray(r_m, float))
        xq = r_m / self.R
        out = np.full(xq.shape, np.nan)
        ok = xq <= 1.0 + 1e-12
        if ok.any():
            xq_ok = np.minimum(xq[ok], 1.0)
            idx = np.clip(np.searchsorted(self.xi, xq_ok), 0, self.N)
            idx_lo = np.clip(idx - 1, 0, self.N)
            use_hi = np.abs(self.xi[idx] - xq_ok) <= np.abs(self.xi[idx_lo] - xq_ok)
            near = np.where(use_hi, idx, idx_lo)
            exact = np.abs(self.xi[near] - xq_ok) < 1e-12
            vals = np.empty(xq_ok.shape)
            vals[exact] = y[near[exact]]
            if (~exact).any():
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(self.xi, y, bc_type=((1, 0.0), 'not-a-knot'))
                vals[~exact] = cs(xq_ok[~exact])
            out[ok] = vals
        return out

    def state(self):
        return dict(t=self.t, T=self.T.copy(), C=self.C.copy(), T_old=None if self.T_old is None
                    else self.T_old.copy(), C_old=None if self.C_old is None else self.C_old.copy(),
                    dt_old=self.dt_old, R=self.R, flux_int=self.flux_int, heat_int=self.heat_int)

    def set_state(self, s):
        self.t, self.T, self.C = s['t'], s['T'].copy(), s['C'].copy()
        self.T_old = None if s['T_old'] is None else s['T_old'].copy()
        self.C_old = None if s['C_old'] is None else s['C_old'].copy()
        self.dt_old, self.R = s['dt_old'], s['R']
        self.flux_int, self.heat_int = s['flux_int'], s['heat_int']


def find_drying_time(solver, t_coarse_end, dt_fine=0.05, target=C_TARGET):
    """
    从当前状态细步推进，捕捉中心 C 降至 target 的时刻 t*（线性插值）。
    返回 (t*, C, T, R)。
    """
    c_prev, t_prev = float(solver.C[0]), solver.t
    C_prev, T_prev, R_prev = solver.C.copy(), solver.T.copy(), solver.R
    while solver.t < t_coarse_end + 1e-9:
        solver.step(dt_fine)
        c_now = float(solver.C[0])
        if c_now < target:
            w = (c_prev - target) / (c_prev - c_now)
            t_star = t_prev + w * (solver.t - t_prev)
            C_star = C_prev + w * (solver.C - C_prev)
            T_star = T_prev + w * (solver.T - T_prev)
            R_star = R_prev + w * (solver.R - R_prev)
            return t_star, C_star, T_star, R_star
        c_prev, t_prev = c_now, solver.t
        C_prev, T_prev, R_prev = solver.C.copy(), solver.T.copy(), solver.R
    return None


def profile_at_r(xi, y, R, r_m):
    """
    由 ξ 网格节点值 y 求物理坐标 r（m）处的值（r > R 返回 nan）
    """
    from scipy.interpolate import CubicSpline
    r_m = np.atleast_1d(np.asarray(r_m, float))
    xq = r_m / R
    out = np.full(xq.shape, np.nan)
    ok = xq <= 1.0 + 1e-12
    if ok.any():
        cs = CubicSpline(xi, y, bc_type=((1, 0.0), 'not-a-knot'))
        out[ok] = cs(np.minimum(xq[ok], 1.0))
        # 恰为节点时直接取节点值（避免插值舍入）
        for m in np.where(ok)[0]:
            j = np.argmin(np.abs(xi - min(xq[m], 1.0)))
            if abs(xi[j] - min(xq[m], 1.0)) < 1e-12:
                out[m] = y[j]
    return out


def simulate(props, chamber, R_func=None, grid=None, t_1s_until=10800.0, cap_fine=0.5,
             cap_coarse=5.0, t_coarse_from=14460.0, until_dry=True, t_max=10800.0,
             on_1s=None, on_60s=None, dt_fine=0.05, verbose=True, solver_kw=None):
    """
    统一推进驱动（q2/q3/q4 共用，保证相同时刻结果一致）。
    t≤t_1s_until 每 1 s 回调 on_1s，之后每 60 s 回调 on_60s。
    until_dry=True 时在中心 C 首次低于 0.15 的 60 s 输出处停，再细步求 t*
    """
    import time as _time
    s = FVSolver(props, chamber, grid=grid, R_func=R_func, **(solver_kw or {}))
    t0 = _time.time()
    dry = None
    prev_state = s.state()
    while True:
        if s.t < t_1s_until - 1e-9:
            t_next = s.t + 1.0
        else:
            t_next = s.t + 60.0
        cap = cap_fine if t_next <= t_coarse_from + 1e-9 else cap_coarse
        is60 = abs(t_next / 60.0 - round(t_next / 60.0)) < 1e-9
        if is60:
            prev_state = s.state() if until_dry else None
        s.advance_to(t_next, cap)
        if on_1s is not None and t_next <= t_1s_until + 1e-9:
            on_1s(s)
        if is60:
            if on_60s is not None:
                on_60s(s)
            if verbose and abs(t_next % 21600.0) < 1e-6:
                print('    t = %5.1f h, 中心 C = %.4f, 平均 C = %.4f, 用时 %.0f s'
                      % (t_next / 3600.0, s.C[0], s.mean_C(), _time.time() - t0), flush=True)
            if until_dry and s.C[0] < C_TARGET:
                after = s.state()
                s.set_state(prev_state)
                dry = find_drying_time(s, t_next, dt_fine=dt_fine)
                s.set_state(after)
                break
        if (not until_dry) and s.t >= t_max - 1e-9:
            break
        if s.t > 400 * 3600.0:
            raise RuntimeError('超过 400 h 仍未烘干，请检查参数')
    return s, dry


# 6. 问题1:温度场解析解（分离变量 + Duhamel 原理）
 
def bessel_eigs(Bi, n_terms):
    """
    解 λ J1(λ) = Bi J0(λ)，第 n 个根在 (j_{1,n-1}, j_{0,n})
    """
    z0 = jn_zeros(0, n_terms)
    z1 = np.concatenate(([0.0], jn_zeros(1, n_terms - 1)))
    f = lambda x: x * j1(x) - Bi * j0(x)
    lam = np.array([brentq(f, z1[i] + 1e-12, z0[i] - 1e-12, xtol=1e-15, rtol=1e-15, maxiter=200)
                    for i in range(n_terms)])
    return lam


def analytic_T_q1(r_m, t_list, chamber, n_terms=400, rho=820.0, cp=2600.0, k=0.36,
                  h=H_HEAT, R=R0, T0=T_INIT):
    """
    阶跃响应 U = 1 − Σ A_n J0(λ_n r/R) exp(−μ_n t)，A_n = 2Bi/[(λ_n$^2$+Bi$^2$)J0(λ_n)]，
    μ_n = αλ_n$^2$/R$^2$。工况分段线性，Duhamel 逐段积出
    """
    alpha = k / (rho * cp)
    Bi = h * R / k
    lam = bessel_eigs(Bi, n_terms)
    An = 2.0 * Bi / ((lam ** 2 + Bi ** 2) * j0(lam))
    mu = alpha * lam ** 2 / R ** 2
    r_m = np.asarray(r_m, float)
    Jr = j0(np.outer(lam, r_m / R)) * An[:, None]          # (n_terms, n_r)
    tk = chamber.knots()
    th_k = np.array([float(chamber.Ta(x)) for x in tk]) - T0
    th0 = float(chamber.Ta(0.0)) - T0
    s = np.diff(th_k) / np.diff(tk)                          # 各段斜率
    out = np.empty((len(t_list), r_m.size))
    for it, t in enumerate(t_list):
        # 初始阶跃（若 Ta(0) ≠ T0）
        val = th0 * (1.0 - np.exp(-mu * t) @ Jr) if th0 != 0.0 else np.zeros(r_m.size)
        jmax = np.searchsorted(tk, t, side='left')
        a = tk[:jmax]
        b = np.minimum(tk[1:jmax + 1], t)
        sj = s[:jmax]
        # Σ_j s_j [(b−a) − Σ_n A_n J0 (e^{−μ(t−b)} − e^{−μ(t−a)})/μ]
        E = (np.exp(-np.outer(t - b, mu)) - np.exp(-np.outer(t - a, mu))) / mu[None, :]
        val = val + np.sum(sj * (b - a)) - (sj @ E) @ Jr
        out[it] = T0 + val
    return out


# 7. 结果写出
 
HEADER_A1 = '时间\\到药材中心的距离'


def r4(x):
    return float(np.round(float(x), 4))


def write_result_workbook(template_keyword, out_name, sheets):
    """
    在附件3模板上填结果。sheets: [(表名/None, 表头, times, data)]；
    nan 写空单元格，数值保留四位
    """
    from openpyxl import Workbook, load_workbook
    ensure_dirs()
    tpl = find_file(template_keyword)
    wb = load_workbook(tpl) if tpl is not None else Workbook()
    for sheet_name, header, times, data in sheets:
        if sheet_name is None:
            ws = wb.worksheets[0]
        elif sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
        else:
            ws = wb.create_sheet(sheet_name)
        from copy import copy
        hfont, halign = copy(ws['A1'].font), copy(ws['A1'].alignment)
        ws.delete_rows(1, ws.max_row + 1)
        c = ws.cell(row=1, column=1, value=HEADER_A1)
        c.font, c.alignment = hfont, halign
        for j, hv in enumerate(header):
            c = ws.cell(row=1, column=j + 2, value=hv)
            c.font, c.alignment = hfont, halign
        data = np.asarray(data, float)
        for i, tv in enumerate(times):
            ws.cell(row=i + 2, column=1, value=int(round(float(tv))))
            row = data[i]
            for j in range(row.size):
                v = row[j]
                if np.isfinite(v):
                    ws.cell(row=i + 2, column=j + 2, value=r4(v)).number_format = '0.0000'
        ws.column_dimensions['A'].width = 20
    path = os.path.join(OUT_DIR, out_name)
    wb.save(path)
    return path


def r_header(r_cm_list):
    """表头 0, 0.1, ..., 2，整数写为 int"""
    out = []
    for v in r_cm_list:
        v = round(float(v), 1)
        out.append(int(v) if abs(v - round(v)) < 1e-12 else v)
    return out


# 8. 绘图风格
 
def setup_plot():
    import logging
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
                                       'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['axes.labelsize'] = 10.5
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 100
    return plt
