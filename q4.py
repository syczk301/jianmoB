"""四问整理版：可直接阅读的合并源码，无隐藏代码包或动态执行源码。
公共物理方程位于q1.py；跨问共享函数通过普通Python模块导入。
数值模型、约束和原参数保持不变；默认命令仅整理已有结果。
"""
import os
import sys
import importlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
if __name__ in ('__main__', '__mp_main__'):
    sys.modules.setdefault('q4', sys.modules[__name__])
def _question_module(name):
    return importlib.import_module(name)
def _project_root():
    return Path(os.environ.get('B_MODEL_WORKSPACE', str(Path(__file__).resolve().parent))).resolve()
def _project_storage():
    return Path(os.environ.get('B_MODEL_STORAGE', str(_project_root()/'work'))).resolve()

from dataclasses import dataclass, asdict, replace
import hashlib
import json
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from dataclasses import dataclass
import time
from scipy.integrate import solve_ivp
import io
import platform
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import openpyxl
from dataclasses import asdict
from matplotlib import font_manager, rcParams
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import qmc
from scipy.linalg import solve_banded
from dataclasses import dataclass, asdict
from collections import deque
from scipy.linalg import eigh_tridiagonal
from dataclasses import replace
import matplotlib
import matplotlib.pyplot as plt
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys, json, time, argparse
import sys, json, argparse
import json, hashlib
from PIL import Image, ImageDraw, ImageFont
import csv, json, hashlib
import pytest

# ==================== reference_table ====================
def reference_voltage(model, T_C, j):
    cfg = model.cfg.changed(T0=T_C + _question_module('q1').TM)
    g = model.g
    y = model.initial(np.full(g.nt, T_C + _question_module('q1').TM))
    z, _, _ = model.unpack(y)
    for ids, side in [(g.anode, 0), (g.cathode, -1)]:
        dx = g.dx[ids]
        T = T_C + _question_module('q1').TM
        D = g.gas_ref[ids] * (T / 298.15) ** 1.75 * g.eps[ids] ** 1.5
        h = _question_module('q1').harmonic_faces(dx, D)
        diag = np.zeros(len(ids))
        diag[:-1] += h
        diag[1:] += h
        rhs = np.zeros(len(ids))
        mask = g.layer[ids] == (1 if side == 0 else 3)
        length = sum(dx[mask])
        rhs[mask] -= j * 10000.0 / (_question_module('q1').F * (2 if side == 0 else 4)) * dx[mask] / length
        b = 0 if side == 0 else -1
        border = D[b] / (dx[b] / 2)
        diag[b] += border
        rhs[b] += border * g.gas_y[ids[b]] * _question_module('q1').P / (_question_module('q1').R * T)
        A = np.zeros((3, len(ids)))
        A[1] = diag
        A[0, 1:] = -h
        A[2, :-1] = -h
        gas = solve_banded((1, 1), A, rhs)
        if np.min(gas) <= 0:
            return np.nan
        z[3, 0, ids] = gas * g.eps[ids]
    obs = model.voltage(model.phases(y), j)
    return float(obs['V'][0]) if obs['ratio'][0] < 1 and obs['kappa_min'][0] > 0 else np.nan

class ReferenceTable:

    def __init__(self, T, j, V, model_id):
        self.T, self.j, self.V, self.model_id = (T, j, V, model_id)

    def lookup(self, T, j):
        T = np.asarray(T, float)
        out = np.full(T.shape, np.nan)
        if not self.j[0] <= j <= self.j[-1]:
            return out
        k = min(np.searchsorted(self.j, j, side='right') - 1, len(self.j) - 2)
        k = max(k, 0)
        for index, t in np.ndenumerate(T):
            if not self.T[0] <= t <= self.T[-1]:
                continue
            i = min(max(np.searchsorted(self.T, t, side='right') - 1, 0), len(self.T) - 2)
            v = self.V[i:i + 2, k:k + 2]
            if not np.isfinite(v).all():
                continue
            a = (t - self.T[i]) / (self.T[i + 1] - self.T[i])
            b = (j - self.j[k]) / (self.j[k + 1] - self.j[k])
            out[index] = (1 - a) * ((1 - b) * v[0, 0] + b * v[0, 1]) + a * ((1 - b) * v[1, 0] + b * v[1, 1])
        return out

def build_vref(cfg, save=True):
    model = _question_module('q1').Model(cfg.changed(cells=1, fixture='H1', lambda0=3.0))
    T = np.arange(-40, 21, 1.0)
    j = np.unique(np.r_[0.0, np.geomspace(1e-09, 0.005, 100), np.arange(0.005, 0.50001, 0.005)])
    for refinement in range(4):
        V = np.array([[reference_voltage(model, t, x) for x in j] for t in T])
        tab = ReferenceTable(T, j, V, cfg.digest)
        rows = []
        bad = []
        for i in range(0, len(T) - 1, 5):
            tc = (T[i] + T[i + 1]) / 2
            for k in range(len(j) - 1):
                jc = (j[k] + j[k + 1]) / 2
                exact = reference_voltage(model, tc, jc)
                interp = tab.lookup(np.array([tc]), jc)[0]
                error = abs(exact - interp)
                rows.append(dict(T_C=tc, j_A_cm2=jc, error_V=error, valid=np.isfinite(error)))
                if np.isfinite(error) and error > 0.001:
                    bad.append(jc)
        if not bad:
            break
        j = np.unique(np.r_[j, bad])
    if save:
        _question_module('q1').OUTPUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_question_module('q1').OUTPUT / 'vref_table.npz', T=tab.T, j=tab.j, V=tab.V, model_id=cfg.digest)
        pd.DataFrame(rows).to_csv(_question_module('q1').OUTPUT / 'vref_interpolation_report.csv', index=False)
        _question_module('q1').write_json(_question_module('q1').OUTPUT / 'vref_manifest.json', dict(model_id=cfg.digest, reference_lambda=3, max_tested_error_V=float(pd.DataFrame(rows).error_V.max()), tested_temperature_stride=5, restriction='midpoint sample validation; no extrapolation; NaN disables residual feature'))
    return tab

def load_reference():
    d = np.load(_question_module('q1').OUTPUT / 'vref_table.npz')
    return ReferenceTable(d['T'], d['j'], d['V'], str(d['model_id']))

# ==================== feedback ====================
@dataclass(frozen=True)
class RuleConfig:
    feedforward: float = 0.5
    KT: float = 0.025
    Kr: float = 0.15
    KV: float = 2.0
    Kd: float = 1.0
    Kh: float = 0.08
    target_C: float = 0.5
    off_C: float = 0.2
    off_V: float = 0.4
    warn_V: float = 0.35
    min_rate: float = 0.2
    warn_drop: float = 0.01
    confirmations: int = 3
    filter_tau: float = 0.4
    derivative_window: float = 2.0
    delay_s: float = 0.0
    shutdown_delay_s: float = 0.0
    noise_T: float = 0.0
    noise_V: float = 0.0
    seed: int = 42

class RuleController:

    def __init__(self, ref, config=None):
        self.ref = ref
        self.cfg = config or RuleConfig()
        assert self.cfg.off_V >= self.cfg.warn_V > 0.3
        self.history = deque()
        self.sensor = deque()
        self.filtered = None
        self.last_t = None
        self.count = 0
        self.command_off = None
        self.off_time = None
        self.rng = np.random.default_rng(self.cfg.seed)
        self.logs = []

    def metadata(self):
        return dict(kind='causal_rule', parameters=asdict(self.cfg), reference_model=self.ref.model_id)

    def step(self, t, T, V, j):
        c = self.cfg
        if self.last_t is not None and t <= self.last_t:
            raise ValueError('nonmonotone measurement clock')
        T = np.asarray(T) + self.rng.normal(0, c.noise_T, len(T))
        V = np.asarray(V) + self.rng.normal(0, c.noise_V, len(V))
        self.sensor.append((t, T, V, j))
        eligible = [s for s in self.sensor if s[0] <= t - c.delay_s + 1e-09]
        measurement = eligible[-1] if eligible else self.sensor[0]
        mt, T, V, mj = measurement
        while len(self.sensor) > 2 and self.sensor[1][0] < t - c.delay_s - 2:
            self.sensor.popleft()
        if not np.isfinite(T).all() or not np.isfinite(V).all():
            raise ValueError('measurement_fault')
        dt = 0 if self.last_t is None else t - self.last_t
        if self.filtered is None:
            self.filtered = (T.copy(), V.copy())
        else:
            a = 1 - np.exp(-dt / max(c.filter_tau, 1e-06))
            self.filtered = (self.filtered[0] + a * (T - self.filtered[0]), self.filtered[1] + a * (V - self.filtered[1]))
        ft, fv = self.filtered
        vr = self.ref.lookup(ft, mj)
        residual = fv - vr
        self.history.append((t, ft.copy(), residual.copy(), fv.copy()))
        while len(self.history) > 2 and self.history[0][0] < t - c.derivative_window:
            self.history.popleft()
        dT = np.zeros(len(T))
        dr = np.zeros(len(T))
        derivative_valid = len(self.history) >= 3
        if derivative_valid:
            h = list(self.history)
            tt = np.array([x[0] for x in h])
            tt -= tt.mean()
            dT = tt @ np.array([x[1] for x in h]) / np.dot(tt, tt)
            dr = tt @ np.array([x[3] for x in h]) / np.dot(tt, tt)
            rs = np.array([x[2] for x in h])
            ok = np.isfinite(rs).all(axis=0)
            dr[ok] = tt @ rs[:, ok] / np.dot(tt, tt)
        risk = (fv < c.warn_V) | (dr < -c.warn_drop) & derivative_valid
        u = c.feedforward + c.KT * np.maximum(c.target_C - ft, 0) + c.KV * np.maximum(c.warn_V - fv, 0) - c.Kh * np.maximum(ft - c.off_C, 0)
        if derivative_valid:
            u += c.Kr * np.maximum(c.min_rate - dT, 0) + c.Kd * np.maximum(-dr - c.warn_drop, 0)
        confirmed = bool(np.all(ft > c.off_C) and np.all(fv > c.off_V) and (not np.any(risk)))
        self.count = self.count + 1 if confirmed else 0
        if self.off_time is None and (not confirmed):
            self.command_off = None
        if self.command_off is None and self.count >= c.confirmations:
            self.command_off = t
        if self.command_off is not None and t + 1e-09 >= self.command_off + c.shutdown_delay_s and (self.off_time is None):
            self.off_time = t
        u = np.zeros(len(T)) if self.off_time is not None else np.clip(u, 0, 1)
        self.logs.append(dict(t=t, measurement_time=mt, j_reference=mj, u=u.tolist(), filtered_T=ft.tolist(), filtered_V=fv.tolist(), derivative_valid=derivative_valid, reference_valid=np.isfinite(vr).tolist(), confirmations=self.count, off_time=self.off_time))
        self.last_t = t
        self.previous_power = u.copy()
        return u

# ==================== precool ====================
def precool(cfg=None, minutes=None):
    cfg = cfg or _question_module('q1').Config(cells=5, fixture='H2', T0=_question_module('q1').TM + 25, ambient=_question_module('q1').TM - 30)
    minutes = np.array(minutes if minutes is not None else list(range(10, 101, 5)), float)
    m = _question_module('q1').Model(cfg)
    g = m.g
    s = m.phases(m.initial())
    k = m.thermal_k(s)
    cap = g.C.copy()
    cap[g.mea_indices] += g.bcap * cfg.lambda0 * 4182
    C = cap * g.tdx
    f = _question_module('q1').harmonic_faces(g.tdx, k)
    diag = np.zeros(g.nt)
    diag[:-1] -= f
    diag[1:] -= f
    diag[0] -= 1 / (1 / cfg.h + g.tdx[0] / (2 * k[0]))
    diag[-1] -= 1 / (1 / cfg.h + g.tdx[-1] / (2 * k[-1]))
    eig, U = eigh_tridiagonal(diag / C, f / np.sqrt(C[:-1] * C[1:]))
    a = U.T @ (np.sqrt(C) * (25 + _question_module('q1').TM - cfg.ambient))
    fields = np.array([cfg.ambient + U @ (np.exp(eig * t * 60) * a) / np.sqrt(C) for t in minutes])
    rows = []
    for t, T in zip(minutes, fields):
        means = np.array([np.average(T[ix], weights=g.dx) for ix in g.mea_indices]) - _question_module('q1').TM
        rows.append(dict(minutes=t, **{f'T{i + 1}_C': v for i, v in enumerate(means)}, mean_C=float(np.mean(means)), delta_T_K=float(np.ptp(means)), total_sensible_J=float(np.dot(C, T - _question_module('q1').TM) * 0.0025)))
    return (minutes, fields, pd.DataFrame(rows), g)

def run_precool():
    minutes, fields, frame, g = precool()
    _question_module('q1').OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(_question_module('q1').OUTPUT / 'precool_scan.csv', index=False)
    np.savez_compressed(_question_module('q1').OUTPUT / 'precool_fields.npz', minutes=minutes, T=fields, x=g.tx, layer=g.names)
    g.manifest().to_csv(_question_module('q1').OUTPUT / 'mesh_manifest.csv', index=False)
    ts, full, curve, _ = precool(minutes=np.arange(0, 100.01, 0.25))
    curve.to_csv(_question_module('q1').OUTPUT / 'precool_dense.csv', index=False)
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'precool_checks.json', dict(energy_monotone=bool(np.all(np.diff(curve.total_sensible_J) <= 1e-08)), max_gradient_K=float(curve.delta_T_K.max()), max_gradient_minutes=float(curve.loc[curve.delta_T_K.idxmax(), 'minutes'])))
    return frame

# ==================== q4 ====================
def rule_worker(payload):
    cfg, rc, field, tag = payload
    controller = RuleController(load_reference(), rc)
    r = _question_module('q1').evaluate(cfg, _question_module('q1').Protocol('fixed'), field, controller, False, label=tag, cache=False)
    return {**r['metrics'], 'rule': asdict(rc)}

def q4(cfg, q3_summary, budget=6, workers=4):
    cfg = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, horizon=300, charge_limit=1000000.0, voltage_limit=0.3, sample=0.5)
    build_vref(cfg)
    minutes, fields, _, _ = precool(cfg, minutes=[20, 40])
    scenarios = [('equilibrium', None), ('20min', fields[0]), ('40min', fields[1])]
    rules = [RuleConfig(feedforward=float(ff), KT=float(kt)) for ff in (0.3, 0.6, 1.0) for kt in (0.02, 0.04)][:max(1, budget)]
    jobs = [(cfg, rc, field, 'q4_dev_' + name) for rc in rules for name, field in scenarios]
    if workers == 1:
        results = list(map(rule_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(rule_worker, jobs))
    groups = []
    for k, rc in enumerate(rules):
        rr = results[k * 3:k * 3 + 3]
        groups.append(dict(config=asdict(rc), all_feasible=all((r['feasibility'] == 'feasible' for r in rr)), E_sum=sum((r['E_aux_total_J'] for r in rr)), results=rr))
    good = [g for g in groups if g['all_feasible']]
    selected = min(good, key=lambda g: g['E_sum']) if good else None
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_rule_search.json', dict(groups=groups, selected=selected))
    rows = []
    coheat = q3_summary.get('coheat')
    inherited = _question_module('q1').Protocol(**coheat['protocol']) if coheat else _question_module('q1').Protocol('fixed', powers=(1.0,) * 5, heat_off=80)
    for k, (name, field) in enumerate(scenarios):
        const = _question_module('q1').evaluate(cfg, inherited, field, label='q4_constant_' + name, stop_success=False)
        rows.append(dict(scenario=name, method='constant_transfer' if coheat else 'constant_reference', **const['metrics']))
        ps = [inherited, _question_module('q1').Protocol('fixed')] + [_question_module('q1').Protocol('fixed', powers=(p,) * 5, heat_off=th) for p in (0.4, 0.7, 1.0) for th in (40, 80, 160)]
        candidates = _question_module('q2').batch(cfg, ps, field, label='q4_constant_reopt_' + name, workers=workers, stop_success=False)
        win = _question_module('q2').save_search('q4_constant_reopt_' + name, candidates, 'E_aux_total_J')
        rows.append(dict(scenario=name, method='constant_reoptimized', **win or {'feasibility': 'unresolved', 'stop_reason': 'not_found'}))
        dynamic = selected['results'][k] if selected else groups[0]['results'][k]
        rows.append(dict(scenario=name, method='dynamic_selected' if selected else 'dynamic_unqualified_reference', **dynamic))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_comparison.json', rows)
    mt, ff, _, _ = precool(cfg)
    jobs = [(cfg, inherited, f, 'q4_scan', False) for f in ff]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scan = list(pool.map(_question_module('q2').worker, jobs))
    for m, r in zip(mt, scan):
        r['precool_minutes'] = float(m)
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_precool_constant_scan.json', scan)
    rc = RuleConfig(**selected['config'] if selected else groups[0]['config'])
    mt, ff, _, _ = precool(cfg, minutes=[15, 30, 60, 90])
    jobs = [(cfg, RuleConfig(**{**asdict(rc), 'noise_T': 0.1, 'noise_V': 0.003, 'delay_s': 0.4, 'seed': 700 + i}), f, 'q4_holdout') for i, f in enumerate(ff)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        holdout = list(pool.map(rule_worker, jobs))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_holdout.json', dict(rule_qualified_on_development=selected is not None, results=holdout))
    fine = cfg.changed(mesh=(12, 6, 8, 10, 12), board_n=4, end_n=8, sample=0.2, max_step=0.1, rtol=2e-06)
    mt, ff, _, _ = precool(fine, minutes=[20, 40])
    jobs = [(fine, rc, f, 'q4_fine') for f in [None, ff[0], ff[1]]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fine_results = list(pool.map(rule_worker, jobs))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_fine_replay.json', fine_results)
    return rows

# ==================== completion ====================
def refine_q4(cfg, workers=4):
    coarse = _question_module('q2').completion_read('q4_precool_constant_scan.json')

    def full_window(row):
        row['delta_T_to_official_success'] = row['delta_T_max']
        row['delta_T_max'] = row.get('shutdown_extrema', {}).get('delta_T_max', row['delta_T_max'])
        return row
    coarse = [full_window(r) for r in coarse]
    by_min = {r['precool_minutes']: r for r in coarse}
    points = set(_question_module('q2').boundary_points(coarse, 1, 'precool_minutes'))
    peak = max(coarse, key=lambda r: r['delta_T_max'])['precool_minutes']
    points.update(range(max(10, int(peak) - 4), min(100, int(peak) + 4) + 1))
    points = sorted(points - set(by_min))
    cfg = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, horizon=300, charge_limit=1000000.0, sample=0.5)
    proto = _question_module('q1').Protocol(**coarse[0]['protocol'])
    _, fields, _, _ = precool(cfg, minutes=points)
    jobs = [(cfg, proto, field, 'q4_scan_refine', False) for field in fields]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        evaluated = list(pool.map(_question_module('q2').worker, jobs))
    added = []
    for minute, r in zip(points, evaluated):
        r['precool_minutes'] = float(minute)
        added.append(full_window(r))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_precool_refined.json', sorted(coarse + added, key=lambda r: r['precool_minutes']))
    return dict(new_cases=len(added), coarse_gradient_peak_minutes=peak)

def controller_latency():
    config = _question_module('q2').completion_read('q4_rule_search.json')
    chosen = config['selected'] or config['groups'][0]
    controller = RuleController(load_reference(), RuleConfig(**chosen['config']))
    durations = []
    for k in range(1000):
        tic = time.perf_counter()
        controller.step(k * 0.2, np.full(5, -10.0), np.full(5, 0.6), 0.2)
        durations.append(time.perf_counter() - tic)
    result = dict(samples=1000, sample_period_s=0.2, max_s=max(durations), p99_s=float(np.quantile(durations, 0.99)), scope='Controller and table only; desktop benchmark excludes sensors, actuator and PDE integration')
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'controller_latency.json', result)
    return result

def q4_discretization():
    rules = _question_module('q2').completion_read('q4_rule_search.json')
    chosen = rules['selected'] or rules['groups'][0]
    fine = _question_module('q2').completion_read('q4_fine_replay.json')
    rows = []
    for scenario, a, b in zip(('equilibrium', '20min', '40min'), chosen['results'], fine):
        rows.append(dict(scenario=scenario, coarse_run=a['candidate_id'], fine_run=b['candidate_id'], coarse_feasibility=a['feasibility'], fine_feasibility=b['feasibility'], coarse_time=a['success_time'], fine_time=b['success_time'], coarse_energy=a['E_aux_total_J'], fine_energy=b['E_aux_total_J'], energy_relative_difference=abs(b['E_aux_total_J'] - a['E_aux_total_J']) / max(b['E_aux_total_J'], 1), coarse_ice=a['ice_max'], fine_ice=b['ice_max']))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_discretization.json', dict(comparison='Combined 14 to 48 MEA grid and 0.5 to 0.2 s controller sampling refinement; not a pure spatial convergence proof', rows=rows))
    return dict(cases=len(rows), fine_feasible=sum((r['fine_feasibility'] == 'feasible' for r in rows)))

def reference_trajectory_error():
    model = _question_module('q1').Model(_question_module('q1').Config(**_question_module('q2').completion_read('selected_model.json')['config']).changed(cells=1, fixture='H1', lambda0=3.0))
    tab = load_reference()
    rules = _question_module('q2').completion_read('q4_rule_search.json')
    chosen = rules['selected'] or rules['groups'][0]
    rows = []
    for result in chosen['results']:
        log = json.loads((_project_storage() / 'runs' / result['candidate_id'] / 'controller_log.json').read_text(encoding='utf-8'))
        errors = []
        times = []
        for entry in log:
            temps = np.array(entry['filtered_T'])
            j = entry['j_reference']
            interp = tab.lookup(temps, j)
            direct = np.array([reference_voltage(model, float(t), j) for t in temps])
            errors.append(interp - direct)
            times.append(entry['t'])
        errors = np.array(errors)
        dt = np.diff(times)
        derivative = np.diff(errors, axis=0) / dt[:, None]
        valid = np.isfinite(errors)
        vd = np.isfinite(derivative)
        rows.append(dict(run=result['candidate_id'], valid_points=int(valid.sum()), invalid_points=int((~valid).sum()), max_voltage_error_V=float(np.max(abs(errors[valid]))) if valid.any() else None, max_error_derivative_V_s=float(np.max(abs(derivative[vd]))) if vd.any() else None, warning_derivative_threshold_V_s=chosen['config']['warn_drop']))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'vref_trajectory_validation.json', rows)
    return dict(trajectories=len(rows), all_tested_derivative_errors_below_threshold=all((r['max_error_derivative_V_s'] is not None and r['max_error_derivative_V_s'] < r['warning_derivative_threshold_V_s'] for r in rows)))

def control_robustness(cfg, workers=4, count=20):
    """Paired stress scenarios, explicitly not a calibrated probability model."""
    summary = _question_module('q2').completion_read('q3_summary.json')
    rules = _question_module('q2').completion_read('q4_rule_search.json')
    coheat = summary.get('coheat')
    selected = rules['selected']
    proto = _question_module('q1').Protocol(**coheat['protocol']) if coheat else _question_module('q1').Protocol('fixed', powers=(1.0,) * 5, heat_off=80.0)
    rc = RuleConfig(**(selected or rules['groups'][0])['config'])
    cfg = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, horizon=300, charge_limit=1000000.0, sample=0.5)
    signature = hashlib.sha256(json.dumps(dict(config=cfg.to_dict(), rule=asdict(rc), protocol=vars(proto), count=count, design_version=1, physical_code=_question_module('q1').code_digest()), sort_keys=True).encode()).hexdigest()
    saved = _question_module('q1').OUTPUT / 'control_robustness.json'
    if saved.exists():
        previous = _question_module('q2').completion_read(saved.name)
        if previous.get('signature') == signature:
            return previous['summary']
    rng = np.random.default_rng(cfg.seed + 71)
    constant = []
    dynamic = []
    design = []
    for k in range(count):
        a = rng.uniform(-1, 1, 6)
        case = cfg.changed(j0=cfg.j0 * np.exp(0.1 * a[0]), tau_b=cfg.tau_b * np.exp(0.3 * a[1]), tau_f=cfg.tau_f * np.exp(0.3 * a[2]), h=cfg.h * (1 + 0.2 * a[3]), lambda0=cfg.lambda0 + 0.15 * a[4], Rc=cfg.Rc * (1 + 0.2 * a[5]))
        rule = RuleConfig(**{**asdict(rc), 'noise_T': 0.1, 'noise_V': 0.003, 'delay_s': 0.4, 'seed': 8100 + k})
        constant.append((case, proto, None, 'robust_constant', False))
        dynamic.append((case, rule, None, 'robust_dynamic'))
        design.append(dict(scenario=k, **{p: getattr(case, p) for p in ('j0', 'tau_b', 'tau_f', 'h', 'lambda0', 'Rc')}))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cr = list(pool.map(_question_module('q2').worker, constant))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        dr = list(pool.map(rule_worker, dynamic))
    rows = []
    for method, results in [('constant', cr), ('dynamic', dr)]:
        for design_row, result in zip(design, results):
            rows.append(dict(method=method, **design_row, **result))
    summary = dict(paired_scenarios=count, constant_success=sum((r['feasibility'] == 'feasible' for r in cr)), dynamic_success=sum((r['feasibility'] == 'feasible' for r in dr)), qualification='conditional stress tests only')
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'control_robustness.json', dict(signature=signature, summary=summary, scenarios=rows, constant_is_selected=coheat is not None, rule_is_qualified=selected is not None, range_status='registered stress ranges, not confidence intervals or real-world probabilities'))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]).to_csv(_question_module('q1').OUTPUT / 'control_robustness.csv', index=False)
    return summary

def complete(cfg, workers=4):
    result = {}
    for name, func in [('trajectory_diagnostics', _question_module('q1').trajectory_diagnostics), ('failure_diagnostics', _question_module('q1').failure_diagnostics), ('horizon_extension', _question_module('q3').horizon_expansion), ('q2_boundary_refinement', lambda: _question_module('q2').refine_q2(cfg, workers)), ('q2_boundary_fine', _question_module('q2').boundary_replay), ('q3_control_convergence', _question_module('q3').q3_control_convergence), ('q3_70s_qualification', _question_module('q3').qualify_q3), ('q4_scan_refinement', lambda: refine_q4(cfg, workers)), ('controller_latency', controller_latency), ('q4_discretization', q4_discretization), ('reference_trajectory_validation', reference_trajectory_error), ('control_robustness', lambda: control_robustness(cfg, workers)), ('extended_energy_optimum', lambda: _question_module('q3').resolve_energy_extension(cfg, workers))]:
        print('START supplementary', name, flush=True)
        result[name] = func()
        _question_module('q1').write_json(_question_module('q1').OUTPUT / 'supplementary_status.json', result)
    return result

# ==================== energy_extension ====================
def precompute_zero_q4(cfg, workers=8):
    """Exploratory zero-heat scan, independent of the final fine-grid acceptance."""
    cfg = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, horizon=600, charge_limit=1000000.0, sample=0.5)
    grid = _question_module('q3').energy_extension_read('q4_precool_refined.json')
    minutes = sorted(set([20.0, 40.0, *[r['precool_minutes'] for r in grid]]))
    _, fields, _, _ = precool(cfg, minutes=minutes)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_question_module('q2').worker, [(cfg, _question_module('q1').Protocol('fixed'), field, 'q4_zero_constant', False) for field in [None, *fields]]))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_zero_exploratory.json', [dict(r, precool_minutes=m) for r, m in zip(results, [None, *minutes])])
    return dict(cases=len(results), feasible=sum((r['feasibility'] == 'feasible' for r in results)))

def zero_heat_robustness(cfg, workers=8):
    """Replay the same registered stress design for the extended zero-heat baseline."""
    import pandas as pd
    previous = _question_module('q3').energy_extension_read('control_robustness.json')
    design = [r for r in previous['scenarios'] if r['method'] == 'constant']
    base = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, horizon=600, charge_limit=1000000.0, sample=0.5)
    cases = [base.changed(**{p: r[p] for p in ('j0', 'tau_b', 'tau_f', 'h', 'lambda0', 'Rc')}) for r in design]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_question_module('q2').worker, [(c, _question_module('q1').Protocol('fixed'), None, 'robust_zero_constant', False) for c in cases]))
    rows = [dict(r, scenario=d['scenario'], method='zero_constant') for r, d in zip(results, design)]
    dynamic = [r for r in previous['scenarios'] if r['method'] == 'dynamic']
    assert all((r['feasibility'] == 'feasible' and r['last_valid_time'] < 300 for r in dynamic))
    summary = dict(paired_scenarios=len(design), zero_constant_success=sum((r['feasibility'] == 'feasible' for r in results)), dynamic_success=len(dynamic), scope='Registered stress cases, not a posterior probability; completed causal dynamic trajectories reused')
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'extended_control_robustness.json', dict(summary=summary, scenarios=rows + dynamic))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows + dynamic]).to_csv(_question_module('q1').OUTPUT / 'extended_control_robustness.csv', index=False)
    return summary

# ==================== test_events ====================
def test_feedback_no_future_and_latched_off():
    ref = ReferenceTable(np.array([-40.0, 20.0]), np.array([0.0, 0.5]), np.ones((2, 2)) * 0.7, 'test')
    ctl = RuleController(ref, RuleConfig(confirmations=3, filter_tau=0.001))
    for t in (0, 0.2):
        assert np.any(ctl.step(t, np.ones(5), np.ones(5) * 0.7, 0.1) > 0)
    assert np.all(ctl.step(0.4, np.ones(5), np.ones(5) * 0.7, 0.1) == 0)
    assert ctl.off_time == 0.4
    assert np.all(ctl.step(0.6, -np.ones(5) * 10, np.ones(5) * 0.7, 0.1) == 0)
    assert all((row['measurement_time'] <= row['t'] for row in ctl.logs))

def test_reference_outside_and_invalid_corner():
    ref = ReferenceTable(np.array([-40.0, 20.0]), np.array([0.0, 0.5]), np.array([[0.5, 0.6], [0.7, np.nan]]), 'test')
    assert np.isnan(ref.lookup(np.array([0.0]), 0.1)[0])
    assert np.isnan(ref.lookup(np.array([-45.0]), 0.1)[0])

def test_delayed_shutdown_confirmation_is_cancelled_on_new_risk():
    ref = ReferenceTable(np.array([-40.0, 20.0]), np.array([0.0, 0.5]), np.ones((2, 2)) * 0.7, 'test')
    ctl = RuleController(ref, RuleConfig(confirmations=2, filter_tau=0.001, shutdown_delay_s=0.4, Kd=0))
    for t in (0, 0.2):
        ctl.step(t, np.ones(1), np.ones(1) * 0.7, 0.1)
    assert ctl.command_off == 0.2 and ctl.off_time is None
    ctl.step(0.4, -np.ones(1), np.ones(1) * 0.7, 0.1)
    assert ctl.command_off is None and ctl.off_time is None
    for t in (0.6, 0.8, 1.0, 1.2):
        ctl.step(t, np.ones(1), np.ones(1) * 0.7, 0.1)
    assert ctl.off_time == 1.2

def test_proposed_constant_power_future_is_discarded_causally():
    ref = ReferenceTable(np.array([-40.0, 20.0]), np.array([0.0, 0.5]), np.ones((2, 2)) * 0.7, 'test')
    rc = RuleConfig(feedforward=0.3, KT=0.01, Kr=0, Kd=0, KV=0)
    c = _question_module('q1').Config(j0=0.3, horizon=1.0, charge_limit=100, T0=_question_module('q1').TM - 2, ambient=_question_module('q1').TM - 2)
    a = _question_module('q1').simulate(c, _question_module('q1').Protocol('fixed'), controller=RuleController(ref, rc), stop_success=False)
    b = _question_module('q1').simulate(c.changed(control_prediction_horizon=0.2), _question_module('q1').Protocol('fixed'), controller=RuleController(ref, rc), stop_success=False)
    assert np.max(abs(a['T'][-1] - b['T'][-1])) < 0.001
    assert abs(a['metrics']['E_aux_total_J'] - b['metrics']['E_aux_total_J']) < 0.01

# ==================== 四问命令入口 ====================
def main(argv=None):
    parser=argparse.ArgumentParser(description='Q4：预冷、参考电压表、动态加热控制及鲁棒性')
    parser.add_argument('command',nargs='?',default='results',choices=['results','check','run','precool','reference','refine','convergence','robustness','extended-robustness','figures'])
    parser.add_argument('--budget',type=int,default=6)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv);core=_question_module('q1')
    if args.command=='check':core.self_check();return
    if args.command=='precool':run_precool()
    elif args.command=='reference':build_vref(core.selected_config())
    elif args.command=='run':
        summary=json.loads((core.OUTPUT/'q3_summary.json').read_text(encoding='utf-8'))
        q4(core.selected_config(),summary,args.budget,args.workers)
    elif args.command=='refine':refine_q4(core.selected_config(),args.workers)
    elif args.command=='convergence':q4_discretization()
    elif args.command=='robustness':control_robustness(core.selected_config(),args.workers,args.budget)
    elif args.command=='extended-robustness':zero_heat_robustness(core.selected_config(),args.workers)
    elif args.command=='figures':core.figures()
    if args.command!='results':core.make_tables()
    core.export_results('q4')

if __name__=='__main__':
    main()
