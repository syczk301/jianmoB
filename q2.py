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
    sys.modules.setdefault('q2', sys.modules[__name__])
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

# ==================== optimize ====================
def worker(payload):
    cfg, proto, Tfield, label, stop = payload
    r = _question_module('q1').evaluate(cfg, proto, Tfield, label=label, stop_success=stop)
    return dict(**r['metrics'], protocol=vars(proto))

def batch(cfg, protocols, Tfield=None, label='search', workers=4, stop_success=True):
    args = [(cfg, p, Tfield, label, stop_success) for p in protocols]
    if workers == 1:
        rows = list(map(worker, args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(worker, args))
    return rows

def best(rows, objective='success_time'):
    good = [r for r in rows if r['feasibility'] == 'feasible' and r.get('numerically_accepted', False)]
    return min(good, key=lambda r: (r[objective], r['success_time'])) if good else None

def save_search(name, rows, objective='success_time'):
    winner = best(rows, objective)
    _question_module('q1').write_json(_question_module('q1').OUTPUT / f'{name}_search.json', dict(candidates=rows, best=winner, conclusion='best admissible search-grid candidate; fine replay required' if winner else 'no feasible candidate found; not an impossibility proof'))
    flat = []
    current = np.inf
    for i, r in enumerate(rows):
        if r['feasibility'] == 'feasible':
            current = min(current, r[objective])
        flat.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))} | dict(evaluation=i + 1, best_so_far=current, protocol_json=json.dumps(r['protocol'])))
    pd.DataFrame(flat).to_csv(_question_module('q1').OUTPUT / f'{name}_evaluations.csv', index=False)
    return winner

def refine_neighbors(cfg, rows, objective, workers=4, Tfield=None, label='local_refine', preheat=False):
    """Two-start bounded coordinate neighborhoods; records every real evaluation."""
    good = sorted([r for r in rows if r['feasibility'] == 'feasible'], key=lambda r: r[objective])[:2]
    extra = []
    for seed in good:
        incumbent = seed
        for fraction in (0.1, 0.04):
            p = _question_module('q1').Protocol(**incumbent['protocol'])
            candidates = []
            if p.kind in ('zero', 'fixed'):
                vector = np.r_[p.powers, p.heat_off]
                steps = np.r_[np.full(5, fraction), fraction * 200]
                count = 5 if preheat else 6
                for k in range(count):
                    for sign in (-1, 1):
                        v = vector.copy()
                        v[k] += sign * steps[k]
                        v[:5] = np.clip(v[:5], 0, 1)
                        v[5] = 300 if preheat else np.clip(v[5], 0, cfg.horizon)
                        candidates.append(_question_module('q1').Protocol(p.kind, powers=tuple(v[:5]), heat_off=float(v[5])))
            else:
                v0 = np.array(p.params, float)
                for k in range(len(v0)):
                    for sign in (-1, 1):
                        v = v0.copy()
                        nlevel = 1 if p.kind == 'constant' else 2 if p.kind == 'ramp' else 3
                        v[k] += sign * fraction * (0.5 if k < nlevel else 100)
                        v[:nlevel] = np.sort(np.clip(v[:nlevel], 0.001, 0.5))
                        if len(v) > nlevel:
                            v[nlevel:] = np.sort(np.clip(v[nlevel:], 0.1, 180))
                        candidates.append(_question_module('q1').Protocol(p.kind, tuple(v)))
            evaluated = batch(cfg, candidates, Tfield, label, workers, stop_success=preheat or p.kind in ('constant', 'ramp', 'step'))
            extra.extend(evaluated)
            winner = best([incumbent, *evaluated], objective)
            if winner:
                incumbent = winner
    return extra

def q2(cfg, budget=80, workers=4):
    cfg = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 10, ambient=_question_module('q1').TM - 10, horizon=300, charge_limit=20, voltage_limit=0.3)
    results = {}
    baseline = None
    for kind in ('constant', 'ramp', 'step'):
        if kind == 'constant':
            candidates = [_question_module('q1').Protocol('constant', (float(j),)) for j in np.linspace(0.005, 0.5, max(50, budget))]
        else:
            d = 3 if kind == 'ramp' else 5
            points = qmc.LatinHypercube(d=d, seed=cfg.seed).random(budget)
            candidates = []
            if baseline:
                j = baseline['protocol']['params'][0]
                candidates.append(_question_module('q1').Protocol(kind, (j, j, 30) if kind == 'ramp' else (j, j, j, 10, 30)))
            for p in points:
                if kind == 'ramp':
                    j0, jp = np.sort(p[:2] * 0.5)
                    pars = (j0, jp, 1 + 119 * p[2])
                else:
                    levels = np.sort(p[:3] * 0.5)
                    switches = np.sort(1 + 119 * p[3:])
                    pars = (*levels, *switches)
                candidates.append(_question_module('q1').Protocol(kind, tuple((float(x) for x in pars))))
        rows = batch(cfg, candidates, label='q2_' + kind, workers=workers)
        rows += refine_neighbors(cfg, rows, 'success_time', workers, label='q2_refine_' + kind)
        winner = save_search('q2_' + kind, rows)
        if kind == 'constant':
            baseline = winner
        if winner:
            proto = _question_module('q1').Protocol(**winner['protocol'])
            fine = cfg.changed(mesh=(12, 6, 8, 10, 12), board_n=4, end_n=8, rtol=2e-06, max_step=0.2)
            replay = _question_module('q1').evaluate(fine, proto, label='q2_fine')
            winner['fine_replay'] = replay['metrics']
        results[kind] = winner
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q2_summary.json', results)
    return results

def q2_temperature_scan(cfg, summary, workers=4, budget=12):
    rows = []
    jobs = []
    groups = []
    for kind, winner in summary.items():
        default = {'constant': (0.2,), 'ramp': (0.02, 0.3, 60.0), 'step': (0.02, 0.1, 0.3, 20.0, 40.0)}
        proto = _question_module('q1').Protocol(**winner['protocol']) if winner else _question_module('q1').Protocol(kind, default[kind])
        for T0 in np.arange(0, -40.01, -2):
            case = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM + T0, ambient=_question_module('q1').TM + T0, charge_limit=20)
            rng = np.random.default_rng(cfg.seed + int(-T0))
            ps = [proto]
            for _ in range(budget):
                p = np.asarray(proto.params, float) * np.exp(rng.normal(0, 0.3, len(proto.params)))
                if kind == 'constant':
                    p[0] = np.clip(p[0], 0.001, 0.5)
                elif kind == 'ramp':
                    p[:2] = np.sort(np.clip(p[:2], 0, 0.5))
                    p[2] = np.clip(p[2], 0.1, 180)
                else:
                    p[:3] = np.sort(np.clip(p[:3], 0, 0.5))
                    p[3:] = np.sort(np.clip(p[3:], 0.1, 180))
                ps.append(_question_module('q1').Protocol(kind, tuple(p)))
            groups.append((kind, float(T0), winner is not None, len(jobs), len(ps)))
            jobs.extend(((case, p, None, 'q2_reopt', True) for p in ps))
    if workers == 1:
        evaluated = list(map(worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(worker, jobs))
    for kind, T0, has_winner, start, count in groups:
        candidates = evaluated[start:start + count]
        transfer = candidates[0]
        win = best(candidates)
        rows.append(dict(strategy=kind, T0_C=T0, mode='fixed_transfer' if has_winner else 'unoptimized_reference', **transfer))
        rows.append(dict(strategy=kind, T0_C=T0, mode='reoptimized', status='feasible' if win else 'not_found', success_time=win['success_time'] if win else None, candidate=win['candidate_id'] if win else None))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q2_temperature_boundary.json', rows)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]).to_csv(_question_module('q1').OUTPUT / 'q2_temperature_boundary.csv', index=False)
    return rows

# ==================== completion ====================
def completion_read(name):
    return json.loads((_question_module('q1').OUTPUT / name).read_text(encoding='utf-8'))

def boundary_points(records, step, key='T0_C'):
    """Refine every observed transition, without assuming monotone feasibility."""
    rows = sorted(records, key=lambda r: r[key])
    points = set()
    for a, b in zip(rows[:-1], rows[1:]):
        good = lambda r: r.get('feasibility') == 'feasible' or r.get('status') == 'feasible'
        if good(a) != good(b):
            points.update((round(float(x), 6) for x in np.arange(a[key] + step, b[key] - 0.01 * step, step)))
    return sorted(points)

def refine_q2(cfg, workers=4, budget=6):
    coarse = completion_read('q2_temperature_boundary.json')
    summary = completion_read('q2_summary.json')
    added = []
    defaults = {'constant': (0.2,), 'ramp': (0.02, 0.3, 60.0), 'step': (0.02, 0.1, 0.3, 20.0, 40.0)}
    for kind in defaults:
        win = summary[kind]
        proto = _question_module('q1').Protocol(**win['protocol']) if win else _question_module('q1').Protocol(kind, defaults[kind])
        for mode in ('fixed_transfer' if win else 'unoptimized_reference', 'reoptimized'):
            records = [r for r in coarse if r['strategy'] == kind and r['mode'] == mode]
            for spacing in (0.5, 0.1):
                for t in boundary_points(records, spacing):
                    case = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM + t, ambient=_question_module('q1').TM + t, charge_limit=20, horizon=300)
                    ps = [proto]
                    if mode == 'reoptimized':
                        nearest = sorted([r for r in records if r.get('candidate')], key=lambda r: abs(r['T0_C'] - t))
                        if nearest:
                            path = _project_storage() / 'runs' / nearest[0]['candidate'] / 'protocol.json'
                            ps.append(_question_module('q1').Protocol(**json.loads(path.read_text(encoding='utf-8'))))
                        rng = np.random.default_rng(cfg.seed + int(round((t + 40) * 100)))
                        for _ in range(budget):
                            p = np.array(ps[-1].params) * np.exp(rng.normal(0, 0.3, len(proto.params)))
                            n = 1 if kind == 'constant' else 2 if kind == 'ramp' else 3
                            p[:n] = np.sort(np.clip(p[:n], 0.001, 0.5))
                            p[n:] = np.sort(np.clip(p[n:], 0.1, 180))
                            ps.append(_question_module('q1').Protocol(kind, tuple(p)))
                    candidates = batch(case, ps, label='q2_boundary_refine', workers=workers)
                    winner = best(candidates)
                    row = dict(strategy=kind, T0_C=t, mode=mode, resolution_C=spacing, status='feasible' if winner else 'not_found', candidate=winner['candidate_id'] if winner else None, success_time=winner['success_time'] if winner else None, candidates=[r['candidate_id'] for r in candidates])
                    records.append(row)
                    added.append(row)
    combined = coarse + added
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q2_temperature_refined.json', combined)
    pd.DataFrame(combined).to_csv(_question_module('q1').OUTPUT / 'q2_temperature_refined.csv', index=False)
    return dict(new_cases=len(added), conclusion='Finite-budget feasibility boundaries, not a proof of impossibility')

def boundary_replay(workers=8):
    rows = completion_read('q2_temperature_refined.json')
    results = []
    jobs = []
    groups = []
    for kind in ('constant', 'ramp', 'step'):
        for mode in ('fixed_transfer', 'unoptimized_reference', 'reoptimized'):
            candidates = [r for r in rows if r['strategy'] == kind and r['mode'] == mode and (r.get('status') == 'feasible' or r.get('feasibility') == 'feasible')]
            if not candidates:
                continue
            row = min(candidates, key=lambda r: r['T0_C'])
            name = row.get('candidate') or row['candidate_id']
            original = _question_module('q1').load_run(name)
            cfg = _question_module('q1').Config(**original['config'])
            proto = _question_module('q1').Protocol(**json.loads((_project_storage() / 'runs' / name / 'protocol.json').read_text(encoding='utf-8')))
            groups.append((kind, mode, row['T0_C'], name, len(jobs)))
            for mesh, step, rtol in [((12, 6, 8, 10, 12), 0.2, 2e-06), ((24, 12, 16, 20, 24), 0.1, 1e-06)]:
                jobs.append((cfg.changed(mesh=mesh, board_n=4, end_n=8, max_step=step, rtol=rtol), proto, None, 'q2_boundary_fine', True))
            for margin in (0.001, 0.01, 0.1):
                jobs.append((cfg.changed(success_margin=margin), proto, None, 'success_margin', True))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        evaluated = list(pool.map(worker, jobs))
    for kind, mode, T0, name, start in groups:
        replays = evaluated[start:start + 2]
        ts = [r['success_time'] for r in replays]
        confirmed = all((r['feasibility'] == 'feasible' for r in replays)) and abs(ts[0] - ts[1]) < max(0.2, 0.01 * ts[1])
        margins = [dict(margin_C=margin, **r) for margin, r in zip((0.001, 0.01, 0.1), evaluated[start + 2:start + 5])]
        results.append(dict(strategy=kind, mode=mode, T0_C=T0, coarse_run=name, replays=replays, startup_time_agreement=bool(confirmed), margin_sensitivity=margins))
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q2_boundary_fine.json', results)
    return dict(boundaries_replayed=len(results), time_agreement=sum((r['startup_time_agreement'] for r in results)), scope='Agreement of tested grids and finite-search candidates; not a global lower bound')

# ==================== test_events ====================
def test_jump_undervoltage_is_latched():
    c = _question_module('q1').Config(j0=0.1, horizon=2, voltage_limit=0.6, charge_limit=100)
    r = _question_module('q1').simulate(c, _question_module('q1').Protocol('step', (0.005, 0.5, 0.005, 0.5, 1.0)))
    assert r['metrics']['stop_reason'] == 'voltage'
    assert abs(r['metrics']['last_valid_time'] - 0.5) < 1e-07
    assert r['metrics']['success_time'] is None

def test_charge_is_series_not_times_five():
    c = _question_module('q1').Config(cells=5, fixture='H2', j0=0.3, horizon=2, charge_limit=0.05)
    r = _question_module('q1').simulate(c, _question_module('q1').Protocol('constant', (0.05,)))
    assert abs(r['metrics']['last_valid_time'] - 1) < 1e-06
    assert abs(r['metrics']['charge_C_cm2'] - 0.05) < 1e-08

def test_invalid_input_cannot_be_a_feasible_control():
    c = _question_module('q1').Config(horizon=1)
    for p in [_question_module('q1').Protocol('constant', (0.6,)), _question_module('q1').Protocol('constant', (-0.1,)), _question_module('q1').Protocol('zero', powers=(1.1,) * 5, heat_off=1)]:
        r = _question_module('q1').simulate(c, p)
        assert r['metrics']['stop_reason'] == 'control_input'
        assert r['metrics']['feasibility'] == 'infeasible'

# ==================== test_completion ====================
def test_refine_disconnected_feasible_intervals():
    rows = [dict(T0_C=t, status=s) for t, s in [(-6, 'not_found'), (-4, 'feasible'), (-2, 'not_found'), (0, 'feasible')]]
    assert boundary_points(rows, 0.5) == [-5.5, -5.0, -4.5, -3.5, -3.0, -2.5, -1.5, -1.0, -0.5]
    assert boundary_points([dict(T0_C=-4, status='not_found'), dict(T0_C=0, status='not_found')], 0.1) == []

# ==================== 四问命令入口 ====================
def main(argv=None):
    parser=argparse.ArgumentParser(description='Q2：加载策略搜索、最低初温及边界复算')
    parser.add_argument('command',nargs='?',default='results',choices=['results','check','run','scan','refine','replay','figures'])
    parser.add_argument('--budget',type=int,default=80)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv);core=_question_module('q1')
    if args.command=='check':core.self_check();return
    if args.command=='run':
        cfg=core.selected_config();summary=q2(cfg,args.budget,args.workers)
        q2_temperature_scan(cfg,summary,args.workers)
    elif args.command=='scan':
        q2_temperature_scan(core.selected_config(),completion_read('q2_summary.json'),args.workers)
    elif args.command=='refine':refine_q2(core.selected_config(),args.workers)
    elif args.command=='replay':boundary_replay()
    elif args.command=='figures':core.figures()
    if args.command!='results':core.make_tables()
    core.export_results('q2')

if __name__=='__main__':
    main()
