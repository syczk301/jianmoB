"""四问整理版：可直接阅读的合并源码，无隐藏代码包或动态执行源码。
公共物理方程位于q1.py；跨问共享函数通过普通Python模块导入。
公共方程保持不变；第二问默认继承当前第一问候选，默认命令仅整理已有结果。
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

# ==================== 当前第一问参数的第二问复算 ====================
CURRENT_KINDS = ('constant', 'ramp', 'step')
CURRENT_NAMES = {'constant': '恒流策略', 'ramp': '线性升载', 'step': '分段阶梯加载'}
CURRENT_MESH = {14: (3, 2, 3, 3, 3), 48: (12, 6, 8, 10, 12),
                96: (24, 12, 16, 20, 24), 192: (48, 24, 32, 40, 48),
                384: (96, 48, 64, 80, 96)}


def current_config(temp=-10, mesh=14):
    """Transfer the current Q1 candidate explicitly; never select the old fit."""
    core = _question_module('q1')
    source = core.readj(core.OUTPUT / 'voltage_optimization/候选模型.json')
    return core.Config(**source['config']).changed(
        cells=5, fixture='H2', T0=core.TM+temp, ambient=core.TM+temp,
        mesh=CURRENT_MESH[mesh], board_n=max(4,mesh//12), end_n=max(8,mesh//6), horizon=600,
        rtol=2e-6, max_step=.2, sample=.5, charge_limit=20,
        current_limit=.5, voltage_limit=.3, success_margin=.01)


def current_worker(job):
    temp, mesh, protocol, stage, changes = job
    core = _question_module('q1')
    cfg = current_config(temp, mesh).changed(**changes)
    p = core.Protocol(**protocol)
    r = core.evaluate(cfg, p, label='q2_current_'+stage)
    m = r['metrics']
    last_t = np.asarray(r['T'][-1]); last_v = np.asarray(r['V'][-1])
    return dict(**m, temp_C=float(temp), mesh=mesh, stage=stage,
                protocol=protocol, final_T_C=last_t.tolist(), final_V=last_v.tolist(),
                final_ice=r['ice'][-1].tolist(), coldest_cells=(np.flatnonzero(last_t <= last_t.min()+.01)+1).tolist(),
                lowest_voltage_cells=(np.flatnonzero(last_v <= last_v.min()+1e-4)+1).tolist(),
                final_min_T_C=float(last_t.min()), changes=changes)


def current_batch(jobs, workers, name):
    core = _question_module('q1'); dest=core.OUTPUT/'q2_current';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(current_worker,j) for j in jobs]
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            core.write_json(dest/(name+'.json'),rows)
            print(json.dumps(dict(batch=name,done=len(rows),total=len(jobs),temp=row['temp_C'],
                  kind=row['protocol']['kind'],reason=row['stop_reason'],time=row['last_valid_time'],
                  min_T=row['final_min_T_C']),ensure_ascii=True),flush=True)
    return rows


def current_protocols(budget=24):
    core=_question_module('q1');ps=[]
    for j in np.unique(np.r_[.005,.01,.02,.03,.04,.05,.075,.1,.15,.2,.3,.4,.5,
                                np.linspace(.01,.5,max(12,budget))]):
        ps.append(core.Protocol('constant',(float(j),)))
    for kind,dim in [('ramp',3),('step',5)]:
        for point in qmc.LatinHypercube(dim,seed=20260923).random(budget):
            if kind=='ramp':
                a=.001+.079*point[0]; b=a+(.5-a)*point[1]; pars=(a,b,2+118*point[2])
            else:
                a=.001+.059*point[0]; b=a+(.3-a)*point[1]; c=b+(.5-b)*point[2]
                t1=2+38*point[3]; pars=(a,b,c,t1,t1+2+78*point[4])
            ps.append(core.Protocol(kind,tuple(float(v) for v in pars)))
        # Include the constant family and slow loading as nested controls.
        for j in (.02,.05,.1,.2,.3):
            ps.append(core.Protocol(kind,(j,j,30.) if kind=='ramp' else (j,j,j,10.,30.)))
    return [vars(p) for p in ps]


def current_search(workers=8,budget=24):
    core=_question_module('q1');dest=core.OUTPUT/'q2_current';dest.mkdir(parents=True,exist_ok=True)
    source=core.OUTPUT/'voltage_optimization/候选模型.json'
    core.write_json(dest/'provenance.json',dict(source=str(source.relative_to(core.io_ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_model=core.readj(source),
        config=current_config().to_dict(),code_hash=core.code_digest(),
        interpretation='Conditional transfer of current Q1 shape_joint; no refitting and no ice15 changes',
        scope='Finite seeded search, not global optimization or an impossibility proof'))
    ps=current_protocols(budget)
    fixed=current_batch([(-10,14,p,'minus10',{}) for p in ps],workers,'minus10')
    # Scan every listed temperature, including disconnected feasibility regions.
    seeds={}
    for kind in CURRENT_KINDS:
        family=[r for r in fixed if r['protocol']['kind']==kind and r['numerically_accepted']]
        warm=sorted(family,key=lambda r:r['final_min_T_C'],reverse=True)[:2]
        seeds[kind]=[r['protocol'] for r in warm]
        seeds[kind]+=[vars(core.Protocol(kind,(j,) if kind=='constant' else
            (j,j,30.) if kind=='ramp' else (j,j,j,10.,30.))) for j in (.03,.1,.3)]
        # A low initial load followed by stronger self-heating.
        if kind=='ramp':seeds[kind].append(vars(core.Protocol(kind,(.02,.3,30.))))
        if kind=='step':seeds[kind].append(vars(core.Protocol(kind,(.02,.1,.3,10.,30.))))
    jobs=[]
    for temp in (0,-2,-4,-6,-8,-12,-20,-30,-40):
        for kind in CURRENT_KINDS:
            unique={json.dumps(p,sort_keys=True):p for p in seeds[kind]}
            jobs.extend((temp,14,p,'temperature',{}) for p in unique.values())
    scan=current_batch(jobs,workers,'temperature')
    return dict(fixed_candidates=len(fixed),temperature_candidates=len(scan))


def current_rank(row):
    if not row['numerically_accepted'] or row['stop_reason']=='numerical_failure':return (2,float('inf'))
    if row['feasibility']=='feasible':return (0,row['success_time'])
    return (1,-row['final_min_T_C'])


def test_current_q2_inherits_q1_without_refitting():
    core=_question_module('q1')
    source=core.readj(core.OUTPUT/'voltage_optimization/候选模型.json')['config']
    cfg=current_config()
    for key in ('j0','tau_b','tau_f','lambda0','cl_proton_loss','cathode_hydration_exponent','vapor_equilibrium','h','shared_bp'):
        assert getattr(cfg,key)==source[key]
    assert (cfg.cells,cfg.fixture,cfg.charge_limit,cfg.current_limit,cfg.voltage_limit)==(5,'H2',20,.5,.3)
    assert cfg.cl_proton_loss and cfg.cathode_hydration_exponent>0


def current_neighbors(protocol,fraction):
    core=_question_module('q1');kind=protocol['kind'];v0=np.array(protocol['params'],float)
    n=1 if kind=='constant' else 2 if kind=='ramp' else 3
    ps=[]
    for k in range(len(v0)):
        for sign in (-1,1):
            v=v0.copy();v[k]+=sign*fraction*(.5 if k<n else 100)
            v[:n]=np.sort(np.clip(v[:n],.001,.5));v[n:]=np.sort(np.clip(v[n:],.1,180))
            ps.append(vars(core.Protocol(kind,tuple(v))))
    return ps


def current_refine(workers=8):
    core=_question_module('q1');dest=core.OUTPUT/'q2_current'
    fixed=core.readj(dest/'minus10.json');extra=[]
    for it,fraction in enumerate((.12,.05,.02)):
        jobs=[]
        for kind in CURRENT_KINDS:
            seeds=sorted([r for r in fixed+extra if r['protocol']['kind']==kind],key=current_rank)[:2]
            ps=[p for r in seeds for p in current_neighbors(r['protocol'],fraction)]
            unique={json.dumps(p,sort_keys=True,default=core.json_default):p for p in ps}
            jobs.extend((-10,14,p,'local'+str(it),{}) for p in unique.values())
        extra+=current_batch(jobs,workers,'local'+str(it))
    core.write_json(dest/'minus10_refined.json',fixed+extra)
    scan=core.readj(dest/'temperature.json')+fixed+extra
    for spacing in (.5,.1):
        jobs=[]
        for kind in CURRENT_KINDS:
            family=[r for r in scan if r['protocol']['kind']==kind]
            temperatures=sorted(set(r['temp_C'] for r in family))
            records=[dict(T0_C=t,status='feasible' if any(r['feasibility']=='feasible' for r in family if r['temp_C']==t) else 'not_found') for t in temperatures]
            for temp in boundary_points(records,spacing):
                nearest=sorted(family,key=lambda r:(abs(r['temp_C']-temp),current_rank(r)))
                feasible=sorted([r for r in family if r['feasibility']=='feasible'],key=lambda r:(abs(r['temp_C']-temp),current_rank(r)))
                ps=[nearest[0]['protocol']]+[r['protocol'] for r in feasible[:2]]
                # Local reoptimization at each transition, not only fixed transfer.
                ps+=current_neighbors(ps[-1],.025 if spacing==.1 else .075)
                unique={json.dumps(p,sort_keys=True,default=core.json_default):p for p in ps}
                jobs.extend((temp,14,p,'boundary_'+str(spacing),{}) for p in unique.values())
        scan+=current_batch(jobs,workers,'boundary_'+str(spacing))
        core.write_json(dest/'temperature_refined.json',scan)
    return dict(fixed=len(fixed+extra),all_rows=len(scan))


def current_verify(workers=8):
    core=_question_module('q1');dest=core.OUTPUT/'q2_current'
    fixed=core.readj(dest/'minus10_refined.json');scan=core.readj(dest/'temperature_refined.json')
    jobs=[]
    for kind in CURRENT_KINDS:
        ordered=sorted([r for r in fixed if r['protocol']['kind']==kind],key=current_rank)
        unique={}
        for r in ordered:
            unique.setdefault(json.dumps(r['protocol'],sort_keys=True),r)
        for representative in list(unique.values())[:3]:
            for mesh in (48,96):jobs.append((-10,mesh,representative['protocol'],'fixed_fine',{}))
        good=[r for r in scan if r['protocol']['kind']==kind and r['feasibility']=='feasible']
        if good:
            cold=min(r['temp_C'] for r in good)
            winner=min([r for r in good if r['temp_C']==cold],key=current_rank)
            for mesh in (48,96):
                for temp in (cold,round(cold-.1,6)):
                    jobs.append((temp,mesh,winner['protocol'],'boundary_fine',{}))
    rows=current_batch(jobs,workers,'verification')
    # If refinement moves the boundary, test a wider bracket on the 96-cell mesh.
    for kind in CURRENT_KINDS:
        edge=[r for r in rows if r['protocol']['kind']==kind and r['stage']=='boundary_fine' and r['mesh']==96]
        if not edge:continue
        last=max(edge,key=lambda r:r['temp_C'])
        if all(r['feasibility']=='feasible' for r in edge):direction=-1;last=min(edge,key=lambda r:r['temp_C'])
        elif all(r['feasibility']!='feasible' for r in edge):direction=1
        else:continue
        for step in range(1,11):
            temp=round(last['temp_C']+direction*.1*step,6)
            trial=current_batch([(temp,96,last['protocol'],'fine_bracket',{})],1,'bracket_'+kind+'_'+str(step))[0]
            rows.append(trial);core.write_json(dest/'verification.json',rows)
            if (trial['feasibility']=='feasible') != (last['feasibility']=='feasible'):break
    # One stricter replay per strategy at its lowest fine-grid feasible temperature.
    jobs=[]
    for kind in CURRENT_KINDS:
        fixed_fine=[r for r in rows if r['mesh']==96 and r['temp_C']==-10 and r['protocol']['kind']==kind]
        if fixed_fine:
            win=min(fixed_fine,key=current_rank)
            jobs.append((-10,192,win['protocol'],'fixed192',{'rtol':5e-7,'max_step':.1}))
        good=[r for r in rows if r['mesh']==96 and r['protocol']['kind']==kind and r['feasibility']=='feasible']
        if good:
            win=min(good,key=lambda r:(r['temp_C'],r['success_time']))
            jobs.append((win['temp_C'],192,win['protocol'],'final192',{'rtol':5e-7,'max_step':.1}))
    rows+=current_batch(jobs,workers,'final192')
    core.write_json(dest/'verification.json',rows)
    return dict(replays=len(rows))


def current_expression(p):
    v=p['params'];kind=p['kind']
    if kind=='constant':return f'j={v[0]:.6f}'
    if kind=='ramp':return f'j={v[0]:.6f}+({v[1]:.6f}−{v[0]:.6f})min(t/{v[2]:.4f},1)'
    return f'j={v[0]:.6f} (t<{v[3]:.4f}); {v[1]:.6f} ({v[3]:.4f}≤t<{v[4]:.4f}); {v[2]:.6f} (t≥{v[4]:.4f})'


def current_fine_search(workers=8,reuse_verification=True):
    """Reoptimize on 96 cells when coarse-grid voltage constraints do not transfer."""
    core=_question_module('q1');dest=core.OUTPUT/'q2_current'
    rows=core.readj(dest/'verification.json') if reuse_verification and (dest/'verification.json').exists() else []
    coarse=core.readj(dest/'minus10_refined.json');jobs=[]
    if not rows:
        initial_jobs=[]
        for kind in CURRENT_KINDS:
            family=[r for r in coarse if r['protocol']['kind']==kind and r['numerically_accepted']]
            # Charge-limited near misses provide stable initial seeds; very high
            # current voltage-failing traces must not dominate the warm-end score.
            ranked=sorted(family,key=lambda r:(r['stop_reason'] not in ('success','charge'),current_rank(r)))
            unique={}
            for r in ranked:unique.setdefault(json.dumps(r['protocol'],sort_keys=True),r['protocol'])
            initial_jobs.extend((-10,96,p,'initial96',{}) for p in list(unique.values())[:3])
        rows=current_batch(initial_jobs,workers,'initial96')
        core.write_json(dest/'verification.json',rows)
    for kind in CURRENT_KINDS:
        available=[r for r in rows if r['temp_C']==-10 and r['mesh']==96 and r['protocol']['kind']==kind]
        if not available:available=[r for r in coarse if r['protocol']['kind']==kind]
        seed=min(available,key=current_rank)['protocol'];ps=[seed]
        n=1 if kind=='constant' else 2 if kind=='ramp' else 3
        for shift in (-.01,-.02):
            p=dict(seed);v=list(p['params']);v[n-1]=max(v[n-2] if n>1 else .001,v[n-1]+shift);p['params']=v;ps.append(p)
        if n>1:
            p=dict(seed);v=list(p['params']);v[0]=min(v[1],v[0]+.01);p['params']=v;ps.append(p)
        for temp in (-10,-9.6,-9.2,-8.8):
            jobs.extend((temp,96,p,'fine_search',{}) for p in ps)
    rows+=current_batch(jobs,workers,'fine_search')
    core.write_json(dest/'verification.json',rows)
    jobs=[]
    for kind in CURRENT_KINDS:
        family=[r for r in rows if r['mesh']==96 and r['protocol']['kind']==kind]
        temperatures=sorted(set(r['temp_C'] for r in family))
        status=[dict(T0_C=t,status='feasible' if any(r['feasibility']=='feasible' for r in family if r['temp_C']==t) else 'not_found') for t in temperatures]
        for temp in boundary_points(status,.1):
            good=sorted([r for r in family if r['feasibility']=='feasible'],key=lambda r:(abs(r['temp_C']-temp),current_rank(r)))
            near=min(family,key=lambda r:(abs(r['temp_C']-temp),current_rank(r)))
            ps=[near['protocol']]+[r['protocol'] for r in good[:2]]
            unique={json.dumps(p,sort_keys=True,default=core.json_default):p for p in ps}
            jobs.extend((temp,96,p,'fine_boundary',{}) for p in unique.values())
    rows+=current_batch(jobs,workers,'fine_boundary')
    core.write_json(dest/'verification.json',rows)
    jobs=[]
    for kind in CURRENT_KINDS:
        family=[r for r in rows if r['mesh']==96 and r['protocol']['kind']==kind]
        representative=min([r for r in family if r['temp_C']==-10],key=current_rank)
        jobs.append((-10,192,representative['protocol'],'fixed192',{'rtol':5e-7,'max_step':.1}))
        good=[r for r in family if r['feasibility']=='feasible']
        if good:
            winner=min(good,key=lambda r:(r['temp_C'],r['success_time']))
            # Include the next colder point with the exact same protocol.
            for t in (winner['temp_C'],round(winner['temp_C']-.1,6)):
                jobs.append((t,192,winner['protocol'],'final192',{'rtol':5e-7,'max_step':.1}))
    rows+=current_batch(jobs,workers,'final192')
    core.write_json(dest/'verification.json',rows)
    # If both points fail, restore a feasible 192-cell point without claiming the old boundary.
    jobs=[]
    for kind in CURRENT_KINDS:
        fine=[r for r in rows if r['protocol']['kind']==kind and r['stage']=='final192']
        if fine and not any(r['feasibility']=='feasible' for r in fine):
            seed=max(fine,key=lambda r:r['temp_C'])
            for delta in (.1,.2,.4):jobs.append((round(seed['temp_C']+delta,6),192,seed['protocol'],'final192_warm',{'rtol':5e-7,'max_step':.1}))
    if jobs:
        rows+=current_batch(jobs,workers,'final192_warm');core.write_json(dest/'verification.json',rows)
    core.write_json(dest/'completion.json',dict(complete=True,verification_rows=len(rows),source_hash=hashlib.sha256((core.OUTPUT/'voltage_optimization/候选模型.json').read_bytes()).hexdigest()))
    return dict(replays=len(rows))


def current_edge_search(workers=8):
    """Retest colder sampled temperatures with newly feasible fine-grid protocols."""
    core=_question_module('q1');dest=core.OUTPUT/'q2_current';rows=core.readj(dest/'verification.json');jobs=[]
    for kind in CURRENT_KINDS:
        good=sorted([r for r in rows if r['mesh']==96 and r['protocol']['kind']==kind and r['feasibility']=='feasible'],key=lambda r:(r['temp_C'],r['success_time']))
        if not good:continue
        cold=good[0]['temp_C'];unique={}
        for r in good:unique.setdefault(json.dumps(r['protocol'],sort_keys=True),r['protocol'])
        for temp in (round(cold-.1,6),round(cold-.2,6)):
            jobs.extend((temp,96,p,'edge96',{}) for p in list(unique.values())[:2])
    edge=current_batch(jobs,workers,'edge96');jobs=[]
    for kind in CURRENT_KINDS:
        good=[r for r in edge if r['protocol']['kind']==kind and r['feasibility']=='feasible']
        if good:
            win=min(good,key=lambda r:(r['temp_C'],r['success_time']))
            for temp in (win['temp_C'],round(win['temp_C']-.1,6)):
                jobs.append((temp,192,win['protocol'],'edge192',{'rtol':5e-7,'max_step':.1}))
    return current_batch(jobs,workers,'edge192')


def current_final_check(workers=6):
    """Test charge-limited near misses where discretization can change classification."""
    core=_question_module('q1');dest=core.OUTPUT/'q2_current';rows=core.readj(dest/'verification.json')
    for name in ('edge96','edge192'):
        if (dest/(name+'.json')).exists():rows+=core.readj(dest/(name+'.json'))
    jobs=[]
    for kind in CURRENT_KINDS:
        good=[r for r in rows if r['mesh']==192 and r['protocol']['kind']==kind and r['feasibility']=='feasible']
        cold=min((r['temp_C'] for r in good),default=0)
        near=sorted([r for r in rows if r['mesh']==96 and r['protocol']['kind']==kind and r['stop_reason']=='charge'
                     and -.12<r['final_min_T_C']<.01 and r['temp_C']<cold],key=lambda r:-r['final_min_T_C'])
        unique={}
        for r in near:unique.setdefault(json.dumps(r['protocol'],sort_keys=True),r)
        for r in list(unique.values())[:2]:
            for temp in (r['temp_C'],round(r['temp_C']+.1,6),round(r['temp_C']-.1,6)):
                if any(x['mesh']==192 and x['temp_C']==temp and x['protocol']==r['protocol'] for x in rows):continue
                jobs.append((temp,192,r['protocol'],'near_miss192',{'rtol':5e-7,'max_step':.1}))
    return current_batch(jobs,workers,'near_miss192')


def current_adaptive_check(workers=2):
    """Resolve a successful 192-cell point whose 96-cell counterpart failed."""
    core=_question_module('q1');dest=core.OUTPUT/'q2_current';rows=core.readj(dest/'verification.json')
    for name in ('edge96','edge192','near_miss192'):
        if (dest/(name+'.json')).exists():rows+=core.readj(dest/(name+'.json'))
    jobs=[];seen=set()
    for high in rows:
        if high['mesh']!=192 or high['feasibility']!='feasible':continue
        low=[r for r in rows if r['mesh']==96 and r['temp_C']==high['temp_C'] and r['protocol']==high['protocol']]
        if low and all(r['feasibility']!='feasible' for r in low):
            for temp in (high['temp_C'],round(high['temp_C']-.1,6)):
                key=(temp,json.dumps(high['protocol'],sort_keys=True))
                if key in seen:continue
                seen.add(key);jobs.append((temp,384,high['protocol'],'adaptive384',{'rtol':5e-7,'max_step':.1}))
    return current_batch(jobs,workers,'adaptive384')


def current_report():
    """All tables and figures derive from saved numerical trajectories."""
    core=_question_module('q1');src=core.OUTPUT/'q2_current'
    dest=core.io_ROOT/'results/q2/current_q1';dest.mkdir(parents=True,exist_ok=True)
    fixed=core.readj(src/'minus10_refined.json');scan=core.readj(src/'temperature_refined.json');fine=core.readj(src/'verification.json')
    for name in ('edge96','edge192','near_miss192','adaptive384'):
        if (src/(name+'.json')).exists():fine+=core.readj(src/(name+'.json'))
    if not all(any(r['stage']=='fixed192' and r['protocol']['kind']==k for r in fine) for k in CURRENT_KINDS):
        raise RuntimeError('192格复算尚未完成；不能导出定稿表3')
    provenance=core.readj(src/'provenance.json')
    source_path=core.io_ROOT/provenance['source']
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==provenance['source_sha256']
    allrows=scan+fine
    for row in allrows:
        if row['feasibility']=='feasible':
            assert row['numerically_accepted'] and row['stop_reason']=='success'
            assert row['charge_C_cm2']<=20+1e-7 and row['V_min']>=.3-1e-7 and row['ice_max']<.99
    headers=['加载策略','最优加载参数','启动时间/s','累计电荷量/C·cm⁻²','最大电流密度/A·cm⁻²','最低电压/V','最大冰体积分数','启动结果']
    table=[];diagnostic=[];boundaries=[];selected={}
    for kind in CURRENT_KINDS:
        rr=[r for r in fine if r['protocol']['kind']==kind and r['temp_C']==-10 and r['mesh']==192]
        r=min(rr,key=current_rank);selected[kind]=r
        run=core.load_run(r['candidate_id']);jmax=float(run['j'].max())
        if r['feasibility']=='feasible':
            table.append([CURRENT_NAMES[kind],current_expression(r['protocol']),f"{r['success_time']:.3f}",f"{r['charge_C_cm2']:.4f}",f'{jmax:.6f}',f"{r['V_min']:.6f}",f"{r['ice_max']:.6g}",'成功（已核查候选）'])
        else:table.append([CURRENT_NAMES[kind],'未获得可行最优参数','—','—','—','—','—','本次搜索未找到可行解'])
        diagnostic.append(dict(策略=CURRENT_NAMES[kind],加载参数=current_expression(r['protocol']),终止原因=core.STOP_REASONS.get(r['stop_reason'],r['stop_reason']),
             终止时间_s=r['last_valid_time'],累计电荷_C_cm2=r['charge_C_cm2'],最大电流_A_cm2=jmax,
             最低电压_V=r['V_min'],最大冰体积分数=r['ice_max'],终止最冷单片温度_C=r['final_min_T_C'],
             最冷单片='、'.join(map(str,r['coldest_cells'])),最低电压单片='、'.join(map(str,r['lowest_voltage_cells'])),运行编号=r['candidate_id']))
        latest={}
        for candidate in [r for r in fine if r['protocol']['kind']==kind and r['mesh']>=192]:
            key=(candidate['temp_C'],json.dumps(candidate['protocol'],sort_keys=True))
            if key not in latest or candidate['mesh']>latest[key]['mesh']:latest[key]=candidate
        good=[r for r in latest.values() if r['feasibility']=='feasible']
        if good:
            cold=min(x['temp_C'] for x in good)
            win=max([x for x in good if x['temp_C']==cold],key=lambda x:(x['mesh'],-x['success_time']))
            fail=[x for x in fine if x['protocol']==win['protocol'] and x['mesh']==win['mesh'] and x['temp_C']<cold and x['feasibility']!='feasible' and x['numerically_accepted']]
            colder=max(fail,key=lambda x:x['temp_C']) if fail else None
            boundaries.append(dict(策略=CURRENT_NAMES[kind],最低已验证可行初温_C=cold,网格=win['mesh'],启动时间_s=win['success_time'],
                累计电荷_C_cm2=win['charge_C_cm2'],最低电压_V=win['V_min'],最大冰体积分数=win['ice_max'],
                同策略较冷失败初温_C=colder['temp_C'] if colder else None,
                较冷失败原因=core.STOP_REASONS[colder['stop_reason']] if colder else '尚未括住',
                较冷终止最冷温度_C=colder['final_min_T_C'] if colder else None,
                较冷最冷单片=colder['coldest_cells'] if colder else [],
                较冷最低电压单片=colder['lowest_voltage_cells'] if colder else [],
                加载参数=current_expression(win['protocol']),运行编号=win['candidate_id']))
    pd.DataFrame(table,columns=headers).to_csv(dest/'表3_当前第一问模型.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostic).to_csv(dest/'负10度代表候选诊断.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(boundaries).to_csv(dest/'最低初温核查.csv',index=False,encoding='utf-8-sig')
    flat=[]
    for r in allrows:
        flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))}|dict(strategy=r['protocol']['kind'],protocol_json=json.dumps(r['protocol']),coldest_cells=str(r['coldest_cells'])))
    pd.DataFrame(flat).to_csv(dest/'全部候选与复算记录.csv',index=False,encoding='utf-8-sig')
    convergence=[]
    for high in [r for r in fine if r['mesh']>=192]:
        matches=[r for r in fine if r['mesh']==high['mesh']//2 and r['temp_C']==high['temp_C'] and r['protocol']==high['protocol']]
        if not matches:continue
        low=matches[-1];a=core.load_run(low['candidate_id']);b=core.load_run(high['candidate_id'])
        end=min(a['t'][-1],b['t'][-1]);t=np.unique(np.r_[a['t'][a['t']<=end],b['t'][b['t']<=end]])
        diffs={key:float(max(np.max(np.abs(np.interp(t,a['t'],a[key][:,k])-np.interp(t,b['t'],b[key][:,k]))) for k in range(5))) for key in ('T','V','ice')}
        same=low['stop_reason']==high['stop_reason'];dt=abs(low['last_valid_time']-high['last_valid_time'])
        relative_ice=abs(low['ice_max']-high['ice_max'])/max(high['ice_max'],1e-8)
        passed=same and dt<=max(.2,.01*high['last_valid_time']) and diffs['T']<.05 and diffs['V']<.005 and relative_ice<.05
        convergence.append(dict(策略=CURRENT_NAMES[high['protocol']['kind']],初温_C=high['temp_C'],低网格=low['mesh'],高网格=high['mesh'],
             终止类型一致=same,终止时间差_s=dt,最大温差_C=diffs['T'],最大电压差_V=diffs['V'],冰峰相对差=relative_ice,
             全部门槛通过=passed,低网格运行=low['candidate_id'],高网格运行=high['candidate_id']))
    pd.DataFrame(convergence).to_csv(dest/'网格核查.csv',index=False,encoding='utf-8-sig')
    mechanisms=[]
    for row in [r for r in fine if r['mesh']>=192]:
        run=core.load_run(row['candidate_id']);model=core.Model(core.Config(**run['config']))
        state=model.phases(run['y'][-1]);voltage=model.voltage(state,run['j'][-1])
        for cell in range(5):
            mechanisms.append(dict(策略=CURRENT_NAMES[row['protocol']['kind']],初温_C=row['temp_C'],网格=row['mesh'],单片=cell+1,
                终止原因=core.STOP_REASONS[row['stop_reason']],终止时间_s=row['last_valid_time'],
                温度_C=run['T'][-1,cell],电压_V=run['V'][-1,cell],局部最大冰体积分数=run['ice'][-1,cell],
                活化损失_V=voltage['activation'][cell],欧姆损失_V=voltage['ohmic'][cell],
                催化层质子传导损失_V=voltage['cl_ohmic'][cell],浓差损失_V=voltage['concentration'][cell],
                阳极催化层最小含水量=float(state['lam'][cell,model.g.cla].min()),
                阴极催化层最小含水量=float(state['lam'][cell,model.g.clc].min()),
                累计反应热_J=float(run['y'][-1,model.il+5+cell]*core.AREA),运行编号=row['candidate_id']))
    pd.DataFrame(mechanisms).to_csv(dest/'关键单片与电压损失分解.csv',index=False,encoding='utf-8-sig')
    # Figure contract: quantitative grid; each panel shows one physical constraint.
    # No smoothing, no extrapolation after a terminal event, all five cells shown.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
       'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'font.size':9,
       'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    colors=['#33658A','#648C75','#C18D53','#8B7298','#A04F56'];samples=[]
    fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for col,kind in enumerate(CURRENT_KINDS):
        r=core.load_run(selected[kind]['candidate_id']);t=r['t']
        for k,color in enumerate(colors):
            for row,key in enumerate(('T','V','ice')):
                axs[row,col].plot(t,r[key][:,k],color=color,lw=1.1,label=f'第{k+1}片')
            for i,time_s in enumerate(t):samples.append(dict(策略=CURRENT_NAMES[kind],时间_s=time_s,单片=k+1,温度_C=r['T'][i,k],电压_V=r['V'][i,k],最大冰体积分数=r['ice'][i,k],电流密度_A_cm2=r['j'][i]))
        axs[0,col].set_title(CURRENT_NAMES[kind]+'：−10 ℃代表候选')
        axs[0,col].axhline(0,color='.5',ls='--',lw=.8);axs[1,col].axhline(.3,color='.5',ls='--',lw=.8)
        if float(r['ice'].max())<1e-12:
            axs[2,col].set_ylim(0,1e-6)
            axs[2,col].text(.5,.65,'冰量仅为数值舍入量级',ha='center',transform=axs[2,col].transAxes,color='.35')
        for row,label in enumerate(('单片平均温度 / ℃','单片电压 / V','局部最大冰体积分数')):
            axs[row,col].set_ylabel(label);axs[row,col].set_xlabel('时间 / s')
        axs[0,col].legend(ncol=3,fontsize=7)
    fig.savefig(dest/'负10度三类策略诊断.png',dpi=300)
    fig.savefig(dest/'负10度三类策略诊断.svg')
    fig.savefig(dest/'负10度三类策略诊断.pdf')
    plt.close(fig)
    if boundaries:
        fig,axs=plt.subplots(3,len(boundaries),figsize=(4*len(boundaries),9),squeeze=False,layout='constrained');boundary_samples=[]
        for col,item in enumerate(boundaries):
            run=core.load_run(item['运行编号']);t=run['t']
            axs[0,col].plot(t,run['j'],color='#33658A',lw=1.4)
            axs[0,col].set_title(f"{item['策略']}：{item['最低已验证可行初温_C']:g} ℃")
            axs[0,col].set_ylabel('电流密度 / (A/cm²)')
            for k,color in enumerate(colors):
                axs[1,col].plot(t,run['T'][:,k],color=color,lw=1.1,label=f'第{k+1}片')
                axs[2,col].plot(t,run['V'][:,k],color=color,lw=1.1)
                for i,tt in enumerate(t):boundary_samples.append(dict(策略=item['策略'],初温_C=item['最低已验证可行初温_C'],时间_s=tt,单片=k+1,温度_C=run['T'][i,k],电压_V=run['V'][i,k],冰体积分数=run['ice'][i,k],电流密度_A_cm2=run['j'][i]))
            axs[1,col].axhline(0,color='.5',ls='--',lw=.8);axs[2,col].axhline(.3,color='.5',ls='--',lw=.8)
            axs[1,col].set_ylabel('单片平均温度 / ℃');axs[2,col].set_ylabel('单片电压 / V')
            axs[1,col].legend(ncol=3,fontsize=7)
            for ax in axs[:,col]:ax.set_xlabel('时间 / s')
        fig.savefig(dest/'最低可行初温下的加载与响应.png',dpi=300)
        fig.savefig(dest/'最低可行初温下的加载与响应.svg')
        fig.savefig(dest/'最低可行初温下的加载与响应.pdf');plt.close(fig)
        pd.DataFrame(boundary_samples).to_csv(dest/'最低初温中文图源数据.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(samples).to_csv(dest/'负10度中文图源数据.csv',index=False,encoding='utf-8-sig')
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for n,kind in enumerate(CURRENT_KINDS):
        family=[r for r in scan if r['protocol']['kind']==kind]
        temps=sorted(set(r['temp_C'] for r in family));times=[];feasible=[]
        for temp in temps:
            good=[r for r in family if r['temp_C']==temp and r['feasibility']=='feasible']
            feasible.append(bool(good));times.append(min((r['success_time'] for r in good),default=np.nan))
        axs[0].plot(temps,times,'o-',ms=3,color=colors[n],label=CURRENT_NAMES[kind])
        axs[1].scatter(temps,np.array(feasible,dtype=float)+n*.06,s=18,color=colors[n],label=CURRENT_NAMES[kind])
    axs[0].set(xlabel='初始温度 / ℃',ylabel='已找到候选的最短启动时间 / s',title='粗网格搜索结果')
    axs[1].set(xlabel='初始温度 / ℃',ylabel='搜索状态',title='逐温度可行性（标记略作错位）',yticks=[0,1],yticklabels=['未找到','已找到'])
    for ax in axs:ax.legend(fontsize=8)
    fig.savefig(dest/'初始温度搜索.png',dpi=300)
    fig.savefig(dest/'初始温度搜索.svg')
    fig.savefig(dest/'初始温度搜索.pdf')
    plt.close(fig)
    def md_table(columns,rows):
        return ['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]
    lines=['# 第二问：基于当前第一问模型的重新计算','',
      '第一问参数直接来自 shape_joint 候选；未重新拟合，未采用已放弃的15秒提前结冰分支。第一问的温度、波形及冰量网格验证仍有未通过项，因此本报告为该模型下的条件性计算。','',
      '## 模型与约束','',
      '五片串联、同一电流密度，十块2 mm双极板、两块10 mm端板；显式板件导热与外表面对流，h=40 W/(m²·K)。继承第一问MEA平均温度定义；端板与双极板参与能量方程。电流密度≤0.5 A/cm²，累计电荷≤20 C/cm²；所有单片全过程电压≥0.30 V，任一跌破即终止。成功要求最冷单片超过0 ℃（数值裕度0.01 ℃），冰体积分数<0.99。搜索时限600 s，达到时限只记未解决，不记物理不可能。','',
      '## 表3：−10 ℃冷启动','']+md_table(headers,table)
    lines+=['','“未找到”表示有限候选搜索及复算未取得可行解，不能把失败轨迹的终止时刻填写成启动时间，也不证明所有可能策略不可行。','',
      '## 代表候选失败诊断','']+md_table(['策略','终止原因','终止时间/s','电荷/C·cm⁻²','最冷温度/℃','最冷单片'],[
       [r['策略'],r['终止原因'],f"{r['终止时间_s']:.3f}",f"{r['累计电荷_C_cm2']:.4f}",f"{r['终止最冷单片温度_C']:.4f}",r['最冷单片']] for r in diagnostic])
    lines+=['','完整加载表达式、电压与冰量见《负10度代表候选诊断.csv》。失败候选按终止时最冷单片温度排序，仅用于诊断，不称作最优可行策略。','',
       '## 最低初温','']+md_table(['策略','最低已验证可行初温/℃','网格','启动时间/s','较冷失败点/℃','失败原因'],[
        [r['策略'],r['最低已验证可行初温_C'],r['网格'],f"{r['启动时间_s']:.3f}",r['同策略较冷失败初温_C'],r['较冷失败原因']] for r in boundaries])
    lines+=['','最低初温是本次搜索发现并在192格或384格复算的可行点；较冷失败点仅针对相同加载参数，不能作为全部策略的不可行下界。粗网格先逐温度搜索，再在观察到的转换区间以0.5 ℃及0.1 ℃细化；不预设可行域严格单调。发现粗网格电压约束不能稳定迁移后，增加96格参数搜索，再以192格核查；对96格失败而192格成功的临界点，补做384格复算。','',
      '## 网格对照','']+md_table(['策略','初温/℃','网格对','终止类型一致','终止时间差/s','最大温差/℃','最大电压差/mV','冰峰相对差/%','全部通过'],[
       [r['策略'],r['初温_C'],f"{r['低网格']}/{r['高网格']}",r['终止类型一致'],f"{r['终止时间差_s']:.3f}",f"{r['最大温差_C']:.4f}",f"{1000*r['最大电压差_V']:.3f}",f"{100*r['冰峰相对差']:.2f}",r['全部门槛通过']] for r in convergence])
    lines+=['','同时细化膜电极、双极板和端板网格；对共同有效时段逐单片比较。门槛为终止类型一致、终止时差≤max(0.2 s,1%)、最大温差<0.05 ℃、最大电压差<5 mV、冰峰相对差<5%。未全部通过时，不声称完整状态轨迹已网格无关；192格成功只表示该离散模型的可行点。','',
      '## 数值核查与来源','',f'−10 ℃粗网格候选评估记录数：{len(fixed)}；全部粗网格搜索记录：{len(scan)}；精细搜索及复算记录：{len(fine)}。',
      f"源参数SHA-256：`{provenance['source_sha256']}`。",
      '原始候选逐条保存终止原因、守恒残差、运行编号和加载参数。数值失败不能当作物理失败，也不能参与可行解排序。图使用完整输出轨迹，终止事件后不外推。','',
      '控制式中t以秒计、j以A/cm²计。线性策略升到峰值后保持；阶梯策略含两个切换时刻。公式显示值经过舍入，复算使用《来源及验证.json》中的全精度参数。小温差边界取0.01 ℃裕度，不等同于题目严格大于0 ℃的数学下确界。','',
      '## 图','', '![负10度三类策略诊断](负10度三类策略诊断.png)','', '![最低初温下的响应](最低可行初温下的加载与响应.png)','', '![初始温度搜索](初始温度搜索.png)']
    lines+=['','## 边界加载参数与关键单片','']
    for item in boundaries:
        lines += [f"- {item['策略']}，{item['最低已验证可行初温_C']} ℃：{item['加载参数']}。"]
        if item['同策略较冷失败初温_C'] is not None:
            lines += [f"  同参数降至{item['同策略较冷失败初温_C']} ℃，因{item['较冷失败原因']}终止；最冷单片为{item['较冷最冷单片']}，其温度仍为{item['较冷终止最冷温度_C']:.4f} ℃。"]
    lines+=['','端板热容和外表面散热使端部升温落后；具体瓶颈由逐单片温度、电压和电荷事件共同识别。阴阳极层次结构保留方向性，第1片与第5片不强行对称。电压分解与各片催化层最低含水量见《关键单片与电压损失分解.csv》；不能仅由电压降低推断已发生严重冰堵。']
    if max(r['最大冰体积分数'] for r in diagnostic)<1e-12:
        lines+=['','本轮−10 ℃代表候选的最大冰体积分数仅为约10⁻¹⁹量级，属于数值舍入量级。故这些轨迹的失败不能归因为冰体积分数达到0.99；限制来自升温、电荷预算和电压约束。图中冰量纵轴统一至少显示到10⁻⁶，防止将舍入噪声放大解释为真实积冰。该结论属于当前模型，并非实测冰量验证。']
    central=[r for r in mechanisms if r['初温_C']==-10 and r['单片']==3]
    if central:
        lines+=['','−10 ℃代表候选终止时，第3片的催化层质子传导损失为'+
            f"{min(r['催化层质子传导损失_V'] for r in central):.3f}–{max(r['催化层质子传导损失_V'] for r in central):.3f} V，"+
            f"阳极催化层最低含水量约{min(r['阳极催化层最小含水量'] for r in central):.2f}–{max(r['阳极催化层最小含水量'] for r in central):.2f}。"+
            '这说明应重点关注含水状态和传导损失；电压下降不等同于冰堵。']
    rejected=[r for r in fine if r['mesh']==384 and r['feasibility']!='feasible' and any(
        low['mesh']==192 and low['temp_C']==r['temp_C'] and low['protocol']==r['protocol'] and low['feasibility']=='feasible' for low in fine)]
    for r in rejected:
        lines += ['',f"**临界点撤回：** {CURRENT_NAMES[r['protocol']['kind']]}在{r['temp_C']} ℃的某候选于192格曾判成功，384格却在{r['last_valid_time']:.3f} s因{core.STOP_REASONS[r['stop_reason']]}终止（最冷单片{r['final_min_T_C']:.5f} ℃）。该点已从最终可行初温表剔除；不使用较粗网格覆盖较细网格的失败结果。"]
    (dest/'第二问计算报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    index=dest.parent/'结果说明.md';legacy=dest.parent/'旧冻结模型结果说明.md'
    if index.exists() and not legacy.exists():legacy.write_text(index.read_text(encoding='utf-8'),encoding='utf-8')
    index.write_text('# Q2 当前结果\n\n本轮第二问直接继承当前第一问 shape_joint 参数。\n\n'
        '[完整计算报告、表3与最低初温](current_q1/第二问计算报告.md)\n\n'
        '[表3 CSV](current_q1/表3_当前第一问模型.csv) · [最低初温 CSV](current_q1/最低初温核查.csv)\n\n'
        '此前旧冻结模型的表格和计算目录作为历史记录保留；见[旧模型说明](旧冻结模型结果说明.md)。'
        '第一问尚有验证未通过项，本轮结果为当前模型下的条件性计算。\n',encoding='utf-8',newline='\n')
    core.write_json(dest/'来源及验证.json',dict(provenance=provenance,fine=fine,selected=selected,boundaries=boundaries,
       figure_contract=dict(archetype='quantitative grid',backend='python',claim='解释当前模型的启动限制与有限搜索边界',
       source='全部候选与复算记录.csv / 负10度中文图源数据.csv',formats=['png','svg','pdf'],smoothing=False,visual_QA='pending')))
    print(json.dumps(dict(table3=table,boundaries=boundaries,report=str(dest/'第二问计算报告.md')),ensure_ascii=True),flush=True)
    return dest


# ==================== 四问命令入口 ====================
def main(argv=None):
    parser=argparse.ArgumentParser(description='Q2：加载策略搜索、最低初温及边界复算')
    parser.add_argument('command',nargs='?',default='results',choices=['results','check','run','legacy-run','scan','refine','replay','figures','current-search','current-refine','current-verify','current-fine','current-edge','current-final','current-adaptive','current-report'])
    parser.add_argument('--budget',type=int,default=80)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv);core=_question_module('q1')
    if args.command=='run':
        current_search(args.workers,args.budget)
        current_refine(args.workers)
        current_fine_search(args.workers,reuse_verification=False)
        current_edge_search(args.workers)
        current_final_check(args.workers)
        current_adaptive_check(min(args.workers,2))
        current_report();return
    if args.command in ('results','figures') and (core.OUTPUT/'q2_current/verification.json').exists():
        current_report();return
    if args.command=='current-search':
        print(current_search(args.workers,args.budget));return
    if args.command=='current-refine':
        print(current_refine(args.workers));return
    if args.command=='current-verify':
        print(current_verify(args.workers));return
    if args.command=='current-fine':
        print(current_fine_search(args.workers));return
    if args.command=='current-edge':
        print({'final_edge_replays':len(current_edge_search(args.workers))});return
    if args.command=='current-final':
        print({'near_miss_replays':len(current_final_check(args.workers))});return
    if args.command=='current-adaptive':
        print({'adaptive_replays':len(current_adaptive_check(min(args.workers,2)))});return
    if args.command=='current-report':
        current_report();return
    if args.command=='check':core.self_check();return
    if args.command=='legacy-run':
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
