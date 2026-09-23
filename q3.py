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
    sys.modules.setdefault('q3', sys.modules[__name__])
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
def q3(cfg, budget=80, workers=4, Tfield=None, label='q3'):
    cfg = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, horizon=300, charge_limit=1000000.0, voltage_limit=0.3)
    output = {}
    for mode in ('preheat', 'coheat'):
        ps = []
        if mode == 'coheat':
            ps.append(_question_module('q1').Protocol('fixed'))
        for power in (0.25, 0.5, 0.75, 1.0):
            for cutoff in (40, 80, 160, 300):
                ps.append(_question_module('q1').Protocol('zero' if mode == 'preheat' else 'fixed', powers=(power,) * 5, heat_off=cutoff))
        for powers in [(1.0, 0.2, 0.0, 0.2, 1.0), (0.9, 0.4, 0.2, 0.4, 0.9), (0.8, 0.3, 0.1, 0.3, 0.8), (1.0, 0.15, 0.1, 0.15, 0.8)]:
            for cutoff in (40, 80, 160, 300):
                ps.append(_question_module('q1').Protocol('zero' if mode == 'preheat' else 'fixed', powers=powers, heat_off=cutoff))
        points = qmc.LatinHypercube(6, seed=cfg.seed + (mode == 'coheat')).random(budget)
        for p in points:
            ps.append(_question_module('q1').Protocol('zero' if mode == 'preheat' else 'fixed', powers=tuple(0.1 + 0.9 * p[:5]), heat_off=float(10 + 290 * p[5])))
        rows = _question_module('q2').batch(cfg, ps, Tfield, label=label + '_' + mode, workers=workers, stop_success=mode == 'preheat')
        rows += _question_module('q2').refine_neighbors(cfg, rows, 'E_aux_total_J', workers, Tfield, label + '_refine_' + mode, mode == 'preheat')
        if mode == 'preheat':
            for row in rows:
                if row['startup_result'] == 'success' and row.get('numerically_accepted'):
                    row['feasibility'] = 'feasible'
                    row['th_startup'] = row['success_time']
                    row['shutdown_status'] = 'at_event'
                    row['protocol']['heat_off'] = row['success_time']
        else:
            for row in rows:
                if row['feasibility'] == 'feasible' and row['th_startup'] > row['success_time'] + 1e-07:
                    row['feasibility'] = 'infeasible'
                    row['stop_reason'] = 'nominal_heat_after_success'
        win = _question_module('q2').save_search(label + '_' + mode, rows, 'E_aux_total_J')
        if win:
            proto = _question_module('q1').Protocol(**win['protocol'])
            fine = cfg.changed(mesh=(12, 6, 8, 10, 12), board_n=4, end_n=8, max_step=0.2, rtol=2e-06)
            if Tfield is None:
                if mode == 'preheat':
                    proto.heat_off = cfg.horizon
                replay = _question_module('q1').evaluate(fine, proto, label=label + '_fine', stop_success=mode == 'preheat')
                win['fine_replay'] = replay['metrics']
            original = _question_module('q1').load_run(win['candidate_id'])
            m = _question_module('q1').Model(cfg)
            state = original['y'][-1].copy()
            state[m.il:] = 0
            qual = _question_module('q1').simulate(cfg.changed(horizon=70), _question_module('q1').Protocol('fixed', load_start=0 if mode == 'preheat' else -original['t'][-1]), stop_success=False, initial=state, complete_horizon=True)
            qual['metrics']['qualification_pass'] = bool(qual['t'][-1] >= 70 - 1e-07 and np.min(qual['T']) > 0 and (qual['metrics']['V_min'] >= 0.3 - 1e-08))
            _question_module('q1').save_run(qual, label + '_' + mode + '_qualification')
            win['qualification'] = qual['metrics']
            if mode == 'coheat':
                cap = _question_module('q1').evaluate(cfg.changed(charge_limit=20), proto, Tfield, label=label + '_charge20', stop_success=False)
                win['charge20_sensitivity'] = cap['metrics']
        output[mode] = win
    _question_module('q1').write_json(_question_module('q1').OUTPUT / f'{label}_summary.json', output)
    return output

# ==================== completion ====================
def qualify_q3():
    """Independent 70-second load replay from the saved fine-grid terminal state."""
    summary = _question_module('q2').completion_read('q3_summary.json')
    results = {}
    for mode, winner in summary.items():
        if winner is None:
            results[mode] = 'no_officially_feasible_candidate'
            continue
        fine = winner.get('second_fine_replay', winner.get('fine_replay', {}))
        if fine.get('feasibility') != 'feasible':
            results[mode] = 'fine_replay_not_feasible'
            continue
        r = _question_module('q1').load_run(fine['candidate_id'])
        cfg = _question_module('q1').Config(**r['config'])
        m = _question_module('q1').Model(cfg)
        state = r['y'][-1].copy()
        state[m.il:] = 0
        proto = _question_module('q1').Protocol('fixed', load_start=0 if mode == 'preheat' else -r['t'][-1])
        qual = _question_module('q1').simulate(cfg.changed(horizon=70), proto, initial=state, stop_success=False, complete_horizon=True)
        name = 'q3_' + mode + '_qualification_70_fine'
        qual['metrics'].update(candidate_id=name, qualification_pass=bool(qual['t'][-1] >= 70 - 1e-07 and np.min(qual['T']) > 0 and (np.min(qual['V']) >= 0.3 - 1e-08)), qualification_min_V=float(np.min(qual['V'])), qualification_min_T_C=float(np.min(qual['T'])))
        _question_module('q1').save_run(qual, name, proto)
        winner['qualification'] = qual['metrics']
        results[mode] = qual['metrics']
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q3_summary.json', summary)
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'load_qualification.json', results)
    return results

def q3_control_convergence():
    summary = _question_module('q2').completion_read('q3_summary.json')
    comparisons = []
    for mode, winner in summary.items():
        if winner is None or winner.get('fine_replay', {}).get('feasibility') != 'feasible':
            continue
        a = _question_module('q1').load_run(winner['fine_replay']['candidate_id'])
        cfg = _question_module('q1').Config(**a['config'])
        proto = _question_module('q1').Protocol(**winner['protocol'])
        if mode == 'preheat':
            proto.heat_off = cfg.horizon
        r = _question_module('q1').evaluate(cfg.changed(mesh=tuple((2 * n for n in cfg.mesh)), max_step=cfg.max_step / 2, rtol=cfg.rtol / 2), proto, label='q3_fine96', stop_success=mode == 'preheat')
        b = r['metrics']
        x = a['metrics']
        ts = b.get('success_time')
        row = dict(mode=mode, run_48=x['candidate_id'], run_96=b['candidate_id'], status_96=b['feasibility'], time_difference_s=None if ts is None else abs(ts - x['success_time']), ice_max_48=x['ice_max'], ice_max_96=b['ice_max'], minimum_voltage_difference_V=abs(b['V_min'] - x['V_min']))
        row['time_pass'] = bool(ts is not None and row['time_difference_s'] < max(0.2, 0.01 * ts))
        row['ice_relative_difference'] = abs(b['ice_max'] - x['ice_max']) / max(abs(b['ice_max']), 1e-08)
        row['ice_peak_5pct_pass'] = row['ice_relative_difference'] < 0.05
        row['voltage_1mV_pass'] = row['minimum_voltage_difference_V'] < 0.001
        comparisons.append(row)
        winner['second_fine_replay'] = b
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q3_summary.json', summary)
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q3_control_convergence.json', comparisons)
    return comparisons

def horizon_expansion():
    results = []
    for path in _question_module('q1').OUTPUT.glob('q[23]_*_search.json'):
        candidates = [r for r in _question_module('q2').completion_read(path.name)['candidates'] if r['stop_reason'] == 'horizon']
        if not candidates:
            continue
        candidate = max(candidates, key=lambda r: np.min(_question_module('q1').load_run(r['candidate_id'])['T'][-1]))
        original = _question_module('q1').load_run(candidate['candidate_id'])
        cfg = _question_module('q1').Config(**original['config'])
        proto = _question_module('q1').Protocol(**candidate['protocol'])
        for horizon in (600, 1200):
            r = _question_module('q1').evaluate(cfg.changed(horizon=horizon), proto, label='horizon_extension', stop_success=True)
            results.append(dict(search=path.stem, original=candidate['candidate_id'], horizon_s=horizon, **r['metrics']))
            if r['metrics']['stop_reason'] != 'horizon':
                break
    selected = _question_module('q2').completion_read('selected_model.json')
    cfg = _question_module('q1').Config(**selected['config']).changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, charge_limit=1000000.0, horizon=600)
    winner = _question_module('q2').completion_read('q3_summary.json').get('coheat')
    probes = [_question_module('q1').Protocol('fixed'), _question_module('q1').Protocol('fixed', powers=(0.05,) * 5, heat_off=40), _question_module('q1').Protocol('fixed', powers=(0.1,) * 5, heat_off=40)]
    if winner:
        p = _question_module('q1').Protocol(**winner['protocol'])
        probes.extend((_question_module('q1').Protocol('fixed', powers=tuple(np.array(p.powers) * scale), heat_off=p.heat_off) for scale in (0.5, 0.75, 1.0)))
    for proto in probes:
        for horizon in (600, 1200):
            r = _question_module('q1').evaluate(cfg.changed(horizon=horizon), proto, label='q3_domain_extension', stop_success=False)
            results.append(dict(search='q3_coheat_domain_extension', protocol=vars(proto), horizon_s=horizon, **r['metrics']))
            if r['metrics']['stop_reason'] != 'horizon':
                break
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'horizon_extension.json', results)
    return dict(evaluations=len(results), feasible=sum((r['feasibility'] == 'feasible' for r in results)), scope='Selected horizon-limited representatives; not an exhaustive infinite-time search')

# ==================== energy_extension ====================
def energy_extension_read(name):
    return json.loads((_question_module('q1').OUTPUT / name).read_text(encoding='utf-8'))

def resolve_energy_extension(cfg, workers=8):
    base = cfg.changed(cells=5, fixture='H2', T0=_question_module('q1').TM - 30, ambient=_question_module('q1').TM - 30, horizon=600, charge_limit=1000000.0)
    cases = [base, base.changed(mesh=(12, 6, 8, 10, 12), board_n=4, end_n=8, max_step=0.2, rtol=2e-06), base.changed(mesh=(24, 12, 16, 20, 24), board_n=4, end_n=8, max_step=0.1, rtol=1e-06)]
    path = _question_module('q1').OUTPUT / 'zero_heat_validation.json'
    rows = sorted(energy_extension_read(path.name), key=lambda r: r['n_MEA']) if path.exists() else []
    matching = len(rows) == len(cases) and all((r.get('config_requested') == c.digest and r.get('code_hash') == _question_module('q1').code_digest() for r, c in zip(rows, cases)))
    if not matching:
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as pool:
            rows = list(pool.map(_question_module('q2').worker, [(c, _question_module('q1').Protocol('fixed'), None, 'q3_zero_heat_validation', False) for c in cases]))
        for c, r in zip(cases, rows):
            r['n_MEA'] = sum(c.mesh)
        _question_module('q1').write_json(path, rows)
    rows = sorted(rows, key=lambda r: r['n_MEA'])
    accepted = all((r['feasibility'] == 'feasible' and r.get('numerically_accepted') and (abs(r['E_aux_total_J']) < 1e-08) for r in rows))
    certificate = dict(accepted_on_tested_grids=accepted, physical_prediction_qualified=False, reason='With u>=0, heater energy is nonnegative. A feasible u=0 trajectory attains that mathematical lower bound in the registered model.', horizon_s=600, validation=rows)
    certificate['grid_comparison'] = dict(startup_difference_48_96_s=abs(rows[1]['success_time'] - rows[2]['success_time']) if rows[1]['success_time'] is not None and rows[2]['success_time'] is not None else None, voltage_min_difference_48_96_V=abs(rows[1]['V_min'] - rows[2]['V_min']), ice_max_48=rows[1]['ice_max'], ice_max_96=rows[2]['ice_max'], note='Grid feasibility and startup agreement do not establish convergence of local peak ice')
    compare = certificate['grid_comparison']
    compare['ice_relative_difference'] = abs(rows[1]['ice_max'] - rows[2]['ice_max']) / max(abs(rows[2]['ice_max']), 1e-08)
    compare['ice_peak_5pct_pass'] = compare['ice_relative_difference'] < 0.05
    compare['voltage_min_1mV_pass'] = compare['voltage_min_difference_48_96_V'] < 0.001
    compare['startup_time_pass'] = compare['startup_difference_48_96_s'] is not None and compare['startup_difference_48_96_s'] < max(0.2, 0.01 * rows[2]['success_time'])
    if not accepted:
        for name in ('q3_extended_summary.json', 'q4_extended_comparison.json', 'q4_extended_scan.json'):
            (_question_module('q1').OUTPUT / name).unlink(missing_ok=True)
        _question_module('q1').write_json(_question_module('q1').OUTPUT / 'energy_lower_bound.json', certificate)
        return certificate
    assert all((r['config_requested'] == c.digest for r, c in zip(rows, cases))), 'Stale zero-heat validation configuration'
    winner = dict(rows[0], fine_replay=rows[1], second_fine_replay=rows[2], energy_lower_bound_J=0.0, search_horizon_s=600)
    r = _question_module('q1').load_run(rows[2]['candidate_id'])
    fine = _question_module('q1').Config(**r['config'])
    m = _question_module('q1').Model(fine)
    state = r['y'][-1].copy()
    state[m.il:] = 0
    q = _question_module('q1').simulate(fine.changed(horizon=70), _question_module('q1').Protocol('fixed', load_start=-float(r['t'][-1])), initial=state, stop_success=False, complete_horizon=True)
    name = 'zero_heat_load_qualification_' + rows[2]['candidate_id'][-12:]
    q['metrics'].update(candidate_id=name, qualification_pass=bool(q['t'][-1] >= 70 - 1e-07 and np.min(q['T']) > 0 and (np.min(q['V']) >= 0.3 - 1e-08)), qualification_min_V=float(np.min(q['V'])), qualification_min_T_C=float(np.min(q['T'])))
    _question_module('q1').save_run(q, name, _question_module('q1').Protocol('fixed', load_start=-float(r['t'][-1])))
    winner['qualification'] = q['metrics']
    old = energy_extension_read('q3_summary.json')
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q3_extended_summary.json', dict(preheat=old['preheat'], coheat=winner))
    certificate['load_qualification'] = q['metrics']
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'energy_lower_bound.json', certificate)
    q4cfg = base.changed(sample=0.5)
    grid = energy_extension_read('q4_precool_refined.json') if (_question_module('q1').OUTPUT / 'q4_precool_refined.json').exists() else energy_extension_read('q4_precool_constant_scan.json')
    minutes = sorted(set([20.0, 40.0, *[r['precool_minutes'] for r in grid]]))
    _, fields, _, _ = _question_module('q4').precool(q4cfg, minutes=minutes)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        computed = list(pool.map(_question_module('q2').worker, [(q4cfg, _question_module('q1').Protocol('fixed'), f, 'q4_zero_constant', False) for f in [None, *fields]]))
    keyed = {None: computed[0], **dict(zip(minutes, computed[1:]))}
    scan = []
    for minute in minutes:
        r = dict(keyed[minute])
        r['precool_minutes'] = minute
        r['delta_T_max'] = r['shutdown_extrema']['delta_T_max']
        scan.append(r)
    peak = max(scan, key=lambda r: r['delta_T_max'])['precool_minutes']
    extra = sorted(set(range(max(10, int(peak) - 4), min(100, int(peak) + 4) + 1)) - set(minutes))
    if extra:
        _, fields, _, _ = _question_module('q4').precool(q4cfg, minutes=extra)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            additional = list(pool.map(_question_module('q2').worker, [(q4cfg, _question_module('q1').Protocol('fixed'), f, 'q4_zero_constant', False) for f in fields]))
        for minute, r in zip(extra, additional):
            r['precool_minutes'] = minute
            r['delta_T_max'] = r['shutdown_extrema']['delta_T_max']
            scan.append(r)
    comparison = []
    original = energy_extension_read('q4_comparison.json')
    for scenario, minute in [('equilibrium', None), ('20min', 20.0), ('40min', 40.0)]:
        for method in ('constant_transfer', 'constant_reoptimized'):
            comparison.append(dict(scenario=scenario, method=method, energy_lower_bound_attained=True, **keyed[minute]))
        dynamic = next((r for r in original if r['scenario'] == scenario and r['method'].startswith('dynamic')))
        assert dynamic['last_valid_time'] < 300 and dynamic['feasibility'] == 'feasible'
        comparison.append({**dynamic, 'method': 'dynamic_speed_tradeoff', 'original_horizon_s': 300, 'comparison_horizon_s': 600, 'reuse_basis': 'completed causal trajectory before old horizon'})
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_extended_comparison.json', comparison)
    _question_module('q1').write_json(_question_module('q1').OUTPUT / 'q4_extended_scan.json', sorted(scan, key=lambda r: r['precool_minutes']))
    robustness = _question_module('q4').zero_heat_robustness(cfg, workers)
    return dict(zero_heat_verified=True, finest_startup_s=rows[-1]['success_time'], energy_lower_bound_J=0.0, robustness=robustness, load_qualification=q['metrics']['qualification_pass'], extended_q4_cases=len(computed) + len(extra), scope='Conditional mathematical optimum; predictive model qualification remains false')

# ==================== test_events ====================
def test_already_warm_without_heat_has_zero_startup_energy():
    c = _question_module('q1').Config(T0=_question_module('q1').TM + 1, ambient=_question_module('q1').TM + 1, horizon=3, j0=0.3)
    r = _question_module('q1').simulate(c, _question_module('q1').Protocol('fixed'), stop_success=False)
    assert r['metrics']['success_time'] == 0
    assert r['metrics']['last_valid_time'] == 0
    assert r['metrics']['E_aux_total_J'] == 0

def test_qualification_really_runs_the_full_window():
    c = _question_module('q1').Config(T0=_question_module('q1').TM + 1, ambient=_question_module('q1').TM + 1, horizon=1, j0=0.3)
    r = _question_module('q1').simulate(c, _question_module('q1').Protocol('fixed'), stop_success=False, complete_horizon=True)
    assert r['metrics']['last_valid_time'] == 1

# ==================== 四问命令入口 ====================
def main(argv=None):
    parser=argparse.ArgumentParser(description='Q3：辅助加热策略、能耗优化及负载资格')
    parser.add_argument('command',nargs='?',default='results',choices=['results','check','run','convergence','qualify','horizon','extend','figures'])
    parser.add_argument('--budget',type=int,default=80)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv);core=_question_module('q1')
    if args.command=='check':core.self_check();return
    if args.command=='run':q3(core.selected_config(),args.budget,args.workers)
    elif args.command=='convergence':q3_control_convergence()
    elif args.command=='qualify':qualify_q3()
    elif args.command=='horizon':horizon_expansion()
    elif args.command=='extend':resolve_energy_extension(core.selected_config(),args.workers)
    elif args.command=='figures':core.figures()
    if args.command!='results':core.make_tables()
    core.export_results('q3')

if __name__=='__main__':
    main()
