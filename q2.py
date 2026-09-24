#!/usr/bin/env python3
"""第二问：从题设出发，使用 q1_merged 两参数标定模型重新求解。
主判据为包含双极板的单片厚度平均温度，端板单列；MEA口径作敏感性。
搜索结果只对声明的控制族与预算有效，优化失败不是全局不可行证明。
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import ast
import zipfile
import hashlib
import json
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix
from scipy.stats import qmc
from scipy.integrate import trapezoid
import sys
from dataclasses import dataclass

# Missing physics is never replaced by another Q1 or by invented coefficients.
# This control-only class allows --help/preflight/check-controls to run without
# the numerical engine. Any physical computation still requires q1_merged.
try:
    import q1_merged as q
except ModuleNotFoundError as exc:
    if exc.name != 'q1_merged':
        raise
    q = None

@dataclass
class _ControlOnlyProtocol:
    kind: str = 'constant'
    params: tuple = (.1,)
    powers: tuple = (0.,)*5
    heat_off: float = 0.
    load_start: float = 0.
    times: object = None
    currents: object = None
    def j(self, t):
        s = t-self.load_start
        if s < 0:
            return 0.
        if self.kind == 'constant':
            return float(self.params[0])
        if self.kind == 'ramp':
            a,b,tr = self.params
            return float(a+(b-a)*min(s/tr, 1.))
        if self.kind == 'zero':
            return 0.
        raise ValueError(self.kind)
    def u(self, t, cells):
        return np.asarray(self.powers[:cells]) if t < self.heat_off else np.zeros(cells)
    def breaks(self, end):
        b = [0.,end,self.load_start,self.heat_off]
        if self.kind == 'ramp':
            b.append(self.load_start+self.params[2])
        return sorted(set(float(t) for t in b if 0 <= t <= end))
    def valid(self, cfg):
        try:
            validate_control(vars(self), current_limit=cfg.current_limit)
            return True
        except (ValueError,TypeError):
            return False

class _MissingPhysicalModel:
    def __init__(self, *args, **kwargs):
        require_engine()

ROOT = Path(__file__).resolve().parent
LEGACY_OUT = ROOT/'results/q2/q1_merged'
OUT = Path(os.environ.get('Q2_OUTPUT_DIR', str(ROOT/'results/q2/control_fix'))).resolve()
FIT = Path(os.environ.get('Q2_FIT_FILE', str(ROOT/'q1_merged_results/calibration/fit.json'))).resolve()
BASE_MODEL = q.Model if q is not None else _MissingPhysicalModel

CODE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
KINDS = ('constant','ramp','step')
NAMES = dict(constant='恒流',ramp='线性升载',step='分段阶梯')

class StepProtocol(q.Protocol if q is not None else _ControlOnlyProtocol):
    """Arbitrary finite piecewise-constant current with explicit switch events."""
    def j(self,t):
        if t < self.load_start:return 0.
        n=(len(self.params)+1)//2
        index=np.searchsorted(self.params[n:],t-self.load_start,side='right')
        return float(self.params[index])
    def breaks(self,end):
        n=(len(self.params)+1)//2
        return sorted(set(float(t) for t in [0.,end,self.heat_off,self.load_start]+[self.load_start+x for x in self.params[n:]] if 0<=t<=end))
    def valid(self,cfg):
        n=(len(self.params)+1)//2
        return (len(self.params)==2*n-1 and n>=2 and np.isfinite(self.params).all()
            and min(self.params[:n])>=0 and max(self.params[:n])<=cfg.current_limit
            and min(self.params[n:])>0 and np.all(np.diff(self.params[n:])>0)
            and self.load_start>=0 and len(self.powers)>=cfg.cells
            and np.isfinite(self.powers).all() and min(self.powers)>=0 and max(self.powers)<=1 and self.heat_off>=0)


def protocol(*args, **kwargs):
    base = q.Protocol if q is not None else _ControlOnlyProtocol
    p = base(*args, **kwargs)
    return StepProtocol(**vars(p)) if p.kind == 'step' else p



def level_count(p):
    return (len(p['params'])+1)//2 if p['kind']=='step' else 2 if p['kind']=='ramp' else 1


class StackModel(BASE_MODEL):
    """Exact assembly of Q1 cell kernels and the original solid-stack heat faces."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.fast = q.USE_NUMBA and cfg.cells == 5 and (cfg.fixture == 'H2') and (not cfg.shared_bp) and (not cfg.contact_r) and (not cfg.half_cl) and (cfg.end_concentration_factor == 1.0)
        if not self.fast:
            return
        self.local = BASE_MODEL(cfg.changed(cells=1, fixture='H1', h=0.0))
        g = self.g
        local = self.local
        self.local.rhs(0.0, self.local.initial(), 0.1, np.zeros(1))
        self.maps = []
        self.scatter = []
        labels = np.full(g.nt, -1, dtype=int)
        for cell, unit in enumerate(g.unit_indices):
            ids = np.r_[np.concatenate([np.arange(b * self.c * self.n + cell * self.n, b * self.c * self.n + (cell + 1) * self.n) for b in range(4)]), self.ih + unit, self.il + cell, self.il + self.c + cell, self.il + 2 * self.c + cell, self.il + 3 * self.c, self.il + 3 * self.c + 1]
            if cfg.nucleation == 'median':
                ids = np.r_[ids, self.ledger_end + cell]
            self.maps.append(ids)
            self.scatter.append(ids[:local.il + 3])
            labels[unit] = cell
        self.links = np.flatnonzero((labels[:-1] != labels[1:]) | (labels[:-1] < 0))
        self.link_k = 1 / (g.tdx[self.links] / (2 * g.k0[self.links]) + g.tdx[self.links + 1] / (2 * g.k0[self.links + 1]))
        self.storage = g.C * g.tdx
        self.edge_k = np.array([1 / (1 / cfg.h + g.tdx[i] / (2 * g.k0[i])) if cfg.h else 0.0 for i in (0, -1)])

    def rhs(self, t, y, j, u, details=False):
        if not self.fast or details:
            return self.reference_rhs(t, y, j, u, details)
        out = np.zeros_like(y)
        energy = 0.0
        for cell, ids in enumerate(self.maps):
            self.local.nucleated[0] = self.nucleated[cell]
            local = self.local.rhs(t, y[ids], j, np.array([u[cell]]))
            out[self.scatter[cell]] = local[:self.local.il + 3]
            if self.cfg.nucleation == 'median':
                out[self.ledger_end + cell] = local[self.local.ledger_end]
            energy += local[self.local.il + 3]
        T = q.TM + y[self.ih:self.il]
        for i, k in zip(self.links, self.link_k):
            flux = k * (T[i] - T[i + 1])
            out[self.ih + i] -= flux / self.storage[i]
            out[self.ih + i + 1] += flux / self.storage[i + 1]
        left = self.edge_k[0] * (self.cfg.ambient - T[0])
        right = self.edge_k[1] * (self.cfg.ambient - T[-1])
        out[self.ih] += left / self.storage[0]
        out[self.il - 1] += right / self.storage[-1]
        out[self.il + 3 * self.c] = energy + left + right
        out[self.ledger_end - 1] = j
        return out

    def jacobian(self, t, y, j, u):
        if not self.fast:
            return self._original_jacobian(t, y, j, u)
        rows = []
        cols = []
        vals = []
        for cell, ids in enumerate(self.maps):
            self.local.nucleated[0] = self.nucleated[cell]
            mat = self.local.jacobian(t, y[ids], j, np.array([u[cell]])).tocoo()
            keep = ((mat.row < self.local.il + 3) | ((mat.row == self.local.ledger_end) & (self.cfg.nucleation == 'median'))) & (mat.col < self.local.il)
            rows.extend(ids[mat.row[keep]])
            cols.extend(ids[mat.col[keep]])
            vals.extend(mat.data[keep])
        for i, k in zip(self.links, self.link_k):
            for row, col, value in ((i, i, -k / self.storage[i]), (i, i + 1, k / self.storage[i]), (i + 1, i, k / self.storage[i + 1]), (i + 1, i + 1, -k / self.storage[i + 1])):
                rows.append(self.ih + row)
                cols.append(self.ih + col)
                vals.append(value)
        for i, k in ((0, self.edge_k[0]), (self.g.nt - 1, self.edge_k[1])):
            rows.append(self.ih + i)
            cols.append(self.ih + i)
            vals.append(-k / self.storage[i])
        mat = coo_matrix((vals, (rows, cols)), shape=(self.size, self.size)).tocsc()
        energy = np.asarray(self.storage @ mat[self.ih:self.il, :]).ravel()
        nonzero = np.flatnonzero(energy)
        mat += csc_matrix((energy[nonzero], (np.full(len(nonzero), self.il + 3 * self.c), nonzero)), shape=mat.shape)
        return mat

@lru_cache(maxsize=1)
def fitted_config():
    require_engine()
    if not FIT.is_file():
        raise FileNotFoundError(f"缺少本版本的标定文件: {FIT}。不能用其他 Q1 分支的参数代替。")
    data = q.load_dataset(q.default_input_files())
    record = q.load_fit(FIT, data, expected_seed=q.current_config())
    return q.Config(**record['config'])



def config(temp=-10., grid='i48', fine=False, **changes):
    require_engine()
    cfg = fitted_config().changed(cells=5,fixture='H2',observation='unit',
        T0=q.TM+temp,ambient=q.TM+temp,horizon=1200.,
        voltage_limit=.3,charge_limit=20.,current_limit=.5,success_margin=.01,
        rtol=5e-7 if fine else 1e-5,max_step=.1 if fine else .5,
        sample=.05 if fine else .5,board_n=4,end_n=8)
    grid={'i14':'3,2,3,3,3','i48':'12,6,8,10,12'}.get(grid,grid)
    return q.apply_grid(cfg,grid).changed(**changes)


def install():
    q.Model=StackModel


def job(cfg,p,stage):
    return (cfg.to_dict(),vars(p),stage)


def evaluate(payload):
    cfgdict, pdict, stage = payload
    require_engine()
    install()
    cfg = q.Config(**cfgdict)
    p = protocol(**pdict)
    assert_no_auxiliary(pdict)
    provenance = dict(config=cfgdict, protocol=pdict, q1_hash=q.code_digest(),
        q2_hash=CODE_HASH, fit_hash=hashlib.sha256(FIT.read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(q.plain(provenance), sort_keys=True).encode()).hexdigest()[:24]
    folder = OUT/'runs'/key
    use_cache = os.environ.get('Q2_REUSE_CACHE', '1') == '1' and not stage.startswith('fresh_')
    cache_hit = False
    if use_cache and (folder/'provenance.json').is_file():
        try:
            cache_hit = q.read_json(folder/'provenance.json') == q.plain(provenance)
            if cache_hit:
                run = q.load_run(folder)
        except (OSError, ValueError, KeyError):
            cache_hit = False
    if not cache_hit:
        run = q.simulate(cfg, p)
        q.save_run(run, folder)
        q.write_json(folder/'provenance.json', provenance)
    m = run['metrics'].copy()
    model = StackModel(cfg)
    state = model.phases(run['y'][-1])
    obs = model.observations(state, run['j'][-1])
    charge_series = run['y'][:, model.ledger_end-1]
    audit = audit_trajectory(run, cfgdict, pdict, charge_series)
    initial_field = model.phases(run['y'][0])['T']
    audit['checks']['all_parts_initial_temperature'] = bool(np.max(np.abs(initial_field-cfg.T0)) <= 1e-6)
    audit['passed'] = bool(all(audit['checks'].values()))
    valid = bool(audit['numerically_accepted'])
    raw_feasibility = m.get('feasibility', 'unresolved')
    # Independent audit is veto-only: no failure flag is ever promoted to success.
    if not valid:
        m['feasibility'] = 'unresolved'
    elif raw_feasibility == 'feasible' and not audit['passed']:
        m['feasibility'] = 'unresolved'
    minimum = np.min(run['T'], axis=1)
    peak_index = int(np.argmax(minimum))
    row = dict(m)
    row.update(candidate_id=key, protocol=pdict, stage=stage,
        temp_C=round(cfg.T0-q.TM, 8), grid=cfg.mesh,
        numerically_accepted=valid, observation=cfg.observation,
        final_min_T_C=float(minimum[-1]), final_T_C=run['T'][-1].tolist(),
        final_MEA_T_C=(obs['T_mean_MEA']-q.TM).tolist(), final_V=run['V'][-1].tolist(),
        coldest_cell=int(run['T'][-1].argmin())+1,
        lowest_voltage_cell=int(run['V'][-1].argmin())+1)
    row.update(best_min_T_C=float(minimum[peak_index]),
        best_min_time_s=float(run['t'][peak_index]),
        raw_feasibility=raw_feasibility, constraint_audit=audit, cache_hit=cache_hit,
        q2_hash=CODE_HASH, q1_hash=q.code_digest(), fit_hash=provenance['fit_hash'])
    q.write_json(folder/'q2_audit.json', row)
    return row



def batch(jobs, name, workers=3):
    # Deduplicate before workers write to content-addressed run directories.
    unique_jobs = {}
    for x in jobs:
        key = json.dumps(local_plain(x[:2]), sort_keys=True, allow_nan=False)
        unique_jobs[key] = x
    jobs = list(unique_jobs.values())
    rows = []
    if not jobs:
        q.write_json(OUT/(name+'.json'), rows)
        return rows
    def record(row):
        rows.append(row)
        q.write_json(OUT/(name+'.json'), rows)
        print(name, len(rows), '/', len(jobs), row['temp_C'], row['protocol']['kind'],
            row['stop_reason'], round(row['last_valid_time'], 3),
            round(row['final_min_T_C'], 5), 'peak Tmin=',
            round(row['best_min_T_C'], 5), flush=True)
    if workers == 1:
        for x in jobs:
            record(evaluate(x))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(evaluate, x) for x in jobs]):
                record(future.result())
    return rows



def rank(r):
    if not r.get('numerically_accepted', False):
        return (2, float('inf'), float('inf'), float('inf'))
    if r.get('feasibility') == 'feasible' and r.get('constraint_audit', {}).get('passed', False):
        return (0, float(r['success_time']), 0., 0.)
    # A coast may warm the coldest cell and then cool again. Rank its best
    # attained minimum, not just the last temperature, without calling it success.
    best = float(r.get('best_min_T_C', r['final_min_T_C']))
    remaining = max(0., 20. - float(r.get('charge_C_cm2', 20.)))
    vmargin = float(r.get('V_min', .3)) - .3
    return (1, max(0., .01-best), -remaining, -vmargin)



def unique(ps):
    # Start delay and auxiliary-power settings are part of a control, too.
    result = {}
    for p in ps:
        result[json.dumps(local_plain(vars(p)), sort_keys=True, allow_nan=False)] = p
    return list(result.values())



def family_candidates(budget=120,seed=20260924):
    ps=[protocol('constant',(float(j),)) for j in np.linspace(.01,.5,max(60,budget))]
    for kind,n in [('ramp',3),('step',5)]:
        for index,x in enumerate(qmc.LatinHypercube(n,seed=seed+n).random(budget)):
            if kind=='ramp':
                a,b=np.sort(.5*x[:2]);pars=(a,b,1+199*x[2])
            else:
                # Figure 3 rising steps, with unrestricted steps separately tested.
                a,b,c=(np.sort(.5*x[:3]) if index % 2 else .5*x[:3]);t1=1+79*x[3];pars=(a,b,c,t1,t1+1+119*x[4])
            ps.append(protocol(kind,tuple(pars)))
    for j in np.linspace(.025,.5,20):
        ps.extend([protocol('ramp',(j,j,30.)),protocol('step',(j,j,j,20.,50.))])
    for peak in (.1,.2,.3,.4,.5):
        for duration in (10.,30.,60.,120.):
            ps.append(protocol('ramp',(0.,peak,duration)))
            ps.append(protocol('step',(.2*peak,.6*peak,peak,duration/3,2*duration/3)))
    return unique(ps)


def neighbors(row,fraction,monotone=True):
    p=protocol(**row['protocol']);a=np.array(p.params);n=level_count(vars(p))
    ps=[p]
    indices=range(len(a)) if len(a)<=15 else sorted(set(np.r_[np.linspace(0,n-1,4,dtype=int),np.linspace(n,len(a)-1,4,dtype=int)]))
    for k in indices:
        for sign in (-1,1):
            v=a.copy();v[k]+=sign*fraction*(.5 if k<n else 200.)
            v[:n]=np.clip(v[:n],0.,.5)
            if monotone and p.kind!='step':v[:n]=np.sort(v[:n])
            if n<len(v):v[n:]=np.sort(np.clip(v[n:],.1,600))
            if p.kind=='step' and np.any(np.diff(v[n:])<.01):continue
            ps.append(protocol(p.kind,tuple(v)))
    return ps


def leaders(rows,count=3):
    return [r for k in KINDS for r in sorted([x for x in rows if x['protocol']['kind']==k],key=rank)[:count]]


def search(workers=3,budget=120,temp=-10.,prefix='minus10'):
    coarse=batch([job(config(temp,'i14'),p,'screen') for p in family_candidates(budget)],prefix+'_screen',workers)
    rows=batch([job(config(temp,'i48'),protocol(**r['protocol']),'refine') for r in leaders(coarse,5)],prefix+'_i48',workers)
    for fraction in (.06,.02,.006,.002):
        ps=unique([p for r in leaders(rows,2) for p in neighbors(r,fraction)])
        rows+=batch([job(config(temp,'i48'),p,'refine') for p in ps],prefix+'_refine_'+str(fraction),workers)
    fine=batch([job(config(temp,'i192',True),protocol(**r['protocol']),'final') for r in leaders(rows,2)],prefix+'_i192',workers)
    return select_with_inclusion(fine,temp,prefix,workers)


def select_with_inclusion(rows,temp,prefix,workers):
    c=min([r for r in rows if r['protocol']['kind']=='constant'],key=rank)
    j=c['protocol']['params'][0]
    ps=[protocol('ramp',(j,j,30.)),protocol('step',(j,j,j,20.,50.))]
    rows+=batch([job(config(temp,'i192',True),p,'inclusion') for p in ps],prefix+'_inclusion',workers)
    selected=leaders(rows,1);q.write_json(OUT/(prefix+'_selected.json'),selected)
    return selected


def check():
    records=[]
    for grid in ('i14','i192'):
        m=StackModel(config(-15,grid));rng=np.random.default_rng(104)
        for flags in ([False]*5,[True,False,True,False,True],[True]*5):
            y=m.initial();z,_,_=m.unpack(y)
            z[0,:,m.g.porous]=rng.uniform(.01,2,z[0,:,m.g.porous].shape)
            z[1,:,m.g.porous]=0. if not any(flags) else rng.uniform(.01,.5,z[1,:,m.g.porous].shape)
            y[m.ih:m.il]+=np.linspace(-2,3,m.g.nt);m.nucleated[:]=flags
            a=m.reference_rhs(0,y,.15,np.zeros(5));b=m.rhs(0,y,.15,np.zeros(5))
            error=float(np.linalg.norm(a-b)/np.linalg.norm(a));assert error<1e-10,error
            assert b[m.ledger_end-1]==.15
            assert abs(m.storage@b[m.ih:m.il]-b[m.ledger_end-2])<1e-6
            dz=b[:m.nw].reshape(4,5,m.n)
            assert max(abs(np.sum(dz[:3]*m.g.dx,axis=(0,2))-b[m.il:m.il+5]))<1e-10
            direction=rng.normal(size=m.size)*np.maximum(abs(y),1e-4);direction[m.il:]=0
            numeric=(m.reference_rhs(0,y+1e-7*direction,.15,np.zeros(5))-a)/1e-7
            jacerr=float(np.linalg.norm(m.jacobian(0,y,.15,np.zeros(5))@direction-numeric)/np.linalg.norm(numeric))
            assert jacerr<3e-4,jacerr
            records.append(dict(grid=grid,nucleated=flags,rhs_error=error,jacobian_error=jacerr))
    cfg=config(-25,'i14',True,horizon=.8,nucleation_prefactor=1e17,tau_b=1e12)
    m=StackModel(cfg);y=m.initial();z,_,_=m.unpack(y);z[0,:,m.g.porous]=1.
    q.Model=BASE_MODEL;ref=q.simulate(cfg,protocol('constant',(.1,)),initial=y)
    install();fast=q.simulate(cfg,protocol('constant',(.1,)),initial=y)
    assert all(t is not None for t in fast['metrics']['nucleation_times_s'])
    assert max(abs(ref['T'][-1]-fast['T'][-1]))<1e-5
    assert max(abs(np.array(ref['metrics']['nucleation_times_s'])-np.array(fast['metrics']['nucleation_times_s'])))<1e-4
    small=config(grid='i14',horizon=3.)
    r=q.simulate(small,protocol('step',(.1,.5,.5,1.,2.)))
    assert r['metrics']['stop_reason']=='voltage' and r['metrics']['success_time'] is None
    r=q.simulate(small.changed(charge_limit=.1),protocol('constant',(.1,)))
    assert abs(r['metrics']['last_valid_time']-1)<1e-6
    assert r['metrics']['stop_reason']=='charge'
    p=protocol('step',(.1,.2,.15,.3,1.,2.,3.))
    assert [p.j(t) for t in (0.,1.,2.,3.)]==[.1,.2,.15,.3]
    r=q.simulate(config(grid='i14',horizon=4.),p)
    assert r['metrics']['stop_reason']=='horizon'
    assert abs(r['metrics']['charge_C_cm2']-.75)<1e-7
    q.write_json(OUT/'checks.json',dict(passed=True,records=records,nucleation_events=fast['metrics']['nucleation_times_s'],
        source_hash=CODE_HASH,first_question_hash=q.code_digest()))
    print('Fusion stack equations, five nucleation events, Jacobian, conservation and safety checks passed',flush=True)


def main():
    return control_main()




def boundary(workers=3,budget=80):
    """Separate unchanged-policy transfer from reoptimization at colder temperatures."""
    selected=q.read_json(OUT/'minus10_selected.json')
    temps=(-40.,-30.,-25.,-20.,-15.,-10.,-5.,-1.)
    fixed_path=OUT/'temperature_fixed.json'
    fixed=q.read_json(fixed_path) if fixed_path.exists() else batch([job(config(t,'i48'),protocol(**r['protocol']),'fixed_transfer') for t in temps for r in selected], 'temperature_fixed',workers)
    feasible_temps=[r['temp_C'] for r in fixed if r['feasibility']=='feasible']
    if feasible_temps:
        warm=min(feasible_temps)
        temps=tuple(float(t) for t in temps if warm-10<=t<=warm)
    # If transfer fails everywhere, retain the full domain for fresh controls.
    ps=family_candidates(max(30,budget//2),seed=20260925)
    # Broad temperature screen does not presume monotonic feasibility or an old boundary.
    short=[p for p in ps if p.kind=='constant'][::3]
    short += [p for p in ps if p.kind=='ramp'][::5]+[p for p in ps if p.kind=='step'][::5]
    rows=batch([job(config(t,'i14'),p,'temperature_screen') for t in temps for p in short],
               'temperature_screen',workers)
    jobs=[]
    for t in temps:
        jobs.extend(job(config(t,'i48'),protocol(**r['protocol']),'temperature_replay')
                    for r in leaders([x for x in rows if x['temp_C']==t],1))
    rows=batch(jobs,'temperature_i48',workers)+fixed
    good=[r for r in rows if r['feasibility']=='feasible']
    if not good:
        q.write_json(OUT/'boundary_status.json',dict(status='no_success_in_search',temperatures=temps));return []
    if min(r['temp_C'] for r in good)==min(temps):
        raise RuntimeError('Cold scan endpoint is feasible; extend temperature domain before reporting a boundary')
    for resolution in (1.,.2,.1):
        jobs=[];temperature_sets={}
        broad=family_candidates(max(30,budget//2),seed=20260926)
        for kind in KINDS:
            family=[r for r in good if r['protocol']['kind']==kind]
            if not family:continue
            cold=min(r['temp_C'] for r in family)
            test=np.round(np.arange(cold-5*resolution,cold+.01,resolution),8)
            temperature_sets[kind]=test
            seeds=sorted(family,key=lambda r:(r['temp_C'],r['success_time']))[:2]
            fresh=[p for p in broad if p.kind==kind]
            fresh=fresh[::max(1,len(fresh)//16)]
            current_selected=q.read_json(OUT/'minus10_selected.json')
            candidates=unique([protocol(**r['protocol']) for r in current_selected if r['protocol']['kind']==kind]+[protocol(**r['protocol']) for r in seeds]+
                [p for r in seeds for f in (.015,.004) for p in neighbors(r,f)]+fresh)
            jobs.extend(job(config(float(t),'i14'),p,'boundary_search') for t in test for p in candidates)
        coarse=batch(jobs,'boundary_screen_'+str(resolution),workers)
        jobs=[]
        for kind,test in temperature_sets.items():
            for t in test:
                candidates=sorted([r for r in coarse if r['protocol']['kind']==kind and r['temp_C']==t],key=rank)[:2]
                jobs.extend(job(config(float(t),'i48'),protocol(**r['protocol']),'boundary_refine') for r in candidates)
        checked=batch(jobs,'boundary_refine_'+str(resolution),workers);rows+=checked
        good=[r for r in rows if r['feasibility']=='feasible']
    # Cold points and warmer fallback points are all replayed; avoid labeling a coarse success as final.
    jobs=[]
    for kind in KINDS:
        family=[r for r in good if r['protocol']['kind']==kind]
        if not family:continue
        lead=min(family,key=lambda r:(r['temp_C'],r['success_time']))
        candidates=[lead]+sorted(family,key=lambda r:(r['temp_C'],r['success_time']))[:3]
        for r in candidates:
            for dt in (-.1,0.,.1):
                jobs.append(job(config(round(r['temp_C']+dt,8),'i192',True),protocol(**r['protocol']),'boundary_final'))
    dedup={json.dumps(q.plain(x[:2]),sort_keys=True):x for x in jobs}
    final=batch(list(dedup.values()),'boundary_i192',workers)
    constant=[r for r in final if r['protocol']['kind']=='constant' and r['feasibility']=='feasible']
    if constant:
        lead=min(constant,key=lambda r:(r['temp_C'],r['success_time']));j=lead['protocol']['params'][0]
        ps=[protocol('ramp',(j,j,30.)),protocol('step',(j,j,j,20.,50.))]
        final+=batch([job(config(lead['temp_C'],'i192',True),p,'boundary_inclusion') for p in ps],
                     'boundary_inclusion',workers)
    chosen=[]
    for kind in KINDS:
        family=[r for r in final if r['protocol']['kind']==kind and r['feasibility']=='feasible']
        if family:chosen.append(min(family,key=lambda r:(r['temp_C'],r['success_time'])))
    q.write_json(OUT/'boundary_final.json',final);q.write_json(OUT/'boundary_selected.json',chosen)
    return chosen


def verify(workers=3,include_boundary=True):
    selected=q.read_json(OUT/'minus10_selected.json')
    if include_boundary:selected+=q.read_json(OUT/'boundary_selected.json')
    sampling_jobs=[];replacements={}
    for row in selected:
        cfg=q.Config(**q.read_json(OUT/'runs'/row['candidate_id']/'config.json'))
        if row['protocol']['kind']=='ramp' and cfg.sample>.025:
            sampling_jobs.append(job(cfg.changed(sample=.025),protocol(**row['protocol']),'sampling_baseline'))
    if sampling_jobs:
        refreshed=batch(sampling_jobs,'sampling_baseline',workers)
        for old in selected:
            for new in refreshed:
                if old['protocol']==new['protocol'] and old['temp_C']==new['temp_C']:
                    replacements[old['candidate_id']]=new
        for name in ('minus10_selected.json','boundary_selected.json'):
            path=OUT/name
            if path.exists():q.write_json(path,[replacements.get(row['candidate_id'],row) for row in q.read_json(path)])
        selected=[replacements.get(row['candidate_id'],row) for row in selected]
        q.write_json(OUT/'sampling_baseline_replacements.json',replacements)
    jobs=[]
    for row in selected:
        p=protocol(**row['protocol']);t=row['temp_C']
        jobs.extend([job(config(t,'i384',True,sample=.025 if p.kind=='ramp' else .05),p,'space'),
            job(config(t,'i192',True,max_step=.05,rtol=2.5e-7,atol_scale=.5,board_n=8,end_n=16,sample=.0125 if p.kind=='ramp' else .025),p,'strict'),
            job(config(t,'i192',True,observation='MEA'),p,'MEA_sensitivity')])
        if row['feasibility']=='feasible':
            jobs.extend(job(config(t,'i192',True,success_margin=margin),p,'margin_'+str(margin)) for margin in (.001,.1))
    cached=[]
    if False:  # Historical equivalence-cache disabled; verify current full source.
        previous=q.read_json(OUT/'verification_minus10.json')
        current=simulation_fingerprint(Path(__file__).read_text())
        allowed={CODE_HASH}
        with zipfile.ZipFile(OUT/'previous_q2_source.zip') as archive:
            for name in archive.namelist():
                source=archive.read(name).decode('utf-8')
                if simulation_fingerprint(source)==current:allowed.add(hashlib.sha256(source.encode()).hexdigest())
        pending=[]
        for payload in jobs:
            match=None
            for row in previous:
                meta=q.read_json(OUT/'runs'/row['candidate_id']/'provenance.json')
                if (meta['q2_hash'] in allowed and meta['q1_hash']==q.code_digest()
                    and meta['fit_hash']==hashlib.sha256(FIT.read_bytes()).hexdigest()
                    and q.plain(meta['config'])==q.plain(payload[0])
                    and q.plain(meta['protocol'])==q.plain(payload[1])):
                    match=dict(row,stage=payload[2]);break
            if match:cached.append(match)
            else:pending.append(payload)
        jobs=pending
        q.write_json(OUT/'verification_reuse.json',dict(reused=len(cached),equivalent_simulation_code_hashes=sorted(allowed)))
    rows=cached+batch(jobs,'verification_pending' if include_boundary else 'verification_pending_minus10',workers)
    q.write_json(OUT/('verification.json' if include_boundary else 'verification_minus10.json'),rows)
    comparisons=[]
    for row in rows:
        base=next(r for r in selected if r['protocol']==row['protocol'] and r['temp_C']==row['temp_C'])
        checks={}
        if row['stage'] in ('space','strict'):
            a=q.load_run(OUT/'runs'/base['candidate_id']);b=q.load_run(OUT/'runs'/row['candidate_id'])
            raw=q.compare_trajectories(a,b,min(a['t'][-1],b['t'][-1]))
            # Q1's completed() expects experiment horizons; Q2 uses terminal events instead.
            checks=raw['individual_checks']
            checks.update(event_type=base['stop_reason']==row['stop_reason'],
                event_time=abs(base['last_valid_time']-row['last_valid_time'])<=.1,
                accepted=base['numerically_accepted'] and row['numerically_accepted'])
            comparisons.append(dict(base=base['candidate_id'],candidate=row['candidate_id'],
                stage=row['stage'],passed=all(checks.values()),checks=checks,
                max_T_difference=raw['next_grid_T_diff'],max_V_difference=raw['next_grid_V_diff'],
                event_time_difference=abs(base['last_valid_time']-row['last_valid_time']),
                ice_curve_difference=raw['phi_max_curve_absolute_diff'],nucleation_converged=raw['nucleation_converged']))
    q.write_json(OUT/('convergence.json' if include_boundary else 'convergence_minus10.json'),comparisons)
    return rows


def formula(p):
    a=p['params']
    if p['kind']=='constant':return f'j={a[0]:.7g}'
    if p['kind']=='ramp':return f'j={a[0]:.7g}+({a[1]:.7g}-{a[0]:.7g})min(t/{a[2]:.7g},1)'
    n=level_count(p)
    return 'j=('+','.join(f'{x:.7g}' for x in a[:n])+'); 切换时刻=('+','.join(f'{x:.7g}' for x in a[n:])+') s'


def diagnostics(row):
    run=q.load_run(OUT/'runs'/row['candidate_id']);model=StackModel(q.Config(**run['config']))
    state=model.phases(run['y'][-1]);obs=model.observations(state,run['j'][-1]);g=model.g
    return dict(candidate_id=row['candidate_id'],temp_C=row['temp_C'],reason=row['stop_reason'],
        nucleation_times_s=row['nucleation_times_s'],cells=[dict(cell=c+1,
            T_unit_C=float(obs['T_mean_unit'][c]-q.TM),T_MEA_C=float(obs['T_mean_MEA'][c]-q.TM),
            voltage=float(obs['V'][c]),activation=float(obs['activation'][c]),ohmic=float(obs['ohmic'][c]),
            CL_ohmic=float(obs['cl_ohmic'][c]),concentration=float(obs['concentration'][c]),
            ice_bulk=float(obs['ice_bulk'][c]),ice_saturation=float(obs['ice_saturation'][c]),
            gas_porosity_min=float(state['eg'][c,g.porous].min()),
            lambda_PEM_min=float(state['lam'][c,g.pem].min()),
            lambda_cCL_mean=float(np.average(state['lam'][c,g.clc],weights=g.dx[g.clc]))) for c in range(5)])


def report():
    selected=q.read_json(OUT/'minus10_selected.json');bounds=q.read_json(OUT/'boundary_selected.json')
    verification=q.read_json(OUT/'verification.json');convergence=q.read_json(OUT/'convergence.json')
    table=[];limits=[]
    for row in selected:
        p=row['protocol'];n=level_count(p)
        checks=[x for x in convergence if x['base']==row['candidate_id']]
        table.append(dict(加载策略=NAMES[p['kind']],加载参数=formula(p),启动时间_s=row['success_time'],
            诊断终止时间_s=row['last_valid_time'],累计电荷_C_cm2=row['charge_C_cm2'],
            最大电流_A_cm2=max(p['params'][:n]),最低电压_V=row['V_min'],最大冰体积分数=row['ice_max'],
            启动结果='成功' if row['feasibility']=='feasible' else '本轮未找到可行解',
            末态最低单片均温_C=row['final_min_T_C'],终止原因=row['stop_reason'],
            数值收敛=bool(len(checks)==2 and all(x['passed'] for x in checks))))
    for row in bounds:
        checks=[x for x in convergence if x['base']==row['candidate_id']]
        limits.append(dict(加载策略=NAMES[row['protocol']['kind']],候选最低初温_C=row['temp_C'],
            加载参数=formula(row['protocol']),启动时间_s=row['success_time'],累计电荷_C_cm2=row['charge_C_cm2'],
            数值收敛=bool(len(checks)==2 and all(x['passed'] for x in checks))))
    pd.DataFrame(table).to_csv(OUT/'表3.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(limits).to_csv(OUT/'最低初温.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(策略=NAMES[r['protocol']['kind']],初温_C=r['temp_C'],检查=r['stage'],
        启动结果=r['feasibility'],启动时间_s=r['success_time'],终止时间_s=r['last_valid_time'],
        最低温度_C=r['final_min_T_C'],最低电压_V=r['V_min']) for r in verification]).to_csv(OUT/'敏感性.csv',index=False,encoding='utf-8-sig')
    failures=[r for r in q.read_json(OUT/'boundary_final.json') if r['feasibility']!='feasible']
    if (OUT/'below_boundary.json').exists():failures+=q.read_json(OUT/'below_boundary.json')
    detail=[diagnostics(r) for r in selected+failures]
    q.write_json(OUT/'逐片失效分解.json',detail)
    causes=[]
    for d in detail:
        cold=min(d['cells'],key=lambda c:c['T_unit_C'])
        low=min(d['cells'],key=lambda c:c['voltage'])
        causes.append(dict(初温_C=d['temp_C'],终止原因=d['reason'],最低温单片=cold['cell'],
            最低单片均温_C=cold['T_unit_C'],最低压单片=low['cell'],最低电压_V=low['voltage'],
            低压片活化损失_V=low['activation'],低压片欧姆损失_V=low['ohmic'],
            低压片浓差损失_V=low['concentration'],最大局部冰体积分数=max(c['ice_bulk'] for c in d['cells'])))
    pd.DataFrame(causes).to_csv(OUT/'失效原因.csv',index=False,encoding='utf-8-sig')
    for row in selected+bounds:
        run=q.load_run(OUT/'runs'/row['candidate_id']);model=StackModel(q.Config(**run['config']))
        series=dict(t_s=run['t'],j_A_cm2=run['j'],Q_C_cm2=run['y'][:,model.ledger_end-1])
        for cell in range(5):
            series[f'T_unit_{cell+1}_C']=run['T'][:,cell]
            series[f'V_{cell+1}_V']=run['V'][:,cell]
            series[f'ice_bulk_{cell+1}']=run['ice'][:,cell]
            series[f'nucleation_hazard_{cell+1}']=run['y'][:,model.ledger_end+cell]
        pd.DataFrame(series).to_csv(OUT/f"轨迹_{row['temp_C']:g}C_{row['protocol']['kind']}.csv",index=False,encoding='utf-8-sig')
    def md(rows):
        if not rows:return '本轮尚无已验证可行点。'
        keys=list(rows[0]);value=lambda x:f'{x:.6g}' if isinstance(x,float) else '—' if x is None else str(x)
        return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+
            ['| '+' | '.join(value(r[k]) for k in keys)+' |' for r in rows])
    text=['# 第二问：融合模型重新求解','',
        '依据原题问题2的五片串联、端板与片间导热要求重新构建优化。唯一物理来源为 `q1_merged.py`，使用其−20 ℃两参数标定结果，不重新拟合；双极板热容系数为1，成核采用累计风险达到ln(2)的中位情景。未引用上一版第二问候选或结论。','',
        '## 题设与明确的建模选择','',
        '- 共用电流密度0≤j≤0.5 A/cm²；累计电荷∫jdt≤20 C/cm²，不乘5；面积25 cm²，无辅助加热。',
        '- 每片保留阳极/阴极各2 mm双极板，两端各10 mm端板，外表面h=40 W/(m²·K)；内部实体界面通量守恒，不重复叠加换热源。',
        '- 主判据Tk为该单片MEA及双极板的厚度平均温度，端板单独建模；这是对题目“单片平均温度”的明确解释。第一问测温口径仍保持MEA，第二问另做MEA判据敏感性。初态所有部件及环境均取T0。',
        '- 全部单片均温越过0 ℃，报告0.01 ℃裕度达成时间，并检查0.001/0.1 ℃。全过程逐片V≥0.30 V；局部冰总体积分数<0.99，同时约束孔隙容量、气体和质子传导有效性。',
        '- 恒流1参数、线性升载3参数、三至四段阶梯（电流水平可增可减，允许零电流阶段），后两类包含恒流退化情形。段数是声明的控制参数化，不是已证明的任意控制最优解。',
        '- 定向改进包括早段省电荷、后段先升后降、四段电流和参数联合扰动；按电压余量构造的多段曲线也须完整开环重放，不能把短段拼接预测直接作为成功证据。',
        '- 先全范围分层采样，再多起点局部精修；可行候选以启动时间排序，失败候选以末态最低温度缺口引导搜索。仿真上限1200 s，达上限与数值失败均不算物理不可行。','',
        '## 表3：−10 ℃','',md(table),'',
        '表中成功解为本轮搜索最优候选，不宣称全局最优；失败行的参数仅作诊断，诊断终止时间不是启动时间。','',
        '## 最低初温','',md(limits),'',
        '先检查固定−10 ℃曲线的迁移，再在不同温度重新搜索；从−40至−1 ℃宽范围筛查并逐步缩小到0.1 ℃温度网格。这里只报告有限搜索的可行温度点及邻域失败证据，不把搜索未找到解当作严格不可行证明。数值收敛为False的行不得称为已验证极限。','',
        '## 约束与原因证据','', '详细的热量分解、改进前后对照与同一曲线降温试验见 [不能启动原因与改进](不能启动原因与改进.md)。', '', md(causes[:len(selected)]), '', '上表为−10 ℃代表候选；更冷失败工况详见 `失效原因.csv`。当最低单片均温仍小于0且电荷已耗尽时，直接约束是热量/电荷预算不足；若电压先触底，应结合损失分解与水合、冰占据判断原因。','',
        '`逐片失效分解.json`给出候选及更冷失败工况的五片温度、电压损失、含水量、孔隙占据及成核时刻；原因必须结合实际先触发的约束判断，不能预先指定为冰堵。',
        '`checks.json`记录融合模型参考方程与加速组装的一致性、逐片成核重启、守恒及约束事件检查；`convergence.json`记录i192/i384及严格时间/热网格下的温压、冰场、成核时刻和终止事件对照。',
        '`敏感性.csv`单列温度口径和严格不等式裕度的影响。成核关系、初始水合及−20 ℃标定向其他温度的外推仍属模型假设，数值收敛不能替代实验验证。','',
        '逐条确定性重放已选控制和实际配置：`python q2.py replay --workers 6`。完整搜索各阶段的源码快照及每条轨迹的来源哈希均保留；重跑整个搜索会按当前搜索实现重新选解，不将旧搜索结果静默视为新结果。', '',
        '重新执行搜索：`python q2.py check` → `python q2.py search --budget 120` → `python q2.py general-step --budget 160` → `python q2.py targeted` → `python q2.py envelope` → `python q2.py joint --workers 6` → `python q2.py idle --workers 6` → `python q2.py boundary --budget 80` → `python q2.py boundary-rescue --workers 6` → `python q2.py verify` → `python q2.py diagnose --workers 6` → `python q2.py report`。','']
    (OUT/'第二问报告.md').write_text('\n'.join(text),encoding='utf-8')
    q.write_json(OUT/'summary.json',dict(table3=table,temperature_limits=limits,q1_hash=q.code_digest(),q2_hash=CODE_HASH))
    print(json.dumps(q.plain(dict(table3=table,limits=limits)),ensure_ascii=False,indent=2))




def general_step(workers=3,budget=160):
    selected=q.read_json(OUT/'minus10_selected.json')
    ps=[p for p in family_candidates(budget,seed=20260927) if p.kind=='step']
    coarse=batch([job(config(grid='i14'),p,'general_step') for p in ps],'general_step_screen',workers)
    rows=batch([job(config(grid='i48'),protocol(**r['protocol']),'general_step') for r in sorted(coarse,key=rank)[:6]],'general_step_i48',workers)
    rows += [r for r in selected if r['protocol']['kind']=='step']
    for fraction in (.04,.012,.003):
        ps=unique([p for r in sorted(rows,key=rank)[:3] for p in neighbors(r,fraction,monotone=False)])
        rows+=batch([job(config(grid='i48'),p,'general_step') for p in ps],'general_step_refine_'+str(fraction),workers)
    final=batch([job(config(grid='i192',fine=True),protocol(**r['protocol']),'general_step') for r in sorted(rows,key=rank)[:2]],'general_step_i192',workers)
    old=next(r for r in selected if r['protocol']['kind']=='step')
    chosen=min(final+[old],key=rank)
    selected=[r if r['protocol']['kind']!='step' else chosen for r in selected]
    q.write_json(OUT/'minus10_selected.json',selected)
    return selected




def targeted(workers=3):
    selected=q.read_json(OUT/'minus10_selected.json');seed=next(r for r in selected if r['protocol']['kind']=='step')
    values=seed['protocol']['params'];n=level_count(seed['protocol'])
    a,b,c=values[:3];t1,t2=values[n:n+2]
    ps=[]
    # Saving early charge and lowering/raising the final load are distinct interventions.
    for first in (0.,.005,.01,.02,.03,.04):
        for middle in (b-.02,b,b+.02):
            for last in (c-.01,c,c+.003,c+.006):
                ps.append(protocol('step',(first,middle,last,t1,t2)))
    for early in (0.,.01,.02,a):
        for peak in (c,.34,.37,.4):
            for switch in (65.,75.,85.):
                for last in (0.,.2,.28,.31):
                    ps.append(protocol('step',(early,b,peak,last,t1,t2,switch)))
    # Linear family: lower first current and altered ramp duration, not a step surrogate.
    ramp=next(r for r in selected if r['protocol']['kind']=='ramp');ra,rb,rt=ramp['protocol']['params']
    for first in (0.,.005,.01,ra):
        for peak in (rb-.01,rb,rb+.005,rb+.015):
            for duration in (rt-12,rt-6,rt,rt+6,rt+12):ps.append(protocol('ramp',(first,peak,duration)))
    coarse=batch([job(config(grid='i14'),p,'targeted') for p in unique(ps)],'targeted_screen',workers)
    rows=batch([job(config(grid='i48'),protocol(**r['protocol']),'targeted') for r in leaders(coarse,5)],'targeted_i48',workers)
    rows += [r for r in selected if r['protocol']['kind']!='constant']
    for fraction in (.02,.008,.002):
        ps=unique([p for r in leaders(rows,2) for p in neighbors(r,fraction,monotone=False)])
        rows+=batch([job(config(grid='i48'),p,'targeted') for p in ps],'targeted_refine_'+str(fraction),workers)
    final=batch([job(config(grid='i192',fine=True),protocol(**r['protocol']),'targeted') for r in leaders(rows,2)],'targeted_i192',workers)
    for kind in ('ramp','step'):
        old=next(r for r in selected if r['protocol']['kind']==kind)
        best=min([old]+[r for r in final if r['protocol']['kind']==kind],key=rank)
        selected=[best if r['protocol']['kind']==kind else r for r in selected]
    q.write_json(OUT/'minus10_selected.json',selected)
    return selected




def envelope(workers=3):
    """State-informed construction only; acceptance always uses a full open-loop replay."""
    install();cfg=config(grid='i48');model=StackModel(cfg);policies=[];construction=[]
    for target in (.305,.32,.35,.4):
        y=model.initial();clock=0.;levels=[];switches=[];history=[]
        for turn in range(100):
            state=model.phases(y);lo=0.;hi=.5
            for _ in range(32):
                mid=(lo+hi)/2
                if model.voltage(state,mid)['V'].min()>=target:lo=mid
                else:hi=mid
            current=lo;remaining=20.-y[model.ledger_end-1]
            if remaining<1e-7 or current<1e-5:break
            saved=y[model.il:model.ledger_end].copy();initial=y.copy()
            initial[model.il:model.ledger_end]=0.
            # A predictor restart must not interpret roundoff ice as a nucleation event.
            z,_,_=model.unpack(initial)
            for cell in range(5):
                if initial[model.ledger_end+cell]<cfg.nucleation_threshold:
                    z[1,cell,np.abs(z[1,cell])<1e-12]=0.
            for retry in range(30):
                run=q.simulate(cfg.changed(horizon=4.,charge_limit=remaining),protocol('constant',(current,)),initial=initial)
                reason=run['metrics']['stop_reason']
                if reason in ('horizon','success','charge'):break
                current*=.96
            if reason not in ('horizon','success','charge'):break
            dt=run['metrics']['last_valid_time']
            if dt<1e-6:break
            if levels:switches.append(clock)
            levels.append(current);clock+=dt;y=run['y'][-1].copy();y[model.il:model.ledger_end]+=saved
            history.append(dict(t=clock,j=current,T_min=float(run['T'][-1].min()),Q=float(y[model.ledger_end-1]),V_min=run['metrics']['V_min']))
            if reason in ('success','charge'):break
        if len(levels)>=2:
            policies.append(protocol('step',tuple(levels+switches)))
            # Controlled simplifications: explicit equal-duration 4/8-stage approximations.
            source=policies[-1]
            for count in (4,8):
                cuts=np.linspace(0,clock,count+1)
                currents=[source.j((cuts[k]+cuts[k+1])/2) for k in range(count)]
                policies.append(protocol('step',tuple(currents+list(cuts[1:-1]))))
        construction.append(dict(target_voltage=target,history=history,construction_only=True))
        q.write_json(OUT/'voltage_guided_construction.json',construction)
        print('voltage guided',target,'stages',len(levels),'Tmin',history[-1]['T_min'] if history else None,flush=True)
    rows=batch([job(config(grid='i48'),p,'voltage_guided') for p in unique(policies)],'voltage_guided_i48',workers)
    selected=q.read_json(OUT/'minus10_selected.json');old=next(r for r in selected if r['protocol']['kind']=='step')
    competitive=[r for r in rows if r['feasibility']=='feasible' or r['final_min_T_C']>=old['final_min_T_C']-.05]
    fine=batch([job(config(grid='i192',fine=True),protocol(**r['protocol']),'voltage_guided') for r in sorted(competitive,key=rank)[:3]],'voltage_guided_i192',workers) if competitive else []
    selected=q.read_json(OUT/'minus10_selected.json');old=next(r for r in selected if r['protocol']['kind']=='step')
    best=min([old]+fine,key=rank)
    q.write_json(OUT/'minus10_selected.json',[best if r['protocol']['kind']=='step' else r for r in selected])
    return fine




def joint(workers=6):
    """Joint parameter perturbations complement coordinate search near active constraints."""
    selected=q.read_json(OUT/'minus10_selected.json');rng=np.random.default_rng(20260928)
    rows=batch([job(config(grid='i48'),protocol(**r['protocol']),'joint_seed') for r in selected if r['protocol']['kind']!='constant'],'joint_seed',workers)
    for iteration,scale in enumerate((1.,.6,.3,.12)):
        ps=[]
        for kind,count in [('ramp',32),('step',64)]:
            parents=sorted([r for r in rows if r['protocol']['kind']==kind],key=rank)[:4]
            for _ in range(count):
                parent=parents[rng.integers(len(parents))];p=parent['protocol'];n=level_count(p)
                a=np.array(p['params']);scales=np.r_[np.full(n,.018),np.full(len(a)-n,8.)]
                v=a+rng.normal(size=len(a))*scales*scale
                v[:n]=np.clip(v[:n],0.,.5)
                if kind=='ramp':v[:n]=np.sort(v[:n])
                v[n:]=np.sort(np.clip(v[n:],.1,300.))
                if kind=='step' and np.any(np.diff(v[n:])<.1):continue
                ps.append(protocol(kind,tuple(v)))
        rows+=batch([job(config(grid='i48'),p,'joint') for p in unique(ps)],'joint_'+str(iteration),workers)
    final=batch([job(config(grid='i192',fine=True),protocol(**r['protocol']),'joint') for r in leaders(rows,2)],'joint_i192',workers)
    selected=q.read_json(OUT/'minus10_selected.json')
    for kind in ('ramp','step'):
        old=next(r for r in selected if r['protocol']['kind']==kind)
        best=min([old]+[r for r in final if r['protocol']['kind']==kind],key=rank)
        selected=[best if r['protocol']['kind']==kind else r for r in selected]
    q.write_json(OUT/'minus10_selected.json',selected)
    return selected




def idle(workers=6):
    ps=[protocol('step',(0.,middle,last,t1,t2)) for t1 in (3.,6.,10.,14.,18.)
        for t2 in (25.,40.,55.) for middle in (.1,.15,.2,.25) for last in (.29,.31,.33,.35)]
    coarse=batch([job(config(grid='i14'),p,'idle') for p in ps],'idle_screen',workers)
    rows=batch([job(config(grid='i48'),protocol(**r['protocol']),'idle') for r in sorted(coarse,key=rank)[:8]],'idle_i48',workers)
    for fraction in (.02,.006):
        ps=unique([p for r in sorted(rows,key=rank)[:2] for p in neighbors(r,fraction,monotone=False)])
        rows+=batch([job(config(grid='i48'),p,'idle') for p in ps],'idle_refine_'+str(fraction),workers)
    fine=batch([job(config(grid='i192',fine=True),protocol(**r['protocol']),'idle') for r in sorted(rows,key=rank)[:2]],'idle_i192',workers)
    selected=q.read_json(OUT/'minus10_selected.json');old=next(r for r in selected if r['protocol']['kind']=='step')
    best=min([old]+fine,key=rank)
    q.write_json(OUT/'minus10_selected.json',[best if r['protocol']['kind']=='step' else r for r in selected])
    return fine




def replay(workers=6):
    rows=q.read_json(OUT/'minus10_selected.json')
    path=OUT/'boundary_selected.json'
    if path.exists():rows+=q.read_json(path)
    jobs=[]
    for row in rows:
        run=q.load_run(OUT/'runs'/row['candidate_id'])
        provenance=q.read_json(OUT/'runs'/row['candidate_id']/'provenance.json')
        assert provenance['q1_hash']==q.code_digest()
        assert provenance['fit_hash']==hashlib.sha256(FIT.read_bytes()).hexdigest()
        jobs.append(job(q.Config(**run['config']),protocol(**row['protocol']),'fresh_replay'))
    return batch(jobs,'selected_replay',workers)




def simulation_fingerprint(source):
    names={'StackModel','StepProtocol','protocol','level_count','config','fitted_config','install','evaluate'}
    nodes=[node for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    return hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def boundary_rescue(workers=6):
    mains=q.read_json(OUT/'minus10_selected.json');bounds=q.read_json(OUT/'boundary_selected.json')
    jobs=[];tests={}
    for kind in KINDS:
        old=next(r for r in bounds if r['protocol']['kind']==kind)
        main=next(r for r in mains if r['protocol']['kind']==kind)
        tests[kind]=np.round(np.arange(old['temp_C']-.3,old['temp_C']+.01,.1),8)
        seeds=[main,old]
        ps=unique([protocol(**r['protocol']) for r in seeds]+[p for r in seeds for f in (.004,.001) for p in neighbors(r,f,False)])
        jobs.extend(job(config(float(t),'i48'),p,'trusted_fine_seed') for t in tests[kind] for p in ps)
    rows=batch(jobs,'boundary_rescue_i48',workers);jobs=[]
    for kind in KINDS:
        good=[r for r in rows if r['protocol']['kind']==kind and r['feasibility']=='feasible']
        if not good:continue
        cold=min(r['temp_C'] for r in good)
        for t in (round(cold-.1,8),cold,round(cold+.1,8)):
            pool=sorted([r for r in rows if r['protocol']['kind']==kind and r['temp_C']==t],key=rank)[:2]
            jobs.extend(job(config(t,'i192',True),protocol(**r['protocol']),'trusted_final') for r in pool)
    extra=batch(jobs,'boundary_rescue_i192',workers)
    final=q.read_json(OUT/'boundary_final.json')+extra
    selected=[]
    for kind in KINDS:
        good=[r for r in final if r['protocol']['kind']==kind and r['feasibility']=='feasible']
        if good:selected.append(min(good,key=lambda r:(r['temp_C'],r['success_time'])))
    q.write_json(OUT/'boundary_final.json',final);q.write_json(OUT/'boundary_selected.json',selected)
    return selected




def energy_diagnosis(row):
    run=q.load_run(OUT/'runs'/row['candidate_id']);m=StackModel(q.Config(**run['config']));g=m.g
    convection=[];produced=[]
    for y,j in zip(run['y'],run['j']):
        state=m.phases(y)
        convection.append(sum(m.edge_k[i]*(state['T'][node]-m.cfg.ambient) for i,node in enumerate((0,-1)))*q.AREA)
        tcl=np.average(state['local_T'][:,g.clc],axis=1,weights=g.dx[g.clc])
        produced.append(q.MW*j*1e4/(2*q.F)*4182*np.sum(tcl-q.TM)*q.AREA)
    delta=run['y'][-1,m.ih:m.il]-run['y'][0,m.ih:m.il];end=g.names=='endplate'
    reaction=float(run['y'][-1,m.il+5:m.il+10].sum()*q.AREA)
    retained=float(np.dot(g.C*g.tdx,delta)*q.AREA)
    plates=float(np.dot((g.C*g.tdx)[end],delta[end])*q.AREA)
    conv=float(trapezoid(convection,run['t']));prod=float(trapezoid(produced,run['t']))
    return dict(kind=row['protocol']['kind'],candidate_id=row['candidate_id'],reaction_heat_J=reaction,
        retained_enthalpy_J=retained,endplate_enthalpy_gain_J=plates,convection_loss_J=conv,
        produced_water_sensible_enthalpy_J=prod,net_water_boundary_enthalpy_out_J=reaction+prod-conv-retained,
        endplate_share_of_retained=plates/retained,
        charge_weighted_stack_mean_voltage=1.48-reaction/(5*q.AREA*1e4*row['charge_C_cm2']))


def diagnose(workers=6):
    mains=q.read_json(OUT/'minus10_selected.json');bounds=q.read_json(OUT/'boundary_selected.json')
    below=batch([job(config(round(r['temp_C']-.1,8),'i192',True,sample=.025 if r['protocol']['kind']=='ramp' else .05),
        protocol(**r['protocol']),'colder_fixed_policy') for r in bounds],'below_boundary',workers)
    energies=[energy_diagnosis(r) for r in mains];q.write_json(OUT/'energy_diagnosis.json',energies)
    detail=[diagnostics(r) for r in mains+below];q.write_json(OUT/'targeted_diagnosis.json',detail)
    best=next(r for r in mains if r['protocol']['kind']=='step');energy=next(r for r in energies if r['kind']=='step')
    cells=next(d for d in detail if d['candidate_id']==best['candidate_id'])['cells']
    cold=min(cells,key=lambda c:c['T_unit_C']);low=min(cells,key=lambda c:c['voltage'])
    before=q.read_json(OUT/'before_targeted.json');comparisons=[]
    for row in mains:
        old=next(r for r in before if r['protocol']['kind']==row['protocol']['kind'])
        comparisons.append(dict(策略=NAMES[row['protocol']['kind']],改进前最低单片均温_C=old['final_min_T_C'],
            改进后最低单片均温_C=row['final_min_T_C'],改进后诊断终止时间_s=row['last_valid_time'],
            累计电荷_C_cm2=row['charge_C_cm2'],最低电压_V=row['V_min']))
    pd.DataFrame(comparisons).to_csv(OUT/'针对性改进对照.csv',index=False,encoding='utf-8-sig')
    def md(rows):
        keys=list(rows[0]);fmt=lambda x:f'{x:.6g}' if isinstance(x,float) else str(x)
        return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+
            ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows])
    cold_table=[]
    for row in below:
        d=next(x for x in detail if x['candidate_id']==row['candidate_id'])
        c=min(d['cells'],key=lambda x:x['T_unit_C']);v=min(d['cells'],key=lambda x:x['voltage'])
        cold_table.append(dict(策略=NAMES[row['protocol']['kind']],初温_C=row['temp_C'],终止原因=row['stop_reason'],
            终止时间_s=row['last_valid_time'],最低温单片=c['cell'],最低温度_C=c['T_unit_C'],
            最低压单片=v['cell'],最低电压_V=v['voltage'],最大冰体积分数=row['ice_max']))
    checks=q.read_json(OUT/'convergence.json') if (OUT/'convergence.json').exists() else []
    text=['# 启动状态、约束与针对性改进','',
        '## 当前结论','',
        f'在固定融合模型、题设热容、h=40、Q≤20 C/cm²和全过程V≥0.30 V条件下，所选策略中可行方案数为{sum(r["feasibility"]=="feasible" for r in mains)}。阶梯候选终止原因为{best["stop_reason"]}，终止时，第{cold["cell"]}片平均温度为{cold["T_unit_C"]:.6f} ℃；这属于有限搜索结果，不是任意控制均不可行的证明。','',
        '## 直接瓶颈','',
        f'最冷单片是第{cold["cell"]}片；第{low["cell"]}片则限制末态电压，其温度已为{low["T_unit_C"]:.3f} ℃，末态电压{low["voltage"]:.6f} V。中位成核时刻为{best["nucleation_times_s"]}；全程最大冰体积分数为{best["ice_max"]:.6g}。是否冰堵应结合孔隙和传质约束判断。',
        f'第{low["cell"]}片末态活化损失约{low["activation"]:.3f} V、欧姆损失约{low["ohmic"]:.3f} V，其中CL欧姆损失约{low["CL_ohmic"]:.3f} V。膜最小含水量λ约{low["lambda_PEM_min"]:.3f}、阴极CL平均λ约{low["lambda_cCL_mean"]:.3f}；后期含水下降与电压裕度缩小有关。冷端需要继续升温，但共用电流不能绕过中间片电压约束。','',
        '## 热量去了哪里','',
        md([dict(能量项='电化学反应热',能量_J=energy['reaction_heat_J']),
            dict(能量项='电堆总焓增量',能量_J=energy['retained_enthalpy_J']),
            dict(能量项='其中：端板焓增量',能量_J=energy['endplate_enthalpy_gain_J']),
            dict(能量项='水分边界交换净带走的焓',能量_J=energy['net_water_boundary_enthalpy_out_J']),
            dict(能量项='外界对流散热',能量_J=energy['convection_loss_J']),
            dict(能量项='生成水显焓输入（相对0 ℃）',能量_J=energy['produced_water_sensible_enthalpy_J'])]),'',
        f'端板吸收约{100*energy["endplate_share_of_retained"]:.1f}%的电堆焓增量；它是储能项，不是额外散热，不能重复相加。水边界焓流包含模型中的气液水能量交换；它不是纯粹的外界对流散热。该轨迹中外界对流只有约{energy["convection_loss_J"]:.1f} J，因此不能把失败主要归因于h过大。',
        '守恒关系：电堆焓增量 = 反应热 + 生成水显焓输入 − 水边界净输出焓 − 外界对流散热。这里没有修改热容或散热系数来制造成功。','',
        '## 已实施的控制改进','',md(comparisons),'',
        '改进前为一般三段阶梯精修后的候选；改进包括前段省电荷、后段先升后降、增加第四段及电流/切换时刻联合扰动。另测试了短暂零电流等待，以及按电压余量构造的多段曲线。后两类没有超过最终四段候选；尽早用大电流耗尽电荷，并不等于端部能及时达温。',
        '所选−10 ℃阶梯参数（电流A/cm²，时间s）：`'+formula(best['protocol'])+'`。可行性以对应轨迹记录为准，有限搜索不证明全局最优。','',
        '## 低于候选边界时，同一条曲线怎样失败','',md(cold_table),'',
        '本表将各策略已验证候选初温降低0.1 ℃，使用完全相同的电流曲线，单独识别先触发的约束。固定曲线失效与重新优化后仍未找到解是不同结论。','',
        '## 数值误差与模型不确定性','',
        f'当前空间与严格积分/热网格对照共{len(checks)}组，全部通过：{bool(checks and all(r["passed"] for r in checks))}。近电压约束的已验证曲线另用i48保留并复查，避免i14切换电压误差造成错误淘汰。',
        '线性升载初始小电流段的输出采样加密到0.025 s，并与0.0125 s对照；保留原1 mV电压差判据，未放宽验收阈值。',
        '主判据为包含双极板的单片厚度平均温度，MEA口径和0.001/0.1 ℃正温度裕度单列于敏感性表。候选边界不是带保持时间和工程裕量的可靠启动保证。',
        f'当前阶梯候选距所设正温裕度的缺口为{max(0.,.01-best["final_min_T_C"]):.6g} ℃。模型预测仍需独立验证；标定RMSE不能直接作为预测误差上界或置信区间。','']
    (OUT/'不能启动原因与改进.md').write_text('\n'.join(text),encoding='utf-8')
    return below


def coast(workers=3):
    seeds = load_control_seeds()
    steps = [p for p in seeds if p['kind'] in ('constant', 'step')]
    if not steps:
        raise ValueError('没有恒流/阶梯种子；先运行 rescue，或用 --seed 指定已有方案文件。')
    jobs = []
    for seed in steps:
        for budget in (16., 18., 19., 19.5, 19.9, 19.99):
            p = cut_at_charge(seed, budget)
            if p is not None:
                jobs.append(job(config(-10., 'i192', True), protocol(**p), 'charge_budgeted_coast'))
    rows = batch(jobs, 'coast_i192', workers)
    q.write_json(OUT/'coast_summary.json', rows)
    coast_report()
    update_step_selection(rows)
    return rows


def coast_report():
    records = []
    for row in q.read_json(OUT/'coast_summary.json'):
        run = q.load_run(OUT/'runs'/row['candidate_id'])
        levels, switches, delay = step_parts(row['protocol'])
        off = delay+switches[-1] if levels[-1] == 0 and switches else None
        indices = np.flatnonzero(run['t'] >= off-1e-9) if off is not None else np.array([], dtype=int)
        reached = bool(indices.size)
        idx = int(indices[0]) if reached else None
        records.append(dict(candidate_id=row['candidate_id'], charge=row['charge_C_cm2'],
            off_s=off, reached_off=reached,
            Tmin_off=float(run['T'][idx].min()) if reached else None,
            best_Tmin_after_off=float(run['T'][idx:].min(axis=1).max()) if reached else None,
            final_Tmin=row['final_min_T_C'], stop_reason=row['stop_reason'],
            feasibility=row['feasibility'], success_time=row['success_time'],
            audit_passed=row['constraint_audit']['passed']))
    q.write_json(OUT/'coast_diagnostics.json', records)
    pd.DataFrame(records).to_csv(OUT/'停载复核.csv', index=False, encoding='utf-8-sig')
    return records




def tail_search(workers=3):
    seeds = load_control_seeds()
    ps = []
    for seed in seeds:
        if seed['kind'] not in ('constant', 'step'):
            continue
        for qfirst in (14., 16., 18., 19.):
            for rest in (0., 2., 8., 20.):
                for current in (.03, .08, .15, .25, .30):
                    p = pulse_tail(seed, qfirst, rest, current, 19.99)
                    if p is not None:
                        ps.append(protocol(**p))
    if not ps:
        raise ValueError('尾段搜索没有可用种子；先运行 rescue 或指定 --seed。')
    rows = batch([job(config(-10., 'i48'), p, 'tail_screen') for p in unique(ps)], 'tail_screen', workers)
    fine = batch([job(config(-10., 'i192', True), protocol(**r['protocol']), 'tail_final')
        for r in sorted(rows, key=rank)[:8]], 'tail_final', workers)
    q.write_json(OUT/'sustained_tail_summary.json', dict(candidates=rows, fine=fine,
        status='finite_search_only', constraints_unchanged=True))
    update_step_selection(fine)
    return fine





# ==================== 受约束的 -10 C 控制修正 ====================
# No thermal mass, transfer coefficient, hydration, nucleation or voltage law
# is changed here. New schedules are evaluated by the same q1_merged engine.

PATCH_VERSION = 'q2-charge-controls-1.0'
Q_SAFE = 19.99  # Deliberately BELOW 20: numerical headroom, not a larger budget.


def local_plain(value):
    if isinstance(value, dict):
        return {str(k):local_plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [local_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return local_plain(value.tolist())
    if isinstance(value, np.generic):
        return local_plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_local_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(local_plain(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def require_engine():
    if q is None:
        raise RuntimeError('缺少 q1_merged.py；本文件只修改控制策略，不包含未提供的物理内核。'
                           '请把实际使用的 q1_merged.py 放在 q2.py 同目录。'
                           '不能把先前四参数 q1.py 改名替代。')
    needed = ('Config','Model','Protocol','simulate','load_fit','load_dataset',
              'default_input_files','current_config','apply_grid','save_run','load_run',
              'read_json','write_json','code_digest')
    missing = [name for name in needed if not hasattr(q,name)]
    if missing:
        raise RuntimeError('q1_merged 接口不匹配: '+', '.join(missing))


def assert_no_auxiliary(p):
    powers = np.asarray(p.get('powers',(0.,)*5), dtype=float)
    if powers.size < 5 or not np.isfinite(powers).all() or np.any(powers != 0.):
        raise ValueError('问题2只允许自冷启动，所有辅助加热功率必须为0。')


def validate_control(p, current_limit=.5):
    assert_no_auxiliary(p)
    a = np.asarray(p.get('params',()), dtype=float)
    delay = float(p.get('load_start',0.))
    if not np.isfinite(a).all() or not np.isfinite(delay) or delay < 0:
        raise ValueError('控制参数必须有限；加载延迟不得为负。')
    kind = p['kind']
    if kind == 'step':
        n = (len(a)+1)//2
        if len(a) != 2*n-1 or n < 2 or np.any(a[n:] <= 0) or np.any(np.diff(a[n:]) <= 0):
            raise ValueError('阶梯参数应为n个电流及n-1个严格递增的正切换时刻。')
        levels = a[:n]
    elif kind == 'constant':
        if len(a) != 1:
            raise ValueError('恒流只接受一个电流水平。')
        levels = a
    elif kind == 'ramp':
        if len(a) != 3 or a[2] <= 0:
            raise ValueError('线性升载需要起点、终点和正升载时间。')
        levels = a[:2]
    elif kind == 'zero':
        levels = np.array([0.])
    else:
        raise ValueError('本控制搜索仅支持 constant/ramp/step/zero。')
    if np.any(levels < 0) or np.any(levels > current_limit):
        raise ValueError(f'电流必须在[0,{current_limit}] A/cm2内。')
    return True


def step_parts(p):
    validate_control(p)
    delay = float(p.get('load_start',0.))
    if p['kind'] == 'constant':
        return [float(p['params'][0])], [], delay
    if p['kind'] == 'zero':
        return [0.], [], delay
    if p['kind'] != 'step':
        raise ValueError('不能把线性升载默认为阶梯控制。')
    a = [float(v) for v in p['params']]
    n = (len(a)+1)//2
    return a[:n],a[n:],delay


def charge_at(p,t):
    """Analytical integral of the prescribed current, counted ONCE for the stack."""
    validate_control(p)
    if not np.isfinite(t):
        raise ValueError('积分时刻必须有限。')
    s = max(0.,float(t)-float(p.get('load_start',0.)))
    if p['kind'] == 'ramp':
        a,b,tr = [float(v) for v in p['params']]
        r = min(s,tr)
        return a*r+.5*(b-a)*r*r/tr+b*max(0.,s-tr)
    levels,switches,_ = step_parts(p)
    bounds = [0.]+switches+[float('inf')]
    return float(sum(j*max(0.,min(s,end)-start)
        for j,start,end in zip(levels,bounds[:-1],bounds[1:]) if s > start))


def time_at_charge(p,target):
    """First time attaining a charge. None means this schedule never attains it."""
    validate_control(p)
    if not np.isfinite(target) or target < 0:
        raise ValueError('电荷目标必须为有限非负数。')
    delay = float(p.get('load_start',0.))
    if target == 0:
        return delay
    if p['kind'] == 'ramp':
        a,b,tr = map(float,p['params'])
        qr = .5*(a+b)*tr
        if target <= qr and qr > 0:
            from scipy.optimize import brentq
            return delay+float(brentq(lambda s: a*s+.5*(b-a)*s*s/tr-target,0.,tr))
        return delay+tr+(target-qr)/b if b > 0 else None
    levels,switches,_ = step_parts(p)
    elapsed = 0.
    for k,j in enumerate(levels):
        left = 0. if k == 0 else switches[k-1]
        duration = switches[k]-left if k < len(switches) else float('inf')
        if j > 0 and target-elapsed <= j*duration+1e-11:
            return delay+left+min(duration,max(0.,(target-elapsed)/j))
        if np.isfinite(duration):
            elapsed += j*duration
    return None


def assemble_step(template, levels, switches):
    levels = [float(v) for v in levels]
    switches = [float(v) for v in switches]
    if len(levels) != len(switches)+1:
        raise ValueError('切换时刻数与电流段数不匹配。')
    if len(levels) == 1:
        result = dict(template,kind='constant',params=(levels[0],))
    else:
        result = dict(template,kind='step',params=tuple(levels+switches))
    result['powers'] = (0.,)*5
    result['heat_off'] = 0.
    result['times'] = None
    result['currents'] = None
    validate_control(result)
    return result


def prefix_until(p,absolute_time):
    """Original step/constant prefix, ending immediately before absolute_time."""
    levels,switches,delay = step_parts(p)
    cut = float(absolute_time)-delay
    if cut <= 0 or not np.isfinite(cut):
        raise ValueError('切断时刻必须晚于加载开始时刻。')
    # A switch exactly at the cutoff belongs to the new tail, not the prefix.
    keep = [s for s in switches if s < cut-1e-10]
    return levels[:len(keep)+1],keep,cut


def cut_at_charge(p,budget=Q_SAFE):
    """Cap a step anywhere, including before its final switch or after zero rests."""
    if not 0 < budget < 20:
        raise ValueError('停载预算必须严格介于0与20 C/cm2之间。')
    if p['kind'] not in ('constant','step','zero'):
        return None
    off = time_at_charge(p,budget)
    if off is None:
        return None
    levels,switches,cut = prefix_until(p,off)
    return assemble_step(p,levels+[0.],switches+[cut])


def pulse_tail(p,first_charge,rest_s,tail_current,total_charge=Q_SAFE):
    """Heat -> optional zero-current redistribution -> reheat -> zero-current tail."""
    if not (0 < first_charge < total_charge < 20 and 0 < tail_current <= .5 and rest_s >= 0):
        raise ValueError('无效的加热/停载/补充加载参数。')
    if p['kind'] not in ('constant','step'):
        return None
    off = time_at_charge(p,first_charge)
    if off is None:
        return None
    levels,switches,cut = prefix_until(p,off)
    if rest_s > 1e-10:
        levels += [0.]
        switches += [cut]
    start = cut+rest_s
    levels += [float(tail_current),0.]
    switches += [start,start+(total_charge-first_charge)/tail_current]
    return assemble_step(p,levels,switches)


def charge_schedule(currents,shares,total_charge,rests=None):
    """Use charge allocations to eliminate infeasible current/time combinations."""
    currents = np.asarray(currents,dtype=float)
    shares = np.asarray(shares,dtype=float)
    if (currents.ndim != 1 or len(currents) < 2 or currents.shape != shares.shape
        or not np.isfinite(currents).all() or not np.isfinite(shares).all()
        or np.any(currents <= 0) or np.any(currents > .5) or np.any(shares <= 0)
        or not 0 < total_charge < 20):
        raise ValueError('电荷参数化输入无效。')
    rests = np.zeros(len(currents)-1) if rests is None else np.asarray(rests,dtype=float)
    if rests.shape != (len(currents)-1,) or not np.isfinite(rests).all() or np.any(rests < 0):
        raise ValueError('停载时间数量不正确或含负数。')
    allocated = total_charge*shares/shares.sum()
    levels = []
    switches = []
    clock = 0.
    for k,j in enumerate(currents):
        levels.append(float(j))
        clock += float(allocated[k]/j)
        switches.append(clock)
        if k < len(currents)-1 and rests[k] > 1e-10:
            levels.append(0.)
            clock += float(rests[k])
            switches.append(clock)
    levels.append(0.)
    return assemble_step(dict(load_start=0.),levels,switches)


def audit_trajectory(run,cfg,p,charge_series=None):
    """Veto-only check of a full run. Does not fix flags or extrapolate a tail.

    Safety combines saved states with solver-maintained extrema/events. This
    audit does not claim a proof of continuous-time safety between solver steps.
    It is additionally checked by tighter integration and finer space in rescue.
    """
    m = run['metrics']
    t = np.asarray(run['t'],float)
    T = np.asarray(run['T'],float)
    V = np.asarray(run['V'],float)
    ice = np.asarray(run['ice'],float)
    j = np.asarray(run['j'],float)
    u = np.asarray(run.get('u',np.array([np.nan])),float)
    shape_ok = (t.ndim == 1 and len(t) > 0 and T.shape == V.shape == ice.shape == (len(t),5)
                and j.shape == (len(t),))
    if not shape_ok:
        return dict(passed=False,numerically_accepted=False,checks={'shape':False})
    finite = bool(all(np.isfinite(a).all() for a in (t,T,V,ice,j,u,run['y'])))
    numerical = bool(finite and m.get('stop_reason') not in ('numerical_failure','control_input')
        and m.get('mass_residual',float('inf')) < 1e-3
        and m.get('energy_residual',float('inf')) < 1e-3)
    ts = m.get('success_time')
    indices = np.flatnonzero(np.abs(t-float(ts)) <= 1e-6) if ts is not None and np.isfinite(ts) else []
    at_event = bool(len(indices))
    temp_event = float(np.min(T[int(indices[-1])])) if at_event else None
    try:
        validate_control(p)
        input_ok = True
        analytic = np.array([charge_at(p,tt) for tt in t])
        q_final = float(analytic[-1])
    except (ValueError,KeyError,TypeError):
        input_ok = False
        analytic = np.full(len(t),np.nan)
        q_final = float('nan')
    qc = np.asarray(charge_series,float) if charge_series is not None else analytic
    max_charge = float(max(np.nanmax(qc),m.get('charge_C_cm2',float('inf'))))
    v_min = float(min(V.min(),m.get('V_min',float('-inf'))))
    ice_max = float(max(ice.max(),m.get('ice_max',float('inf'))))
    charge_matches = bool(qc.shape == (len(t),) and np.isfinite(qc).all()
        and np.max(np.abs(qc-analytic)) <= 1e-6
        and abs(float(m.get('charge_C_cm2',float('inf')))-q_final) <= 1e-6)
    checks = dict(shape=shape_ok, finite=finite, numerical=numerical,
        zero_time_origin=bool(abs(t[0]) <= 1e-9), chronological=bool(np.all(np.diff(t) >= -1e-10)),
        initial_unit_temperature=bool(np.max(np.abs(T[0]-(cfg['T0']-273.15))) <= 1e-6),
        current_input_valid=input_ok, no_auxiliary=bool(u.shape == (len(t),5) and np.all(u == 0.)
            and abs(m.get('E_aux_total_J',float('inf'))) <= 1e-9),
        voltage_limit=bool(v_min >= .3-1e-8), ice_limit=bool(ice_max < .99),
        current_limit=bool(j.min() >= -1e-12 and j.max() <= .5+1e-12),
        charge_limit=bool(max_charge <= 20.+1e-7), charge_accounting=charge_matches,
        solver_success=bool(m.get('feasibility') == 'feasible' and m.get('stop_reason') == 'success'),
        temperature_event=bool(at_event and temp_event > 0. and temp_event >= cfg['success_margin']-1e-7),
        full_to_success=bool(at_event and abs(t[-1]-float(ts)) <= 1e-6),
        registered_limits=bool(cfg['cells'] == 5 and cfg['fixture'] == 'H2'
            and cfg['voltage_limit'] == .3 and cfg['charge_limit'] == 20.
            and cfg['current_limit'] == .5 and cfg['h'] == 40.
            and cfg['observation'] == 'unit' and cfg['success_margin'] > 0.))
    return dict(passed=bool(all(checks.values())),numerically_accepted=numerical,checks=checks,
        min_voltage=v_min,max_ice=ice_max,max_charge=max_charge,
        temperature_at_success=temp_event,analytic_charge=q_final,
        tolerance_note='voltage 1e-8 V; charge 1e-7 C/cm2; event temperature 1e-7 K; no state clipping',
        scope='Saved states plus original solver event/extrema records; no empirical validation implied')


def load_control_seeds():
    explicit = os.environ.get('Q2_SEED_FILE')
    paths = [Path(explicit)] if explicit else [OUT/'minus10_selected.json',LEGACY_OUT/'minus10_selected.json']
    result = []
    for path in paths:
        if not path.is_file():
            if explicit:
                raise FileNotFoundError(path)
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(value,dict):
            value = value.get('candidates',value.get('selected',[value]))
        for row in value:
            p = row.get('protocol',row)
            try:
                validate_control(p)
            except (ValueError,KeyError,TypeError):
                continue
            result.append(p)
    # Read old controls only. No old outcome is accepted as a new simulation result.
    return [vars(p) for p in unique([protocol(**p) for p in result])]


def update_step_selection(rows):
    path = OUT/'minus10_selected.json'
    previous = q.read_json(path) if path.is_file() else []
    # Only rows with an audit from THIS source may compete.
    fit_hash = hashlib.sha256(FIT.read_bytes()).hexdigest()
    eligible = [r for r in previous+rows if r.get('constraint_audit') is not None
        and r.get('q2_hash') == CODE_HASH and r.get('q1_hash') == q.code_digest()
        and r.get('fit_hash') == fit_hash]
    selected = leaders(eligible,1)
    q.write_json(path,selected)
    return selected


def fresh_controls(count,seed=20260930):
    rng = np.random.default_rng(seed)
    result = []
    for k in range(count):
        n = (3,4,5)[k%3]
        currents = rng.uniform(.06,.40,n)
        currents[0] = rng.uniform(.015,.12)
        if k%2:
            currents = np.sort(currents)
        shares = rng.dirichlet(np.ones(n)*1.5)
        shares[0] *= .35
        shares /= shares.sum()
        rests = np.zeros(n-1)
        if k%3 == 0:
            rests[-1] = rng.uniform(1.,20.)
        result.append(charge_schedule(currents,shares,float(rng.uniform(18.,Q_SAFE)),rests))
    return result


def mutate_charge_controls(p,count,rng):
    """Joint duration/current perturbations, re-normalised to a sub-limit charge."""
    levels,switches,_ = step_parts(p)
    if not switches:
        p = cut_at_charge(p,Q_SAFE)
        if p is None:
            return []
        levels,switches,_ = step_parts(p)
    # Finite heating phases plus a terminal zero. Never mutate an infinite load.
    if levels[-1] != 0:
        p = cut_at_charge(p,Q_SAFE)
        if p is None:
            return []
        levels,switches,_ = step_parts(p)
    duration = np.diff([0.]+switches)
    current = np.array(levels[:-1],float)
    positives = current > 0.
    total = float(np.dot(current,duration))
    result = []
    for k in range(count):
        scale = (.015,.04,.10,.2)[k%4]
        a = current.copy()
        a[positives] = np.clip(a[positives]*np.exp(rng.normal(0,scale,positives.sum())),.005,.5)
        dt = duration*np.exp(rng.normal(0,scale,len(duration)))
        target = float(np.clip(total+rng.normal(0,.08),.5,Q_SAFE))
        dt[positives] *= target/float(np.dot(a,dt))
        if np.sum(dt) >= 1100:
            continue
        result.append(assemble_step(p,list(a)+[0.],list(np.cumsum(dt))))
    return result


def rescue(workers=3,budget=120):
    """Finite, state-law-preserving search; success is never guaranteed by the code."""
    if budget < 12:
        raise ValueError('rescue --budget 至少12。')
    seed_controls = load_control_seeds()
    stages = []
    ps = [protocol(**p) for p in seed_controls]
    # Extra coast/reheat controls are a STEP family, not renamed constant/ramp.
    for p in seed_controls:
        if p['kind'] not in ('constant','step'):
            continue
        for cap in (18.,19.,19.5,19.9,Q_SAFE):
            cut = cut_at_charge(p,cap)
            if cut is not None:
                ps.append(protocol(**cut))
        for first,rest,current in ((16.,0.,.1),(18.,0.,.08),(18.,4.,.15),
                                   (18.,12.,.2),(19.,2.,.08),(19.,8.,.15)):
            tail = pulse_tail(p,first,rest,current,Q_SAFE)
            if tail is not None:
                ps.append(protocol(**tail))
    # No selected JSON is required: small broad baseline plus fresh charge schedules.
    for j in (.02,.04,.06,.08,.10,.12,.16,.20,.25,.30,.35):
        ps.append(protocol('constant',(j,)))
        ps.append(protocol(**cut_at_charge(dict(kind='constant',params=(j,)),Q_SAFE)))
    for p in fresh_controls(budget,seed=20260930):
        ps.append(protocol(**p))
    ps = unique(ps)
    write_local_json(OUT/'search_plan.json',dict(version=PATCH_VERSION,
        physical_code=q.code_digest(),fit_sha256=hashlib.sha256(FIT.read_bytes()).hexdigest(),
        q2_sha256=CODE_HASH,temperature_C=-10.,screen_grid='i48',final_grid='i192',
        unchanged_limits=dict(Q=20.,j=.5,V=.3,h=40.,success_margin=.01),
        seed_controls=seed_controls,n_initial=len(ps),budget_per_mutation_round=budget,
        note='The budget controls generated schedules per round, not global optimality.'))
    rows = batch([job(config(-10.,'i48'),p,'rescue_screen') for p in ps], 'rescue_screen', workers)
    rng = np.random.default_rng(20261001)
    for round_id in range(2):
        parents = [r for r in sorted(rows,key=rank) if r['protocol']['kind'] in ('constant','step')][:6]
        ps = []
        for r in parents:
            ps += [protocol(**p) for p in mutate_charge_controls(r['protocol'],max(2,budget//max(1,len(parents))),rng)]
        extra = batch([job(config(-10.,'i48'),p,f'rescue_joint_{round_id}') for p in unique(ps)],
                      f'rescue_joint_{round_id}',workers)
        rows += extra
    winners = sorted(rows,key=rank)[:8]
    # Keep the best original family representatives as controls for the comparison.
    winners += leaders(rows,1)
    fine = batch([job(config(-10.,'i192',True),protocol(**r['protocol']),'rescue_fine') for r in winners],
                 'rescue_i192',workers)
    if not any(r['feasibility']=='feasible' for r in fine):
        # Fine-grid near-boundary attempts are NOT eliminated by coarse ranking.
        parents = [r for r in sorted(fine,key=rank) if r['protocol']['kind'] in ('constant','step')][:3]
        ps = [protocol(**p) for r in parents for p in mutate_charge_controls(r['protocol'],4,rng)]
        fine += batch([job(config(-10.,'i192',True),p,'rescue_fine_neighbour') for p in unique(ps)],
                      'rescue_fine_neighbours',workers)
    selected = update_step_selection(fine)
    q.write_json(OUT/'rescue_candidates.json',fine)
    summary = verify_rescue(fine,workers)
    q.write_json(OUT/'rescue_summary.json',summary)
    export_rescue_report(summary, fine)
    return summary


def verify_rescue(rows,workers=3):
    feasible = [r for r in rows if r.get('feasibility')=='feasible'
                and r.get('constraint_audit',{}).get('passed',False)]
    best = min(rows,key=rank) if rows else None
    if not feasible:
        (OUT/'verified_success.json').unlink(missing_ok=True)
        summary = dict(status='not_found_in_this_search',verified_success=False,
            model_preserved=True,best=best,verified_candidates=[],
            reason='No currently audited i192 feasible trajectory; not a proof of impossibility.')
        write_local_json(OUT/'status.json',summary)
        print('本轮尚未找到通过约束审查的−10 ℃解；不是全局不可行证明。',flush=True)
        return summary
    choices = sorted(feasible,key=rank)[:3]
    validations = []
    for r in choices:
        p = protocol(**r['protocol'])
        jobs = [job(config(-10.,'i384',True),p,'fresh_rescue_space'),
                job(config(-10.,'i192',True,rtol=2.5e-7,max_step=.05,atol_scale=.5,
                    board_n=8,end_n=16,sample=.025),p,'fresh_rescue_strict')]
        other = batch(jobs,'rescue_check_'+r['candidate_id'],workers)
        a = q.load_run(OUT/'runs'/r['candidate_id'])
        checks = []
        for checked in other:
            b = q.load_run(OUT/'runs'/checked['candidate_id'])
            raw = q.compare_trajectories(a,b,min(float(a['t'][-1]),float(b['t'][-1])))
            individual = dict(raw['individual_checks'])
            individual.update(both_feasible=bool(checked['feasibility']=='feasible'
                and checked['constraint_audit']['passed']),
                event_type=r['stop_reason']==checked['stop_reason'],
                event_time=abs(r['last_valid_time']-checked['last_valid_time']) <= .1)
            checks.append(dict(stage=checked['stage'],candidate_id=checked['candidate_id'],
                passed=bool(all(individual.values())),individual_checks=individual,
                max_T_difference=raw.get('next_grid_T_diff'),
                max_V_difference=raw.get('next_grid_V_diff'),
                event_time_difference=abs(r['last_valid_time']-checked['last_valid_time'])))
        validations.append(dict(base=r,checks=checks,passed=all(c['passed'] for c in checks)))
        if validations[-1]['passed']:
            break  # Candidates are ordered by startup time; retain slower ones only as fallbacks.
    passed = [v for v in validations if v['passed']]
    chosen = min(passed,key=lambda z:z['base']['success_time']) if passed else None
    summary = dict(status='verified_numerical_success' if chosen else 'candidate_success_not_converged',
        verified_success=bool(chosen),model_preserved=True,best=chosen['base'] if chosen else choices[0],
        verified_candidates=validations,limits=dict(T0_C=-10.,Q_C_cm2=20.,j_A_cm2=.5,V_V=.3,h=40.),
        scope='Numerical success under the supplied model, not independent experimental validation.')
    write_local_json(OUT/'status.json',summary)
    if chosen:
        write_local_json(OUT/'verified_success.json',summary)
    else:
        # No stale "verified" label from an earlier run is retained.
        (OUT/'verified_success.json').unlink(missing_ok=True)
    print(summary['status'], 'best ts=',summary['best']['success_time'],flush=True)
    return summary


def export_rescue_report(summary, rows):
    columns = []
    for row in rows:
        columns.append(dict(candidate_id=row['candidate_id'],
            family=NAMES[row['protocol']['kind']],parameters=formula(row['protocol']),
            success_time_s=row.get('success_time') if row['feasibility']=='feasible' else None,
            diagnostic_stop_time_s=row['last_valid_time'],
            cumulative_charge_C_cm2=row['charge_C_cm2'],minimum_voltage_V=row['V_min'],
            maximum_ice_fraction=row['ice_max'],final_minimum_temperature_C=row['final_min_T_C'],
            highest_attained_minimum_temperature_C=row['best_min_T_C'],
            feasibility=row['feasibility'],audit_passed=row['constraint_audit']['passed']))
    pd.DataFrame(columns).to_csv(OUT/'minus10_candidates.csv',index=False,encoding='utf-8-sig')
    titles={'verified_numerical_success':'已找到通过本轮数值复核的可行策略',
            'candidate_success_not_converged':'找到i192可行候选，但数值收敛复核尚未通过',
            'not_found_in_this_search':'本轮有限搜索尚未找到i192可行解'}
    best=summary.get('best')
    text=['# −10 ℃自冷启动控制搜索','','## 状态','',titles[summary['status']],'',
        '始终使用原q1_merged模型及原标定参数。电荷上限20 C/cm²，电流上限0.5 A/cm²，'
        '电压下限0.30 V，h=40，无辅助加热；成功判据为五片含双极板的平均温度同时达到0.01 ℃。','',
        '停载和补充加载均计入同一条完整阶梯曲线；不重置电荷、温度或含水状态。'
        '带零电流尾段的方案不归入纯恒流或纯线性升载。','']
    if best:
        text += ['## 当前代表方案','',formula(best['protocol']),'',
            f"积分终止原因：{best['stop_reason']}；末态最低温度：{best['final_min_T_C']:.8g} ℃；"
            f"累计电荷：{best['charge_C_cm2']:.8g} C/cm²；最低电压：{best['V_min']:.8g} V。",'',
            '启动时间仅在feasibility为feasible且独立约束审查通过时填写；失败轨迹的终止时间不是启动时间。','']
    text += ['## 适用范围','','本次搜索不是全局最优证明；数值成功不是独立实验验证。'
             'status.json保存完整状态及本次细网格验证条目。'
             '只有状态为verified_numerical_success时才写入verified_success.json。','']
    if summary.get('verified_success') and best:
        levels,switches,delay=step_parts(best['protocol'])
        rows_control=[]
        for k,current in enumerate(levels):
            start=delay+(switches[k-1] if k else 0.)
            end=min(delay+switches[k] if k<len(switches) else best['success_time'],best['success_time'])
            if start>=end:continue
            rows_control.append(dict(阶段=k+1,开始_s=start,结束_s=end,电流_A_cm2=current))
        pd.DataFrame(rows_control).to_csv(OUT/'成功方案电流表.csv',index=False,encoding='utf-8-sig')
        text+=['## 已验证成功方案的电流时序','',
            '|阶段|开始 s|结束 s|电流 A/cm²|','|---|---|---|---|']
        for r in rows_control:
            text.append(f"|{r['阶段']}|{r['开始_s']:.6f}|{r['结束_s']:.6f}|{r['电流_A_cm2']:.9f}|")
        text+=['','达到启动条件时结束冷启动策略。精确参数以 verified_success.json 为准，表格仅为显示舍入。','',
            '## 约束及数值验证','',
            f"启动时间 {best['success_time']:.9f} s；电荷 {best['charge_C_cm2']:.9f} C/cm²；全程最低电压 {best['V_min']:.9f} V；最大电流 {max(levels):.9f} A/cm²；最大冰体积分数 {best['ice_max']:.6g}。",'',
            '|单片|含双极板的平均温度 ℃|MEA平均温度 ℃|','|---|---|---|']
        for k,(unit,mea) in enumerate(zip(best['final_T_C'],best['final_MEA_T_C'])):
            text.append(f'|{k+1}|{unit:.6f}|{mea:.6f}|')
        text+=['','本次仅证明当前单片平均温度判据下数值启动成功，不表示各层局部温度均为正，也不表示温差已经消除。', '',
            '梯度优化允许计算越限试探曲线以评估约束方向，这些轨迹仅存于 polish_trials.json，不参与成功认定。正式候选从初始状态重新积分，配置保持原0.30 V终止事件、20 C/cm²上限和全部物理参数，并经过独立约束审查、i384空间加密及严格积分检查。', '',
            '## 表3更新','']
        old=q.read_json(OUT/'rescue_i192.json') if (OUT/'rescue_i192.json').is_file() else []
        compared=[]
        for kind in KINDS:
            candidates=[best] if kind=='step' else [r for r in old if r['protocol']['kind']==kind and r.get('q1_hash')==best['q1_hash'] and r.get('fit_hash')==best['fit_hash']]
            if not candidates:continue
            r=min(candidates,key=rank)
            maximum=max(step_parts(r['protocol'])[0]) if kind!='ramp' else max(r['protocol']['params'][:2])
            compared.append(dict(加载策略=NAMES[kind],启动时间_s=r['success_time'] if r['feasibility']=='feasible' else None,
                最大电流密度_A_cm2=maximum,累计电荷_C_cm2=r['charge_C_cm2'],最低电压_V=r['V_min'],
                最大冰体积分数=r['ice_max'],最大温差_C=r['delta_T_max'],
                启动结果='成功' if r['feasibility']=='feasible' else '本轮未找到成功策略',candidate_id=r['candidate_id']))
        pd.DataFrame(compared).to_csv(OUT/'表3.csv',index=False,encoding='utf-8-sig')
        text+=['恒流、线性升载保留同一物理模型此前真实复算结果；阶梯更新为本次已通过数值复核的成功方案。详见表3.csv。','']
    (OUT/'minus10_report.md').write_text('\n'.join(text),encoding='utf-8')
    if summary.get('verified_success'):
        (OUT/'启动成功报告.md').write_text('\n'.join(text),encoding='utf-8')


def preflight():
    status = dict(version=PATCH_VERSION,source_sha256=CODE_HASH,
        q1_merged_imported=q is not None,fit_path=str(FIT),fit_exists=FIT.is_file(),
        output=str(OUT),physical_simulation_executed=False)
    if q is not None:
        status['physical_source'] = str(Path(q.__file__).resolve())
        try:
            cfg = fitted_config()
            status['fit_validated'] = True
            status['physical_config'] = cfg.to_dict()
        except Exception as exc:
            status['fit_validated'] = False
            status['error'] = f'{type(exc).__name__}: {exc}'
    else:
        status['fit_validated'] = False
        status['error'] = 'Missing actual q1_merged.py; no fallback physics used.'
    if not status['fit_validated']:
        (OUT/'verified_success.json').unlink(missing_ok=True)
    write_local_json(OUT/'preflight.json',status)
    print(json.dumps(local_plain(status),ensure_ascii=False,indent=2))
    return status


def control_checks():
    """Analytical control checks only. Never described as a cold-start replay."""
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    write_local_json(OUT/'control_tests.json',dict(tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        physical_simulation_executed=False,scope='Analytical protocols and independent audit logic only'))
    return result.wasSuccessful()


# Small, deterministic regression checks intentionally need no physical kernel.
import unittest
class ControlTests(unittest.TestCase):
    def test_serial_charge_not_multiplied(self):
        self.assertAlmostEqual(charge_at(dict(kind='constant',params=(.2,)),50.),10.)
    def test_ramp_integral(self):
        self.assertAlmostEqual(charge_at(dict(kind='ramp',params=(0.,.3,60.)),60.),9.)
    def test_delayed_ramp_inverse(self):
        p=dict(kind='ramp',params=(.01,.3,60.),load_start=7.)
        self.assertAlmostEqual(charge_at(p,time_at_charge(p,14.)),14.,places=9)
    def test_cut_before_last_switch(self):
        p=dict(kind='step',params=(.1,.3,.4,10.,50.))
        c=cut_at_charge(p,2.)
        self.assertAlmostEqual(charge_at(c,500.),2.)
        self.assertLess(c['params'][-1],50.)
    def test_cut_exactly_at_switch(self):
        p=dict(kind='step',params=(.1,.3,.4,10.,50.))
        c=cut_at_charge(p,1.)
        self.assertAlmostEqual(charge_at(c,500.),1.)
    def test_cut_with_initial_delay(self):
        p=dict(kind='constant',params=(.2,),load_start=11.)
        c=cut_at_charge(p,19.99)
        self.assertAlmostEqual(time_at_charge(c,19.99),110.95)
        self.assertAlmostEqual(charge_at(c,500.),19.99)
    def test_terminal_zero_does_not_divide(self):
        p=dict(kind='step',params=(.2,0.,10.))
        self.assertIsNone(cut_at_charge(p,19.99))
    def test_internal_zero_rest(self):
        p=dict(kind='step',params=(.1,0.,.2,10.,15.))
        self.assertAlmostEqual(time_at_charge(p,2.),20.)
    def test_plateau_first_attainment(self):
        p=dict(kind='step',params=(.1,0.,.2,10.,15.))
        self.assertAlmostEqual(time_at_charge(p,1.),10.)
    def test_pulse_tail_charge(self):
        p=dict(kind='constant',params=(.25,))
        c=pulse_tail(p,18.,8.,.1)
        self.assertAlmostEqual(charge_at(c,1000.),19.99)
        self.assertEqual(protocol(**c).j(76.),0.)
        self.assertAlmostEqual(protocol(**c).j(82.),.1)
    def test_no_rest_duplicate_switch(self):
        c=pulse_tail(dict(kind='constant',params=(.25,)),18.,0.,.1)
        self.assertTrue(validate_control(c))
        self.assertAlmostEqual(charge_at(c,1000.),19.99)
    def test_upper_charge_not_relaxed(self):
        with self.assertRaises(ValueError):
            cut_at_charge(dict(kind='constant',params=(.25,)),20.)
    def test_no_auxiliary_allowed(self):
        with self.assertRaises(ValueError):
            validate_control(dict(kind='constant',params=(.1,),powers=(.01,0,0,0,0)))
    def test_negative_current_rejected(self):
        with self.assertRaises(ValueError):
            validate_control(dict(kind='constant',params=(-.1,)))
    def test_excess_current_rejected(self):
        with self.assertRaises(ValueError):
            validate_control(dict(kind='constant',params=(.501,)))
    def test_ordered_switches(self):
        with self.assertRaises(ValueError):
            validate_control(dict(kind='step',params=(.1,.2,.3,10.,9.)))
    def test_unique_keeps_delay(self):
        p=protocol('constant',(.1,));r=protocol('constant',(.1,),load_start=4.)
        self.assertEqual(len(unique([p,r,p])),2)
    def test_step_left_and_right(self):
        p=protocol('step',(.1,.2,0.,5.,10.))
        self.assertAlmostEqual(p.j(np.nextafter(5.,0.)),.1)
        self.assertAlmostEqual(p.j(5.),.2)
        self.assertEqual(p.j(10.),0.)
    def test_charge_parameterisation(self):
        c=charge_schedule([.02,.3,.1],[1.,3.,2.],19.99,[0.,8.])
        self.assertAlmostEqual(charge_at(c,5000.),19.99)
        self.assertTrue(validate_control(c))
    def test_random_controls_preserve_bounds(self):
        for c in fresh_controls(200):
            self.assertTrue(validate_control(c))
            self.assertLessEqual(charge_at(c,100000.),19.99+1e-10)
    def test_mutations_preserve_budget(self):
        c=charge_schedule([.02,.3,.1],[1.,3.,2.],19.99,[0.,8.])
        for p in mutate_charge_controls(c,100,np.random.default_rng(7)):
            self.assertTrue(validate_control(p))
            self.assertLessEqual(charge_at(p,100000.),19.99+1e-10)
    @staticmethod
    def example():
        # Manufactured arrays test accounting logic, not the fuel-cell equations.
        cfg=dict(cells=5,fixture='H2',T0=263.15,voltage_limit=.3,charge_limit=20.,
                 current_limit=.5,h=40.,observation='unit',success_margin=.01)
        p=dict(kind='constant',params=(.2,))
        r=dict(t=np.array([0.,10.]),T=np.array([[-10.]*5,[.01]*5]),
            V=np.full((2,5),.5),ice=np.zeros((2,5)),j=np.full(2,.2),u=np.zeros((2,5)),
            y=np.zeros((2,5)),metrics=dict(feasibility='feasible',stop_reason='success',
                success_time=10.,mass_residual=0.,energy_residual=0.,V_min=.5,ice_max=0.,
                charge_C_cm2=2.,E_aux_total_J=0.))
        return r,cfg,p
    def test_audit_manufactured_pass(self):
        r,cfg,p=self.example()
        self.assertTrue(audit_trajectory(r,cfg,p,[0.,2.])['passed'])
    def test_audit_veto_false_success_temperature(self):
        r,cfg,p=self.example();r['T'][-1,0]=-.001
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,2.])['passed'])
    def test_audit_veto_voltage_breach(self):
        r,cfg,p=self.example();r['metrics']['V_min']=.299
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,2.])['passed'])
    def test_audit_veto_counter_reset(self):
        r,cfg,p=self.example()
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,1.])['passed'])
    def test_audit_no_promote_failure(self):
        r,cfg,p=self.example();r['metrics']['feasibility']='infeasible'
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,2.])['passed'])
    def test_audit_veto_hidden_heater(self):
        r,cfg,p=self.example();r['u'][-1,0]=.01
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,2.])['passed'])
    def test_audit_veto_numerical_failure(self):
        r,cfg,p=self.example();r['metrics']['stop_reason']='numerical_failure'
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,2.])['numerically_accepted'])
    def test_audit_veto_wrong_temperature_basis(self):
        r,cfg,p=self.example();cfg['observation']='MEA'
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,2.])['passed'])
    def test_audit_veto_changed_charge_limit(self):
        r,cfg,p=self.example();cfg['charge_limit']=21.
        self.assertFalse(audit_trajectory(r,cfg,p,[0.,2.])['passed'])
    def test_failure_ranking_retains_coast_peak(self):
        a=dict(numerically_accepted=True,feasibility='unresolved',final_min_T_C=-2.,best_min_T_C=-.03,
               charge_C_cm2=19.99,V_min=.4)
        b=dict(a,final_min_T_C=-.1,best_min_T_C=-.1)
        self.assertLess(rank(a),rank(b))


def control_main():
    global OUT,FIT
    parser=argparse.ArgumentParser(description='Q2控制修正版：保持q1_merged物理参数，搜索−10℃自启动。')
    choices=['polish','rescue','check-controls','preflight','check','pilot','search','general-step','targeted',
             'envelope','joint','idle','coast','tail-search','boundary-rescue','boundary',
             'verify-minus10','verify','diagnose','replay','report','results']
    parser.add_argument('command',nargs='?',default='rescue',choices=choices)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--budget',type=int,default=120)
    parser.add_argument('--fit',type=Path,default=FIT)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--seed',type=Path,help='可选旧方案JSON；只读其控制，不沿用旧成败结论。')
    parser.add_argument('--fresh',action='store_true',help='强制重新积分，不读当前源码缓存。')
    args=parser.parse_args()
    if args.workers < 1:
        parser.error('--workers必须为正整数')
    OUT=args.output.resolve();FIT=args.fit.resolve();OUT.mkdir(parents=True,exist_ok=True)
    os.environ['Q2_OUTPUT_DIR']=str(OUT)
    os.environ['Q2_FIT_FILE']=str(FIT)
    if args.seed:
        os.environ['Q2_SEED_FILE']=str(args.seed.resolve())
    if args.fresh:
        os.environ['Q2_REUSE_CACHE']='0'
    fitted_config.cache_clear()
    if args.command=='check-controls':
        raise SystemExit(0 if control_checks() else 1)
    if args.command=='preflight':
        status=preflight()
        raise SystemExit(0 if status['fit_validated'] else 2)
    if args.command=='results':
        status=OUT/'status.json'
        print(status.read_text(encoding='utf-8') if status.exists() else '尚未生成本修正版的真实仿真结果。')
        return
    try:
        require_engine()
        fitted_config()
    except (RuntimeError,FileNotFoundError,ValueError) as exc:
        (OUT/'verified_success.json').unlink(missing_ok=True)
        write_local_json(OUT/'status.json',dict(status='blocked_missing_or_mismatched_model',
            verified_success=False,physical_simulation_executed=False,error=str(exc)))
        parser.exit(2,str(exc)+'\n')
    actions={
        'rescue':lambda:rescue(args.workers,args.budget),
        'polish':lambda:polish(args.workers,args.budget),
        'check':check,
        'pilot':lambda:batch([job(config(grid='i48'),protocol('constant',(j,)),'pilot')
                            for j in (.05,.1,.15,.2,.25,.3,.4,.5)],'pilot',args.workers),
        'search':lambda:search(args.workers,args.budget),
        'general-step':lambda:general_step(args.workers,args.budget),
        'targeted':lambda:targeted(args.workers),
        'envelope':lambda:envelope(args.workers),
        'joint':lambda:joint(args.workers),'idle':lambda:idle(args.workers),
        'coast':lambda:coast(args.workers),'tail-search':lambda:tail_search(args.workers),
        'boundary-rescue':lambda:boundary_rescue(args.workers),
        'boundary':lambda:boundary(args.workers,args.budget),
        'verify-minus10':lambda:verify(args.workers,False),'verify':lambda:verify(args.workers),
        'diagnose':lambda:diagnose(args.workers),'replay':lambda:replay(args.workers),'report':report}
    actions[args.command]()



def polish_trial(payload):
    """Optimizer-only continuation, NEVER an accepted Q2 trajectory.

    The voltage event is moved to 0.15 V only for evaluating gradients of
    infeasible trial controls. All 0.301 V constraints remain in SLSQP and
    every delivered candidate is replayed with the original 0.30 V event.
    No thermal, water, kinetic, initial-state or charge parameter is changed.
    """
    currents,durations=payload
    currents=np.asarray(currents);durations=np.asarray(durations)
    spent=float(np.dot(currents[:-1],durations))
    if spent>=19.95:
        return dict(objective=10.+spent,voltage_margins=[-1.]*len(currents),
                    charge_margin=19.95-spent,eligible=False)
    end=float(durations.sum()+(20.-spent)/currents[-1])
    p=protocol('step',tuple(currents)+tuple(np.cumsum(durations)))
    install()
    cfg=config(-10.,'i48',rtol=2e-6,max_step=.25,sample=.1,
               horizon=end+.001,voltage_limit=.15)
    run=q.simulate(cfg,p,stop_success=False,complete_horizon=True)
    complete=run['t'][-1]>=end-1e-5
    margins=[];boundaries=np.r_[0.,np.cumsum(durations),end]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        mask=(run['t']>=a-1e-8)&(run['t']<=b+1e-8)
        margins.append(float(run['V'][mask].min()-.301) if mask.any() else -1.)
    if not complete:
        margins=np.minimum(margins,-.01).tolist()
    objective=-float(run['T'][-1].min())+max(0.,end-run['t'][-1])*.2
    valid=bool(run['metrics']['stop_reason']!='numerical_failure' and
        run['metrics']['mass_residual']<1e-3 and run['metrics']['energy_residual']<1e-3)
    if not valid:objective+=20.
    return dict(objective=objective,voltage_margins=margins,charge_margin=19.95-spent,
        eligible=bool(valid and complete and min(margins)>=-1e-7),
        final_min_T_C=float(run['T'][-1].min()),V_min=float(run['V'].min()),
        end_s=float(run['t'][-1]),protocol=vars(p),numerically_accepted=valid,
        note='Derivative trial only; not eligible for success reporting before original-limit replay.')


def polish(workers=3,budget=40):
    """Constrained local optimization exploits unused voltage margins between switches."""
    from scipy.optimize import minimize
    base=next(p for p in load_control_seeds() if p['kind']=='step' and level_count(p)==4)
    switch=np.array([base['params'][4],30.,42.,base['params'][5],60.,72.,base['params'][6]])
    initial=np.array([protocol(**base).j(t) for t in np.r_[0.,switch]+1e-7])
    original_duration=np.diff(np.r_[0.,switch])
    history=[];candidates=[];stages=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for variable_times in (False,True):
            cache={};n=len(initial)
            x0=np.r_[initial/.1,original_duration/10.] if variable_times else initial/.1
            def unpack(x):
                return x[:n]*.1,x[n:]*10. if variable_times else original_duration
            def consume(xs):
                missing={tuple(np.round(x,10)):x.copy() for x in xs if tuple(np.round(x,10)) not in cache}
                if missing:
                    answers=pool.map(polish_trial,[unpack(x) for x in missing.values()])
                    for key,row in zip(missing,answers):
                        cache[key]=row;history.append(row)
                        if row['eligible']:candidates.append(row)
                    q.write_json(OUT/'polish_trials.json',history)
                return [cache[tuple(np.round(x,10))] for x in xs]
            def vector(x):
                r=consume([x])[0]
                return np.r_[r['objective'],np.array(r['voltage_margins'])*10.,r['charge_margin']]
            def derivative(x):
                step=.002
                xs=[x.copy() for _ in x]
                for k in range(len(x)):xs[k][k]+=step
                rows=consume([x]+xs)
                vectors=[np.r_[r['objective'],np.array(r['voltage_margins'])*10.,r['charge_margin']] for r in rows]
                return (np.array(vectors[1:])-vectors[0]).T/step
            iteration=[0]
            def callback(x):
                r=consume([x])[0];iteration[0]+=1
                print('polish', 'times+current' if variable_times else 'current',iteration[0],
                      'Tmin',round(r.get('final_min_T_C',-100.),6),
                      'Vmin',round(r.get('V_min',0.),6),'eligible',r['eligible'],flush=True)
            bounds=[(.05,5.)]*n+([(.2,6.)]*(n-1) if variable_times else [])
            result=minimize(lambda x:vector(x)[0],x0,jac=lambda x:derivative(x)[0],
                bounds=bounds,method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:vector(x)[1:],jac=lambda x:derivative(x)[1:])],
                callback=callback,options=dict(maxiter=max(8,min(budget,45)),ftol=1e-6,disp=True))
            stages.append(dict(variable_times=variable_times,message=str(result.message),iterations=result.nit))
            if candidates:
                lead=min(candidates,key=lambda r:r['objective']);v=lead['protocol']['params']
                initial=np.array(v[:n]);original_duration=np.diff(np.r_[0.,v[n:]])
                # Success candidates are checked immediately; no need for the second stage if verified.
                if lead['final_min_T_C']>.03:
                    break
    best=sorted(candidates,key=lambda r:r['objective'])[:3]
    q.write_json(OUT/'polish_optimization.json',dict(stages=stages,trials=len(history),candidates=best))
    fine=batch([job(config(-10.,'i192',True),protocol(**r['protocol']),'polish_original_limits') for r in best],
               'polish_i192',workers)
    q.write_json(OUT/'polish_candidates.json',fine)
    if fine:
        update_step_selection(fine)
        summary=verify_rescue(fine,workers)
        q.write_json(OUT/'polish_summary.json',summary)
        export_rescue_report(summary,fine)
    return fine

if __name__ == "__main__":
    main()
