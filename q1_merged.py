#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一问融合版：守恒焓模型 + 中位成核情景 + 单温度两参数标定。

默认只用 -20℃ 拟合，-25℃仅作回顾性跨温度检验；不使用原联合拟合参数。
保留题设双极板热容、局部CL水合、电压不截断、指数拟合通量与BDF。
成核及CL结合水析冰沿用Notebook情景关系，不视为独立实测定律。
仅针对一次冷启动；不建模完全融化后再次冷却的成核重置。
python q1_merged.py fit --budget 60
python q1_merged.py run
python q1_merged.py check
python q1_merged.py convergence
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from dataclasses import dataclass, asdict, replace, fields
from pathlib import Path
import hashlib
import io
import json
import math
import platform
import posixpath
import re
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from typing import Any
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, brentq
from scipy.sparse import lil_matrix, csc_matrix


try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate
USE_NUMBA = NUMBA_AVAILABLE and os.environ.get('Q1_REFERENCE_BACKEND', '0') != '1'

VERSION = "q1-merged-median-holdout-1.0"
# Source-file hash is based on the ACTUAL running file, independent of its name.
_LOADED_CODE_DIGEST = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

def code_digest() -> str:
    return _LOADED_CODE_DIGEST

def list_numeric_config(cfg) -> list[float]:
    return [float(v) for v in asdict(cfg).values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]

def layer_faces(length: float, n: int, stretch: float = 0.0) -> np.ndarray:
    """Nested exponential face mapping; material interfaces are exact endpoints."""
    if n < 1 or length <= 0 or not 0 <= stretch <= 10:
        raise ValueError("Invalid layer-grid specification")
    z = np.linspace(0.0, 1.0, n + 1)
    e = length * z if stretch == 0 else length * np.expm1(stretch*z) / np.expm1(stretch)
    e[0], e[-1] = 0.0, length
    return e

# ==================== config ====================
@dataclass(frozen=True)
class Config:
    cells: int = 1
    fixture: str = 'H1'
    basis: str = 'B'
    proton_face_scheme: str = 'exact'  # legacy_center is for regression comparison only
    mesh: tuple = (3, 2, 3, 3, 3)
    cgdl_stretch: float = 0.0  # exp mapping, 0 recovers the source uniform layer
    atol_scale: float = 1.0
    board_n: int = 2
    end_n: int = 4
    shared_bp: bool = False
    water: str = 'W1'
    outlet: str = 'closed'
    thermal: str = 'phase'
    observation: str = 'MEA'
    j0: float = 0.01
    tau_b: float = 1.0
    tau_f: float = 1.0
    tau_m: float = 1.0
    mu_factor: float = 1.0
    ice_connectivity: float = 3.0
    h: float = 40.0
    contact_r: float = 0.0
    Rc: float = 1e-06
    lambda0: float = 3.0
    concentration: str = 'given'
    end_concentration_factor: float = 1.0
    cl_proton_loss: bool = False
    cathode_hydration_exponent: float = 0.0
    vapor_equilibrium: str = 'ice_reference'
    half_cl: bool = False
    T0: float = 253.15
    ambient: float = 253.15
    rtol: float = 1e-05
    max_step: float = 0.5
    voltage_limit: float = 0.3
    charge_limit: float = 20.0
    current_limit: float = 0.5
    success_margin: float = 0.01
    horizon: float = 300.0
    sample: float = 0.2
    control_prediction_horizon: float = 5.0
    seed: int = 20260923
    cl_conductivity_factor: float = 1.0
    sorption_temp_coefficient: float = 0.0
    bound_flux_scheme: str = 'upwind'
    bp_heat_capacity_factor: float = 1.0
    nucleation: str = 'immediate'  # legacy comparison; merged seed uses median
    nucleation_prefactor: float = 1.127e10
    nucleation_barrier: float = 403000.0
    nucleation_threshold: float = math.log(2.0)
    bound_ice_rate: float = 0.0  # s^-1, scenario coefficient, not fitted

    def changed(self, **kwargs):
        return replace(self, **kwargs)

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def validate(self):
        if self.nucleation not in ('median', 'immediate'):
            raise ValueError('Unknown nucleation scenario')
        if min(self.nucleation_prefactor, self.nucleation_barrier, self.nucleation_threshold) <= 0 or self.bound_ice_rate < 0:
            raise ValueError('Invalid nucleation/bound-ice scenario')
        if self.bound_flux_scheme not in ('upwind', 'exponential'):
            raise ValueError('Unknown bound-water flux scheme')
        if not 0.5 <= self.bp_heat_capacity_factor <= 1.5:
            raise ValueError('Effective fixture capacity factor must be in [0.5, 1.5]')
        if not np.isfinite(list_numeric_config(self)).all():
            raise ValueError('Config contains a non-finite number')
        if self.cl_conductivity_factor <= 0:
            raise ValueError('cl_conductivity_factor must be positive')
        if self.cgdl_stretch < 0 or self.cgdl_stretch > 10:
            raise ValueError('cgdl_stretch must be in [0, 10]')
        if self.atol_scale <= 0 or self.rtol <= 0 or self.max_step <= 0 or self.sample <= 0 or self.horizon <= 0:
            raise ValueError('Solver tolerances, horizon and time steps must be positive')
        if self.board_n < 1 or self.end_n < 1 or self.h < 0:
            raise ValueError('Invalid plate grid or convection coefficient')
        if any(int(n) != n or n < 1 for n in self.mesh):
            raise ValueError('Five positive integer layer counts are required')
        assert self.cells in (1, 5)
        assert self.fixture in ('H1', 'H2')
        assert self.basis in ('A', 'B')
        assert self.proton_face_scheme in ('exact', 'legacy_center')
        assert self.water in ('W0', 'W1')
        assert self.outlet in ('closed', 'drain')
        assert self.thermal in ('phase', 'effective')
        assert self.observation in ('MEA', 'unit')
        assert len(self.mesh) == 5 and min(self.mesh) >= 1
        assert self.j0 > 0 and min(self.tau_b, self.tau_f, self.tau_m) > 0
        assert self.cathode_hydration_exponent >= 0
        assert self.vapor_equilibrium in ('ice_reference', 'supercooled_liquid')

AREA = 0.0025

F = 96485.33212

R = 8.314462618

MW = 0.018

# Attachment 1, parameter table C27: 1000 g/mol = 1 kg/mol.
EW = 1.0

TM = 273.15

LV = 2500000.0

LF = 333600.0

P = 101325.0

YO2 = 0.233 / 0.032 / (0.233 / 0.032 + 0.767 / 0.028)

LAYER_NAMES = ('aGDL', 'aCL', 'PEM', 'cCL', 'cGDL')

THICKNESS = (0.00015, 3.4e-06, 1.2e-05, 1.13e-05, 0.00015)

POROSITY = (0.8, 0.3916, 0.0, 0.4207, 0.8)

DRY_C = (185 * 545, 970 * 240, 2150 * 1050, 970 * 240, 185 * 545)

K_SOLID = (0.3, 0.27, 0.24, 0.27, 0.3)

# ==================== properties ====================
def psat(T, liquid=False):
    c = np.clip(np.asarray(T) - TM, -90, 150)
    l = 611.21 * np.exp((18.678 - c / 234.5) * c / (257.14 + c))
    if liquid:
        return l
    i = 611.15 * np.exp((23.036 - c / 333.7) * c / (279.82 + c))
    return np.where(c >= 0, l, i)

def partition(w, ice, T, eps, liquid_reference=False):
    """Analytic saturation partition including the liquid-induced pore change.

    w=vapor+liquid, per total porous volume. Tiny negative trial states during
    Newton are extended only in constitutive evaluation, never projected into y.
    """
    wp = np.maximum(w, 0)
    rs = MW * psat(T, liquid=liquid_reference) / (R * T)
    sat = rs * (eps - ice / 920 - wp / 990) / (1 - rs / 990)
    mv = np.minimum(wp, np.maximum(0, sat))
    ml = wp - mv
    return (mv, ml, eps - ice / 920 - ml / 990)

def lambda_eq(aw):
    a = np.clip(aw, 0, 1)
    return 0.043 + 17.81 * a - 39.85 * a * a + 36 * a * a * a

def membrane_D(T, lam):
    return 1e-10 * np.exp(2416 * (1 / 303.15 - 1 / T)) * (2.563 - 0.33 * lam + 0.0264 * lam ** 2 - 0.000671 * lam ** 3)

def conductivity(T, lam):
    return (0.5139 * lam - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / T))

def sigma(T):
    tau = 1 - np.asarray(T) / 647.096
    return 0.2358 * tau ** 1.256 * (1 - 0.625 * tau)

def liquid_mu(T, factor=1.0):
    tr = np.maximum(np.asarray(T), 253.15) / 300
    return factor * 1e-06 * (280.68 * tr ** (-1.9) + 511.45 * tr ** (-7.7) + 61.131 * tr ** (-19.6) + 0.45903 * tr ** (-40))

def hydraulic_pressure(ml, ice, T, eps, K, angle):
    """p_l-p_g; opposite sign to plan's p_c=p_g-p_l."""
    free = np.maximum(eps - ice / 920, 1e-12)
    s = np.clip(ml / 990 / free, 0, 1)
    J = 1.417 * s - 2.12 * s * s + 1.263 * s * s * s
    return -sigma(T) * np.cos(np.deg2rad(angle)) * np.sqrt(eps / K) * J

def water_enthalpies(T):
    dt = T - TM
    return (2000 * dt + LV, 4182 * dt, 2050 * dt - LF, 4182 * dt)

def harmonic_faces(dx, coeff):
    return 1 / (dx[:-1] / (2 * np.maximum(coeff[:-1], 1e-30)) + dx[1:] / (2 * np.maximum(coeff[1:], 1e-30)))

def liquid_flux(ml, ice, T, eps, K, angle, dx, factor=1.0, exponent=3.0, drain=False, side=0):
    p = hydraulic_pressure(ml, ice, T, eps, K, angle)
    saturation = np.clip(ml / (990 * np.maximum(eps - ice / 920, 1e-12)), 0, 1)
    connect = np.maximum(1 - ice / (920 * eps), 0) ** exponent
    mob = saturation ** 3 / liquid_mu(T, factor)
    f = np.zeros(len(dx) + 1)
    dp = p[:-1] - p[1:]
    f[1:-1] = 990 * harmonic_faces(dx, K * connect) * dp * np.where(dp >= 0, mob[:-1], mob[1:])
    if drain:
        b = 0 if side == 0 else -1
        sign = -1 if side == 0 else 1
        f[b] = sign * 990 * K[b] * connect[b] * mob[b] * max(p[b], 0) / (dx[b] / 2)
    return f

# ==================== geometry ====================
class Geometry:

    def __init__(self, cfg):
        self.cfg = cfg
        self.layer_faces = [layer_faces(d, int(n), cfg.cgdl_stretch if k == 4 else 0.0)
                            for k, (d, n) in enumerate(zip(THICKNESS, cfg.mesh))]
        self.dx = np.concatenate([np.diff(e) for e in self.layer_faces])
        self.layer = np.repeat(np.arange(5), cfg.mesh)
        self.n = len(self.dx)
        self.x = np.cumsum(self.dx) - self.dx / 2
        self.eps = np.take(POROSITY, self.layer)
        self.omega = np.take([0, 0.3, 1, 0.3, 0], self.layer)
        self.bcap = self.omega * 2150 * MW / EW
        self.cla = np.flatnonzero(self.layer == 1)
        self.clc = np.flatnonzero(self.layer == 3)
        self.pem = np.flatnonzero(self.layer == 2)
        self.bound = np.flatnonzero(self.omega > 0)
        self.anode = np.flatnonzero(self.layer < 2)
        self.cathode = np.flatnonzero(self.layer > 2)
        self.porous = self.eps > 0
        self.K = np.where(np.isin(self.layer, [1, 3]), 6.2e-13, 6.2e-12)
        self.angle = np.where(np.isin(self.layer, [1, 3]), 100.0, 110.0)
        self.im = np.zeros(self.n)
        self.im[self.cla] = (self.x[self.cla] - THICKNESS[0]) / THICKNESS[1]
        self.im[self.pem] = 1
        self.im[self.clc] = 1 - (self.x[self.clc] - sum(THICKNESS[:3])) / THICKNESS[3]
        # Evaluate current at physical faces, including the two CL/PEM corners.
        # Layer-local linspace avoids roundoff in cumulative coordinates at corners.
        self.im_faces = np.concatenate((
            np.zeros(cfg.mesh[0]), (self.layer_faces[1] / THICKNESS[1])[:-1],
            np.ones(cfg.mesh[2]), (1 - self.layer_faces[3] / THICKNESS[3])[:-1],
            np.zeros(cfg.mesh[4] + 1)))
        self.gas_ref = np.where(self.layer < 2, 0.00011, 2.2e-05)
        self.vap_ref = np.where(self.layer < 2, 8.69e-05, 2.48e-05)
        self.gas_y = np.where(self.layer < 2, 1.0, 0.233 / 0.032 / (0.233 / 0.032 + 0.767 / 0.028))
        td, tc, tk, names, owners = ([], [], [], [], [])
        self.mea_indices = []
        self.heater_indices = []
        self.unit_indices = []

        def block(name, thickness, n, heatcap, k, owner, widths=None):
            ix = np.arange(len(td), len(td) + n)
            td.extend([thickness / n] * n if widths is None else widths)
            tc.extend([heatcap] * n)
            tk.extend([k] * n)
            names.extend([name] * n)
            owners.extend([owner] * n)
            return ix
        if cfg.fixture == 'H2' or cfg.cells == 5:
            block('endplate', 0.01, cfg.end_n, 7900 * 500, 15, -1)
        pending_bp = None
        for cell in range(cfg.cells):
            left = pending_bp if pending_bp is not None else block('BP', 0.002, cfg.board_n, 1980 * 766, 95, cell)
            mea = []
            for i in range(5):
                mea.extend(block(LAYER_NAMES[i], THICKNESS[i], cfg.mesh[i], DRY_C[i], K_SOLID[i], cell, np.diff(self.layer_faces[i])))
            right = block('BP', 0.002, cfg.board_n, 1980 * 766, 95, cell)
            pending_bp = right if cfg.shared_bp and cell < cfg.cells - 1 else None
            self.mea_indices.append(mea)
            self.heater_indices.append(right)
            self.unit_indices.append(np.r_[left, mea, right])
        if cfg.fixture == 'H2' or cfg.cells == 5:
            block('endplate', 0.01, cfg.end_n, 7900 * 500, 15, -1)
        self.tdx, self.C, self.k0 = [np.array(a, dtype=float) for a in (td, tc, tk)]
        self.tx = np.cumsum(self.tdx) - self.tdx / 2
        self.names, self.owners = (np.array(names), np.array(owners))
        self.C[self.names == 'BP'] *= cfg.bp_heat_capacity_factor
        self.mea_indices = np.array(self.mea_indices)
        self.nt = len(td)

    def manifest(self):
        import pandas as pd
        return pd.DataFrame(dict(x_m=self.tx, dx_m=self.tdx, layer=self.names, cell=self.owners, dry_C_J_m3K=self.C, k_table=self.k0))

# ==================== model ====================
class Model:

    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.g = Geometry(cfg)
        self.c, self.n = (cfg.cells, self.g.n)
        self.nw = 4 * self.c * self.n
        self.ih = self.nw
        self.il = self.ih + self.g.nt
        self.ledger_end = self.il + 3 * self.c + 2
        self.size = self.ledger_end + (self.c if cfg.nucleation == 'median' else 0)
        self.nucleated = np.full(self.c, cfg.nucleation == 'immediate', dtype=bool)
        self.active_columns = []
        for block, mask in enumerate((self.g.porous, self.g.porous, self.g.omega > 0, self.g.porous)):
            for c in range(self.c):
                self.active_columns.extend(block * self.c * self.n + c * self.n + np.flatnonzero(mask))
        self.active_columns.extend(range(self.ih, self.il))
        self._jac_pattern = None
        self._jac_groups = None
        self._jac_scale = np.ones(self.size)
        self._jac_scale[:self.c * self.n] = 0.0001
        self._jac_scale[self.c * self.n:2 * self.c * self.n] = 0.01
        self._jac_scale[3 * self.c * self.n:self.nw] = 0.01

    def unpack(self, y):
        z = y[:self.nw].reshape(4, self.c, self.n)
        return (z, y[self.ih:self.il] * self.g.C, y[self.il:self.ledger_end])

    def phases(self, y):
        z, H, led = self.unpack(y)
        w, ice, b, gas = z
        T = TM + H / self.g.C
        ix = self.g.mea_indices
        for _ in range(6):
            mv, ml, eg = partition(w, ice, T[ix], self.g.eps, self.cfg.vapor_equilibrium == 'supercooled_liquid')
            cap = self.g.C[ix] + 2000 * mv + 4182 * ml + 2050 * ice + 4182 * b
            new = TM + (H[ix] - LV * mv + LF * ice) / cap
            T[ix] = new
        mv, ml, eg = partition(w, ice, T[ix], self.g.eps, self.cfg.vapor_equilibrium == 'supercooled_liquid')
        lam = np.divide(b, self.g.bcap, out=np.zeros_like(b), where=self.g.bcap > 0)
        conc = gas / np.maximum(eg, 1e-12)
        return dict(T=T, local_T=T[ix], mv=mv, ml=ml, mi=ice, mb=b, eg=eg, lam=lam, gas=conc, led=led, H=H, mobile=w)

    def initial(self, Tfield=None):
        y = np.zeros(self.size)
        z = y[:self.nw].reshape(4, self.c, self.n)
        z[2] = self.g.bcap * self.cfg.lambda0
        T = np.full(self.g.nt, self.cfg.T0) if Tfield is None else np.array(Tfield).copy()
        assert T.shape == (self.g.nt,)
        z[3] = self.g.eps * self.g.gas_y * P / (R * T[self.g.mea_indices])
        H = self.g.C * (T - TM)
        H[self.g.mea_indices] += z[2] * 4182 * (T[self.g.mea_indices] - TM)
        y[self.ih:self.il] = H / self.g.C
        return y

    def thermal_k(self, s):
        k = self.g.k0.copy()
        if self.cfg.thermal == 'effective':
            return k
        layer = self.g.layer
        ks = np.take(K_SOLID, layer)
        kg = np.where(layer < 2, 0.1672, YO2 * 0.0246 + (1 - YO2) * 0.0235)
        solid = 1 - self.g.eps - self.g.omega
        km = solid * ks + self.g.omega * 0.24 + s['eg'] * kg + s['ml'] / 990 * 0.6 + s['mi'] / 920 * 2.3
        km[:, self.g.pem] = 0.24
        k[self.g.mea_indices] = km
        return k

    def _original_voltage(self, s, j):
        g = self.g
        T = s['local_T']
        J = j * 10000.0
        tc = np.average(T[:, g.clc], axis=1, weights=g.dx[g.clc])
        ca = np.average(s['gas'][:, g.cla], axis=1, weights=g.dx[g.cla])
        co = np.average(s['gas'][:, g.clc], axis=1, weights=g.dx[g.clc])
        ta = np.average(T[:, g.cla], axis=1, weights=g.dx[g.cla])
        ph, po = (ca * R * ta / P, co * R * tc / P)
        E = 1.229 - 0.00085 * (tc - 298.15) + R * tc / (2 * F) * np.log(np.maximum(ph, 1e-20) * np.sqrt(np.maximum(po, 1e-20)))
        si = s['mi'][:, g.clc] / (920 * g.eps[g.clc])
        hydration = np.clip(s['lam'][:, g.clc] / 14.0, 1e-12, 1.0) ** self.cfg.cathode_hydration_exponent
        active = np.average(np.maximum(1 - si, 1e-12) ** 3.5 * hydration, axis=1, weights=g.dx[g.clc])
        exchange = self.cfg.j0 * np.exp(-67000 / R * (1 / tc - 1 / 298.15)) * active
        act = R * tc / (0.5 * F) * np.arcsinh(J / (2 * exchange))
        kap = conductivity(T[:, g.pem], s['lam'][:, g.pem])
        ohm = J * (np.sum(g.dx[g.pem] / np.maximum(kap, 1e-12), axis=1) + self.cfg.Rc)
        cl_ohm = np.zeros(self.c)
        cl_kmin = np.full(self.c, np.inf)
        if self.cfg.cl_proton_loss:
            ids = np.r_[g.cla, g.clc]
            kcl = conductivity(T[:, ids], s['lam'][:, ids]) * g.omega[ids] ** 1.5
            lengths = np.where(g.layer[ids] == 1, THICKNESS[1], THICKNESS[3])
            weights = g.im[ids] ** 2 + (g.dx[ids] / lengths) ** 2 / 12
            cl_ohm = J * np.sum(g.dx[ids] * weights / np.maximum(kcl, 1e-12), axis=1)
            cl_kmin = np.min(kcl, axis=1)
            ohm += cl_ohm
        Dc = 2.2e-05 * (T[:, g.cathode] / 298.15) ** 1.75 * np.maximum(s['eg'][:, g.cathode], 1e-12) ** 1.5
        path = THICKNESS[4] + THICKNESS[3] * (0.5 if self.cfg.half_cl else 1.0)
        deff = np.sum(g.dx[g.cathode]) / np.sum(g.dx[g.cathode] / Dc, axis=1)
        jlim = 4 * F * deff * co / path
        ratio = J / np.maximum(jlim, 1e-20)
        con = -R * tc / (4 * F) * np.log(np.maximum(1 - ratio, 1e-12))
        if self.cfg.concentration == 'unified':
            con[:] = 0
        if self.c == 5:
            con[[0, -1]] *= self.cfg.end_concentration_factor
        V = E - act - ohm - con
        return dict(V=V, reversible=E, activation=act, ohmic=ohm, concentration=con, cl_ohmic=cl_ohm, active_fraction=active, jlim=jlim / 10000.0, ratio=ratio, kappa_min=np.minimum(np.min(kap, axis=1), cl_kmin))

    def observations(self, s, j):
        g = self.g
        tm = np.average(s['local_T'], axis=1, weights=g.dx)
        tu = np.array([np.average(s['T'][ix], weights=g.tdx[ix]) for ix in g.unit_indices])
        temp = tm if self.cfg.observation == 'MEA' else tu
        return dict(T_mean=temp, T_mean_MEA=tm, T_mean_unit=tu, ice_bulk=np.max(s['mi'] / 920, axis=1), ice_saturation=np.max(s['mi'][:, g.porous] / (920 * g.eps[g.porous]), axis=1), **self.voltage(s, j))

    def _original_margins(self, y, j):
        s = self.phases(y)
        o = self.observations(s, j)
        g = self.g
        D = membrane_D(s['local_T'][:, g.bound], s['lam'][:, g.bound])
        return dict(voltage=float(np.min(o['V']) - self.cfg.voltage_limit), pore=float(np.min(s['eg'][:, g.porous]) - 1e-08), gas=float(np.min(s['gas'][:, g.porous]) - 1e-08), oxygen=float(1 - np.max(o['ratio'])), membrane=float(min(np.min(o['kappa_min']), np.min(D) * 10000000000.0)), nonnegative=float(min(np.min(s['mobile']), np.min(s['mi']), np.min(s['mb'])) + 1e-05), charge=float(self.cfg.charge_limit - s['led'][-1]), success=float(np.min(o['T_mean']) - TM - self.cfg.success_margin))

    def _original_rhs(self, t, y, j, u, details=False):
        g = self.g
        cfg = self.cfg
        J = j * 10000.0
        s = self.phases(y)
        v = self.voltage(s, j)
        out = np.zeros_like(y)
        dz = out[:self.nw].reshape(4, self.c, self.n)
        dw, di, db, dg = dz
        dH = np.zeros(g.nt)
        ledger = out[self.il:self.ledger_end]
        massheat = np.zeros((self.c, self.n))
        fluxes = {name: np.zeros((self.c, self.n + 1)) for name in ('vapor', 'liquid', 'bound', 'gas')}
        water_external = np.zeros(self.c)
        net_energy_mass = 0.0
        for c in range(self.c):
            T = s['local_T'][c]
            hv, hl, hi, hb = water_enthalpies(T)
            for ids, side in ((g.anode, 0), (g.cathode, -1)):
                dx = g.dx[ids]
                eg = np.maximum(s['eg'][c, ids], 1e-12)
                gasD = g.gas_ref[ids] * (T[ids] / 298.15) ** 1.75 * eg ** 1.5
                cv = s['mv'][c, ids] / (MW * eg)
                dv = g.vap_ref[ids] * (T[ids] / 298.15) ** 1.75 * eg ** 1.5
                fv = np.zeros(len(ids) + 1)
                fg = fv.copy()
                fv[1:-1] = MW * harmonic_faces(dx, dv) * (cv[:-1] - cv[1:])
                cg = s['gas'][c, ids]
                fg[1:-1] = harmonic_faces(dx, gasD) * (cg[:-1] - cg[1:])
                if side == 0:
                    fv[0] = -MW * dv[0] * cv[0] / (dx[0] / 2)
                    fg[0] = gasD[0] * (g.gas_y[ids[0]] * P / (R * T[ids[0]]) - cg[0]) / (dx[0] / 2)
                else:
                    fv[-1] = MW * dv[-1] * cv[-1] / (dx[-1] / 2)
                    fg[-1] = gasD[-1] * (cg[-1] - g.gas_y[ids[-1]] * P / (R * T[ids[-1]])) / (dx[-1] / 2)
                fl = np.zeros_like(fv)
                if cfg.water == 'W1':
                    fl = liquid_flux(s['ml'][c, ids], s['mi'][c, ids], T[ids], g.eps[ids], g.K[ids], g.angle[ids], dx, cfg.mu_factor, cfg.ice_connectivity, cfg.outlet == 'drain', side)
                dw[c, ids] += (fv[:-1] + fl[:-1] - fv[1:] - fl[1:]) / dx
                dg[c, ids] += (fg[:-1] - fg[1:]) / dx
                for name, f in [('vapor', fv), ('liquid', fl), ('gas', fg)]:
                    fluxes[name][c, ids[0]:ids[-1] + 2] = f
                water_external[c] += fv[0] + fl[0] - fv[-1] - fl[-1]
                e = np.zeros_like(fv)
                for f, h in [(fv, hv[ids]), (fl, hl[ids])]:
                    e[1:-1] += f[1:-1] * np.where(f[1:-1] >= 0, h[:-1], h[1:])
                    e[0] += f[0] * h[0]
                    e[-1] += f[-1] * h[-1]
                massheat[c, ids] += (e[:-1] - e[1:]) / dx
                net_energy_mass += e[0] - e[-1]
            ids = g.bound
            lam = s['lam'][c, ids]
            D = np.maximum(membrane_D(T[ids], lam), 1e-20)
            conduct = g.omega[ids] ** 1.5 * 2150 * MW / EW * D
            f = np.zeros(len(ids) + 1)
            f[1:-1] = harmonic_faces(g.dx[ids], conduct) * (lam[:-1] - lam[1:])
            fm = g.im_faces[ids[:-1] + 1]
            if self.cfg.proton_face_scheme == 'legacy_center':
                fm = (g.im[ids[:-1]] + g.im[ids[1:]]) / 2
            donor = lam[:-1]
            f[1:-1] += MW * (2.5 * donor / 22) * J * fm / F
            if cfg.bound_flux_scheme == 'exponential':
                diffusive = harmonic_faces(g.dx[ids], conduct)
                drift = MW * (2.5 / 22) * J * fm / F
                pe = drift / diffusive
                # Exact constant-coefficient drift-diffusion face flux.
                # B(-Pe) = B(Pe) + Pe avoids overflow for positive drift.
                bernoulli = np.array([bernoulli_positive(float(p)) for p in pe])
                f[1:-1] = diffusive * bernoulli * (lam[:-1] - lam[1:]) + drift * lam[:-1]
            db[c, ids] += (f[:-1] - f[1:]) / g.dx[ids]
            fluxes['bound'][c, ids[0]:ids[-1] + 2] = f
            e = np.zeros_like(f)
            e[1:-1] = f[1:-1] * np.where(f[1:-1] >= 0, hb[ids[:-1]], hb[ids[1:]])
            massheat[c, ids] += (e[:-1] - e[1:]) / g.dx[ids]
            cls = np.r_[g.cla, g.clc]
            pv = s['mv'][c, cls] / np.maximum(s['eg'][c, cls], 1e-12) * R * T[cls] / MW
            beq = g.bcap[cls] * lambda_eq(pv / psat(T[cls], liquid=True))
            rb = (beq - s['mb'][c, cls]) / cfg.tau_b
            dw[c, cls] -= rb
            db[c, cls] += rb
            freeze = s['ml'][c] * np.maximum(TM - T, 0) / (20 * cfg.tau_f) * self.nucleated[c]
            melt = np.maximum(s['mi'][c], 0) * np.maximum(T - TM, 0) / (5 * cfg.tau_m)
            di[c] = freeze - melt
            dw[c] -= di[c]
            if self.nucleated[c] and cfg.bound_ice_rate:
                eq_ice = lambda_eq(psat(T[cls]) / psat(T[cls], liquid=True))
                bound_freeze = cfg.bound_ice_rate * g.bcap[cls] * np.maximum(s['lam'][c, cls] - eq_ice, 0) * (T[cls] < TM)
                db[c, cls] -= bound_freeze
                di[c, cls] += bound_freeze
                # Internal phase transfer at unchanged total enthalpy: no extra latent source.
            if cfg.nucleation == 'median':
                cold = TM - T
                valid = cold > .01
                out[self.ledger_end + c] = cfg.nucleation_prefactor * AREA * np.sum(
                    s['ml'][c, valid] / 990 * g.dx[valid] *
                    np.exp(-cfg.nucleation_barrier / (T[valid] * cold[valid]**2)))
            sw = MW * J / (2 * F * THICKNESS[3])
            dw[c, g.clc] += sw
            massheat[c, g.clc] += sw * hl[g.clc]
            net_energy_mass += np.sum(sw * hl[g.clc] * g.dx[g.clc])
            dg[c, g.cla] -= J / (2 * F * THICKNESS[1])
            dg[c, g.clc] -= J / (4 * F * THICKNESS[3])
            qrx = J * (1.48 - v['V'][c])
            dH[g.mea_indices[c]] += qrx / sum(THICKNESS) + massheat[c]
            heater = g.heater_indices[c]
            dH[heater] += u[c] * 10000.0 / sum(g.tdx[heater])
            ledger[c] = MW * J / (2 * F) + water_external[c]
            ledger[self.c + c] = qrx
            ledger[2 * self.c + c] = u[c] * 10000.0
        k = self.thermal_k(s)
        conduct = harmonic_faces(g.tdx, k)
        if cfg.contact_r:
            interface = (g.names[:-1] == 'endplate') != (g.names[1:] == 'endplate')
            conduct[interface] = 1 / (1 / conduct[interface] + cfg.contact_r)
        heatflux = np.zeros(g.nt + 1)
        heatflux[1:-1] = conduct * (s['T'][:-1] - s['T'][1:])
        heatflux[0] = (cfg.ambient - s['T'][0]) / (1 / cfg.h + g.tdx[0] / (2 * k[0])) if cfg.h else 0.0
        heatflux[-1] = (s['T'][-1] - cfg.ambient) / (1 / cfg.h + g.tdx[-1] / (2 * k[-1])) if cfg.h else 0.0
        dH += (heatflux[:-1] - heatflux[1:]) / g.tdx
        out[self.ih:self.il] = dH / g.C
        ledger[-2] = np.sum(ledger[self.c:3 * self.c]) + net_energy_mass + heatflux[0] - heatflux[-1]
        ledger[-1] = j
        if details:
            return dict(derivative=out, fluxes=fluxes, heatflux=heatflux, water_external=water_external, reaction_W_m2=ledger[self.c:2 * self.c], massheat=massheat)
        return out

    def balances(self, y, initial):
        z, H, led = self.unpack(y)
        zi, Hi, _ = self.unpack(initial)
        dw = np.sum((z[:3] - zi[:3]) * self.g.dx, axis=(0, 2))
        re_w = dw - led[:self.c]
        re_h = np.dot(H - Hi, self.g.tdx) - led[-2]
        wscale = np.maximum(np.sum(zi[:3] * self.g.dx, axis=(0, 2)) + MW * 10000.0 * led[-1] / (2 * F), 1e-08)
        hscale = max(abs(led[-2]), abs(np.dot(Hi, self.g.tdx)), 1.0)
        return (float(np.max(abs(re_w) / wscale)), float(abs(re_h) / hscale))

    def sparsity(self):
        a = lil_matrix((self.size, self.size), dtype=int)
        g = self.g
        for c in range(self.c):
            ix = np.r_[np.concatenate([np.arange(b * self.c * self.n + c * self.n, b * self.c * self.n + (c + 1) * self.n) for b in range(4)]), self.ih + g.mea_indices[c]]
            a[np.ix_(ix, ix)] = 1
        for i in range(g.nt):
            a[self.ih + i, self.ih + max(0, i - 1):self.ih + min(g.nt, i + 2)] = 1
        a[self.il:, :self.il] = 1
        return a.tocsr()

    def _original_jacobian(self, t, y, j, u):
        """Bounded forward differences avoid unbounded perturbations of zero slots.

        All ledger columns are analytically zero. The invariant off-material slots
        remain zero, so their Newton corrections and columns can also be omitted.
        A fixed relative step is independent of SciPy's adaptive num_jac factors.
        """
        if self._jac_pattern is None:
            pattern = lil_matrix((self.size, self.size), dtype=int)
            for c in range(self.c):
                ix = np.r_[np.concatenate([np.arange(b * self.c * self.n + c * self.n, b * self.c * self.n + (c + 1) * self.n) for b in range(4)]), self.ih + self.g.unit_indices[c]]
                pattern[np.ix_(ix, ix)] = 1
                pattern[self.il + c, ix] = 1
                pattern[self.il + self.c + c, ix] = 1
                if self.cfg.nucleation == 'median':
                    pattern[self.ledger_end + c, ix] = 1
            for i in range(self.g.nt):
                pattern[self.ih + i, self.ih + max(0, i - 1):self.ih + min(self.g.nt, i + 2)] = 1
            self._jac_pattern = pattern.tocsc()
            occupied = []
            groups = []
            for col in self.active_columns:
                ix = set(self._jac_pattern.indices[self._jac_pattern.indptr[col]:self._jac_pattern.indptr[col + 1]])
                for k, used in enumerate(occupied):
                    if not ix & used:
                        groups[k].append(col)
                        used.update(ix)
                        break
                else:
                    groups.append([col])
                    occupied.append(ix)
            self._jac_groups = groups
        f0 = self.rhs(t, y, j, u)
        rows = []
        cols = []
        data = []
        for group in self._jac_groups:
            yp = y.copy()
            delta = 1e-08 * np.maximum(self._jac_scale[group], np.abs(y[group]))
            yp[group] += delta
            diff = self.rhs(t, yp, j, u) - f0
            for col, step in zip(group, delta):
                ix = self._jac_pattern.indices[self._jac_pattern.indptr[col]:self._jac_pattern.indptr[col + 1]]
                values = diff[ix] / step
                nz = values != 0
                ix = ix[nz]
                values = values[nz]
                rows.extend(ix)
                cols.extend([col] * len(ix))
                data.extend(values)
        matrix = csc_matrix((data, (rows, cols)), shape=(self.size, self.size))
        heat_row = np.asarray(self.g.C * self.g.tdx @ matrix[self.ih:self.il, :]).ravel()
        nz = np.flatnonzero(heat_row)
        matrix += csc_matrix((heat_row[nz], (np.full(len(nz), self.il + 3 * self.c), nz)), shape=matrix.shape)
        return matrix

    # Current branch: explicit CL conductivity scaling and equivalent JIT implementation.
    def voltage(self,s,j):
        out=self._original_voltage(s,j)
        if self.cfg.cl_proton_loss:
            factor=self.cfg.cl_conductivity_factor
            new=out['cl_ohmic']/factor
            out['V']+=out['cl_ohmic']-new
            out['ohmic']+=new-out['cl_ohmic']
            out['cl_ohmic']=new
            ids=np.r_[self.g.cla,self.g.clc]
            kp=conductivity(s['local_T'][:,self.g.pem],s['lam'][:,self.g.pem])
            kc=factor*conductivity(s['local_T'][:,ids],s['lam'][:,ids])*self.g.omega[ids]**1.5
            out['kappa_min']=np.minimum(kp.min(axis=1),kc.min(axis=1))
        return out

    def reference_rhs(self,t,y,j,u,details=False):
        out=self._original_rhs(t,y,j,u,details)
        beta=self.cfg.sorption_temp_coefficient
        if beta:
            derivative=out['derivative'] if details else out
            dz=derivative[:self.nw].reshape(4,self.c,self.n)
            st=self.phases(y);ids=np.r_[self.g.cla,self.g.clc]
            for c in range(self.c):
                T=st['local_T'][c,ids]
                pv=st['mv'][c,ids]/np.maximum(st['eg'][c,ids],1e-12)*R*T/MW
                old=(self.g.bcap[ids]*lambda_eq(pv/psat(T,liquid=True))-st['mb'][c,ids])/self.cfg.tau_b
                delta=old*(np.exp(-beta*(T-253.15))-1)
                dz[0,c,ids]-=delta;dz[2,c,ids]+=delta
        return out

    def rhs(self,t,y,j,u,details=False):
        c=self.cfg;g=self.g
        if not USE_NUMBA or details or self.c!=1 or c.half_cl or c.contact_r:
            return self.reference_rhs(t,y,j,u,details)
        if not hasattr(self,'_compiled_par'):
            self._compiled_par=np.array([c.j0,c.tau_b,c.tau_f,c.tau_m,c.mu_factor,c.ice_connectivity,c.h,c.ambient,c.Rc,
              float(c.vapor_equilibrium=='supercooled_liquid'),float(c.thermal=='effective'),float(c.water=='W1'),float(c.outlet=='drain'),
              float(c.concentration=='unified'),c.cathode_hydration_exponent,float(c.cl_proton_loss),EW,c.cl_conductivity_factor,c.sorption_temp_coefficient,float(c.bound_flux_scheme=='exponential'),
              float(c.nucleation=='median'), c.nucleation_prefactor, c.nucleation_barrier,
              float(self.nucleated[0]), c.bound_ice_rate])
            self._compiled_im_faces=g.im_faces.copy()
            if c.proton_face_scheme=='legacy_center':
                ids=g.bound;self._compiled_im_faces[ids[:-1]+1]=(g.im[ids[:-1]]+g.im[ids[1:]])/2
        self._compiled_par[23] = float(self.nucleated[0])
        return kernel(y,float(j),float(u[0]),g.dx,g.layer,g.eps,g.omega,g.bcap,g.K,g.angle,self._compiled_im_faces,g.im,
           g.mea_indices[0],g.tdx,g.C,g.k0,g.heater_indices[0],self._compiled_par)

    def margins(self,y,j):
        last=getattr(self,'_margin_memo',None)
        if last is not None and float(j)==last[0] and np.array_equal(y,last[1]):return last[2].copy()
        result=self._original_margins(y,j);self._margin_memo=(float(j),y.copy(),result.copy());return result

    def jacobian(self,t,y,j,u):
        if not USE_NUMBA or self.c!=1 or self.cfg.fixture!='H1' or self.cfg.contact_r or self.cfg.half_cl:
            return self._original_jacobian(t,y,j,u)
        self.rhs(t,y,j,u)
        g=self.g
        return csc_matrix(jac_kernel(y,float(j),float(u[0]),g.dx,g.layer,g.eps,g.omega,g.bcap,g.K,g.angle,
           self._compiled_im_faces,g.im,g.mea_indices[0],g.tdx,g.C,g.k0,g.heater_indices[0],
           self._compiled_par,np.asarray(self.active_columns,dtype=np.int64),self._jac_scale))



# ==================== Equivalent compiled numerical kernels ====================
@njit(cache=True)
def bernoulli_positive(x):
    if abs(x) < 1e-4:
        return 1.0 - x/2.0 + x*x/12.0 - x**4/720.0
    if x > 50.0:
        return x*np.exp(-x)/(1.0-np.exp(-x))
    return x/np.expm1(x)

@njit(cache=True)
def sat_pressure(T, liquid):
    c=min(max(T-273.15,-90.),150.)
    if liquid or c>=0:
        return 611.21*np.exp((18.678-c/234.5)*c/(257.14+c))
    return 611.15*np.exp((23.036-c/333.7)*c/(279.82+c))

@njit(cache=True)
def partition_one(w, ice, T, eps, liquid):
    wp=max(w,0.)
    rs=.018*sat_pressure(T,liquid)/(8.314462618*T)
    sat=rs*(eps-ice/920.-wp/990.)/(1.-rs/990.)
    mv=min(wp,max(0.,sat)); ml=wp-mv
    return mv,ml,eps-ice/920.-ml/990.

@njit(cache=True)
def kernel(y,j,u,dx,lay,eps,omega,bcap,K,angle,imface,imcenter,ix,tdx,C,k0,heaters,par):
    # par: j0,tau_b,tau_f,tau_m,mu_factor,ice_connectivity,h,ambient,Rc,
    #      vapor_liquid,thermal_effective,water_enabled,drain,concentration_unified,
    #      hydration_gamma,cl_proton_loss,EW_kg_mol
    R=8.314462618; F=96485.33212; MW=.018; TM=273.15; P=101325.
    n=len(dx); nt=len(C); ih=4*n; il=ih+nt; J=j*1e4
    w=y[:n]; ice=y[n:2*n]; bound=y[2*n:3*n]; gas=y[3*n:4*n]
    T=TM+y[ih:il]; H=y[ih:il]*C
    mv=np.zeros(n); ml=np.zeros(n); eg=np.zeros(n); lam=np.zeros(n); cg=np.zeros(n)
    for turn in range(6):
        for k in range(n):
            a,b,c=partition_one(w[k],ice[k],T[ix[k]],eps[k],par[9]>0.)
            T[ix[k]]=TM+(H[ix[k]]-2500000.*a+333600.*ice[k])/(C[ix[k]]+2000.*a+4182.*b+2050.*ice[k]+4182.*bound[k])
    for k in range(n):
        mv[k],ml[k],eg[k]=partition_one(w[k],ice[k],T[ix[k]],eps[k],par[9]>0.)
        if bcap[k]>0: lam[k]=bound[k]/bcap[k]
        cg[k]=gas[k]/max(eg[k],1e-12)
    tc=0.;ta=0.;ca=0.;co=0.;wa=0.;wc=0.;active=0.;res=0.;dr=0.;cpath=0.
    for k in range(n):
        tk=T[ix[k]]
        if lay[k]==1:
            wa+=dx[k];ta+=dx[k]*tk;ca+=dx[k]*cg[k]
        elif lay[k]==3:
            wc+=dx[k];tc+=dx[k]*tk;co+=dx[k]*cg[k]
            active+=dx[k]*max(1.-ice[k]/(920.*eps[k]),1e-12)**3.5*min(max(lam[k]/14.,1e-12),1.)**par[14]
        if lay[k]==2:
            kap=(.5139*lam[k]-.326)*np.exp(1268.*(1./303.15-1./tk))
            res+=dx[k]/max(kap,1e-12)
        if par[15]>0 and (lay[k]==1 or lay[k]==3):
            kcl=par[17]*(.5139*lam[k]-.326)*np.exp(1268.*(1./303.15-1./tk))*omega[k]**1.5
            thickness=.0000034 if lay[k]==1 else .0000113
            weight=imcenter[k]**2+(dx[k]/thickness)**2/12.
            res+=dx[k]*weight/max(kcl,1e-12)
        if lay[k]>2:
            dc=2.2e-5*(tk/298.15)**1.75*max(eg[k],1e-12)**1.5
            dr+=dx[k]/dc;cpath+=dx[k]
    tc/=wc;ta/=wa;co/=wc;ca/=wa;active/=wc
    ph=ca*R*ta/P;po=co*R*tc/P
    E=1.229-.00085*(tc-298.15)+R*tc/(2.*F)*np.log(max(ph,1e-20)*np.sqrt(max(po,1e-20)))
    exchange=par[0]*np.exp(-67000./R*(1./tc-1./298.15))*active
    act=R*tc/(.5*F)*np.arcsinh(J/(2.*exchange))
    ohm=J*(res+par[8]);jlim=4.*F*(cpath/dr)*co/(.00015+.0000113)
    con=-R*tc/(4.*F)*np.log(max(1.-J/max(jlim,1e-20),1e-12))
    if par[13]>0: con=0.
    V=E-act-ohm-con; qrx=J*(1.48-V)
    out=np.zeros(len(y));dw=out[:n];di=out[n:2*n];db=out[2*n:3*n];dg=out[3*n:4*n]
    massheat=np.zeros(n);dH=np.zeros(nt);extwater=0.;extheat=0.
    for side in range(2):
        start=0;end=0
        if side==0:
            start=0
            for k in range(n):
                if lay[k]<2:end=k+1
        else:
            end=n
            for k in range(n):
                if lay[k]==3:
                    start=k;break
        count=end-start
        fv=np.zeros(count+1);fg=np.zeros(count+1);fl=np.zeros(count+1)
        gd=np.zeros(count);vd=np.zeros(count);cv=np.zeros(count)
        pressure=np.zeros(count);hydro=np.zeros(count);mob=np.zeros(count)
        for a in range(count):
            k=start+a;tk=T[ix[k]];ep=max(eg[k],1e-12)
            gd[a]=(1.1e-4 if side==0 else 2.2e-5)*(tk/298.15)**1.75*ep**1.5
            vd[a]=(8.69e-5 if side==0 else 2.48e-5)*(tk/298.15)**1.75*ep**1.5
            cv[a]=mv[k]/(MW*ep)
            s=min(max(ml[k]/(990.*max(eps[k]-ice[k]/920.,1e-12)),0.),1.)
            sig=.2358*(1.-tk/647.096)**1.256*(1.-.625*(1.-tk/647.096))
            pressure[a]=-sig*np.cos(angle[k]*np.pi/180.)*np.sqrt(eps[k]/K[k])*(1.417*s-2.12*s*s+1.263*s*s*s)
            hydro[a]=K[k]*max(1.-ice[k]/(920.*eps[k]),0.)**par[5]
            tr=max(tk,253.15)/300.
            mu=par[4]*1e-6*(280.68*tr**(-1.9)+511.45*tr**(-7.7)+61.131*tr**(-19.6)+.45903*tr**(-40))
            mob[a]=s**3/mu
        for a in range(count-1):
            k=start+a
            fv[a+1]=MW*(cv[a]-cv[a+1])/(dx[k]/(2.*max(vd[a],1e-30))+dx[k+1]/(2.*max(vd[a+1],1e-30)))
            fg[a+1]=(cg[k]-cg[k+1])/(dx[k]/(2.*max(gd[a],1e-30))+dx[k+1]/(2.*max(gd[a+1],1e-30)))
            if par[11]>0:
                dp=pressure[a]-pressure[a+1]
                fl[a+1]=990.*dp*(mob[a] if dp>=0 else mob[a+1])/(dx[k]/(2.*max(hydro[a],1e-30))+dx[k+1]/(2.*max(hydro[a+1],1e-30)))
        a=0 if side==0 else count-1;k=start+a;f=0 if side==0 else count;sign=-1. if side==0 else 1.
        gy=1. if side==0 else (.233/.032)/(.233/.032+.767/.028)
        fv[f]=sign*MW*vd[a]*cv[a]/(dx[k]/2.)
        fg[f]=sign*gd[a]*(cg[k]-gy*P/(R*T[ix[k]]))/(dx[k]/2.)
        if par[11]>0 and par[12]>0:
            fl[f]=sign*990.*hydro[a]*mob[a]*max(pressure[a],0.)/(dx[k]/2.)
        en=np.zeros(count+1)
        for a in range(1,count):
            for phase in range(2):
                flux=fv[a] if phase==0 else fl[a]
                donor=start+a-1 if flux>=0 else start+a
                dt=T[ix[donor]]-TM
                enth=2000.*dt+2500000. if phase==0 else 4182.*dt
                en[a]+=flux*enth
        en[0]=fv[0]*(2000.*(T[ix[start]]-TM)+2500000.)+fl[0]*4182.*(T[ix[start]]-TM)
        en[count]=fv[count]*(2000.*(T[ix[end-1]]-TM)+2500000.)+fl[count]*4182.*(T[ix[end-1]]-TM)
        for a in range(count):
            k=start+a
            dw[k]+=(fv[a]+fl[a]-fv[a+1]-fl[a+1])/dx[k]
            dg[k]+=(fg[a]-fg[a+1])/dx[k]
            massheat[k]+=(en[a]-en[a+1])/dx[k]
        extwater+=fv[0]+fl[0]-fv[count]-fl[count]
        extheat+=en[0]-en[count]
    ids=np.where(omega>0)[0];count=len(ids);conduct=np.zeros(count);bf=np.zeros(count+1);be=np.zeros(count+1)
    for a in range(count):
        k=ids[a];la=lam[k];tk=T[ix[k]]
        D=max(1e-10*np.exp(2416.*(1./303.15-1./tk))*(2.563-.33*la+.0264*la**2-.000671*la**3),1e-20)
        conduct[a]=omega[k]**1.5*2150.*MW/par[16]*D
    for a in range(count-1):
        k=ids[a];kk=ids[a+1]
        bf[a+1]=(lam[k]-lam[kk])/(dx[k]/(2.*max(conduct[a],1e-30))+dx[kk]/(2.*max(conduct[a+1],1e-30)))+MW*(2.5*lam[k]/22.)*J*imface[k+1]/F
        if par[19] > 0:
            diffusion=1./(dx[k]/(2.*max(conduct[a],1e-30))+dx[kk]/(2.*max(conduct[a+1],1e-30)))
            drift=MW*(2.5/22.)*J*imface[k+1]/F
            bf[a+1]=diffusion*bernoulli_positive(drift/diffusion)*(lam[k]-lam[kk])+drift*lam[k]
        donor=k if bf[a+1]>=0 else kk
        be[a+1]=bf[a+1]*4182.*(T[ix[donor]]-TM)
    for a in range(count):
        k=ids[a];db[k]+=(bf[a]-bf[a+1])/dx[k];massheat[k]+=(be[a]-be[a+1])/dx[k]
    for k in range(n):
        tk=T[ix[k]]
        if lay[k]==1 or lay[k]==3:
            pv=mv[k]/max(eg[k],1e-12)*R*tk/MW
            a=min(max(pv/sat_pressure(tk,True),0.),1.)
            eq=.043+17.81*a-39.85*a*a+36.*a*a*a
            rb=(bcap[k]*eq-bound[k])/(par[1]*np.exp(par[18]*(tk-253.15)))
            dw[k]-=rb;db[k]+=rb
        di[k]=par[23]*ml[k]*max(TM-tk,0.)/(20.*par[2])-max(ice[k],0.)*max(tk-TM,0.)/(5.*par[3])
        dw[k]-=di[k]
        if par[23]>0 and par[24]>0 and (lay[k]==1 or lay[k]==3) and tk<TM:
            ai=min(sat_pressure(tk,False)/sat_pressure(tk,True),1.)
            leq=.043+17.81*ai-39.85*ai*ai+36.*ai*ai*ai
            rate=par[24]*bcap[k]*max(lam[k]-leq,0.)
            db[k]-=rate;di[k]+=rate
        if par[20]>0 and tk<TM-.01:
            out[il+5]+=par[21]*.0025*ml[k]/990.*dx[k]*np.exp(-par[22]/(tk*(TM-tk)**2))
        if lay[k]==3:
            sw=MW*J/(2.*F*.0000113)
            dw[k]+=sw;massheat[k]+=sw*4182.*(tk-TM);extheat+=sw*4182.*(tk-TM)*dx[k]
            dg[k]-=J/(4.*F*.0000113)
        if lay[k]==1:dg[k]-=J/(2.*F*.0000034)
        dH[ix[k]]+=qrx/.0003267+massheat[k]
    heatwidth=0.
    for k in heaters:heatwidth+=tdx[k]
    for k in heaters:dH[k]+=u*1e4/heatwidth
    kk=k0.copy()
    if par[10]==0:
        ks=np.array([.3,.27,.24,.27,.3]);yo=(.233/.032)/(.233/.032+.767/.028)
        for k in range(n):
            kg=.1672 if lay[k]<2 else yo*.0246+(1.-yo)*.0235
            kk[ix[k]]=(1.-eps[k]-omega[k])*ks[lay[k]]+omega[k]*.24+eg[k]*kg+ml[k]/990.*.6+ice[k]/920.*2.3
            if lay[k]==2:kk[ix[k]]=.24
    hf=np.zeros(nt+1)
    for k in range(nt-1):
        hf[k+1]=(T[k]-T[k+1])/(tdx[k]/(2.*max(kk[k],1e-30))+tdx[k+1]/(2.*max(kk[k+1],1e-30)))
    if par[6]!=0:
        hf[0]=(par[7]-T[0])/(1./par[6]+tdx[0]/(2.*kk[0]));hf[-1]=(T[-1]-par[7])/(1./par[6]+tdx[-1]/(2.*kk[-1]))
    for k in range(nt):out[ih+k]=(dH[k]+(hf[k]-hf[k+1])/tdx[k])/C[k]
    out[il]=MW*J/(2.*F)+extwater;out[il+1]=qrx;out[il+2]=u*1e4
    out[il+3]=qrx+u*1e4+extheat+hf[0]-hf[-1];out[il+4]=j
    return out

@njit(cache=True)
def jac_kernel(y,j,u,dx,lay,eps,omega,bcap,K,angle,imface,imcenter,ix,tdx,C,k0,heaters,par,active,scale):
    s=len(y);n=len(dx);ih=4*n;il=ih+len(C)
    base=kernel(y,j,u,dx,lay,eps,omega,bcap,K,angle,imface,imcenter,ix,tdx,C,k0,heaters,par)
    mat=np.zeros((s,s))
    for k in active:
        yp=y.copy();h=1e-8*max(scale[k],abs(y[k]));yp[k]+=h
        f=kernel(yp,j,u,dx,lay,eps,omega,bcap,K,angle,imface,imcenter,ix,tdx,C,k0,heaters,par)
        for r in range(il+2):mat[r,k]=(f[r]-base[r])/h
        heat=0.
        for r in range(len(C)):heat+=C[r]*tdx[r]*mat[ih+r,k]
        mat[il+3,k]=heat
        if par[20]>0:mat[il+5,k]=(f[il+5]-base[il+5])/h
    return mat

# ==================== protocols ====================
@dataclass
class Protocol:
    kind: str = 'constant'
    params: tuple = (0.1,)
    powers: tuple = (0.0,) * 5
    heat_off: float = 0.0
    load_start: float = 0.0
    times: object = None
    currents: object = None

    def j(self, t):
        if self.kind == 'experiment':
            return float(np.interp(t, self.times, self.currents))
        s = max(0.0, t - self.load_start)
        if t < self.load_start:
            return 0.0
        if self.kind == 'constant':
            return self.params[0]
        if self.kind == 'ramp':
            a, b, tr = self.params
            return a + (b - a) * min(s / max(tr, 1e-08), 1)
        if self.kind == 'step':
            a, b, c, t1, t2 = self.params
            return a if s < t1 else b if s < t2 else c
        if self.kind == 'fixed':
            return min(0.005 * s, 0.3)
        if self.kind == 'zero':
            return 0.0
        raise ValueError(self.kind)

    def u(self, t, cells):
        return np.asarray(self.powers[:cells], float) if t < self.heat_off else np.zeros(cells)

    def breaks(self, end):
        b = [0.0, end, self.heat_off, self.load_start]
        if self.kind == 'step':
            b += [self.load_start + self.params[3], self.load_start + self.params[4]]
        if self.kind in ('ramp', 'fixed'):
            b += [self.load_start + (60 if self.kind == 'fixed' else self.params[2])]
        return sorted(set((float(v) for v in b if 0 <= v <= end)))

    def valid(self, cfg):
        if len(self.powers) < cfg.cells or not np.isfinite(self.powers).all():
            return False
        if min(self.powers) < 0 or max(self.powers) > 1 or self.heat_off < 0:
            return False
        if self.kind == 'experiment':
            levels = np.asarray(self.currents)
        elif self.kind == 'constant':
            levels = np.asarray(self.params[:1])
        elif self.kind == 'ramp':
            levels = np.asarray(self.params[:2])
        elif self.kind == 'step':
            levels = np.asarray(self.params[:3])
        elif self.kind == 'fixed':
            levels = np.array([0.0, 0.3])
        elif self.kind == 'zero':
            levels = np.array([0.0])
        else:
            return False
        return bool(np.isfinite(levels).all() and np.min(levels) >= 0 and (np.max(levels) <= cfg.current_limit + 1e-12))

# ==================== solver ====================
def simulate(cfg, protocol, Tfield=None, controller=None, stop_success=True, initial=None, complete_horizon=False):
    model = Model(cfg)
    y0 = model.initial(Tfield) if initial is None else initial.copy()
    y = y0.copy()
    nucleation_times = [None] * model.c
    if cfg.nucleation == 'median':
        for cell in range(model.c):
            existing_ice = np.any(model.phases(y)['mi'][cell] > 0)
            if y[model.ledger_end + cell] >= cfg.nucleation_threshold or existing_ice:
                model.nucleated[cell] = True
                nucleation_times[cell] = 0.0
    ts = None
    exception = None
    reason = 'horizon'
    wall = time.perf_counter()
    times, states, currents, powers = ([], [], [], [])
    bounds = protocol.breaks(cfg.horizon)

    def append(t, state, j, u):
        times.append(float(t))
        states.append(state.copy())
        currents.append(j)
        powers.append(np.array(u).copy())

    def events_for(jfun):
        keys = ['voltage', 'pore', 'gas', 'oxygen', 'membrane', 'nonnegative', 'charge', 'success']
        events = []
        for key in keys:

            def event(t, state, key=key):
                return model.margins(state, jfun(t))[key] + (1e-09 if key == 'voltage' else 0.0)
            event.direction = 1 if key == 'success' else -1
            heat_finished = controller.off_time is not None if controller is not None else a >= protocol.heat_off
            event.terminal = key != 'success' or (not complete_horizon and (stop_success or heat_finished))
            events.append(event)
        if cfg.nucleation == 'median':
            for cell in range(model.c):
                if model.nucleated[cell]:
                    continue
                def nucleation_event(t, state, cell=cell):
                    return state[model.ledger_end + cell] - cfg.nucleation_threshold
                nucleation_event.direction = 1
                nucleation_event.terminal = True
                keys.append(f'nucleation_{cell}')
                events.append(nucleation_event)
        return (keys, events)
    nfev = 0
    extrema = dict(V_min=float('inf'), ice_max=0.0, delta_T_max=0.0)
    extrema_to_success = None

    def update_ext(t, state, j):
        nonlocal extrema_to_success
        obs = model.observations(model.phases(state), j)
        extrema['V_min'] = min(extrema['V_min'], float(min(obs['V'])))
        extrema['ice_max'] = max(extrema['ice_max'], float(max(obs['ice_bulk'])))
        extrema['delta_T_max'] = max(extrema['delta_T_max'], float(np.ptp(obs['T_mean'])))
        if ts is None or t <= ts + 1e-08:
            extrema_to_success = extrema.copy()
    try:
        a = 0.0
        while a < cfg.horizon - 1e-10:
            if not protocol.valid(cfg):
                reason = 'control_input'
                break
            b = next((x for x in bounds if x > a + 1e-10))
            if controller is not None:
                b = min(b, a + cfg.control_prediction_horizon)
            j0 = protocol.j(a)
            if controller is not None:
                obs = model.observations(model.phases(y), j0)
                if controller.last_t is not None and abs(controller.last_t - a) < 1e-09:
                    u = controller.previous_power.copy()
                else:
                    u = controller.step(a, obs['T_mean'] - TM, obs['V'], j0)
            else:
                u = protocol.u((a + b) / 2, cfg.cells)
            append(a, y, j0, u)
            update_ext(a, y, j0)
            margins = model.margins(y, j0)
            failed = [k for k, value in margins.items() if k != 'success' and value < -1e-09]
            if failed:
                reason = failed[0]
                break
            if ts is None and margins['success'] >= 0:
                ts = a
                heat_finished = controller.off_time is not None if controller is not None else a >= protocol.heat_off
                if not complete_horizon and (stop_success or heat_finished):
                    reason = 'success'
                    break

            def jf(t):
                return protocol.j(min(t, np.nextafter(b, a)))
            keys, events = events_for(jf)
            atol = np.full(model.size, 1e-07)
            atol[2 * model.c * model.n:3 * model.c * model.n] = 1e-06
            atol[model.ih:model.il] = 1e-07
            atol[model.il:] = 1e-09
            atol *= cfg.atol_scale
            sol = solve_ivp(lambda t, state: model.rhs(t, state, jf(t), u), (a, b), y, method='BDF', rtol=cfg.rtol, atol=atol, max_step=cfg.max_step, jac=lambda t, state: model.jacobian(t, state, jf(t), u), dense_output=True, events=events)
            nfev += sol.nfev
            accepted_end = float(sol.t[-1])
            if controller is not None:
                first = (np.floor((a + 1e-09) / cfg.sample) + 1) * cfg.sample
                for tc in np.arange(first, sol.t[-1] + 1e-09, cfg.sample):
                    tc = min(float(tc), float(sol.t[-1]))
                    sample_state = sol.sol(tc)
                    obs = model.observations(model.phases(sample_state), protocol.j(tc))
                    new_u = controller.step(tc, obs['T_mean'] - TM, obs['V'], protocol.j(tc))
                    if not np.array_equal(new_u, u):
                        accepted_end = tc
                        break
            for ie, key in enumerate(keys):
                if len(sol.t_events[ie]) and sol.t_events[ie][0] <= accepted_end + 1e-09:
                    te = float(sol.t_events[ie][0])
                    if key == 'success' and ts is None:
                        ts = te
                    if events[ie].terminal:
                        reason = key
            check_times = list(sol.t[sol.t <= accepted_end + 1e-09]) + [accepted_end]
            if ts is not None and a <= ts <= accepted_end:
                check_times.append(ts)
            for t in sorted(set(check_times)):
                update_ext(t, sol.sol(t), jf(t))
            tt = np.arange(a + cfg.sample, accepted_end - 1e-09, cfg.sample)
            output_times = list(tt) + [accepted_end]
            if ts is not None and a <= ts <= accepted_end:
                output_times.append(ts)
            for t in sorted(set(output_times)):
                append(t, sol.sol(t), jf(t), u)
            y = sol.sol(accepted_end)
            if not sol.success:
                reason = 'numerical_failure'
                break
            if sol.status == 1 and accepted_end >= sol.t[-1] - 1e-09:
                if reason.startswith('nucleation_'):
                    cell = int(reason.split('_')[1])
                    model.nucleated[cell] = True
                    nucleation_times[cell] = accepted_end
                    model._margin_memo = None
                    reason = 'horizon'
                    a = accepted_end
                    continue  # restart BDF exactly at event; never toggle inside Newton trials
                marg = model.margins(y, jf(sol.t[-1]))
                if reason == 'charge' and marg['success'] >= -1e-07 and (marg['voltage'] >= -1e-08):
                    ts = float(sol.t[-1])
                    reason = 'success'
                break
            if not complete_horizon and controller is not None and (ts is not None) and (controller.off_time is not None) and (accepted_end >= max(ts, controller.off_time)):
                reason = 'success'
                break
            if not complete_horizon and (not stop_success) and (ts is not None) and (controller is None) and (accepted_end >= max(ts, protocol.heat_off)):
                reason = 'success'
                break
            a = accepted_end
    except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        reason = 'numerical_failure'
        exception = repr(exc)
    else:
        exception = None
    if not times:
        append(0.0, y, protocol.j(0), protocol.u(0, cfg.cells))
    T, V, ice, si, lam, ph = ([], [], [], [], [], [])
    balances = []
    for state, j in zip(states, currents):
        s = model.phases(state)
        obs = model.observations(s, j)
        T.append(obs['T_mean'] - TM)
        V.append(obs['V'])
        ice.append(obs['ice_bulk'])
        si.append(obs['ice_saturation'])
        lam.append(np.average(s['lam'][:, model.g.pem], axis=1, weights=model.g.dx[model.g.pem]))
        balances.append(model.balances(state, y0))
    T, V, ice, si, lam, balances = map(np.array, (T, V, ice, si, lam, balances))
    y = states[-1].copy()
    ledger = y[model.il:model.ledger_end]
    last = times[-1]
    off = controller.off_time if controller is not None else min(protocol.heat_off, last) if protocol.heat_off <= last else None
    if protocol.kind == 'zero' and stop_success and (ts is not None):
        off = ts
    energies = ledger[2 * model.c:3 * model.c] * AREA
    safe_reason = reason in ('success', 'horizon')
    feasible = ts is not None and safe_reason and (off is not None)
    feasibility = 'feasible' if feasible else 'unresolved' if reason in ('horizon', 'numerical_failure') else 'infeasible'
    official = ts is not None
    result = dict(case_hash=cfg.digest, feasibility=feasibility, stop_reason=reason, success_time=ts, last_valid_time=last, th_startup=off, startup_result='success' if official else reason, shutdown_status='passed' if feasible else 'unresolved' if safe_reason else reason, E_aux_total_J=float(sum(energies)), E_cells_J=energies.tolist(), charge_C_cm2=float(ledger[-1]), wall_s=time.perf_counter() - wall, nfev=nfev, mass_residual=float(max(balances[:, 0])), energy_residual=float(max(balances[:, 1])), exception=exception, **extrema_to_success or extrema)
    if ts is not None:
        k = np.searchsorted(np.asarray(times), ts)
        k = min(k, len(states) - 1)
        result['E_aux_to_success_J'] = float(sum(states[k][model.il + 2 * model.c:model.il + 3 * model.c]) * AREA)
    result['shutdown_extrema'] = extrema
    result['nucleation_times_s'] = nucleation_times
    result['nucleation_scenario'] = cfg.nucleation
    return dict(metrics=result, config=cfg.to_dict(), t=np.array(times), y=np.array(states), j=np.array(currents), u=np.array(powers), T=T, V=V, ice=ice, ice_saturation=si, lambda_mean=lam, balances=balances, x=model.g.tx, layer=model.g.names, cell=model.g.owners)

# ==================== Input and provenance ====================
# Only the reporting rows supplied in the question; NEVER used to infer current.
OFFICIAL_TABLES = {
    -20.0: dict(t=[0,5,10,15,20,25,30,35],
                V=[.799,.801,.538,.521,.599,.626,.633,.666],
                T=[-20,-19.96,-19.65,-18.41,-17.08,-15.90,-14.27,-12.67]),
    -25.0: dict(t=[0,5,10,15,20,25,30,35],
                V=[.799,.800,.537,.518,.592,.618,.621,.644],
                T=[-25,-24.96,-24.65,-23.38,-22.03,-20.85,-19.20,-17.59])}
CANONICAL_COLUMNS = ('T0_C','t_s','I_A','V_V','T_C','j_A_cm2')
GRIDS = {
    'u14': ((3,2,3,3,3),0.0), 'u48': ((12,6,8,10,12),0.0),
    'u96': ((24,12,16,20,24),0.0), 'u192': ((48,24,32,40,48),0.0),
    'u384': ((96,48,64,80,96),0.0),
    'cgdl240': ((48,24,32,40,96),0.0),
    'cgdl336': ((48,24,32,40,192),0.0),
    'ccl232': ((48,24,32,80,48),0.0),
    'i96': ((24,12,16,20,24),4.0), 'i192': ((48,24,32,40,48),4.0),
    'i384': ((96,48,64,80,96),4.0), 'i768': ((192,96,128,160,192),4.0)}


def plain(value: Any) -> Any:
    """Strict JSON: non-finite diagnostic values become null, not invalid NaN."""
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload = json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)
    # Per-process temporary file plus atomic replacement: no partial JSON cache.
    tmp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        tmp.write_text(payload,encoding='utf-8'); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def apply_grid(cfg: Config, grid: str) -> Config:
    if grid in GRIDS:
        mesh, stretch = GRIDS[grid]
    else:
        try:
            mesh = tuple(int(v) for v in grid.split(',')); stretch = cfg.cgdl_stretch
        except ValueError as exc:
            raise ValueError('Grid must be a preset or five comma-separated integers') from exc
        if len(mesh) != 5: raise ValueError('Grid needs five layer counts')
    out = cfg.changed(mesh=tuple(mesh), cgdl_stretch=float(stretch))
    out.validate(); return out


def source_members(path: Path | list[Path]):
    """Read files without extracting archives or running workbook formulas."""
    if isinstance(path, (list, tuple)):
        for item in sorted(map(Path, path), key=lambda p: p.name):
            yield from source_members(item)
        return
    if not path.exists(): raise FileNotFoundError(f'Input not found: {path}')
    def visit(name, blob, depth=0):
        suffix = Path(name).suffix.lower()
        if suffix == '.zip':
            if depth > 4: raise ValueError('Archive nesting exceeds four levels')
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for item in sorted(z.infolist(), key=lambda x:x.filename):
                    if item.is_dir() or item.filename.startswith('__MACOSX'): continue
                    if item.file_size > 128*1024*1024: raise ValueError('Archive member exceeds 128 MiB')
                    if Path(item.filename).suffix.lower() in ('.zip','.xlsx','.csv'):
                        yield from visit(name+'!'+item.filename,z.read(item),depth+1)
        elif suffix in ('.csv','.xlsx'):
            yield name, blob
    if path.is_dir():
        names = sorted(p for p in path.rglob('*') if p.is_file() and
                       p.suffix.lower() in ('.csv','.xlsx','.zip'))
        for p in names: yield from visit(str(p.relative_to(path)), p.read_bytes())
    else:
        yield from visit(path.name,path.read_bytes())


def xlsx_cells(blob: bytes):
    """Minimal read-only OOXML reader for numeric tables and cached formulas.

    No formula evaluation, no workbook rewriting, no dependency on Excel.
    Returns each sheet as {row_number: {column_letters: cached_value}}, formulas.
    """
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    relns='{http://schemas.openxmlformats.org/package/2006/relationships}'
    rid='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            shared=[''.join(el.itertext()) for el in
                    ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        rels={el.get('Id'):el.get('Target') for el in
              ET.fromstring(z.read('xl/_rels/workbook.xml.rels')).findall(relns+'Relationship')}
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        for sheet in workbook.findall('m:sheets/m:sheet',ns):
            target=rels[sheet.get(rid)]
            target=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            rows={}; formulas={}
            root=ET.fromstring(z.read(target))
            for row in root.findall('m:sheetData/m:row',ns):
                rn=int(row.get('r')); entries={}
                for cell in row.findall('m:c',ns):
                    ref=cell.get('r'); col=re.match(r'[A-Z]+',ref).group()
                    typ=cell.get('t'); v=cell.find('m:v',ns); f=cell.find('m:f',ns)
                    if f is not None: formulas[ref]='='+(f.text or '')
                    if typ=='inlineStr':
                        val=''.join(t.text or '' for t in cell.findall('m:is//m:t',ns))
                    elif v is None or v.text is None: val=None
                    elif typ=='s': val=shared[int(v.text)]
                    elif typ in ('str','e'): val=v.text
                    else:
                        try: val=float(v.text)
                        except ValueError: val=v.text
                    entries[col]=val
                rows[rn]=entries
            yield sheet.get('name'), rows, formulas


def finite_number(x) -> bool:
    return isinstance(x,(int,float,np.number)) and bool(np.isfinite(x))


def normalized_experiment(df: pd.DataFrame, temp: float, origin: str) -> pd.DataFrame:
    d=df.copy()
    for k in ('t','I','V','T','j'):
        if k not in d: d[k]=np.nan
        d[k]=pd.to_numeric(d[k],errors='raise')
    if len(d)<2: raise ValueError(f'{origin}: need at least two data points')
    if not np.isfinite(d[['t','V','T']].to_numpy()).all():
        raise ValueError(f'{origin}: missing/non-finite time, voltage or temperature')
    if abs(float(d.t.iloc[0]))>1e-9 or np.any(np.diff(d.t)<=0):
        raise ValueError(f'{origin}: time must begin at zero and increase strictly')
    # Do not infer one current column from the other. A and B remain distinct.
    for k in ('I','j'):
        nonmissing=d[k].notna()
        if (d.loc[nonmissing,k]<0).any() or not np.isfinite(d.loc[nonmissing,k]).all():
            raise ValueError(f'{origin}: invalid {k} input')
    if not (d.I.notna().all() or d.j.notna().all()):
        raise ValueError(f'{origin}: neither current nor current-density series is complete')
    d['j_A']=d.I/(AREA*1e4)
    d['T0_C']=float(temp)
    d['source']=origin
    return d.reset_index(drop=True)


@dataclass
class Dataset:
    experiments: dict[float,pd.DataFrame]
    source_hash: str
    members: list[dict]
    parameters: list[dict]
    formulas: list[dict]


def load_dataset(path: str | Path | list[Path]) -> Dataset:
    path=[Path(p).resolve() for p in path] if isinstance(path,(list,tuple)) else Path(path).resolve()
    experiments={}; records=[]; params=[]; forms=[]
    def add(temp, frame, origin):
        temp=float(temp)
        if temp in experiments:
            raise ValueError(f'Duplicate condition {temp} C ({origin}); supply only one copy of each experiment')
        experiments[temp]=normalized_experiment(frame,temp,origin)
    for name,blob in source_members(path):
        records.append(dict(name=name,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
        if name.lower().endswith('.csv'):
            frame=pd.read_csv(io.BytesIO(blob),encoding='utf-8-sig')
            needed={'T0_C','t_s','V_V','T_C'}
            if not needed.issubset(frame.columns):
                raise ValueError(f'{name}: CSV needs T0_C,t_s,V_V,T_C and I_A and/or j_A_cm2')
            if frame.empty: raise ValueError(f'{name}: template is empty; experimental values must be supplied')
            frame=frame.rename(columns={'t_s':'t','I_A':'I','V_V':'V','T_C':'T','j_A_cm2':'j'})
            if frame.T0_C.isna().any(): raise ValueError(f'{name}: T0_C cannot be missing')
            for temp,part in frame.groupby('T0_C',sort=False): add(temp,part,name)
        else:
            for title,rows,formula in xlsx_cells(blob):
                valid=[rn for rn,c in rows.items() if all(finite_number(c.get(k)) for k in ('A','B','C','D'))]
                is_experiment=len(valid)>=2 and (('20' in title or '25' in title) or len(valid)>=20)
                if not is_experiment:
                    for rn,c in rows.items():
                        if c.get('B') is not None and c.get('C') is not None:
                            params.append(dict(workbook=name,sheet=title,row=rn,name=c.get('B'),
                                               value=c.get('C'),unit=c.get('D')))
                    continue
                parsed=[]
                for rn in sorted(rows):
                    if rn<min(valid): continue
                    c=rows[rn]
                    if all(c.get(k) is None for k in ('A','B','C','D','E')): continue
                    if not all(finite_number(c.get(k)) for k in ('A','B','C','D')):
                        raise ValueError(f'{name}!{title}:{rn}: incomplete experimental row; not silently skipped')
                    j=c.get('E')
                    if j is not None and not finite_number(j):
                        raise ValueError(f'{name}!{title}:E{rn}: nonnumeric cached density')
                    parsed.append(dict(t=c['A'],I=c['B'],V=c['C'],T=c['D'],j=j,source_row=rn))
                    if f'E{rn}' in formula:
                        forms.append(dict(workbook=name,sheet=title,row=rn,formula=formula[f'E{rn}'],
                                          cached_density=j))
                if '25' in title: temp=-25.0
                elif '20' in title: temp=-20.0
                else:
                    raise ValueError(f'{title}: ambiguous initial temperature; use canonical CSV with T0_C')
                add(temp,pd.DataFrame(parsed),f'{name}!{title}')
    if not experiments: raise ValueError('No experiment found; provide B题.zip, original XLSX or canonical CSV')
    h=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    return Dataset(experiments,h,records,params,forms)


def audit_dataset(data: Dataset, dest: Path) -> dict:
    rows=[]
    for temp,d in sorted(data.experiments.items()):
        mask=d.I.notna() & d.j.notna()
        ja=d.loc[mask,'j_A'].to_numpy(); jb=d.loc[mask,'j'].to_numpy()
        rel=np.abs(ja-jb)/np.maximum(np.abs(ja),1e-12)
        positive=mask & (d.j>0)
        implied=(d.loc[positive,'I']/d.loc[positive,'j']).to_numpy()
        rows.append(dict(T0_C=temp,n=len(d),first_t_s=float(d.t.iloc[0]),last_t_s=float(d.t.iloc[-1]),
             I_complete=bool(d.I.notna().all()),j_complete=bool(d.j.notna().all()),
             mismatched_rows_gt_1percent=int((rel>.01).sum()),
             maximum_relative_difference=float(rel.max()) if len(rel) else None,
             implied_area_median_cm2=float(np.median(implied)) if len(implied) else None,
             first_observed_temperature_C=float(d['T'].iloc[0])))
    ew=[p for p in data.parameters if p['row']==27 and finite_number(p['value'])
        and ('mol' in str(p['unit']).lower())]
    ew_check=[]
    for p in ew:
        unit=str(p['unit']).lower().replace(' ','')
        value=float(p['value'])*(1.0 if 'kg' in unit else .001)
        ew_check.append(dict(source=p,value_kg_mol=value,matches_code=abs(value-EW)<1e-12))
        if abs(value-EW)>1e-12:
            raise ValueError(f'Attachment EW={value} kg/mol disagrees with code EW={EW}; review inputs before fitting')
    report=dict(source_hash=data.source_hash,members=data.members,current_audit=rows,
        EW_code_kg_mol=EW,EW_checks=ew_check,area_code_cm2=AREA*1e4,
        current_convention_resolved=False,fixture_verified=False,
        interpretation='A=I/25; B=provided density. Numerical fit cannot determine the intended source definition.',
        physics_source='User-supplied q1(1).py; original constitutive values retained',
        missing_formula_cache=[f for f in data.formulas if f['cached_density'] is None])
    write_json(dest/'input_audit.json',report)
    pd.DataFrame(rows).to_csv(dest/'current_audit.csv',index=False,encoding='utf-8-sig')
    write_json(dest/'original_formulas.json',data.formulas)
    write_json(dest/'parameter_source.json',data.parameters)
    write_json(dest/'runtime.json',dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,code_hash=code_digest()))
    return report


# ==================== Runs and cache ====================
def run_experiment(cfg: Config, temp: float, data: Dataset, out: Path,
                   label: str='run', cache: bool=True) -> dict:
    if cfg.cells!=1: raise ValueError('This workflow is restricted to Question 1, one cell')
    d=data.experiments[float(temp)]
    current=d.j_A.to_numpy(float) if cfg.basis=='A' else d.j.to_numpy(float)
    if not np.isfinite(current).all():
        raise ValueError(f'{temp} C: basis {cfg.basis} input is incomplete. Do not reconstruct it silently.')
    actual=cfg.changed(T0=float(d['T'].iloc[0])+TM,ambient=float(temp)+TM,horizon=float(d.t.iloc[-1]),
                       charge_limit=1e6,current_limit=1e6,voltage_limit=0.0)
    protocol=Protocol('experiment',times=d.t.to_numpy(float),currents=current)
    signature=dict(config=actual.to_dict(),code_hash=code_digest(),source_hash=data.source_hash,
                   condition=float(temp),complete_horizon=True,backend='numba' if USE_NUMBA else 'reference',
                   runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    key=hashlib.sha256(json.dumps(plain(signature),sort_keys=True).encode()).hexdigest()[:24]
    folder=Path(out)/'runs'/key
    if cache and all((folder/n).exists() for n in ('metrics.json','config.json','trajectory.npz')):
        r=load_run(folder)
        if r['metrics'].get('code_hash')==code_digest() and r['metrics'].get('source_hash')==data.source_hash:
            return r
    # No hidden numerical retry: every convergence case uses exactly its registered tolerances.
    r=simulate(actual,protocol,stop_success=False,complete_horizon=True)
    m=r['metrics']
    finite=all(np.isfinite(r[k]).all() for k in ('t','y','T','V','ice','balances'))
    accepted=bool(finite and m['stop_reason']!='numerical_failure' and
                  m['mass_residual']<1e-3 and m['energy_residual']<1e-3)
    m.update(candidate_id=key,label=label,source_hash=data.source_hash,code_hash=code_digest(),
             numerically_accepted=accepted,full_period=bool(accepted and m['stop_reason']=='horizon'
             and r['t'][-1]>=d.t.iloc[-1]-1e-7))
    save_run(r,folder)
    return r


def save_run(run: dict, folder: Path) -> None:
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/f'trajectory.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp,**{k:v for k,v in run.items() if k not in ('config','metrics')})
    os.replace(tmp,folder/'trajectory.npz')
    write_json(folder/'config.json',run['config'])
    # Completion marker written LAST.
    write_json(folder/'metrics.json',run['metrics'])


def load_run(folder: str | Path) -> dict:
    folder=Path(folder)
    with np.load(folder/'trajectory.npz',allow_pickle=False) as archive:
        r={k:archive[k] for k in archive.files}
    r['config']=read_json(folder/'config.json'); r['metrics']=read_json(folder/'metrics.json')
    return r


def completed(r: dict, end: float) -> bool:
    return bool(len(r['t']) and np.isfinite(r['t']).all() and r['t'][-1]>=end-1e-7
        and r['metrics']['stop_reason']=='horizon' and r['metrics'].get('numerically_accepted',False)
        and all(np.isfinite(r[k]).all() for k in ('y','T','V')))


def safe_interp(t, original_t, values):
    """Linear alignment with missing values outside the actually computed interval."""
    return np.interp(t,original_t,values,left=np.nan,right=np.nan)


def waveform(result: dict, d: pd.DataFrame) -> dict:
    if not completed(result,float(d.t.iloc[-1])):
        return dict(complete=False,stop_reason=result['metrics']['stop_reason'],
                    last_time=float(result['t'][-1]))
    v=safe_interp(d.t,result['t'],result['V'][:,0]); temp=safe_interp(d.t,result['t'],result['T'][:,0])
    vi=int(np.argmin(v)); obs=d.loc[d.V==d.V.min(),'t']
    terr=max(float(obs.min()-d.t.iloc[vi]),float(d.t.iloc[vi]-obs.max()),0.0)
    depth=v[0]-v.min(); recovery=v[-1]-v.min()
    r=dict(complete=True,V_RMSE=float(np.sqrt(np.mean((v-d.V.to_numpy())**2))),
           T_RMSE=float(np.sqrt(np.mean((temp-d['T'].to_numpy())**2))),
           V_MAE=float(np.mean(abs(v-d.V.to_numpy()))),T_MAE=float(np.mean(abs(temp-d['T'].to_numpy()))),
           valley_time_s=float(d.t.iloc[vi]),valley_time_error_s=terr,V_min=float(v.min()),
           depth_error=float(abs(depth-(d.V.iloc[0]-d.V.min()))),
           recovery_error=float(abs(recovery-(d.V.iloc[-1]-d.V.min()))))
    r['prediction_pass']=bool(r['V_RMSE']<=.03 and r['T_RMSE']<=.3 and terr<=1
                             and r['depth_error']<=.03 and r['recovery_error']<=.03)
    r['gate_scope']='Inherited engineering targets, not competition-mandated acceptance criteria'
    return r
# ==================== Conservative grid diagnostics ====================
ICE_NUMERICAL_TARGETS = dict(voltage_V=.001,temperature_K=.02,
    phi_absolute=1e-5,phi_relative=.02,inventory_absolute_kg=1e-10,
    inventory_relative=.02,onset_threshold=1e-6,onset_time_s=.4,
    interface_halfwidth_m=20e-6)


def ice_observables(run: dict) -> dict:
    cfg=Config(**run['config']); g=Geometry(cfg)
    state=run['y'][:,:4*cfg.cells*g.n].reshape(-1,4,cfg.cells,g.n)
    ice=state[:,1]; phi=ice/920.0
    return dict(phi=phi,maximum=phi.max(axis=(1,2)),
                inventory_kg=AREA*np.sum(ice*g.dx,axis=(1,2)),dx=g.dx)


def threshold_onset(t,values,threshold):
    t=np.asarray(t); values=np.asarray(values)
    idx=np.flatnonzero(values>=threshold)
    if not len(idx): return None
    k=int(idx[0])
    if k==0: return float(t[0])
    frac=(threshold-values[k-1])/(values[k]-values[k-1])
    return float(t[k-1]+frac*(t[k]-t[k-1]))


def overlap_matrix(source_dx, target_dx) -> np.ndarray:
    """Map source FV averages to target FV averages by exact geometric overlaps."""
    a=np.asarray(source_dx,float); b=np.asarray(target_dx,float)
    if np.any(a<=0) or np.any(b<=0) or not np.isclose(a.sum(),b.sum(),rtol=1e-12,atol=1e-15):
        raise ValueError('Projection requires positive widths and the same physical domain')
    ea=np.r_[0,np.cumsum(a)]; eb=np.r_[0,np.cumsum(b)]
    overlap=np.maximum(0,np.minimum(eb[1:,None],ea[None,1:])-np.maximum(eb[:-1,None],ea[None,:-1]))
    return overlap/b[:,None]


def conservative_project(values,source_dx,target_dx) -> np.ndarray:
    """The last axis is space; this projection never replaces the raw peak output."""
    x=np.asarray(values)
    if x.shape[-1]!=len(source_dx): raise ValueError('Wrong source spatial dimension')
    return x @ overlap_matrix(source_dx,target_dx).T


def _aligned_field(run,values,t):
    return np.stack([np.column_stack([safe_interp(t,run['t'],values[:,c,k])
        for k in range(values.shape[2])]) for c in range(values.shape[1])],axis=1)


def compare_trajectories(a: dict,b: dict,expected_end: float,targets=None) -> dict:
    limits=dict(ICE_NUMERICAL_TARGETS if targets is None else targets)
    start=max(float(a['t'][0]),float(b['t'][0])); end=min(float(a['t'][-1]),float(b['t'][-1]))
    tt=np.unique(np.r_[a['t'],b['t']]); tt=tt[(tt>=start)&(tt<=end)]
    if not len(tt): return dict(complete=False,output_converged=False,ice_converged=False,reason='no common samples')
    ia,ib=ice_observables(a),ice_observables(b)
    pa=safe_interp(tt,a['t'],ia['maximum']); pb=safe_interp(tt,b['t'],ib['maximum'])
    ma=safe_interp(tt,a['t'],ia['inventory_kg']); mb=safe_interp(tt,b['t'],ib['inventory_kg'])
    pscale=float(max(abs(pa).max(),abs(pb).max())); mscale=float(max(abs(ma).max(),abs(mb).max()))
    dp=float(abs(pa-pb).max()); dm=float(abs(ma-mb).max())
    fa=_aligned_field(a,ia['phi'],tt); fb=_aligned_field(b,ib['phi'],tt)
    ea=np.r_[0,np.cumsum(ia['dx'])]; eb=np.r_[0,np.cumsum(ib['dx'])]
    interface=sum(THICKNESS[:4]); half=limits['interface_halfwidth_m']
    lo=max(0,interface-half); hi=min(sum(THICKNESS),interface+half)
    edges=np.unique(np.r_[ea,eb,lo,hi]); mids=(edges[:-1]+edges[1:])/2
    ka=np.clip(np.searchsorted(ea,mids,side='right')-1,0,len(ia['dx'])-1)
    kb=np.clip(np.searchsorted(eb,mids,side='right')-1,0,len(ib['dx'])-1)
    diff=np.abs(fa[:,:,ka]-fb[:,:,kb]); widths=np.diff(edges)
    l1=float(np.max(np.sum(diff*widths,axis=2)/sum(THICKNESS)))
    near=(mids>=lo)&(mids<=hi)
    local_l1=float(np.max(np.sum(diff[:,:,near]*widths[near],axis=2)/(hi-lo)))
    local_linf=float(diff[:,:,near].max())
    projected=conservative_project(fb,ib['dx'],ia['dx'])
    pdiff=abs(fa-projected)
    projection_error=float(np.max(pdiff))
    projection_inventory_error=float(np.max(abs(np.sum(projected*ia['dx'],axis=2)-np.sum(fb*ib['dx'],axis=2))))
    def output_diff(key):
        return max(float(np.max(abs(safe_interp(tt,a['t'],a[key][:,c])-safe_interp(tt,b['t'],b[key][:,c]))))
                   for c in range(a[key].shape[1]))
    dv,dt=output_diff('V'),output_diff('T')
    oa=threshold_onset(tt,pa,limits['onset_threshold']); ob=threshold_onset(tt,pb,limits['onset_threshold'])
    od=None if oa is None or ob is None else abs(oa-ob)
    onset_ok=(oa is None and ob is None) or (od is not None and od<=limits['onset_time_s'])
    phi_tol=limits['phi_absolute']+limits['phi_relative']*pscale
    mass_tol=limits['inventory_absolute_kg']+limits['inventory_relative']*mscale
    is_complete=completed(a,expected_end) and completed(b,expected_end)
    checks=dict(voltage=dv<=limits['voltage_V'],temperature=dt<=limits['temperature_K'],
                phi_max=dp<=phi_tol,ice_inventory=dm<=mass_tol,spatial_mean=l1<=phi_tol,
                interface_mean=local_l1<=phi_tol,onset=onset_ok)
    nuc_a=a['metrics'].get('nucleation_times_s',[None])
    nuc_b=b['metrics'].get('nucleation_times_s',[None])
    nuc_ok=all((x is None and y is None) or (x is not None and y is not None and abs(x-y)<=limits['onset_time_s'])
               for x,y in zip(nuc_a,nuc_b))
    checks['nucleation']=nuc_ok
    ice_ok=nuc_ok and all(checks[k] for k in ('phi_max','ice_inventory','spatial_mean','interface_mean','onset'))
    return dict(complete=is_complete,common_end_s=end,next_grid_V_diff=dv,next_grid_T_diff=dt,
        phi_max_curve_absolute_diff=dp,phi_max_curve_relative_diff=dp/pscale if pscale else None,
        peak_a=float(pa.max()),peak_b=float(pb.max()),
        signed_peak_change_relative_to_a=(float(pb.max()-pa.max())/abs(float(pa.max())) if pa.max()!=0 else None),
        inventory_curve_absolute_diff_kg=dm,inventory_curve_relative_diff=dm/mscale if mscale else None,
        spatial_phi_mean_L1_diff=l1,interface_phi_mean_L1_diff=local_l1,interface_phi_Linf_diff=local_linf,
        conservative_projection_Linf_diff=projection_error,
        conservative_projection_volume_residual_m=projection_inventory_error,
        nucleation_a_s=nuc_a,nucleation_b_s=nuc_b,nucleation_converged=nuc_ok,
        onset_a_s=oa,onset_b_s=ob,onset_time_diff_s=od,onset_converged=bool(onset_ok),
        ice_converged=bool(is_complete and ice_ok),output_converged=bool(is_complete and all(checks.values())),
        individual_checks={k:bool(v) for k,v in checks.items()},targets=limits,
        interpretation='Raw FV maxima are primary. Projection is a diagnostic, never a substitute for the maximum.')


def peak_location(run: dict) -> dict:
    g=Geometry(Config(**run['config'])); ice=ice_observables(run)
    k,c,i=np.unravel_index(np.argmax(ice['phi']),ice['phi'].shape)
    return dict(phi=float(ice['phi'][k,c,i]),time_s=float(run['t'][k]),
                cell=int(c),layer=LAYER_NAMES[g.layer[i]],x_um=float(g.x[i]*1e6),
                width_um=float(g.dx[i]*1e6),n_MEA=g.n,inventory_max_kg=float(ice['inventory_kg'].max()))


def convergence(cfg: Config,data: Dataset,out: Path,grids: list[str],temps=None,
                time_grid: str | None=None) -> dict:
    """Same physical parameters on all grids; no refitting inside this routine."""
    if len(grids)<3: raise ValueError('At least three spatial grids are required')
    temps=sorted(data.experiments) if temps is None else list(temps)
    out=Path(out); records=[]; pairs=[]; final_axes={}
    base=cfg.changed(board_n=4,end_n=8,max_step=.1,rtol=5e-7,atol_scale=1.0)
    def execute(condition,axis,specs):
        previous=None
        for name,setting in specs:
            r=run_experiment(setting,condition,data,out,label=f'{axis}_{name}')
            entry=dict(temp_C=condition,axis=axis,grid=name,config=setting.to_dict(),
                run=r['metrics']['candidate_id'],peak=peak_location(r),**r['metrics'])
            records.append(entry)
            if previous is not None:
                old_name,old=previous
                comparison=compare_trajectories(old,r,float(data.experiments[condition].t.iloc[-1]))
                pair=dict(temp_C=condition,axis=axis,grid_a=old_name,grid_b=name,
                          run_a=old['metrics']['candidate_id'],run_b=r['metrics']['candidate_id'],**comparison)
                pairs.append(pair); final_axes[(condition,axis)]=pair
            previous=name,r
            write_json(out/'convergence_progress.json',dict(records=records,pairs=pairs))
            print(f'{condition:g} C | {axis} | {name} | {r["metrics"]["stop_reason"]}',flush=True)
    for temp in temps:
        execute(temp,'space',[(name,apply_grid(base,name)) for name in grids])
        fine=apply_grid(base,time_grid or grids[-1])
        execute(temp,'time',[(f'dt{dt:g}',fine.changed(max_step=dt,rtol=rt,atol_scale=at))
                            for dt,rt,at in ((.2,1e-6,2),(.1,5e-7,1),(.05,2.5e-7,.5))])
        execute(temp,'thermal_space',[(f'board{n}',fine.changed(board_n=n,end_n=2*n)) for n in (2,4,8)])
        execute(temp,'sampling',[(f'sample{sample:g}',fine.changed(max_step=.05,rtol=2.5e-7,
                      atol_scale=.5,sample=sample)) for sample in (.2,.1)])
        # Verify the exact registered/exported configuration too. It must not
        # inherit a pass belonging only to some unrelated finer configuration.
        reference=apply_grid(base,grids[-1]).changed(board_n=8,end_n=16,
                    max_step=.05,rtol=2.5e-7,atol_scale=.5,sample=min(.1,cfg.sample))
        execute(temp,'registered_vs_reference',[('registered',cfg),('finest_reference',reference)])
    all_ok=all(x['output_converged'] for x in final_axes.values())
    result=dict(config=cfg.to_dict(),parameter_policy='fixed; no fit during comparisons',
         source_hash=data.source_hash,code_hash=code_digest(),grids=grids,
         time_check_grid=time_grid or grids[-1],records=records,pairs=pairs,
         last_pairs=list(final_axes.values()),numerical_checks_passed=bool(all_ok),
         independent_ice_validation=False,
         qualification='Finite grid/solver checks only; not experimental validation or an error bound')
    write_json(out/'convergence.json',result)
    pd.DataFrame([{k:v for k,v in p.items() if not isinstance(v,(dict,list))} for p in pairs]).to_csv(
        out/'convergence_pairs.csv',index=False,encoding='utf-8-sig')
    return result


def interface_refinement(cfg: Config,data: Dataset,out: Path) -> dict:
    """cGDL-only and cCL-only diagnostics, distinct from a whole-domain convergence test."""
    specs=['u96','u192','cgdl240','cgdl336','ccl232','i192']
    base=cfg.changed(board_n=4,end_n=8,max_step=.1,rtol=5e-7,atol_scale=1)
    runs={}; records=[]; comparisons=[]
    for temp in sorted(data.experiments):
        for grid in specs:
            setting=apply_grid(base,grid); r=run_experiment(setting,temp,data,out,label='interface')
            runs[grid]=r
            records.append(dict(temp_C=temp,grid=grid,run=r['metrics']['candidate_id'],peak=peak_location(r),
                                **waveform(r,data.experiments[temp])))
            print(f'interface {temp:g} C {grid}: {r["metrics"]["stop_reason"]}',flush=True)
        for a,b in [('u96','u192'),('u192','cgdl240'),('cgdl240','cgdl336'),
                    ('u192','ccl232'),('u192','i192')]:
            comparisons.append(dict(temp_C=temp,grid_a=a,grid_b=b,
                **compare_trajectories(runs[a],runs[b],float(data.experiments[temp].t.iloc[-1]))))
    result=dict(code_hash=code_digest(),source_hash=data.source_hash,records=records,pairs=comparisons,
                whole_domain_convergence_claim=False,parameters_refitted=False)
    write_json(out/'interface_refinement.json',result)
    pd.DataFrame([{k:v for k,v in p.items() if not isinstance(v,(dict,list))} for p in comparisons]).to_csv(
        out/'interface_refinement.csv',index=False,encoding='utf-8-sig')
    return result


def hydraulic_benchmark(out: Path,horizon: float=.2) -> dict:
    """Isothermal, no-freezing two-layer diagnostic using the original liquid flux.

    Closed boundaries; prescribed initial water in cCL, dry cGDL. This is a
    manufactured numerical experiment, NOT an experimental cold-start dataset.
    """
    equilibrium=[]; dynamics=[]
    for temp in (-20.,-25.):
        t=np.full(2,temp+TM); eps=np.array([POROSITY[3],POROSITY[4]])
        k=np.array([6.2e-13,6.2e-12]); angle=np.array([100.,110.]); s_left=.02
        pressure=hydraulic_pressure(990*eps*np.array([s_left,0]),np.zeros(2),t,eps,k,angle)[0]
        def fun(s):
            return hydraulic_pressure(990*eps*np.array([s_left,s]),np.zeros(2),t,eps,k,angle)[1]-pressure
        s_right=brentq(fun,0,1,xtol=1e-15)
        f=liquid_flux(990*eps*np.array([s_left,s_right]),np.zeros(2),t,eps,k,angle,
                      np.array([THICKNESS[3],THICKNESS[4]]))
        equilibrium.append(dict(temp_C=temp,s_left=s_left,s_right=s_right,pressure_Pa=float(pressure),
                                 interface_flux_kg_m2_s=float(f[1]),passed=bool(abs(f[1])<1e-12)))
    for ncl,ngdl in ((10,24),(20,48),(40,96)):
        dx=np.r_[np.full(ncl,THICKNESS[3]/ncl),np.full(ngdl,THICKNESS[4]/ngdl)]
        eps=np.r_[np.full(ncl,POROSITY[3]),np.full(ngdl,POROSITY[4])]
        k=np.r_[np.full(ncl,6.2e-13),np.full(ngdl,6.2e-12)]
        angle=np.r_[np.full(ncl,100.),np.full(ngdl,110.)]; temp=np.full(len(dx),253.15)
        initial=np.r_[990*eps[:ncl]*.02,np.zeros(ngdl)]
        def rhs(t,y):
            f=liquid_flux(y,np.zeros_like(y),temp,eps,k,angle,dx)
            return (f[:-1]-f[1:])/dx
        pattern=lil_matrix((len(dx),len(dx)),dtype=int)
        for i in range(len(dx)): pattern[i,max(0,i-1):min(len(dx),i+2)]=1
        sol=solve_ivp(rhs,(0,horizon),initial,method='BDF',rtol=1e-7,atol=1e-9,
                      max_step=min(.01,horizon/5),jac_sparsity=pattern.tocsr())
        residual=float(np.max(abs(sol.y.T@dx-initial@dx))/max(initial@dx,1e-30))
        dynamics.append(dict(n_cCL=ncl,n_cGDL=ngdl,success=bool(sol.success),
              mass_relative_residual=residual,minimum_liquid=float(sol.y.min()),
              final_interface_flux=float(liquid_flux(sol.y[:,-1],np.zeros(len(dx)),temp,eps,k,angle,dx)[ncl]),
              final_liquid=sol.y[:,-1].tolist(),dx=dx.tolist()))
    report=dict(scope='Manufactured isothermal liquid-only benchmark; no freeze, sorption, vapor or reaction',
                equilibrium=equilibrium,dynamics=dynamics,
                passed=all(e['passed'] for e in equilibrium) and all(d['success'] and
                       d['mass_relative_residual']<1e-6 and d['minimum_liquid']>-1e-5 for d in dynamics),
                physical_validation=False)
    write_json(out/'hydraulic_benchmark.json',report)
    return report
# ==================== Calibration and parameter diagnostics ====================




def parallel_map(function,jobs,workers):
    if workers<=1: return [function(job) for job in jobs]
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        return list(pool.map(function,jobs))




def registry_config(out: Path,data: Dataset | None=None) -> Config:
    record=read_json(Path(out)/'fit_registry.json')
    if record.get('code_hash')!=code_digest(): raise ValueError('Stale code hash in registry; explicitly refit or import as an initial guess')
    if data is not None and record.get('source_hash')!=data.source_hash:
        raise ValueError('Registry belongs to different experimental input; do not mix data and parameters')
    cfg=Config(**record['config'])
    if cfg.proton_face_scheme!='exact': raise ValueError('Legacy proton-face parameters cannot be used as the registered model')
    cfg.validate(); return cfg


def _scenario_worker(job):
    name,cfg,temp,data,out=job
    r=run_experiment(cfg,temp,data,Path(out),label='sensitivity')
    return dict(case=name,temp_C=temp,run=r['metrics']['candidate_id'],peak=peak_location(r),
                **waveform(r,data.experiments[temp]))


def sensitivity(cfg: Config,data: Dataset,out: Path,workers=1) -> list[dict]:
    choices=[('baseline',cfg)]
    for key in ('j0','tau_b','cathode_hydration_exponent','cl_conductivity_factor','tau_f'):
        for factor in (.9,1.1): choices.append((f'{key}_{factor:g}x',cfg.changed(**{key:getattr(cfg,key)*factor})))
    for factor in (.1,10.): choices.append((f'tau_f_{factor:g}x',cfg.changed(tau_f=cfg.tau_f*factor)))
    choices += [('alternate_current_basis',cfg.changed(basis='A' if cfg.basis=='B' else 'B')),
                ('alternate_fixture',cfg.changed(fixture='H2' if cfg.fixture=='H1' else 'H1')),
                ('alternate_temperature_observation',cfg.changed(observation='unit' if cfg.observation=='MEA' else 'MEA'))]
    jobs=[]; skipped=[]
    for name,setting in choices:
        for temp,d in data.experiments.items():
            needed=d.I if setting.basis=='A' else d.j
            if not needed.notna().all():
                skipped.append(dict(case=name,temp_C=temp,reason='Required current column absent')); continue
            jobs.append((name,setting,temp,data,str(out)))
    rows=parallel_map(_scenario_worker,jobs,workers)
    write_json(out/'sensitivity.json',dict(rows=rows,skipped=skipped,parameter_policy='Fixed except named perturbation'))
    return rows


def _response_worker(job):
    cfg,data,out=job; result=[]; ids=[]
    for temp,d in sorted(data.experiments.items()):
        r=run_experiment(cfg,temp,data,Path(out),label='identifiability'); ids.append(r['metrics']['candidate_id'])
        if not completed(r,float(d.t.iloc[-1])): return None,ids
        result.extend(safe_interp(d.t,r['t'],r['V'][:,0])/.03)
        result.extend(safe_interp(d.t,r['t'],r['T'][:,0])/.3)
    return np.array(result),ids


def identifiability(cfg: Config,data: Dataset,out: Path,workers=1) -> dict:
    names=('j0','tau_b','cathode_hydration_exponent','cl_conductivity_factor'); steps=(.03,.015)
    choices=[cfg]+[cfg.changed(**{k:getattr(cfg,k)*np.exp(h)}) for h in steps for k in names]
    results=parallel_map(_response_worker,[(c,data,str(out)) for c in choices],workers)
    if any(r[0] is None for r in results):
        report=dict(status='unresolved',reason='At least one perturbed trajectory is incomplete')
    else:
        matrices=[np.column_stack([(results[1+s*len(names)+i][0]-results[0][0])/h for i in range(len(names))])
                  for s,h in enumerate(steps)]
        change=np.linalg.norm(matrices[0]-matrices[1],axis=0)/np.maximum(np.linalg.norm(matrices[1],axis=0),1e-12)
        singular=np.linalg.svd(matrices[-1],compute_uv=False)
        norm=np.linalg.norm(matrices[-1],axis=0)
        report=dict(status='computed',parameters=names,log_steps=steps,singular_values=singular,
              condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
              numerical_rank=int(np.linalg.matrix_rank(matrices[-1])),
              sensitivity_column_cosines=matrices[-1].T@matrices[-1]/np.maximum(norm[:,None]*norm[None,:],1e-20),
              derivative_relative_change=change,derivative_step_stable=(change<.2))
    report.update(config=cfg.to_dict(),code_hash=code_digest(),source_hash=data.source_hash,
        runs=[r[1] for r in results],interpretation='Local finite-difference sensitivity only; no confidence-interval or global-identifiability claim')
    write_json(out/'identifiability.json',report)
    return report


# ==================== Results, mechanisms and portable plotting ====================
def export_run(run: dict,temp: float,data: Dataset,out: Path,role='conditional_prediction') -> dict:
    dest=out/'results'/f'{temp:g}C'; dest.mkdir(parents=True,exist_ok=True)
    cfg=Config(**run['config']); model=Model(cfg); g=model.g; d=data.experiments[temp]
    allpoints=d.copy()
    for name,key in [('model_V','V'),('model_T_C','T'),('model_phi_max','ice')]:
        allpoints[name]=safe_interp(d.t,run['t'],run[key][:,0])
    allpoints['V_relative_error_percent']=100*abs(allpoints.model_V-d.V)/abs(d.V).replace(0,np.nan)
    allpoints['T_relative_error_percent']=100*abs(allpoints.model_T_C-d['T'])/abs(d['T']).replace(0,np.nan)
    allpoints['run']=run['metrics']['candidate_id']; allpoints['role']=role
    allpoints.to_csv(dest/'all_observations.csv',index=False,encoding='utf-8-sig')
    if temp in OFFICIAL_TABLES:
        table=pd.DataFrame(OFFICIAL_TABLES[temp]).rename(columns={'t':'time_s','V':'experiment_V','T':'experiment_T_C'})
        for name,key in [('model_V','V'),('model_T_C','T'),('max_ice_bulk','ice')]:
            table[name]=safe_interp(table.time_s,run['t'],run[key][:,0])
        table['V_relative_error_percent']=100*abs(table.model_V-table.experiment_V)/abs(table.experiment_V)
        table['T_relative_error_percent']=100*abs(table.model_T_C-table.experiment_T_C)/abs(table.experiment_T_C).replace(0,np.nan)
        table['workbook_V_at_time']=safe_interp(table.time_s,d.t,d.V)
        table['workbook_T_at_time']=safe_interp(table.time_s,d.t,d['T'])
        table['run']=run['metrics']['candidate_id']; table['role']=role
        table=table[['time_s','experiment_V','model_V','V_relative_error_percent',
             'experiment_T_C','model_T_C','T_relative_error_percent','max_ice_bulk',
             'workbook_V_at_time','workbook_T_at_time','run','role']]
        table.to_csv(dest/'question_table.csv',index=False,encoding='utf-8-sig')
    rows=[]; interface=[]; snapshots=[]
    initial=model.phases(run['y'][0]); initial_mass=AREA*sum(np.sum(initial[k]*g.dx) for k in ('mv','ml','mi','mb'))
    face=int(g.cathode[g.layer[g.cathode]==4][0])
    for ti,(t,y,j) in enumerate(zip(run['t'],run['y'],run['j'])):
        s=model.phases(y)
        if cfg.nucleation == 'median':
            onset = run['metrics'].get('nucleation_times_s', [None])[0]
            model.nucleated[0] = onset is not None and t >= onset - 1e-9
        extra=model.rhs(t,y,j,np.zeros(1),details=True)
        inventory={key:float(AREA*np.sum(s[key]*g.dx)) for key in ('mv','ml','mi','mb')}
        generated=float(s['led'][-1]*1e4*MW/(2*F)*AREA)
        freeze=s['ml']*np.maximum(TM-s['local_T'],0)/(20*cfg.tau_f)*model.nucleated[:,None]
        cls=np.r_[g.cla,g.clc]
        bound_freeze=cfg.bound_ice_rate*g.bcap[cls]*np.maximum(s['lam'][:,cls]-lambda_eq(
            psat(s['local_T'][:,cls])/psat(s['local_T'][:,cls],liquid=True)),0)*(s['local_T'][:,cls]<TM)*model.nucleated[:,None]
        melt=np.maximum(s['mi'],0)*np.maximum(s['local_T']-TM,0)/(5*cfg.tau_m)
        rows.append(dict(time_s=float(t),**inventory,generated_kg=generated,
                net_outward_kg=generated-(sum(inventory.values())-initial_mass),
                bound_freeze_kg_s=float(AREA*np.sum(bound_freeze*g.dx[cls])),
                freeze_kg_s=float(AREA*np.sum(freeze*g.dx)),melt_kg_s=float(AREA*np.sum(melt*g.dx)),
                phi_max=float(np.max(s['mi']/920)),
                T_min_K=float(s['local_T'].min()),T_max_K=float(s['local_T'].max()),
                water_residual=float(run['balances'][ti,0]),energy_residual=float(run['balances'][ti,1])))
        left,right=face-1,face
        ids=np.array([left,right])
        pressure=hydraulic_pressure(s['ml'][0,ids],s['mi'][0,ids],s['local_T'][0,ids],
                                    g.eps[ids],g.K[ids],g.angle[ids])
        interface.append(dict(time_s=float(t),liquid_flux=float(extra['fluxes']['liquid'][0,face]),
            vapor_flux=float(extra['fluxes']['vapor'][0,face]),p_left_Pa=float(pressure[0]),p_right_Pa=float(pressure[1]),
            ml_left=float(s['ml'][0,left]),ml_right=float(s['ml'][0,right]),
            phi_left=float(s['mi'][0,left]/920),phi_right=float(s['mi'][0,right]/920),
            width_left_um=float(g.dx[left]*1e6),width_right_um=float(g.dx[right]*1e6)))
        if ti in (0,len(run['t'])//2,len(run['t'])-1):
            for k in range(g.n):
                snapshots.append(dict(time_s=float(t),layer=LAYER_NAMES[g.layer[k]],x_um=g.x[k]*1e6,dx_um=g.dx[k]*1e6,
                    T_K=s['local_T'][0,k],mv=s['mv'][0,k],ml=s['ml'][0,k],mi=s['mi'][0,k],mb=s['mb'][0,k],
                    phi=s['mi'][0,k]/920,epsilon_g=s['eg'][0,k]))
    pd.DataFrame(rows).to_csv(dest/'water_budget.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(interface).to_csv(dest/'cCL_cGDL_interface.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(snapshots).to_csv(dest/'spatial_snapshots.csv',index=False,encoding='utf-8-sig')
    summary=dict(temp_C=temp,run=run['metrics']['candidate_id'],config=run['config'],metrics=run['metrics'],
                 peak=peak_location(run),prediction=waveform(run,d),role=role,independent_ice_validation=False)
    write_json(dest/'summary.json',summary)
    return summary


def export_results(cfg: Config,data: Dataset,out: Path,role='conditional_prediction',plots=False) -> list[dict]:
    results=[]
    for temp in sorted(data.experiments):
        r=run_experiment(cfg,temp,data,out,label='export')
        results.append(export_run(r,temp,data,out,role))
        if plots: plot_run(r,temp,data.experiments[temp],out/'results'/f'{temp:g}C')
    write_json(out/'result_summary.json',results)
    text=['# 第一问计算结果','','参数、离散网格和数据来源见同目录 JSON。冰量为模型推断，没有直接实测冰量验证。',
          '相对误差按题目表格的摄氏温度数值计算；分母为零时留空，并另报温度 RMSE。',
          'question_table.csv 保留题目指定实验值；另外列出工作簿在该时刻的插值值，以暴露二者差异。',
          '不对提前终止后的时段进行电压、温度或冰分数外推。','','## 配置',
          '```json',json.dumps(cfg.to_dict(),ensure_ascii=False,indent=2),'```','','## 结果']
    for s in results:
        p=s['prediction']; text += ['',f"### {s['temp_C']:g} ℃",'',
             f"完整轨迹：{p['complete']}；终止原因：{s['metrics']['stop_reason']}。",
             f"模型峰值冰分数：{s['peak']['phi']:.8g}；位置：{s['peak']['layer']}。"]
        if p['complete']:
            text.append(f"电压 RMSE：{p['V_RMSE']:.6g} V；温度 RMSE：{p['T_RMSE']:.6g} ℃。")
    text += ['','## 解释边界','','电流 A/B 口径、H1/H2 热域及温度观测定义需由实验说明确定。',
             '本程序采用单温度两参数标定、干气外边界及显式成核和相变情景，不宣称这些假设已获实验验证。',
             '测试通过、优化器收敛、网格收敛和物理预测能力是四个不同结论。']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return results


def plot_run(run,temp,d,dest):
    # Optional dependency; no font lookup occurs when importing the model or testing it.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    chinese=next((name for name in ('Microsoft YaHei','Noto Sans CJK SC','Source Han Sans SC','SimHei') if name in fonts),None)
    labels={'V':('电压 / V','Voltage / V'),'T':('平均温度 / ℃','Mean temperature / C'),
            'ice':('最大冰体积分数','Maximum ice volume fraction')}
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':[chinese] if chinese else ['DejaVu Sans'],
                         'axes.unicode_minus':False,'svg.fonttype':'none'}):
        for key in ('V','T','ice'):
            fig,ax=plt.subplots(figsize=(6.4,4.1))
            ax.plot(run['t'],run[key][:,0],label='模型' if chinese else 'Model')
            if key in ('V','T'):
                ax.plot(d.t,d[key],linestyle='--',label='实验' if chinese else 'Experiment')
            ax.set(xlabel='时间 / s' if chinese else 'Time / s',ylabel=labels[key][0 if chinese else 1],title=f'{temp:g} C')
            ax.legend(); fig.tight_layout()
            fig.savefig(dest/f'{key}.png',dpi=600); fig.savefig(dest/f'{key}.svg'); plt.close(fig)
# ==================== Self-contained command line ====================
def read_config_initial(path: str | Path) -> Config:
    record=read_json(path); values=record.get('config',record)
    unknown=set(values)-{f.name for f in fields(Config)}
    if unknown: raise ValueError(f'Unsupported configuration fields: {sorted(unknown)}')
    values=dict(values)
    if 'mesh' in values: values['mesh']=tuple(values['mesh'])
    cfg=Config(**values); cfg.validate(); return cfg




def smoke_worker(spec):
    grid,horizon=spec
    cfg=apply_grid(Config(j0=.3,horizon=horizon,sample=min(.2,horizon),max_step=.05,
                         voltage_limit=0,rtol=1e-6),grid)
    run=simulate(cfg,Protocol('constant',(.02,)),stop_success=False,complete_horizon=True)
    if run['metrics']['stop_reason']!='horizon' or run['t'][-1]<horizon-1e-8:
        raise RuntimeError(f'Smoke integration failed: {run["metrics"]}')
    if run['metrics']['mass_residual']>=1e-3 or run['metrics']['energy_residual']>=1e-3:
        raise RuntimeError('Smoke integration did not close the ledgers')
    return dict(grid=grid,horizon=horizon,metrics=run['metrics'],
                scope='Manufactured constant-current runtime check; not an experimental prediction')


def self_check(out: Path) -> dict:
    """Actual test collection, execution and count; never a hard-coded pass count."""
    out.mkdir(parents=True,exist_ok=True)
    report=out/'tests.xml'
    env=dict(os.environ); env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; env['PYTHONUTF8']='1'
    # pytest imports by a synthetic package name; its Numba cache must not
    # replace caches later loaded by the standalone/importable module.
    env['NUMBA_CACHE_DIR']=str(out/'numba_test_cache')
    result=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve()),'-q',
        '-o','python_functions=test_*','--import-mode=importlib','-p','no:cacheprovider',
        '--junitxml',str(report)],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
    log=result.stdout+'\n'+result.stderr
    (out/'tests.log').write_text(log,encoding='utf-8'); print(log)
    info=dict(returncode=result.returncode,code_hash=code_digest(),python=sys.version,
              platform=platform.platform(),numpy=np.__version__)
    if report.exists():
        root=ET.parse(report).getroot(); suites=list(root.iter('testsuite'))
        for key in ('tests','failures','errors','skipped'):
            info[key]=sum(int(s.get(key,0)) for s in suites)
        info['passed']=info['tests']-info['failures']-info['errors']-info['skipped']
    write_json(out/'tests.json',info)
    if result.returncode: raise RuntimeError(f'Tests failed; see {out / "tests.log"}')
    return info







def default_input_files() -> Path | list[Path]:
    """Find only original inputs, never recursively read old result workbooks."""
    script_dir = Path(__file__).resolve().parent
    for folder in dict.fromkeys((script_dir, Path.cwd())):
        if (folder / '附件2.xlsx').is_file():
            return [p for p in (folder/'附件1.xlsx', folder/'附件2.xlsx') if p.is_file()]
        if (folder/'inputs'/'附件2.xlsx').is_file():
            return [p for p in (folder/'inputs'/'附件1.xlsx', folder/'inputs'/'附件2.xlsx') if p.is_file()]
        if (folder/'B题.zip').is_file():
            return folder/'B题.zip'
    project_inputs=script_dir.parents[1]/'中文题目/B题/氢燃料电池低温冷启动建模与控制策略研究  附件'
    if (project_inputs/'附件2.xlsx').is_file():
        return [p for p in (project_inputs/'附件1.xlsx',project_inputs/'附件2.xlsx') if p.is_file()]
    raise FileNotFoundError('请把附件1.xlsx、附件2.xlsx放在q1.py旁，或用 --input 指定仅含原始输入的目录。')



CURRENT_CANDIDATE_NAME = 'merged_median_two_parameter_holdout'


def current_config() -> Config:
    """Data-independent starting values, not the former two-temperature optimum."""
    return apply_grid(Config(
        j0=float(np.exp(-2.5)*(14/3)**3), cathode_hydration_exponent=3.,
        tau_b=1., tau_f=5., tau_m=1., cl_proton_loss=True,
        cl_conductivity_factor=1., vapor_equilibrium='supercooled_liquid',
        bound_flux_scheme='exponential', bp_heat_capacity_factor=1.,
        nucleation='median', bound_ice_rate=.01, board_n=4, end_n=8,
        rtol=5e-7, max_step=.1, sample=.05), 'i192')


def full_metrics(r, d):
    m = waveform(r, d)
    m.update(mass_residual=r['metrics']['mass_residual'],
             energy_residual=r['metrics']['energy_residual'],
             nucleation_time_s=r['metrics'].get('nucleation_times_s', [None])[0],
             ice_peak=float(r['ice'].max()))
    return m


def fit_merged(base: Config, data: Dataset, out: Path, budget=60):
    """Only -20 C enters the objective. No held-out outcome selects parameters."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if -20. not in data.experiments:
        raise ValueError('The merged workflow requires the -20 C calibration condition')
    if budget < 1:
        raise ValueError('budget must be positive')
    base.validate()
    d = data.experiments[-20.]
    x0 = np.array([np.log(base.j0)-base.cathode_hydration_exponent*np.log(14/3),
                   base.cathode_hydration_exponent])
    bounds = (np.array([-6., 0.]), np.array([3., 6.]))
    records, memo = [], {}
    def unpack(x):
        return base.changed(j0=float(np.exp(x[0])*(14/3)**x[1]),
                            cathode_hydration_exponent=float(x[1]))
    def residual(x):
        key = tuple(x)
        if key in memo:
            return memo[key]
        run = run_experiment(unpack(x), -20., data, out/'cache', label='calibration_only')
        if completed(run, float(d.t.iloc[-1])):
            v = safe_interp(d.t, run['t'], run['V'][:, 0])
            temp = safe_interp(d.t, run['t'], run['T'][:, 0])
            values = np.r_[(v-d.V.to_numpy())/.02, (temp-d['T'].to_numpy())/.4]
        else:
            # Penalize incomplete trajectories; never extrapolate them to the data horizon.
            values = np.full(2*len(d), 100.+10.*(float(d.t.iloc[-1])-run['t'][-1]))
        memo[key] = values
        records.append(dict(evaluation=len(records)+1, x=list(x), cost=float(.5*values@values),
                            **full_metrics(run, d)))
        write_json(out/'evaluations.json', records)
        print(f"fit -20 C: evaluation={len(records)}, cost={records[-1]['cost']:.6g}", flush=True)
        return values
    def jac(x):
        f0 = residual(x)
        columns = []
        for k in range(2):
            xp = x.copy()
            step = .002 if x[k]+.002 <= bounds[1][k] else -.002
            xp[k] += step
            columns.append((residual(xp)-f0)/step)
        return np.column_stack(columns)
    opt = least_squares(residual, x0, jac=jac, bounds=bounds, max_nfev=budget,
                        ftol=2e-4, xtol=2e-4, gtol=2e-4)
    fitted = unpack(opt.x)
    training_run = run_experiment(fitted, -20., data, out/'cache', label='calibration_final')
    accepted = bool(opt.success and completed(training_run, float(d.t.iloc[-1])))
    singular = np.linalg.svd(opt.jac, compute_uv=False)
    report = dict(version=VERSION, config=fitted.to_dict(), initial_config=base.to_dict(),
        train=[-20.], fitted_parameters=['ln_j0_at_lambda3', 'cathode_hydration_exponent'],
        residual_scales={'V': .02, 'T_C': .4}, x=opt.x, initial_x=x0,
        optimizer_success=bool(opt.success), accepted=accepted, message=opt.message,
        cost=float(opt.cost), nfev=int(opt.nfev), actual_evaluations=len(records),
        jacobian=opt.jac, singular_values=singular,
        condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
        source_hash=data.source_hash, code_hash=code_digest(),
        training_metrics=full_metrics(training_run,d),
        qualification='Retrospective condition holdout, not new blinded experimental validation. '
        'Nucleation and bound-ice coefficients are fixed scenarios; no measured ice validation.')
    write_json(out/'fit.json', report)
    if not accepted:
        raise RuntimeError('Calibration did not converge to a complete trajectory; see fit.json. No held-out success is claimed.')
    return report


def load_fit(path, data, expected_seed=None):
    record = read_json(path)
    if record.get('code_hash') != code_digest() or record.get('source_hash') != data.source_hash:
        raise ValueError('Calibration cache does not match source code or inputs; run fit again')
    if not record.get('accepted') or record.get('train') != [-20.] or len(record.get('fitted_parameters', [])) != 2:
        raise ValueError('Expected a completed -20 C two-parameter calibration')
    if expected_seed is not None and Config(**record['initial_config']).digest != expected_seed.digest:
        raise ValueError('Calibration settings changed; use fit to recalibrate explicitly')
    return record


def export_merged_diagnostics(run, data_frame, dest):
    """One source of truth for the notebook, residuals and physical budgets."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    model=Model(Config(**run['config']));g=model.g
    initial=model.phases(run['y'][0]);rows=[]
    onset=run['metrics'].get('nucleation_times_s',[None])[0]
    for t,y,j in zip(run['t'],run['y'],run['j']):
        state=model.phases(y);obs=model.observations(state,j);led=state['led']
        stored=float(AREA*np.dot(state['H']-initial['H'],g.tdx))
        recorded=float(AREA*led[-2]);reaction=float(AREA*led[1])
        rows.append(dict(time_s=t,j_A_cm2=j,T_C=obs['T_mean'][0]-TM,V=obs['V'][0],
            reversible_V=obs['reversible'][0],activation_V=obs['activation'][0],
            ohmic_V=obs['ohmic'][0],concentration_V=obs['concentration'][0],
            ice_fraction=obs['ice_bulk'][0],lambda_PEM=np.average(state['lam'][0,g.pem],weights=g.dx[g.pem]),
            lambda_cCL=np.average(state['lam'][0,g.clc],weights=g.dx[g.clc]),
            hazard=y[model.ledger_end] if model.cfg.nucleation=='median' else np.nan,
            nucleated=bool(model.cfg.nucleation=='immediate' or (onset is not None and t>=onset-1e-9)),
            reaction_J=reaction,auxiliary_J=float(AREA*led[2]),stored_enthalpy_J=stored,
            net_boundary_and_water_enthalpy_J=recorded-reaction-float(AREA*led[2]),
            energy_residual_J=stored-recorded))
    frame=pd.DataFrame(rows)
    frame.to_csv(dest/'time_series.csv',index=False,encoding='utf-8-sig')
    residuals=data_frame[['t','V','T']].copy()
    residuals['model_V']=safe_interp(data_frame.t,run['t'],run['V'][:,0])
    residuals['model_T_C']=safe_interp(data_frame.t,run['t'],run['T'][:,0])
    residuals['V_residual']=residuals.model_V-residuals.V
    residuals['T_residual_C']=residuals.model_T_C-residuals['T']
    residuals.to_csv(dest/'residuals.csv',index=False,encoding='utf-8-sig')
    return frame


def replay_merged(cfg, data, out, plots=True):
    rows=[]
    for temp in sorted(data.experiments, reverse=True):
        role='calibration' if temp == -20. else 'retrospective_cross_temperature_check'
        run=run_experiment(cfg,temp,data,out,label=role)
        export_run(run,temp,data,out,role=role)
        export_merged_diagnostics(run,data.experiments[temp],Path(out)/'results'/f'{temp:g}C')
        if plots:
            plot_run(run,temp,data.experiments[temp],Path(out)/'results'/f'{temp:g}C')
        rows.append(dict(temp_C=temp, role=role, **full_metrics(run,data.experiments[temp])))
        print(f'{temp:g} C: {rows[-1]}',flush=True)
    write_json(Path(out)/'replay_summary.json',dict(version=VERSION,config=cfg.to_dict(),
        source_hash=data.source_hash,code_hash=code_digest(),results=rows,
        qualification='Only -20 C fitted. Fixed phase-change scenarios; retrospective holdout, not independent ice validation.'))
    pd.DataFrame(rows).to_csv(Path(out)/'replay_summary.csv',index=False,encoding='utf-8-sig')
    return rows


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',nargs='?',default='run',choices=['run','fit','check','convergence','audit','smoke'])
    p.add_argument('--input',help='Original XLSX directory, archive or canonical CSV')
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'q1_merged_results'))
    p.add_argument('--params',help='Validated merged fit.json; run/convergence require matching code and input hashes')
    p.add_argument('--grid',default=None,help='Explicit calibration grid; default i192')
    p.add_argument('--budget',type=int,default=60)
    p.add_argument('--nucleation',choices=['median','immediate'],default=None)
    p.add_argument('--bound-ice-rate',type=float,default=None,help='Scenario coefficient in s^-1, not fitted')
    p.add_argument('--bp-heat-capacity-factor',type=float,default=None,help='Default 1 = given material value; overrides are explicit scenarios')
    p.add_argument('--max-step',type=float,default=None)
    p.add_argument('--rtol',type=float,default=None)
    p.add_argument('--sample',type=float,default=None)
    p.add_argument('--reference',action='store_true')
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--convergence-grids',nargs='+',default=['i96','i192','i384'])
    return p


def command_config(args, data=None, initial=False):
    cfg=current_config()
    if args.grid:
        cfg=apply_grid(cfg,args.grid)
    changes={key:getattr(args,key) for key in ['nucleation','bound_ice_rate','bp_heat_capacity_factor','max_step','rtol','sample']
             if getattr(args,key,None) is not None}
    cfg=cfg.changed(**changes)
    cfg.validate()
    return cfg


def main(argv=None):
    global USE_NUMBA
    args=parser().parse_args(argv)
    if args.reference:
        USE_NUMBA=False
        os.environ['Q1_REFERENCE_BACKEND']='1'
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    try:
        if args.command=='check':
            self_check(out/'validation')
            return 0
        if args.command=='smoke':
            cfg=apply_grid(current_config(),'u14').changed(horizon=.4,voltage_limit=0.)
            r=simulate(cfg,Protocol('constant',(.05,)),stop_success=False,complete_horizon=True)
            write_json(out/'smoke.json',r['metrics'])
            return 0 if r['metrics']['stop_reason']=='horizon' else 3
        data=load_dataset(Path(args.input) if args.input else default_input_files())
        audit_dataset(data,out/'audit')
        if args.command=='audit':return 0
        seed=command_config(args,data)
        fit_path=Path(args.params) if args.params else out/'calibration'/'fit.json'
        if args.command=='fit' or (args.command=='run' and not fit_path.exists() and not args.params):
            if args.params:raise ValueError('fit starts from the explicit data-independent seed; omit --params')
            report=fit_merged(seed,data,out/'calibration',args.budget)
        else:
            # A supplied fit carries its own mesh/scenario. Explicit overrides must agree.
            expected=None if args.params else seed
            report=load_fit(fit_path,data,expected_seed=expected)
            if args.params:
                actual=Config(**report['initial_config'])
                if args.grid and (tuple(actual.mesh) != tuple(seed.mesh) or actual.cgdl_stretch != seed.cgdl_stretch):
                    raise ValueError('Grid override differs from supplied calibration')
                for key in ['nucleation','bound_ice_rate','bp_heat_capacity_factor','max_step','rtol','sample']:
                    if getattr(args,key,None) is not None and getattr(actual,key)!=getattr(seed,key):
                        raise ValueError(f'{key} differs from supplied calibration')
        cfg=Config(**report['config'])
        if args.command=='convergence':
            result=convergence(cfg,data,out/'convergence',args.convergence_grids)
            return 0 if result['numerical_checks_passed'] else 4
        rows=replay_merged(cfg,data,out,plots=not args.no_plots)
        return 0 if all(row.get('complete') for row in rows) else 3
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2


# ==================== Embedded regression tests ====================
try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    def test_geometry_and_units():
        m = Model(Config())
        assert sum(m.g.dx) == pytest.approx(0.0003267)
        assert np.dot(m.g.C, m.g.tdx) == pytest.approx(6127.47966)
        s = Model(Config(cells=5, fixture='H2'))
        assert np.dot(s.g.C, s.g.tdx) == pytest.approx(109637.3983)
        assert sum(s.g.tdx) == pytest.approx(0.0416335)
        assert AREA * 10000.0 == 25
        assert EW == 1000 * 0.001
        assert m.g.bcap[m.g.pem[0]] == pytest.approx(2150 * MW / EW)

    @pytest.mark.parametrize('mesh', [(3,2,3,3,3), (12,6,8,10,12), (24,12,16,20,24)])
    def test_proton_face_current_and_local_faraday_balance(mesh):
        m = Model(Config(mesh=mesh))
        g = m.g
        assert g.im_faces[g.pem[0]] == 1.0
        assert g.im_faces[g.clc[0]] == 1.0
        # Uniform CL reaction implies exact face-current divergence in every cell.
        divergence = np.diff(g.im_faces) / g.dx
        expected = np.zeros(g.n)
        expected[g.cla] = 1 / THICKNESS[1]
        expected[g.clc] = -1 / THICKNESS[3]
        np.testing.assert_allclose(divergence, expected, rtol=1e-12, atol=1e-8)
        y = m.initial()
        details = m.rhs(0, y, 0.1, np.zeros(1), details=True)
        # Uniform lambda cancels diffusion, isolating the actual implemented drag flux.
        expected_flux = MW * (2.5 * m.cfg.lambda0 / 22) * 1000 / F
        flux = details['fluxes']['bound'][0]
        assert flux[g.pem[0]] == pytest.approx(expected_flux, rel=1e-12)
        assert flux[g.clc[0]] == pytest.approx(expected_flux, rel=1e-12)

    def test_ice_comparison_cannot_pass_on_voltage_temperature_alone():
        cfg = Config()
        m = Model(cfg)
        y = np.stack([m.initial(), m.initial()])
        a = dict(config=cfg.to_dict(), t=np.array([0., 1.]), y=y,
                 T=np.zeros((2,1)), V=np.ones((2,1)),
                 metrics=dict(stop_reason='horizon', numerically_accepted=True))
        b = {**a, 'y': y.copy()}
        assert compare_trajectories(a,b,1)['output_converged']
        b['y'][1, m.n + m.g.clc[0]] = 920 * 0.01
        check = compare_trajectories(a,b,1)
        assert check['next_grid_V_diff'] == check['next_grid_T_diff'] == 0
        assert not check['ice_converged'] and not check['output_converged']
        assert check['inventory_curve_absolute_diff_kg'] > 0
        assert not compare_trajectories(a,a,2)['output_converged']

    def test_exact_saturation_and_enthalpy_inversion():
        m = Model(Config())
        y = m.initial()
        z, H, _ = m.unpack(y)
        z[0, 0, m.g.clc] = 100
        z[1, 0, m.g.clc] = 20
        T = np.full(m.g.nt, 253.15)
        mv, ml, eg = partition(z[0], z[1], T[m.g.mea_indices], m.g.eps)
        h = m.g.C * (T - TM)
        h[m.g.mea_indices] += (2000 * mv + 4182 * ml + 2050 * z[1] + 4182 * z[2]) * -20 + LV * mv - LF * z[1]
        y[m.ih:m.il] = h / m.g.C
        s = m.phases(y)
        assert np.max(abs(s['T'] - T)) < 1e-07
        assert np.max(abs(s['mv'] + s['ml'] - z[0])) < 1e-12
        assert np.min(eg[:, m.g.porous]) > 0

    def test_conservative_residual_at_nonuniform_state():
        m = Model(Config(cells=5, fixture='H2', outlet='drain'))
        y = m.initial()
        z, _, _ = m.unpack(y)
        z[0, :, m.g.clc] = 2
        z[1, :, m.g.clc] = 1
        d = m.rhs(0, y, 0.1, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        dz = d[:m.nw].reshape(4, m.c, m.n)
        water = np.sum(dz[:3] * m.g.dx, axis=(0, 2))
        assert np.max(abs(water - d[m.il:m.il + m.c])) < 1e-13
        thermal = np.dot(d[m.ih:m.il] * m.g.C, m.g.tdx)
        assert thermal == pytest.approx(d[-2], rel=1e-12, abs=1e-07)
        detail = m.rhs(0, y, 0.1, np.array([0.1, 0.2, 0.3, 0.4, 0.5]), details=True)
        for c in range(m.c):
            for layer in range(5):
                ix = np.flatnonzero(m.g.layer == layer)
                left, right = (ix[0], ix[-1] + 1)
                external = sum((f[c, left] - f[c, right] for name, f in detail['fluxes'].items() if name != 'gas'))
                generated = MW * 1000 / (2 * F) if layer == 3 else 0.0
                local = np.sum(dz[:3, c, ix] * m.g.dx[ix], axis=None)
                assert local == pytest.approx(external + generated, abs=1e-13)

    def test_liquid_flux_dry_equal_pressure_and_drain():
        args = dict(T=np.array([253.15] * 2), eps=np.array([0.8] * 2), K=np.array([6.2e-12] * 2), angle=np.array([110.0] * 2), dx=np.array([1e-05] * 2), ice=np.zeros(2))
        assert np.all(liquid_flux(np.zeros(2), **args, drain=True) == 0)
        assert liquid_flux(np.array([100.0, 100.0]), **args)[1] == 0
        assert liquid_flux(np.array([100.0, 0.0]), **args)[1] > 0
        assert liquid_flux(np.array([0.0, 100.0]), **args)[1] < 0
        assert liquid_flux(np.array([100.0, 100.0]), **args, drain=True, side=0)[0] < 0
        blocked = {**args, 'ice': np.array([0.0, 920 * 0.8])}
        assert abs(liquid_flux(np.array([100.0, 0.0]), **blocked)[1]) < 1e-15

    def test_plate_flux_cancels_and_sealed_energy():
        m = Model(Config(cells=5, fixture='H2', h=0))
        y = m.initial(Tfield=np.linspace(250, 280, m.g.nt))
        d = m.rhs(0, y, 0, np.zeros(5))
        assert np.dot(d[m.ih:m.il] * m.g.C, m.g.tdx) == pytest.approx(d[-2], abs=1e-07)

    def test_fixed_charge_and_current_limits():
        p = Protocol('fixed')
        t = np.linspace(0, 60, 6001)
        assert np.trapezoid([p.j(ti) for ti in t], t) == pytest.approx(9)
        assert 0.3 * (290 / 3) - 9 == pytest.approx(20)

    def test_initial_temperature_is_not_fitted():
        m = Model(Config(T0=248.15))
        y = m.initial()
        assert np.max(abs(m.phases(y)['T'] - 248.15)) < 1e-10
        assert np.max(abs(m.phases(y)['lam'][:, m.g.pem] - 3)) < 1e-10

    def test_colored_jacobian_matches_independent_directional_difference():
        m = Model(Config(cells=5, fixture='H2'))
        y = m.initial()
        z, _, _ = m.unpack(y)
        z[0, :, m.g.porous] = 2
        z[1, :, m.g.porous] = 1
        u = np.arange(1, 6) * 0.1
        jac = m.jacobian(0, y, 0.1, u)
        rng = np.random.default_rng(7)
        direction = np.zeros(m.size)
        direction[m.active_columns] = rng.normal(size=len(m.active_columns))
        h = 1e-06
        direct = (m.rhs(0, y + h * direction, 0.1, u) - m.rhs(0, y - h * direction, 0.1, u)) / (2 * h)
        estimated = jac @ direction
        ix = np.r_[m.ih + np.flatnonzero(m.g.names == 'BP'), m.ih + np.flatnonzero(m.g.names == 'endplate')]
        assert np.linalg.norm((estimated - direct)[ix]) / np.linalg.norm(direct[ix]) < 0.0001
        assert len(m._jac_groups) < len(m.active_columns) / 2

    # ==================== test_voltage_closure ====================
    def test_uniform_cl_dissipation_resistance_is_mesh_independent():
        expected = 0.2 * 10000.0 * (THICKNESS[1] + THICKNESS[3]) / (3 * conductivity(253.15, 3) * 0.3 ** 1.5)
        for mesh in [(1, 1, 1, 1, 1), (3, 2, 3, 3, 3), (8, 8, 8, 8, 8)]:
            cfg = Config(mesh=mesh, cl_proton_loss=True)
            m = Model(cfg)
            s = m.phases(m.initial())
            new = m.voltage(s, 0.2)
            old = Model(cfg.changed(cl_proton_loss=False)).voltage(s, 0.2)
            assert new['cl_ohmic'][0] == pytest.approx(expected, rel=1e-12)
            assert old['V'][0] - new['V'][0] == pytest.approx(expected)
            assert m.voltage(s, 0)['cl_ohmic'][0] == 0

    def test_hydration_factor_limits_and_reference_normalization():
        for gamma in (0.0, 3.0):
            m = Model(Config(cathode_hydration_exponent=gamma))
            s = m.phases(m.initial())
            assert m.voltage(s, 0.1)['active_fraction'][0] == pytest.approx((3 / 14) ** gamma)
            s['lam'][:, m.g.clc] = 28.0
            assert m.voltage(s, 0.1)['active_fraction'][0] == pytest.approx(1.0)

    @pytest.mark.parametrize('equilibrium', ['ice_reference', 'supercooled_liquid'])
    def test_extended_closure_still_closes_energy_and_water(equilibrium):
        m = Model(Config(cl_proton_loss=True, cathode_hydration_exponent=3, j0=15, vapor_equilibrium=equilibrium))
        y = m.initial()
        d = m.rhs(0, y, 0.1, np.zeros(1))
        dz = d[:m.nw].reshape(4, m.c, m.n)
        assert np.sum(dz[:3] * m.g.dx) == pytest.approx(d[m.il], abs=1e-12)
        assert np.dot(d[m.ih:m.il] * m.g.C, m.g.tdx) == pytest.approx(d[-2], abs=1e-07)

    def test_supercooled_flash_conserves_water_and_uses_liquid_saturation():
        w = np.array([10.0])
        ice = np.array([1.0])
        eps = np.array([0.4])
        T = np.array([253.15])
        mv, ml, eg = partition(w, ice, T, eps, True)
        assert mv + ml == pytest.approx(w)
        assert mv / eg == pytest.approx(0.018 * psat(T, liquid=True) / (8.314462618 * T))
        assert mv[0] > partition(w, ice, T, eps, False)[0][0]


# Added manufactured numerical tests. They are NOT experimental validation.
def _manufactured_run(cfg=None,phi=0.0,end=1.0):
    cfg=Config() if cfg is None else cfg
    m=Model(cfg); y=np.stack([m.initial(),m.initial()])
    y[:,m.c*m.n:2*m.c*m.n]=920*phi
    return dict(config=cfg.to_dict(),t=np.array([0.0,end]),y=y,
        T=np.full((2,m.c),-20.),V=np.full((2,m.c),.7),
        metrics=dict(stop_reason='horizon',numerically_accepted=True))


def test_stretched_grid_has_exact_layer_widths_and_shared_thermal_grid():
    cfg=apply_grid(Config(),'i192'); g=Geometry(cfg)
    for i,length in enumerate(THICKNESS):
        assert np.sum(g.dx[g.layer==i])==pytest.approx(length,rel=1e-13)
    np.testing.assert_allclose(g.tdx[g.mea_indices[0]],g.dx,rtol=0,atol=0)
    assert g.dx[-cfg.mesh[-1]]<.3e-6
    assert np.all(np.diff(g.layer_faces[4])>0)


def test_stretched_grid_is_nested_when_doubled():
    a=Geometry(apply_grid(Config(),'i96')); b=Geometry(apply_grid(Config(),'i192'))
    for i in range(5):
        np.testing.assert_allclose(a.layer_faces[i],b.layer_faces[i][::2],rtol=1e-13,atol=1e-18)


def test_stretched_geometry_dry_heat_capacity_is_unchanged():
    a=Geometry(apply_grid(Config(),'u192')); b=Geometry(apply_grid(Config(),'i192'))
    assert np.dot(a.C,a.tdx)==pytest.approx(np.dot(b.C,b.tdx),rel=1e-14)


def test_exact_proton_faces_on_stretched_grid():
    m=Model(apply_grid(Config(),'i96')); g=m.g
    d=np.diff(g.im_faces)/g.dx
    np.testing.assert_allclose(d[g.cla],1/THICKNESS[1],rtol=1e-12)
    np.testing.assert_allclose(d[g.clc],-1/THICKNESS[3],rtol=1e-12)
    assert g.im_faces[g.pem[0]]==g.im_faces[g.clc[0]]==1


def test_uniform_mapping_matches_uniform_widths():
    np.testing.assert_allclose(np.diff(layer_faces(3.,6,0)),np.full(6,.5),rtol=0,atol=0)


def test_grid_argument_validation():
    for grid in ('1,2,3','0,1,1,1,1','wrong'):
        with pytest.raises(ValueError): apply_grid(Config(),grid)
    with pytest.raises(ValueError): Config(cgdl_stretch=-1).validate()
    with pytest.raises(ValueError): Config(atol_scale=0).validate()


def test_projection_conserves_inventory_on_nonmatching_grids():
    a=np.array([.1,.2,.3,.4]); b=np.array([.17,.33,.5])
    x=np.arange(24,dtype=float).reshape(2,3,4)
    np.testing.assert_allclose(conservative_project(x,a,b)@b,x@a,rtol=1e-14,atol=1e-14)


def test_projection_preserves_constant_and_bounds():
    a=np.array([.2,.3,.5]); b=np.array([.1,.2,.3,.4])
    np.testing.assert_allclose(conservative_project(np.ones(3)*.02,a,b),.02,atol=1e-15)
    y=conservative_project(np.array([1.,2.,3.]),a,b)
    assert y.min()>=1-1e-14 and y.max()<=3+1e-14


def test_projection_refuses_different_domains():
    with pytest.raises(ValueError): conservative_project(np.ones(2),[.5,.5],[1.,1.])


def test_equal_peak_does_not_hide_wrong_spatial_distribution():
    a=_manufactured_run(); b=_manufactured_run(); n=sum(a['config']['mesh'])
    a['y'][:,n:n+3]=920*.1; b['y'][:,2*n-3:2*n]=920*.1
    check=compare_trajectories(a,b,1)
    assert check['phi_max_curve_absolute_diff']==0
    assert check['inventory_curve_absolute_diff_kg']<1e-18
    assert check['spatial_phi_mean_L1_diff']>.09
    assert not check['ice_converged']


def test_constant_field_agrees_on_uniform_and_stretched_grids():
    a=_manufactured_run(apply_grid(Config(),'u48'),.02)
    b=_manufactured_run(apply_grid(Config(),'i96'),.02)
    check=compare_trajectories(a,b,1)
    assert check['output_converged']
    assert check['conservative_projection_Linf_diff']<1e-12


def test_failed_trajectory_never_passes_convergence():
    a=_manufactured_run(); b=_manufactured_run()
    b['metrics']['stop_reason']='numerical_failure'
    assert not compare_trajectories(a,b,1)['output_converged']
    assert not compare_trajectories(a,a,2)['output_converged']


def test_onset_interpolation_is_not_a_nucleation_model():
    t=np.array([0.,.2,.4]); y=np.array([0.,0.,2e-6])
    assert threshold_onset(t,y,1e-6)==pytest.approx(.3)
    assert threshold_onset(t,np.zeros(3),1e-6) is None


def test_no_prediction_extrapolation_after_termination():
    y=safe_interp([0,.5,1,2],[0,1],[10,20])
    assert np.isnan(y[-1]); assert y[1]==15
    r=_manufactured_run(end=1)
    d=pd.DataFrame(dict(t=[0.,2.],V=[.7,.6],T=[-20.,-19.]))
    assert not waveform(r,d)['complete']
    assert 'V_RMSE' not in waveform(r,d)


def test_current_columns_are_not_silently_reconciled():
    d=normalized_experiment(pd.DataFrame(dict(t=[0.,1.],I=[1.,2.],V=[.7,.6],T=[-20.,-19.],j=[.1,.2])),-20,'test')
    np.testing.assert_allclose(d.j_A,[.04,.08]); np.testing.assert_allclose(d.j,[.1,.2])


def test_input_requires_monotonic_zero_based_time():
    d=pd.DataFrame(dict(t=[1.,2.],I=[1.,2.],V=[.7,.6],T=[-20.,-19.]))
    with pytest.raises(ValueError): normalized_experiment(d,-20,'test')
    d.t=[0.,0.]
    with pytest.raises(ValueError): normalized_experiment(d,-20,'test')


def _csv_fixture(path):
    pd.DataFrame(dict(T0_C=[-20.,-20.],t_s=[0.,.04],I_A=[.5,.5],
                     V_V=[.76,.75],T_C=[-20.,-19.99],j_A_cm2=[.02,.02])).to_csv(path,index=False)


def test_canonical_csv_loading_and_hash(tmp_path):
    p=tmp_path/'input.csv'; _csv_fixture(p)
    a=load_dataset(p); assert set(a.experiments)=={-20.}
    text=p.read_text().replace('.76','.77'); p.write_text(text)
    b=load_dataset(p); assert a.source_hash!=b.source_hash


def test_duplicate_condition_is_rejected(tmp_path):
    _csv_fixture(tmp_path/'a.csv'); _csv_fixture(tmp_path/'b.csv')
    with pytest.raises(ValueError,match='Duplicate'): load_dataset(tmp_path)


def test_empty_template_is_not_treated_as_experimental_data(tmp_path):
    p=tmp_path/'empty.csv'; p.write_text(','.join(CANONICAL_COLUMNS)+'\n')
    with pytest.raises(ValueError,match='empty'): load_dataset(p)


def _minimal_xlsx_fixture(cached=True):
    # OOXML parser fixture, not a scientific dataset.
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'; rows=[]
    for row,t in ((3,0.),(4,.2),(5,.4)):
        cells=''.join(f'<c r="{col}{row}"><v>{value}</v></c>' for col,value in zip('ABCD',[t,.5,.75,-20.]))
        cells+=f'<c r="E{row}"><f>B{row}/25</f>'+('<v>0.02</v>' if cached else '')+'</c>'
        rows.append(f'<row r="{row}">{cells}</row>')
    parts={
      'xl/workbook.xml':f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="-20" sheetId="1" r:id="rId1"/></sheets></workbook>',
      'xl/_rels/workbook.xml.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/></Relationships>',
      'xl/worksheets/sheet1.xml':f'<worksheet xmlns="{ns}"><sheetData>'+''.join(rows)+'</sheetData></worksheet>'}
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,content in parts.items(): z.writestr(name,content)
    return stream.getvalue()


def test_original_xlsx_cached_formulas_preserved(tmp_path):
    p=tmp_path/'test.xlsx'; p.write_bytes(_minimal_xlsx_fixture())
    data=load_dataset(p)
    assert data.formulas[0]['formula']=='=B3/25'
    assert data.experiments[-20.].j.iloc[0]==pytest.approx(.02)


def test_missing_formula_cache_is_not_recomputed(tmp_path):
    p=tmp_path/'test.xlsx'; p.write_bytes(_minimal_xlsx_fixture(False))
    data=load_dataset(p); assert data.experiments[-20.].j.isna().all()
    with pytest.raises(ValueError,match='incomplete'):
        run_experiment(Config(basis='B'),-20,data,tmp_path/'out')


def test_nested_original_archive_reader(tmp_path):
    inner=io.BytesIO()
    with zipfile.ZipFile(inner,'w') as z: z.writestr('experiment.xlsx',_minimal_xlsx_fixture())
    p=tmp_path/'B.zip'
    with zipfile.ZipFile(p,'w') as z: z.writestr('附件2.zip',inner.getvalue())
    data=load_dataset(p); assert len(data.experiments[-20.])==3
    assert '!附件2.zip!' in data.members[0]['name']


def test_heterogeneous_equal_pressure_zero_liquid_flux():
    t=np.full(2,253.15); eps=np.array([POROSITY[3],POROSITY[4]])
    k=np.array([6.2e-13,6.2e-12]); angle=np.array([100.,110.]); s0=.02
    def diff(sr):
        pressure=hydraulic_pressure(990*eps*np.array([s0,sr]),np.zeros(2),t,eps,k,angle)
        return pressure[1]-pressure[0]
    sr=brentq(diff,0,1,xtol=1e-15)
    f=liquid_flux(990*eps*np.array([s0,sr]),np.zeros(2),t,eps,k,angle,np.array([1e-6,2e-6]))
    assert abs(f[1])<1e-12 and abs(sr-s0)>1e-4


def test_nonuniform_manufactured_diffusion_has_constant_flux():
    dx=np.array([.1,.2,.4,.3]); x=np.cumsum(dx)-dx/2
    c=3*x+1; f=harmonic_faces(dx,np.full(4,2.))*(c[:-1]-c[1:])
    np.testing.assert_allclose(f,-6.,rtol=1e-14)


def test_initial_freezing_source_vanishes_without_liquid():
    m=Model(Config(j0=.3)); dy=m.rhs(0,m.initial(),.02,np.zeros(1))
    np.testing.assert_allclose(dy[m.n:2*m.n],0,atol=0)


def test_renamed_file_code_digest_uses_actual_source():
    assert code_digest()==hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def test_registry_rejects_stale_hash_and_preserves_grid(tmp_path):
    record=dict(config=apply_grid(Config(),'i192').to_dict(),code_hash='stale')
    write_json(tmp_path/'fit_registry.json',record)
    with pytest.raises(ValueError,match='Stale'): registry_config(tmp_path)
    record['code_hash']=code_digest(); write_json(tmp_path/'fit_registry.json',record)
    cfg=registry_config(tmp_path); assert sum(cfg.mesh)==192 and cfg.cgdl_stretch==4


def test_registry_rejects_wrong_dataset_hash(tmp_path):
    record=dict(config=Config().to_dict(),code_hash=code_digest(),source_hash='a')
    write_json(tmp_path/'fit_registry.json',record)
    with pytest.raises(ValueError,match='different'): registry_config(tmp_path,Dataset({},'b',[],[],[]))


def test_config_import_unknown_fields_is_not_silent(tmp_path):
    p=tmp_path/'config.json'; write_json(p,dict(config=dict(j0=.3,unknown_physics=1)))
    with pytest.raises(ValueError,match='Unsupported'): read_config_initial(p)


def test_json_outputs_are_strict_and_atomic(tmp_path):
    p=tmp_path/'out.json'; write_json(p,dict(x=np.nan,y=np.array([1.,np.inf])))
    assert read_json(p)==dict(x=None,y=[1.,None]); assert not list(tmp_path.glob('*.tmp'))


def test_single_cell_integrator_smoke():
    r=smoke_worker(('u14',.04))
    assert r['metrics']['last_valid_time']==pytest.approx(.04)
    assert r['metrics']['exception'] is None


def test_early_invalid_initial_state_does_not_crash():
    r=simulate(Config(j0=1e-8,horizon=.04,voltage_limit=.3),Protocol('constant',(.2,)),
               stop_success=False,complete_horizon=True)
    assert r['metrics']['stop_reason']=='voltage'
    assert r['metrics']['exception'] is None and r['t'][-1]==0


def test_experimental_runner_cache_roundtrip(tmp_path):
    p=tmp_path/'data.csv'; _csv_fixture(p); d=load_dataset(p)
    cfg=Config(j0=.3,max_step=.02)
    a=run_experiment(cfg,-20.,d,tmp_path/'out'); b=run_experiment(cfg,-20.,d,tmp_path/'out')
    np.testing.assert_array_equal(a['y'],b['y'])
    assert a['metrics']['candidate_id']==b['metrics']['candidate_id']
    assert a['metrics']['full_period']


def test_calibration_mesh_is_explicit_in_cli():
    args=parser().parse_args(['fit','--grid','i192','--input','unused.csv'])
    c=command_config(args,None,initial=True)
    assert tuple(c.mesh)==(48,24,32,40,48) and c.cgdl_stretch==4 and c.h==40



def test_stretched_grid_local_and_global_water_energy_balances():
    m=Model(apply_grid(Config(j0=.3),'i96')); g=m.g
    y=m.initial(); z,_,_=m.unpack(y)
    z[0,0,g.cathode]=.2
    z[1,0,g.cathode]=.1
    d=m.rhs(0,y,.05,np.zeros(1),details=True)
    dy=d['derivative']; dz=dy[:m.nw].reshape(4,m.c,m.n)
    total=np.sum(dz[:3]*g.dx)
    assert total==pytest.approx(dy[m.il],abs=1e-12)
    assert np.dot(dy[m.ih:m.il]*g.C,g.tdx)==pytest.approx(dy[-2],rel=1e-12,abs=1e-7)
    for layer in range(5):
        ids=np.flatnonzero(g.layer==layer); left,right=ids[0],ids[-1]+1
        inflow=sum(f[0,left]-f[0,right] for name,f in d['fluxes'].items() if name!='gas')
        source=MW*.05*1e4/(2*F) if layer==3 else 0.
        assert np.sum(dz[:3,0,ids]*g.dx[ids])==pytest.approx(inflow+source,abs=1e-12)



# ==================== Inherited core and merged configuration checks ====================
if pytest is not None:
    @pytest.mark.parametrize('grid', ['u14', 'i96'])
    @pytest.mark.parametrize('scheme', ['upwind', 'exponential'])
    @pytest.mark.parametrize('factor,beta', [(1.,0.),(.5,0.),(1.,-.2),(2.,.15)])
    @pytest.mark.parametrize('state', ['initial','wet'])
    def test_current_extension_reference_kernel_and_balances(grid, factor, beta, state, scheme):
        cfg = apply_grid(current_config(), grid).changed(cl_conductivity_factor=factor,
                                                         sorption_temp_coefficient=beta,
                                                         bound_flux_scheme=scheme)
        model = Model(cfg)
        y = model.initial()
        if state == 'wet':
            rng = np.random.default_rng(68)
            z,_,_ = model.unpack(y)
            z[0,0,model.g.porous] = rng.uniform(.01,10,sum(model.g.porous))
            z[1,0,model.g.porous] = rng.uniform(.01,5,sum(model.g.porous))
        reference = model.reference_rhs(0,y,.15,np.zeros(1))
        actual = model.rhs(0,y,.15,np.zeros(1))
        error = np.linalg.norm(reference-actual)/max(np.linalg.norm(reference),1e-30)
        assert error < 1e-10
        dz = actual[:model.nw].reshape(4,model.c,model.n)
        assert abs(np.sum(dz[:3]*model.g.dx)-actual[model.il]) < 1e-10
        assert abs(np.dot(actual[model.ih:model.il]*model.g.C,model.g.tdx)-actual[model.il+3*model.c]) < 1e-6

    def test_merged_seed_uses_given_capacity_and_no_joint_optimum():
        cfg=current_config()
        assert cfg.bp_heat_capacity_factor == 1.0 and cfg.cl_conductivity_factor == 1.0
        assert cfg.nucleation == 'median' and cfg.bound_ice_rate == .01
        assert cfg.tau_b == 1.0 and cfg.tau_f == 5.0
        assert cfg.cathode_hydration_exponent == 3.0
        assert parser().parse_args([]).command == 'run'

    def test_unit_extension_recovers_original_voltage_and_rhs():
        m = Model(current_config().changed(cl_conductivity_factor=1.,sorption_temp_coefficient=0.))
        y = m.initial()
        a = m._original_voltage(m.phases(y),.1)
        b = m.voltage(m.phases(y),.1)
        np.testing.assert_allclose(a['V'],b['V'],rtol=0,atol=1e-14)
        reference = m._original_rhs(0,y,.1,np.zeros(1))
        actual = m.rhs(0,y,.1,np.zeros(1))
        assert np.linalg.norm(reference-actual)/np.linalg.norm(reference) < 1e-10

    def test_catalyst_factor_scales_cl_not_membrane_loss():
        c = current_config().changed(cl_conductivity_factor=1.)
        a = Model(c); b = Model(c.changed(cl_conductivity_factor=.5))
        s = a.phases(a.initial())
        va = a.voltage(s,.1); vb = b.voltage(s,.1)
        np.testing.assert_allclose(vb['cl_ohmic'],2*va['cl_ohmic'],rtol=1e-13)
        np.testing.assert_allclose(vb['ohmic']-vb['cl_ohmic'],va['ohmic']-va['cl_ohmic'],rtol=1e-13)

    def test_input_selection_accepts_explicit_file_list(tmp_path):
        p = tmp_path/'source.csv'
        _csv_fixture(p)
        data = load_dataset([p])
        assert set(data.experiments) == {-20.}

    def test_source_contains_no_local_imports_or_encoded_program():
        import ast
        tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        forbidden = {'q1_complete','q1_state_closure','q1_bias_solver','q1_voltage_backend',
                     'fit_endpoint','fit_state_closure','check_extension','q2','q3','q4'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert not any(a.name.split('.')[0] in forbidden for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in forbidden
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                assert node.func.id not in ('exec','eval')


if pytest is not None:
    @pytest.mark.parametrize('pe', [0., 1e-8, .01, 1., 10., 100.])
    def test_exponential_flux_preserves_exact_stationary_solution(pe):
        # u(x)=a exp(v*x/D)+b has spatially constant flux v*b.
        left = 2.0 + np.exp(-pe)
        right = 3.0
        flux = bernoulli_positive(pe)*(left-right)+pe*left
        assert flux == pytest.approx(2.0*pe, abs=1e-12, rel=1e-12)

    def test_effective_fixture_capacity_changes_only_bp_storage():
        base=current_config().changed(bp_heat_capacity_factor=1.)
        a=Model(base);b=Model(base.changed(bp_heat_capacity_factor=.9))
        mask=a.g.names=='BP'
        np.testing.assert_allclose(b.g.C[mask],.9*a.g.C[mask])
        np.testing.assert_array_equal(b.g.C[~mask],a.g.C[~mask])
        np.testing.assert_allclose(b.phases(b.initial())['T'],base.T0,atol=1e-9)
        d=b.rhs(0,b.initial(),.1,np.zeros(1))
        assert np.dot(d[b.ih:b.il]*b.g.C,b.g.tdx)==pytest.approx(d[b.il+3*b.c],abs=1e-7)


if pytest is not None:
    def wet_merged_state(model, lam=12.):
        y=model.initial();g=model.g
        z,_,_=model.unpack(y)
        z[0,0,g.porous]=2.
        z[2,0]=g.bcap*lam
        T=np.full(g.n,model.cfg.T0)
        mv,ml,_=partition(z[0,0],z[1,0],T,g.eps,True)
        hv,hl,hi,hb=water_enthalpies(T)
        H=g.C*(model.cfg.T0-TM)
        H[g.mea_indices[0]]+=mv*hv+ml*hl+z[1,0]*hi+z[2,0]*hb
        y[model.ih:model.il]=H/g.C
        return y

    @pytest.mark.parametrize('nucleated',[False,True])
    def test_merged_phase_transfer_conserves_mass_enthalpy_and_matches_jit(nucleated):
        cfg=apply_grid(current_config(),'u14').changed(T0=248.15,ambient=248.15)
        m=Model(cfg);y=wet_merged_state(m);m.nucleated[0]=nucleated
        ref=m.reference_rhs(0,y,.05,np.zeros(1))
        fast=m.rhs(0,y,.05,np.zeros(1))
        np.testing.assert_allclose(fast,ref,rtol=1e-9,atol=1e-8)
        dz=fast[:m.nw].reshape(4,m.c,m.n)
        assert np.sum(dz[:3]*m.g.dx)==pytest.approx(fast[m.il],abs=1e-10)
        assert np.dot(fast[m.ih:m.il]*m.g.C,m.g.tdx)==pytest.approx(fast[m.il+3],abs=1e-7)
        assert fast[m.ledger_end]>0
        if not nucleated:
            assert np.max(abs(dz[1]))==0
        else:
            assert np.max(dz[1])>0
            without=Model(cfg.changed(bound_ice_rate=0));without.nucleated[0]=True
            delta=fast-without.rhs(0,y,.05,np.zeros(1))
            dz_delta=delta[:m.nw].reshape(4,m.c,m.n)
            assert np.sum(dz_delta[1])>0
            np.testing.assert_allclose(dz_delta[1],-dz_delta[2],atol=1e-9)
            np.testing.assert_allclose(delta[m.ih:m.il],0,atol=1e-9)

    def test_merged_event_restarts_and_no_ice_before_nucleation():
        cfg=apply_grid(current_config(),'u14').changed(T0=248.15,ambient=248.15,
            nucleation_prefactor=1e13,tau_b=1e12,bound_ice_rate=0.,
            horizon=.02,sample=.001,max_step=.005,voltage_limit=-10.,h=0.)
        m=Model(cfg);initial=wet_merged_state(m)
        run=simulate(cfg,Protocol('constant',(.01,)),initial=initial,stop_success=False,complete_horizon=True)
        assert run['metrics']['stop_reason']=='horizon',run['metrics']
        onset=run['metrics']['nucleation_times_s'][0]
        assert onset is not None and 0<onset<cfg.horizon
        assert run['ice'][run['t']<onset-1e-10].max()<1e-12
        assert run['ice'][-1].max()>1e-9
        event_idx=np.argmin(abs(run['t']-onset))
        assert run['y'][event_idx,m.ledger_end]==pytest.approx(cfg.nucleation_threshold,abs=1e-8)
        assert run['metrics']['mass_residual']<1e-8
        assert run['metrics']['energy_residual']<1e-8

    def test_merged_reference_jacobian_includes_hazard_row():
        m=Model(apply_grid(current_config(),'u14').changed(T0=248.15))
        y=wet_merged_state(m)
        a=m._original_jacobian(0,y,.05,np.zeros(1)).toarray()
        b=m.jacobian(0,y,.05,np.zeros(1)).toarray()
        np.testing.assert_allclose(a[m.ledger_end],b[m.ledger_end],rtol=1e-5,atol=1e-6)
        assert np.max(abs(a[m.ledger_end]))>0
        # The hazard is a passive integral between discrete events.
        assert np.max(abs(b[:,m.ledger_end]))==0

    def test_merged_calibration_never_queries_held_out_condition(tmp_path,monkeypatch):
        from types import SimpleNamespace
        cfg=apply_grid(current_config(),'u14')
        p=tmp_path/'source.csv';_csv_fixture(p);data=load_dataset(p)
        data.experiments[-25.]=data.experiments[-20.].copy()
        calls=[]
        original=run_experiment
        def spy(setting,temp,*args,**kwargs):
            calls.append(temp)
            assert temp == -20.
            return original(setting,temp,*args,**kwargs)
        def optimizer(fun,x,jac,**kwargs):
            values=fun(x);J=jac(x)
            return SimpleNamespace(x=x,jac=J,success=True,message='test optimizer',
                cost=float(.5*values@values),nfev=1)
        monkeypatch.setitem(globals(),'run_experiment',spy)
        monkeypatch.setitem(globals(),'least_squares',optimizer)
        record=fit_merged(cfg,data,tmp_path/'fit',budget=1)
        assert set(calls)=={-20.} and record['train']==[-20.]
        assert record['config']['bp_heat_capacity_factor']==1.
        load_fit(tmp_path/'fit'/'fit.json',data,expected_seed=cfg)
        with pytest.raises(ValueError):
            load_fit(tmp_path/'fit'/'fit.json',data,expected_seed=cfg.changed(bound_ice_rate=0.))
        record['train']=[-20.,-25.]
        write_json(tmp_path/'fit'/'fit.json',record)
        with pytest.raises(ValueError):load_fit(tmp_path/'fit'/'fit.json',data)

    def test_merged_failed_optimizer_is_not_accepted(tmp_path,monkeypatch):
        from types import SimpleNamespace
        p=tmp_path/'source.csv';_csv_fixture(p);data=load_dataset(p)
        def optimizer(fun,x,jac,**kwargs):
            return SimpleNamespace(x=x,jac=np.eye(2),success=False,message='budget',cost=1.,nfev=1)
        monkeypatch.setitem(globals(),'least_squares',optimizer)
        with pytest.raises(RuntimeError):
            fit_merged(apply_grid(current_config(),'u14'),data,tmp_path/'fit',budget=1)
        assert not read_json(tmp_path/'fit'/'fit.json')['accepted']

if __name__=='__main__':
    raise SystemExit(main())
