from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


QUESTION_ID = "Q1"
COMPUTE_ENGINEERING_VERSION = "mcm-compute-runtime-v2"
RENDER_ENGINEERING_VERSION = "mcm-render-runtime-v3"
RUNTIME_OUTPUT_NAME = "代码运行输出.txt"
TABLE_OUTPUT_NAME = "表格输出.md"
DEFAULT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)


class CompactLogger:


    TERMINAL_CHANNELS = {"CORE", "STATUS", "CACHE", "WARN", "ERROR"}

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def emit(self, message: str, channel: str = "DETAIL") -> None:
        normalized = str(message).strip()
        line = f"[{channel}] {normalized}"
        print(line, file=self._handle, flush=True)
        if channel in self.TERMINAL_CHANNELS:
            print(line, flush=True)

    def core(self, message: str) -> None:
        self.emit(message, "CORE")

    def status(self, message: str) -> None:
        self.emit(message, "STATUS")

    def cache(self, message: str) -> None:
        self.emit(message, "CACHE")

    def warning(self, message: str) -> None:
        self.emit(message, "WARN")

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_region_hash(tag: str) -> str:
    source = Path(__file__).resolve().read_text(encoding="utf-8-sig")
    source = source.replace("\r\n", "\n")
    return hashlib.sha256((tag + "\n" + source).encode("utf-8")).hexdigest()


def _dependency_signature(names: Sequence[str]) -> dict[str, str]:
    signature = {"python": platform.python_version()}
    for name in sorted(set(names)):
        try:
            signature[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            signature[name] = "not-installed"
    return signature


def _fingerprint_inputs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for raw_path in sorted(paths, key=lambda item: str(item.resolve()).lower()):
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{path}")
        stat = path.stat()
        fingerprints.append(
            {
                "path": path.as_posix(),
                "size": stat.st_size,
                "sha256": file_sha256(path),
            }
        )
    return fingerprints


def _resolve_cache_root(output_dir: Path) -> Path:
    task_root = output_dir.resolve().parent
    cache_root = (task_root / ".mcm-cache" / QUESTION_ID).resolve()
    cache_root.relative_to(task_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _safe_remove_cache_dir(path: Path, cache_root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(cache_root.resolve())
    if resolved == cache_root.resolve():
        raise RuntimeError("拒绝删除缓存根目录。")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def _load_compute_cache(cache_dir: Path) -> Any | None:
    payload_path = cache_dir / "results.pkl"
    ready_path = cache_dir / "READY"
    if not payload_path.is_file() or not ready_path.is_file():
        return None
    try:
        expected = ready_path.read_text(encoding="utf-8").strip()
        if not expected or file_sha256(payload_path) != expected:
            return None
        with payload_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_compute_cache(cache_dir: Path, results: Any, cache_root: Path) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"{cache_dir.name[:12]}-", dir=cache_dir.parent)
    )
    try:
        payload_path = temp_dir / "results.pkl"
        with payload_path.open("wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (temp_dir / "READY").write_text(
            file_sha256(payload_path), encoding="utf-8"
        )
        if cache_dir.exists():
            _safe_remove_cache_dir(cache_dir, cache_root)
        os.replace(temp_dir, cache_dir)
    finally:
        if temp_dir.exists():
            _safe_remove_cache_dir(temp_dir, cache_root)


def _render_manifest(paths: Sequence[Path]) -> list[str]:
    return [f"{path.name}\t{file_sha256(path)}" for path in paths]


def _load_render_cache(cache_dir: Path, output_dir: Path) -> list[Path] | None:
    ready_path = cache_dir / "READY"
    if not ready_path.is_file():
        return None
    entries: list[tuple[Path, Path]] = []
    try:
        for line in ready_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, expected = line.split("\t", 1)
            if Path(name).name != name:
                return None
            source = (cache_dir / name).resolve()
            source.relative_to(cache_dir.resolve())
            target = (output_dir / name).resolve()
            target.relative_to(output_dir.resolve())
            if source.suffix.lower() != ".png" or not source.is_file():
                return None
            if file_sha256(source) != expected:
                return None
            entries.append((source, target))
    except (OSError, ValueError):
        return None
    if not entries or len({target.name for _, target in entries}) != len(entries):
        return None
    restored: list[Path] = []
    for source, target in entries:
        shutil.copy2(source, target)
        restored.append(target)
    return restored


def _save_render_cache(
    cache_dir: Path,
    figure_paths: Sequence[Path],
    cache_root: Path,
) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"{cache_dir.name[:12]}-", dir=cache_dir.parent)
    )
    try:
        copied: list[Path] = []
        for figure_path in figure_paths:
            source = figure_path.resolve()
            if source.suffix.lower() != ".png" or not source.is_file():
                raise RuntimeError(f"绘图函数返回了无效 PNG：{source}")
            target = temp_dir / source.name
            shutil.copy2(source, target)
            copied.append(target)
        (temp_dir / "READY").write_text(
            "\n".join(_render_manifest(copied)) + "\n", encoding="utf-8"
        )
        if cache_dir.exists():
            _safe_remove_cache_dir(cache_dir, cache_root)
        os.replace(temp_dir, cache_dir)
    finally:
        if temp_dir.exists():
            _safe_remove_cache_dir(temp_dir, cache_root)


def _managed_inventory_path(cache_root: Path) -> Path:
    return cache_root / "managed-files.txt"


def _clear_previous_managed_outputs(output_dir: Path, cache_root: Path) -> None:
    inventory_path = _managed_inventory_path(cache_root)
    if not inventory_path.is_file():
        return
    for name in inventory_path.read_text(encoding="utf-8").splitlines():
        if not name.strip():
            continue
        target = (output_dir / name).resolve()
        target.relative_to(output_dir.resolve())
        if target.parent != output_dir.resolve():
            raise RuntimeError(f"缓存清单包含非法路径：{name}")
        if target.suffix.lower() in {".csv", ".md", ".png"} and target.is_file():
            target.unlink()


def _write_managed_inventory(paths: Sequence[Path], cache_root: Path) -> None:
    names = sorted({path.resolve().name for path in paths})
    target = _managed_inventory_path(cache_root)
    temp = target.with_suffix(".tmp")
    temp.write_text("\n".join(names) + "\n", encoding="utf-8")
    os.replace(temp, target)


def _validate_output_paths(
    paths: Iterable[Path],
    output_dir: Path,
    suffixes: set[str],
) -> list[Path]:
    validated: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        path.relative_to(output_dir.resolve())
        if path.parent != output_dir.resolve() or path.suffix.lower() not in suffixes:
            raise RuntimeError(f"导出函数返回了不合规路径：{path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"导出文件不存在或为空：{path}")
        validated.append(path)
    return validated


def write_dataframe_csv(dataframe: Any, path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def dataframe_markdown(dataframe: Any) -> str:
    columns = [_markdown_cell(column) for column in dataframe.columns]
    rows = list(dataframe.itertuples(index=False, name=None))
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_markdown_tables(
    tables: Sequence[tuple[str, Any, str | None]],
    path: Path,
) -> Path:
    if not 1 <= len(tables) <= 3:
        raise ValueError("表格输出.md 必须包含 1–3 张小表。")
    sections: list[str] = []
    for title, dataframe, note in tables:
        if len(dataframe) > 20:
            raise ValueError(f"Markdown 小表超过 20 行：{title}")
        section = [f"## {title}", "", dataframe_markdown(dataframe)]
        if note:
            section.extend(["", f"注：{note}"])
        sections.append("\n".join(section))
    path = path.resolve()
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def setup_plotting() -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if font_path.is_file():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except (OSError, RuntimeError):
                pass

    candidates = (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
        "Arial",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = tuple(name for name in candidates if name in available)
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(selected or ("DejaVu Sans",)),
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return selected


def save_publication_figure(figure: Any, path: Path, dpi: int = 600) -> Path:
    path = path.resolve()
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    return path


import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares, differential_evolution


def _attachment(name):
    root = Path(__file__).resolve().parent
    candidates = [root/name, root/'B题'/name, root.parent/name, root.parent/'B题'/name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'请将 {name} 放在脚本目录或其上一级目录。')


def get_input_paths():
    root = Path(__file__).resolve().parent
    return [_attachment('附件1.xlsx'), _attachment('附件2.xlsx')]


def get_analysis_contract():
    return {
        'question': 'Q1', 'input_sheets': ['参数清单', '启动温度为-20℃', '启动温度为-25℃'],
        'model': 'five-layer one-dimensional finite volume, semi-implicit water/ice/heat and electrochemistry',
        'calibration': 'DE global search and bounded trust-region refinement on -20℃ only; -25℃ external-temperature validation',
        'drive': 'recorded current-density column; current/25cm2 inconsistency retained as audit',
        'parameters': ['log10_j0_ref', 'thermal_inertia_scale', 'ramp_polarization_gain',
                       'polarization_relaxation_s', 'initial_conditioning_loss_V',
                       'conditioning_charge_C_cm2', 'ramp_decay_charge_C_cm2'],
        'grid': [8, 3, 4, 4, 8], 'dt_s': 0.2, 'seed': 20260924,
        'phase': 'mobile water to vapor/liquid by Buck saturation; kinetic liquid-ice conversion; pore-volume cap',
        'validation_times_s': [0, 5, 10, 15, 20, 25, 30, 35],
    }


def get_compute_dependency_names():
    return ['numpy', 'pandas', 'scipy', 'openpyxl']


def _number_map(frame):
    return {str(r[1]).strip(): r[2] for r in frame.itertuples(index=False, name=None)}


def _read_experiment(path, sheet, label):
    raw = pd.read_excel(path, sheet_name=sheet)
    d = pd.DataFrame({
        'time_s': pd.to_numeric(raw.iloc[:, 0], errors='coerce'),
        'current_A': pd.to_numeric(raw.iloc[:, 1], errors='coerce'),
        'voltage_V': pd.to_numeric(raw.iloc[:, 2], errors='coerce'),
        'temperature_C': pd.to_numeric(raw.iloc[:, 3], errors='coerce'),
        'current_density_A_cm2': pd.to_numeric(raw.iloc[:, 4], errors='coerce'),
    }).dropna().sort_values('time_s').reset_index(drop=True)
    if len(d) < 100 or d.time_s.duplicated().any() or not d.time_s.is_monotonic_increasing:
        raise ValueError(f'{label}实验时间序列缺失或重复')
    d['condition'] = label
    return d


def _grid(par):
    specs = [
        ('aGDL', float(par['阳极 GDL 厚度']), 8, float(par['阳极 GDL 孔隙率']), 0.30, 185*545),
        ('aCL', float(par['阳极 CL 厚度']), 3, float(par['阳极 CL 孔隙率']), 0.27, 970*240),
        ('PEM', float(par['质子交换膜厚度']), 4, 0.0, 0.24, 2150*1050),
        ('cCL', float(par['阴极 CL 厚度']), 4, float(par['阴极 CL 孔隙率']), 0.27, 970*240),
        ('cGDL', float(par['阴极 GDL 厚度']), 8, float(par['阴极 GDL 孔隙率']), 0.30, 185*545),
    ]
    layer, dx, eps, k, cp = [], [], [], [], []
    for name, thick, count, pore, conduct, heatcap in specs:
        layer.extend([name]*count); dx.extend([thick/count]*count)
        eps.extend([pore]*count); k.extend([conduct]*count); cp.extend([heatcap]*count)
    dx=np.array(dx); eps=np.array(eps); k=np.array(k); cp=np.array(cp)
    x=np.cumsum(dx)-dx/2
    return {'layer': np.array(layer), 'dx':dx, 'x':x, 'eps':eps, 'k':k, 'cp':cp,
            'length':float(np.sum(dx)), 'porous':eps>0, 'mem':np.array(layer)=='PEM',
            'ccl':np.array(layer)=='cCL', 'cath':np.isin(layer,['cCL','cGDL'])}


def _operator(dx, diffusivity, capacity, dt, h=0.0):
    n=len(dx); a=np.eye(n)
    for i in range(n-1):
        g=1.0/(dx[i]/(2*diffusivity[i])+dx[i+1]/(2*diffusivity[i+1]))
        a[i,i]+=dt*g/(capacity[i]*dx[i]); a[i,i+1]-=dt*g/(capacity[i]*dx[i])
        a[i+1,i+1]+=dt*g/(capacity[i+1]*dx[i+1]); a[i+1,i]-=dt*g/(capacity[i+1]*dx[i+1])
    bc=np.zeros(n)
    if h:
        for i in (0,n-1):
            g=1.0/(dx[i]/(2*diffusivity[i])+1/h)
            bc[i]=dt*g/(capacity[i]*dx[i]); a[i,i]+=bc[i]
    return lu_factor(a), bc


def _sat_water(tk, gas_pore):
    tc=tk-273.15
    if tc>=0:
        ps=611.21*np.exp((18.678-tc/234.5)*tc/(257.14+tc))
    else:
        ps=611.15*np.exp((23.036-tc/333.7)*tc/(279.82+tc))
    return gas_pore*0.018*ps/(8.314*tk)


def _voltage(j_cm, tk, mobile, ice, grid, par, ptrans, logj0):
    faraday=96485.0; rgas=8.314; j=j_cm*1e4
    eps=grid['eps']; porous=grid['porous']; ccl=grid['ccl']; cath=grid['cath']
    rho_i=920.0; rho_l=990.0
    m_sat=np.array([_sat_water(float(t), max(float(e),1e-9)) if p else 0.0
                    for t,e,p in zip(tk,eps,porous)])
    liq=np.where(porous,np.maximum(mobile-m_sat,0.0),0.0)
    gas_pore=np.maximum(eps-liq/rho_l-ice/rho_i, 1e-5)
    ice_vol=np.where(porous,ice/rho_i,0.0)
    pore_ice=np.where(porous,ice_vol/np.maximum(eps,1e-8),0.0)
    area=np.average(np.maximum(1.0-pore_ice[ccl],1e-5)**float(par['阴极冰覆盖活性面积指数']),
                    weights=grid['dx'][ccl])
    dref=np.where(grid['layer']=='cGDL',2.2e-5,2.2e-5)
    deff=dref*(tk/298.15)**1.75*gas_pore**1.5
    resistance=np.sum(grid['dx'][cath]/np.maximum(deff[cath],1e-12))
    c_o2=0.21*float(par['初始压力及操作压力'])/(rgas*np.average(tk[cath],weights=grid['dx'][cath]))
    jlim=4*faraday*c_o2/max(resistance,1e-9)
    tj=float(np.average(tk,weights=grid['dx']))
    p0=float(par['初始压力及操作压力'])
    po2=max(0.21*(1-min(j/max(jlim,1e-6),0.999)),1e-6)*p0
    erev=1.229-8.5e-4*(tj-298.15)+rgas*tj/(2*faraday)*np.log((p0/101325)*(po2/101325)**0.5)
    j0=10**logj0*np.exp(-67000/rgas*(1/tj-1/298.15))
    eta_act=rgas*tj/(0.5*faraday)*np.arcsinh(j/max(2*j0*area,1e-12))
    m_mem=float(np.average(mobile[grid['mem']],weights=grid['dx'][grid['mem']]))

    lam=np.clip(m_mem*float(par['膜当量质量'])/1000/(2150*0.018),float(par['初始膜含水量']),22.0)
    kappa=max((0.5139*lam-0.326)*np.exp(1268*(1/303.15-1/tj)),0.02)
    eta_ohm=j*(float(par['质子交换膜厚度'])/kappa+0.01e-4)
    eta_con=-rgas*tj/(4*faraday)*np.log(max(1-j/max(jlim,1e-6),1e-6))
    v=erev-eta_act-eta_ohm-eta_con-ptrans
    return v, {'ice_vol':ice_vol,'pore_ice':pore_ice,'gas_pore':gas_pore,
               'diff_o2':deff,'area_eff':area,'jlim_A_m2':jlim,
               'v_rev':erev,'eta_act':eta_act,'eta_ohm':eta_ohm,'eta_con':eta_con,
               'eta_trans':ptrans,'lambda':lam,'liquid':liq,'vapor':np.where(porous,np.minimum(mobile,m_sat),0.0)}


def _simulate(exp, par, grid, theta, keep_fields=False):
    logj0, cscale, gain, tau, cold_loss, cold_q, ramp_q=theta
    times=exp.time_s.to_numpy(float); load=exp.current_density_A_cm2.to_numpy(float)
    dt=0.2; n=len(grid['dx']); eps=grid['eps']; dx=grid['dx']
    rho_i=920.0; latent=float(par['水冻结潜热']); h=float(par['端板与外界对流换热系数'])
    bp_areal=(float(par['阳极双极板厚度'])+float(par['阴极双极板厚度']))*1980*766
    cp=(grid['cp']+bp_areal/grid['length'])*cscale
    heat_lu, heat_bc=_operator(dx,grid['k'],cp,dt,h)
    dwater=np.where(grid['mem'],1e-11,2e-8)
    water_lu,_=_operator(dx,dwater,np.ones(n),dt)
    temp=np.full(n,273.15+float(exp.temperature_C.iloc[0]))
    mobile=np.zeros(n); ice=np.zeros(n)
    mobile[grid['mem']]=float(par['初始膜含水量'])*2150*0.018/(float(par['膜当量质量'])/1000)
    ambient=temp[0]; polar=0.0; charge=0.0; last_j=float(load[0]); previous_time=0.0
    out={'time_s':times.copy(),'sim_voltage_V':np.empty(len(times)),
         'sim_temperature_C':np.empty(len(times)), 'max_ice_volume_fraction':np.empty(len(times)),
         'mean_ice_volume_fraction':np.empty(len(times)), 'min_gas_porosity':np.empty(len(times)),
         'effective_ccl_area_fraction':np.empty(len(times)), 'eta_transient_V':np.empty(len(times)),
         'eta_act_V':np.empty(len(times)), 'eta_ohm_V':np.empty(len(times)),
         'eta_con_V':np.empty(len(times))}
    fields={'temperature_C':[],'ice_volume_fraction':[],'gas_porosity':[],
            'liquid_kg_m3':[],'vapor_kg_m3':[],'solid_ice_kg_m3':[]}
    def record(idx,j_now):
        total_polar=polar+cold_loss*np.exp(-charge/cold_q)
        v,aux=_voltage(j_now,temp,mobile,ice,grid,par,total_polar,logj0)
        out['sim_voltage_V'][idx]=v
        out['sim_temperature_C'][idx]=np.average(temp,weights=dx)-273.15
        out['max_ice_volume_fraction'][idx]=np.max(aux['ice_vol'])
        out['mean_ice_volume_fraction'][idx]=np.average(aux['ice_vol'],weights=dx)
        out['min_gas_porosity'][idx]=np.min(aux['gas_pore'][grid['porous']])
        out['effective_ccl_area_fraction'][idx]=aux['area_eff']
        for key,auxkey in [('eta_transient_V','eta_trans'),('eta_act_V','eta_act'),('eta_ohm_V','eta_ohm'),('eta_con_V','eta_con')]:
            out[key][idx]=aux[auxkey]
        if keep_fields:
            for key,val in [('temperature_C',temp-273.15),('ice_volume_fraction',aux['ice_vol']),
                            ('gas_porosity',aux['gas_pore']),('liquid_kg_m3',aux['liquid']),
                            ('vapor_kg_m3',aux['vapor']),('solid_ice_kg_m3',ice)]:
                fields[key].append(val.copy())
        return v
    record(0,last_j)
    for idx in range(1,len(times)):
        t=float(times[idx]); gap=t-previous_time
        steps=max(1,int(round(gap/dt))); step=gap/steps
        if abs(step-dt)>1e-7: raise ValueError('实验时间网格与0.2s计算步长不匹配')
        j_now=float(load[idx]); j_prev=float(load[idx-1])
        for sub in range(steps):
            f=(sub+1)/steps; j_step=j_prev+f*(j_now-j_prev)
            polar=polar*np.exp(-dt/tau)+gain*np.exp(-charge/ramp_q)*max(j_step-last_j,0.0)
            charge+=j_step*dt
            source=np.zeros(n)
            source[grid['ccl']]=0.018*(j_step*1e4)/(2*96485*float(par['阴极 CL 厚度']))
            mobile=lu_solve(water_lu,mobile+dt*source)
            mobile=np.maximum(mobile,0)
            sat=np.array([_sat_water(float(tt),max(float(ee),1e-9)) if pp else 0.0
                          for tt,ee,pp in zip(temp,eps,grid['porous'])])
            unfrozen=np.maximum(mobile-sat,0.0)
            capacity=np.maximum(eps*rho_i-ice,0.0)
            freeze=np.minimum(unfrozen*(1-np.exp(-0.6*dt)),capacity)
            freeze[~grid['porous']]=0.0
            mobile-=freeze; ice+=freeze
            melt=np.where(temp>273.15,np.minimum(ice,(1-np.exp(-0.6*dt))*ice),0.0)
            mobile+=melt; ice-=melt
            v,aux=_voltage(j_step,temp,mobile,ice,grid,par,
                           polar+cold_loss*np.exp(-charge/cold_q),logj0)
            qgen=(j_step*1e4)*(1.48-v)/grid['length']
            rhs=temp+dt*(qgen+latent*(freeze-melt)/dt)/cp+heat_bc*ambient
            temp=lu_solve(heat_lu,rhs)
            last_j=j_step
        previous_time=t; record(idx,j_now)
    if keep_fields:
        out['fields']={k:np.array(v) for k,v in fields.items()}
    return out


def compute_results(logger):
    p1,p2=get_input_paths()
    par=_number_map(pd.read_excel(p1,sheet_name='参数清单'))
    exps={
        '-20℃':_read_experiment(p2,'启动温度为-20℃','-20℃'),
        '-25℃':_read_experiment(p2,'启动温度为-25℃','-25℃'),
    }
    grid=_grid(par)
    logger.emit(f'参数表={len(par)}条；实验=-20℃ {len(exps["-20℃"])}行，-25℃ {len(exps["-25℃"])}行；空间网格={len(grid["dx"])}格', 'DETAIL')
    area=float(par['燃料电池活化面积'])
    audit={label:float(np.median(d.current_A/d.current_density_A_cm2)) for label,d in exps.items()}
    logger.warning(f'附件1活化面积={area:g} cm²；附件2电流/电流密度对应面积中位数：-20℃ {audit["-20℃"]:.1f} cm²，-25℃ {audit["-25℃"]:.1f} cm²；建模采用实验电流密度列')
    train=exps['-20℃']
    def residual(theta):
        sim=_simulate(train,par,grid,theta)
        rv=(sim['sim_voltage_V']-train.voltage_V.to_numpy())/0.045
        rt=(sim['sim_temperature_C']-train.temperature_C.to_numpy())/1.5
        return np.r_[rv,rt]
    lower=np.array([-4,0.5,0,0.3,0,0.05,0.1],dtype=float)
    upper=np.array([1.3,3,6,18,0.3,10,20],dtype=float)


    objective=lambda z: float(np.dot(residual(z),residual(z)))
    global_fit=differential_evolution(objective,list(zip(lower,upper)),seed=20260924,
                                      popsize=2,maxiter=2,polish=False,workers=1,
                                      updating='immediate',tol=0.03)
    fit=least_squares(residual,x0=global_fit.x,bounds=(lower,upper),
                      method='trf',max_nfev=24,ftol=1e-4,xtol=1e-4)
    logger.status(f'差分进化搜索：评估{global_fit.nfev}次，目标={global_fit.fun:.4f}；信赖域精修')
    logger.status(f'参数校准完成：函数评估{fit.nfev}次；收敛={fit.success}；目标={np.linalg.norm(fit.fun):.3f}')
    theta=fit.x
    if theta[0]>1.25:
        logger.warning('有效交换电流密度接近设定上界；其与经验调理极化项存在辨识耦合，不能作为独立物性常数解释')
    sims={label:_simulate(d,par,grid,theta,keep_fields=True) for label,d in exps.items()}
    full=[]; tables={}; metrics=[]
    for label,d in exps.items():
        sim=sims[label]
        fr=d.copy()
        for key in ('sim_voltage_V','sim_temperature_C','max_ice_volume_fraction','mean_ice_volume_fraction',
                    'min_gas_porosity','effective_ccl_area_fraction','eta_transient_V','eta_act_V','eta_ohm_V','eta_con_V'):
            fr[key]=sim[key]
        fr['voltage_relative_error_pct']=(fr.sim_voltage_V-fr.voltage_V)/fr.voltage_V.abs()*100
        fr['temperature_relative_error_pct']=(fr.sim_temperature_C-fr.temperature_C)/fr.temperature_C.abs()*100
        full.append(fr)
        rows=[]
        for tt in (0,5,10,15,20,25,30,35):
            ix=int(np.argmin(abs(d.time_s.to_numpy()-tt)))
            ev=round(float(d.voltage_V.iloc[ix]),3); et=round(float(d.temperature_C.iloc[ix]),2)
            sv=float(sim['sim_voltage_V'][ix]); st=float(sim['sim_temperature_C'][ix])
            rows.append({'时间 /s':tt,'实验电压 /V':ev,'模型电压 /V':round(sv,3),
                         '电压相对误差/%':round((sv-ev)/abs(ev)*100,2),
                         '实验温度 /℃':et,'模型温度 /℃':round(st,2),
                         '温度相对误差/%':round((st-et)/abs(et)*100,2),
                         '模型最大冰体积分数':round(float(sim['max_ice_volume_fraction'][ix]),5)})
        tables[label]=pd.DataFrame(rows)
        metrics.append({'工况':label,'角色':'校准' if label=='-20℃' else '跨温度验证',
                        '电压MAE/V':float(np.mean(abs(fr.sim_voltage_V-fr.voltage_V))),
                        '电压RMSE/V':float(np.sqrt(np.mean((fr.sim_voltage_V-fr.voltage_V)**2))),
                        '温度MAE/℃':float(np.mean(abs(fr.sim_temperature_C-fr.temperature_C))),
                        '温度RMSE/℃':float(np.sqrt(np.mean((fr.sim_temperature_C-fr.temperature_C)**2))),
                        '最大冰体积分数':float(np.max(fr.max_ice_volume_fraction))})
    metrics=pd.DataFrame(metrics)
    logger.core(f'-20℃校准：电压MAE={metrics.iloc[0]["电压MAE/V"]:.4f} V，温度MAE={metrics.iloc[0]["温度MAE/℃"]:.3f} ℃')
    logger.core(f'-25℃跨温度验证：电压MAE={metrics.iloc[1]["电压MAE/V"]:.4f} V，温度MAE={metrics.iloc[1]["温度MAE/℃"]:.3f} ℃')
    logger.core(f'两工况模型最大冰体积分数={metrics["最大冰体积分数"].max():.5f}；冰无实验观测，仅作模型预测')
    params=pd.DataFrame([{'参数':'参考交换电流密度','值':10**theta[0],'单位':'A/m²'},
                         {'参数':'等效热容倍率','值':theta[1],'单位':'-'},
                         {'参数':'升载暂态极化增益','值':theta[2],'单位':'V/(A/cm²)'},
                         {'参数':'暂态极化松弛时间','值':theta[3],'单位':'s'},
                         {'参数':'初始调理极化损失','值':theta[4],'单位':'V'},
                         {'参数':'调理极化电荷尺度','值':theta[5],'单位':'C/cm²'},
                         {'参数':'升载极化衰减电荷尺度','值':theta[6],'单位':'C/cm²'},
                         {'参数':'液水冻结一阶速率','值':0.6,'单位':'1/s（固定，未由实验辨识）'}])
    return {'full':pd.concat(full,ignore_index=True),'tables':tables,'metrics':metrics,
            'params':params,'sims':sims,'grid':grid,'fit_success':bool(fit.success),
            'audit_area':audit,'area':area}


def validate_computation(results,logger):
    metrics=results['metrics']
    if float(metrics['电压MAE/V'].max())>0.08 or float(metrics['温度MAE/℃'].max())>1.0:
        raise ValueError('电压或温度误差过大，当前结果不应交付')
    for label,frame in results['tables'].items():
        if len(frame)!=8 or not np.all(np.isfinite(frame.select_dtypes('number'))):
            raise ValueError(f'{label}题定表行数或数值异常')
    for label,sim in results['sims'].items():
        if np.min(sim['max_ice_volume_fraction']) < -1e-9 or np.max(sim['max_ice_volume_fraction']) >= 1:
            raise ValueError(f'{label}冰体积分数不在物理范围')
        if np.min(sim['min_gas_porosity'])<=0 or np.min(sim['effective_ccl_area_fraction'])<=0:
            raise ValueError(f'{label}孔隙率或活性面积异常')
        fields=sim['fields']
        if np.min(fields['liquid_kg_m3']) < -1e-6 or np.min(fields['vapor_kg_m3']) < -1e-6 or np.min(fields['solid_ice_kg_m3']) < -1e-6:
            raise ValueError(f'{label}相态质量为负')
    logger.status('计算验收通过：两张8行题定表、三相非负、孔隙率和活性面积为正')


def get_figure_contract():
    return {
        'evidence': 'two-condition measured/simulated voltage and temperature; simulated ice, porosity and spatial ice/temperature field',
        'layout': 'new evidence-led Chinese composite figures at 600 DPI; no old-image reuse',
    }


def get_render_dependency_names():
    return ['matplotlib', 'numpy']


def render_figures(results, output_dir, logger):
    return render_question('Q1', results, output_dir, logger)

from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
INK = '#203047'
TEAL = '#087F8C'
ORANGE = '#D38343'
PLUM = '#785988'
GREY = '#687988'
PALE = '#E7EDEF'

def _style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'Noto Sans SC', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 9, 'axes.titlesize': 11, 'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': '#A9B6BF', 'axes.spines.top': False, 'axes.spines.right': False, 'grid.color': PALE, 'grid.linewidth': 0.6, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'figure.dpi': 120})

def _save(fig, output_dir, q, name):
    output_dir = Path(output_dir)
    path = output_dir / name
    fig.savefig(path, dpi=600)
    plt.close(fig)
    dest = output_dir.parent / 'figures' / q
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest / name)
    return path

def _panel(ax, title):
    ax.set_title(title, loc='left', weight='bold', pad=10)
    ax.grid(alpha=0.75)
    ax.set_axisbelow(True)

def render_question(q, results, output_dir, logger):
    _style()
    return _q1(results, output_dir, logger)

def _q1(r, out, log):
    full = r['full']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for label, color in [('-20℃', TEAL), ('-25℃', ORANGE)]:
        d = full[full['condition'] == label]
        ax[0].scatter(d['voltage_V'], d['sim_voltage_V'], s=12, color=color, alpha=0.65, label=label)
        residual = (d['sim_voltage_V'] - d['voltage_V']).to_numpy()
        ax[1].plot(d['time_s'], residual * 1000, lw=1.6, color=color, label=label)
    bounds = [0.45, 0.85]
    ax[0].plot(bounds, bounds, color=INK, lw=1, ls='--')
    ax[0].set(xlabel='实验电压 / V', ylabel='模型电压 / V')
    ax[1].axhline(0, color=INK, lw=0.9)
    ax[1].set(xlabel='时间 / s', ylabel='电压残差 / mV')
    _panel(ax[0], 'a  两组工况的电压一致性')
    _panel(ax[1], 'b  跨温度残差随时间变化')
    for a in ax:
        a.legend(frameon=False)
    paths = [_save(fig, out, 'Q1', 'Q1_电压一致性与跨温度残差.png')]
    space = r['grid']['x'] * 1000000.0
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for label, color in [('-20℃', TEAL), ('-25℃', ORANGE)]:
        d = full[full['condition'] == label]
        ax[0].plot(d['time_s'], d['max_ice_volume_fraction'], color=color, lw=2, label=label)
    sim = r['sims']['-25℃']
    ice = sim['fields']['ice_volume_fraction']
    time = full[full['condition'] == '-25℃']['time_s'].to_numpy()
    mesh = ax[1].pcolormesh(time, space, ice.T, shading='auto', cmap='magma', rasterized=True)
    fig.colorbar(mesh, ax=ax[1], label='局部冰体积分数')
    ax[0].set(xlabel='时间 / s', ylabel='最大局部冰体积分数')
    ax[1].set(xlabel='时间 / s', ylabel='膜电极厚度坐标 / μm')
    _panel(ax[0], 'a  冰相峰值形成过程')
    _panel(ax[1], 'b  −25℃工况的空间结冰位置')
    ax[0].legend(frameon=False)
    paths.append(_save(fig, out, 'Q1', 'Q1_冰相时空演化.png'))
    log.status('Q1重设计证据图生成：2张600 DPI PNG')
    return paths


def export_tables(results, output_dir, logger):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    out=Path(output_dir)
    files=[]
    for name,df in [
        ('Q1_全时段验证.csv',results['full']),
        ('Q1_题定表_20℃.csv',results['tables']['-20℃']),
        ('Q1_题定表_25℃.csv',results['tables']['-25℃']),
        ('Q1_误差指标.csv',results['metrics']),
        ('Q1_模型参数.csv',results['params']),
    ]:
        files.append(write_dataframe_csv(df,out/name))
    grid=results['grid']; spatial=[]
    for label,sim in results['sims'].items():
        nt=len(sim['time_s']); nx=len(grid['x'])
        fields=sim['fields']
        spatial.append(pd.DataFrame({
            '工况':[label]*(nt*nx),
            '时间/s':np.repeat(sim['time_s'],nx),
            '位置/μm':np.tile(grid['x']*1e6,nt),
            '层':np.tile(grid['layer'],nt),
            '局部温度/℃':fields['temperature_C'].ravel(),
            '局部冰体积分数':fields['ice_volume_fraction'].ravel(),
            '有效气孔率':fields['gas_porosity'].ravel(),
            '水蒸气质量浓度/kg_m3':fields['vapor_kg_m3'].ravel(),
            '液态水质量浓度/kg_m3':fields['liquid_kg_m3'].ravel(),
            '冰质量浓度/kg_m3':fields['solid_ice_kg_m3'].ravel(),
        }))
    files.append(write_dataframe_csv(pd.concat(spatial,ignore_index=True),out/'Q1_空间场.csv'))
    note='实验列来自附件2对应时刻，温度按题定表保留2位；相对误差=(模型−实验)/|实验|×100%。冰为模型预测，非实验测量。'
    metric=results['metrics'].copy()
    for col in metric.select_dtypes('number').columns:
        metric[col]=metric[col].map(lambda v: f'{v:.4f}')
    md=write_markdown_tables([
        ('表1 一维单电池瞬态自冷启动模型预测与实验验证结果（−20 ℃，n=8）',results['tables']['-20℃'],note),
        ('表2 一维单电池瞬态自冷启动模型预测与实验验证结果（−25 ℃，n=8）',results['tables']['-25℃'],note),
        ('两工况模型误差与预测能力对比（每组184个时刻）',metric,'−20 ℃用于参数校准；−25 ℃保持参数不变作跨温度验证。'),
    ],out/'表格输出.md')
    files.append(md)
    logger.status(f'已导出{len(files)-1}个CSV与表格输出.md')
    return files


def run_pipeline() -> int:
    code_path = Path(__file__).resolve()
    output_dir = code_path.parent / QUESTION_ID
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = CompactLogger(output_dir / RUNTIME_OUTPUT_NAME)
    try:
        logger.status(
            f"开始执行 {QUESTION_ID}；计算工程版本={COMPUTE_ENGINEERING_VERSION}；"
            f"绘图工程版本={RENDER_ENGINEERING_VERSION}。"
        )
        input_paths = [Path(path).resolve() for path in get_input_paths()]
        analysis_contract = get_analysis_contract()
        figure_contract = get_figure_contract()
        compute_dependency_signature = _dependency_signature(
            list(get_compute_dependency_names())
        )
        render_dependency_signature = _dependency_signature(
            list(get_render_dependency_names())
        )
        input_fingerprints = _fingerprint_inputs(input_paths)

        compute_key = stable_digest(
            {
                "question": QUESTION_ID,
                "engineering": COMPUTE_ENGINEERING_VERSION,
                "compute_source": _source_region_hash("COMPUTE_BLOCK"),
                "inputs": input_fingerprints,
                "analysis_contract": analysis_contract,
                "figure_evidence": figure_contract.get("evidence", {}),
                "dependencies": compute_dependency_signature,
            }
        )
        cache_root = _resolve_cache_root(output_dir)
        compute_cache_dir = cache_root / "compute" / compute_key
        results = _load_compute_cache(compute_cache_dir)

        if results is None:
            logger.cache(f"计算缓存未命中：{compute_key[:12]}；开始统计计算。")
            results = compute_results(logger)
            validate_computation(results, logger)
            _save_compute_cache(compute_cache_dir, results, cache_root)
            logger.cache(f"计算缓存已写入：{compute_key[:12]}。")
        else:
            try:
                validate_computation(results, logger)
                logger.cache(f"计算缓存命中：{compute_key[:12]}；跳过统计计算。")
            except Exception:
                logger.warning("缓存计算验收失败；废弃缓存并重新计算。")
                _safe_remove_cache_dir(compute_cache_dir, cache_root)
                results = compute_results(logger)
                validate_computation(results, logger)
                _save_compute_cache(compute_cache_dir, results, cache_root)

        _clear_previous_managed_outputs(output_dir, cache_root)
        table_paths = _validate_output_paths(
            export_tables(results, output_dir, logger),
            output_dir,
            {".csv", ".md"},
        )

        render_key = stable_digest(
            {
                "compute_key": compute_key,
                "engineering": RENDER_ENGINEERING_VERSION,
                "render_source": _source_region_hash("RENDER_BLOCK"),
                "figure_contract": figure_contract,
                "dependencies": render_dependency_signature,
            }
        )
        render_cache_dir = cache_root / "render" / render_key
        figure_paths = _load_render_cache(render_cache_dir, output_dir)
        if figure_paths is None:
            logger.cache(f"绘图缓存未命中：{render_key[:12]}；开始生成图片。")
            figure_paths = _validate_output_paths(
                render_figures(results, output_dir, logger),
                output_dir,
                {".png"},
            )
            _save_render_cache(render_cache_dir, figure_paths, cache_root)
            logger.cache(f"绘图缓存已写入：{render_key[:12]}。")
        else:
            figure_paths = _validate_output_paths(
                figure_paths, output_dir, {".png"}
            )
            logger.cache(f"绘图缓存命中：{render_key[:12]}；跳过图片渲染。")

        managed_paths = [*table_paths, *figure_paths]
        _write_managed_inventory(managed_paths, cache_root)

        logger.emit(f"最终代码：{code_path}", "DETAIL")
        logger.emit(f"TXT：{logger.path}", "DETAIL")
        for path in table_paths:
            label = "Markdown" if path.suffix.lower() == ".md" else "CSV"
            logger.emit(f"{label}：{path}", "DETAIL")
        for path in figure_paths:
            logger.emit(f"图片：{path}", "DETAIL")
        logger.status(
            f"{QUESTION_ID} 完成；计算键={compute_key[:12]}，绘图键={render_key[:12]}。"
        )
        return 0
    except Exception as exc:
        logger.emit(f"执行失败：{type(exc).__name__}: {exc}", "ERROR")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(run_pipeline())
