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


QUESTION_ID = "Q2"
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


def _number_map(frame):
    return {str(r[1]).strip(): r[2] for r in frame.itertuples(index=False, name=None)}

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

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import differential_evolution, minimize, minimize_scalar


def _attachment(name):
    root = Path(__file__).resolve().parent
    candidates = [root/name, root/'B题'/name, root.parent/name, root.parent/'B题'/name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'请将 {name} 放在脚本目录或其上一级目录。')


def get_input_paths():
    root=Path(__file__).resolve().parent
    return [_attachment('附件1.xlsx'),root/'Q1'/'Q1_模型参数.csv']


def get_analysis_contract():
    return {
        'question':'Q2',
        'upstream':'Q1 finite-volume constitutive relations and frozen calibrated parameters',
        'physics':'five independently resolved cells; eq.(2) intercell conduction and eq.(3) h=40 end boundary, endplate heat storage as boundary state',
        'strategies':['constant','strictly rising linear ramp and hold','three strictly rising steps and hold'],
        'constraints':{'jmax_A_cm2':0.5,'qmax_C_cm2':20,'Tcell_C_strictly_gt':0,
                       'local_ice_strictly_lt':0.99,'Vpath_V_gte':0.30},
        'initial_temperature_C':-10,
        'ambient':'equal to initial temperature in each scanned case, per planning assumption',
        'search':'coarse five-layer Sobol plus feasible-first pattern search; full 27-volume/0.1s recalc; 0.1℃ threshold refinement; 0.05s confirmation',
        'seed':20260924,
    }


def get_compute_dependency_names():
    return ['numpy','pandas','scipy','openpyxl']


def _coarse_grid(full):
    names=['aGDL','aCL','PEM','cCL','cGDL']
    dx=np.array([float(np.sum(full['dx'][full['layer']==name])) for name in names])
    eps=np.array([float(np.mean(full['eps'][full['layer']==name])) for name in names])
    k=np.array([float(np.mean(full['k'][full['layer']==name])) for name in names])
    cp=np.array([float(np.mean(full['cp'][full['layer']==name])) for name in names])
    layer=np.array(names)
    return {'layer':layer,'dx':dx,'x':np.cumsum(dx)-dx/2,'eps':eps,'k':k,'cp':cp,
            'length':float(np.sum(dx)),'porous':eps>0,'mem':layer=='PEM','ccl':layer=='cCL',
            'cath':np.isin(layer,['cCL','cGDL'])}


def _calibrated_theta(path):
    values=pd.read_csv(path).set_index('参数')['值']
    return np.array([np.log10(float(values['参考交换电流密度'])),
                     float(values['等效热容倍率']),float(values['升载暂态极化增益']),
                     float(values['暂态极化松弛时间']),float(values['初始调理极化损失']),
                     float(values['调理极化电荷尺度']),float(values['升载极化衰减电荷尺度'])])


def _context(par,grid,theta,dt):
    length=grid['length']; dx=grid['dx']; n=len(dx)
    bp_areal=(float(par['阳极双极板厚度'])+float(par['阴极双极板厚度']))*1980*766
    cp=(grid['cp']+bp_areal/length)*theta[1]
    heat_lu,_=_operator(dx,grid['k'],cp,dt,h=0)
    water_lu,_=_operator(dx,np.where(grid['mem'],1e-11,2e-8),np.ones(n),dt)


    R_mea=float(np.sum(dx/grid['k']))
    R_bp=(float(par['阳极双极板厚度'])+float(par['阴极双极板厚度']))/95
    G_inter=1/(R_mea+R_bp)

    R_endplate=float(par['端板厚度'])/float(par['导热系数'])
    G_end=1.0/(0.5*R_mea+0.5*R_bp+0.5*R_endplate)
    C_end=float(par['端板厚度'])*float(par['密度'])*float(str(par['比热容']).replace('\xa0','').strip())
    return {'par':par,'grid':grid,'theta':theta,'dt':dt,'cp':cp,'heat_lu':heat_lu,
            'water_lu':water_lu,'G_inter':G_inter,'G_end':G_end,'C_end':C_end,
            'h':float(par['端板与外界对流换热系数'])}


def _current(kind,x,t):
    if kind=='恒流策略': return float(x[0])
    if kind=='线性升载': return float(x[0]+(x[1]-x[0])*min(t/x[2],1.0))

    j1=float(x[0]); j2=j1+float(x[1])*(0.5-j1);j3=j2+float(x[2])*(0.5-j2)
    if t<x[3]: return j1
    if t<x[3]+x[4]: return j2
    return j3


def _simulate_stack(ctx,kind,x,T0,record=False,max_time=180):
    p=ctx['par']; g=ctx['grid']; theta=ctx['theta']; dt=ctx['dt']; dx=g['dx']; n=len(dx)
    temp=np.full((5,n),273.15+T0,dtype=float)
    mobile=np.zeros((5,n)); ice=np.zeros((5,n))
    mobile[:,g['mem']]=float(p['初始膜含水量'])*2150*0.018/(float(p['膜当量质量'])/1000)
    end=np.full(2,273.15+T0,dtype=float); ambient=273.15+T0
    polar=0.0; charge=0.0; last_j=_current(kind,x,0)
    min_v=np.inf; max_ice=0.0; max_j=last_j; first_fail_cell=0; reason='时限内未成功'
    rows=[]; cell_ice=np.zeros(5)
    def snapshot(t,j,volts,icecell,meanT):
        row={'时间/s':t,'电流密度/A_cm2':j,'累计电荷量/C_cm2':charge,
             '最低温度/℃':float(min(meanT)-273.15),'最低电压/V':float(min(volts)),
             '最大冰体积分数':float(max(icecell)),'左端板温度/℃':float(end[0]-273.15),
             '右端板温度/℃':float(end[1]-273.15)}
        for kk in range(5):
            row[f'电池{kk+1}温度/℃']=float(meanT[kk]-273.15)
            row[f'电池{kk+1}电压/V']=float(volts[kk])
            row[f'电池{kk+1}最大冰体积分数']=float(icecell[kk])
        rows.append(row)
    def electrical(j,charge,polar):
        v=[]; ai=[]
        loss=polar+theta[4]*np.exp(-charge/theta[5])
        for kk in range(5):
            vv,aa=_voltage(j,temp[kk],mobile[kk],ice[kk],g,p,loss,theta[0])
            v.append(vv);ai.append(aa)
        return np.array(v),ai
    volts,aux=electrical(last_j,charge,polar)
    meanT=np.average(temp,axis=1,weights=dx)
    cell_ice=np.array([float(np.max(a['ice_vol'])) for a in aux])
    min_v=float(np.min(volts)); first_fail_cell=int(np.argmin(volts))+1
    if record:snapshot(0,last_j,volts,cell_ice,meanT)
    if min_v<0.30:
        return {'success':False,'reason':'初始电压低于0.30 V','failure_cell':first_fail_cell,
                'time_s':0.0,'charge_C_cm2':0.0,'max_j_A_cm2':last_j,'min_voltage_V':min_v,
                'max_ice_fraction':0.0,'min_temp_C':T0,'end_cell_ice':cell_ice,
                'history':pd.DataFrame(rows) if record else None}
    steps=int(np.ceil(max_time/dt))
    for ii in range(1,steps+1):
        t=ii*dt; j=_current(kind,x,t); prev_charge=charge
        polar=polar*np.exp(-dt/theta[3])+theta[2]*np.exp(-charge/theta[6])*max(j-last_j,0.0)
        charge+=j*dt; max_j=max(max_j,j)
        previous_minT=float(np.min(meanT)-273.15)
        source=np.zeros(n)
        source[g['ccl']]=0.018*(j*1e4)/(2*96485*float(p['阴极 CL 厚度']))
        freeze=np.zeros((5,n)); melt=np.zeros((5,n))
        for kk in range(5):
            mobile[kk]=np.maximum(lu_solve(ctx['water_lu'],mobile[kk]+dt*source),0)
            sat=np.array([_sat_water(float(tt),max(float(ee),1e-9)) if pp else 0.0
                          for tt,ee,pp in zip(temp[kk],g['eps'],g['porous'])])
            free=np.maximum(mobile[kk]-sat,0)
            freeze[kk]=np.minimum(free*(1-np.exp(-0.6*dt)),np.maximum(g['eps']*920-ice[kk],0))
            freeze[kk,~g['porous']]=0
            freeze[kk,temp[kk]>=273.15]=0
            mobile[kk]-=freeze[kk];ice[kk]+=freeze[kk]
            melt[kk]=np.where(temp[kk]>=273.15,ice[kk]*(1-np.exp(-0.6*dt)),0)
            mobile[kk]+=melt[kk];ice[kk]-=melt[kk]
        volts,aux=electrical(j,charge,polar)
        cell_ice=np.array([float(np.max(a['ice_vol'])) for a in aux])
        max_ice=max(max_ice,float(np.max(cell_ice)))
        if float(np.min(volts))<min_v:
            min_v=float(np.min(volts));first_fail_cell=int(np.argmin(volts))+1
        meanT=np.average(temp,axis=1,weights=dx)
        inter=ctx['G_inter']*(meanT[:-1]-meanT[1:])
        flux=np.zeros(5);flux[:-1]-=inter;flux[1:]+=inter
        left=ctx['G_end']*(end[0]-meanT[0]);right=ctx['G_end']*(end[1]-meanT[-1])

        flux[0]+=left-ctx['h']*(meanT[0]-ambient)
        flux[-1]+=right-ctx['h']*(meanT[-1]-ambient)
        end[0]+=dt*(-left-ctx['h']*(end[0]-ambient))/ctx['C_end']
        end[1]+=dt*(-right-ctx['h']*(end[1]-ambient))/ctx['C_end']
        for kk in range(5):
            qgen=j*1e4*(1.48-volts[kk])/g['length']
            rhs=temp[kk]+dt*(qgen+flux[kk]/g['length']+float(p['水冻结潜热'])*(freeze[kk]-melt[kk])/dt)/ctx['cp']
            temp[kk]=lu_solve(ctx['heat_lu'],rhs)
        meanT=np.average(temp,axis=1,weights=dx)
        current_minT=float(np.min(meanT)-273.15)
        if record:snapshot(t,j,volts,cell_ice,meanT)
        if min_v<0.30:
            reason='电压下限';break
        if max_ice>=0.99:
            first_fail_cell=int(np.argmax(cell_ice))+1;reason='严重冰堵';break
        if current_minT>0:
            frac=np.clip(-previous_minT/max(current_minT-previous_minT,1e-9),0,1)
            ts=t-dt+dt*frac
            qsuccess=prev_charge+j*dt*frac
            if qsuccess<=20+1e-9:
                return {'success':True,'reason':'五片同时满足','failure_cell':0,'time_s':ts,
                        'charge_C_cm2':qsuccess,'max_j_A_cm2':max_j,'min_voltage_V':min_v,
                        'max_ice_fraction':max_ice,'min_temp_C':current_minT,
                        'end_cell_ice':cell_ice.copy(),'history':pd.DataFrame(rows) if record else None}
        if charge>20:
            first_fail_cell=int(np.argmin(meanT))+1;reason='累计电荷上限'
            cap_fraction=np.clip((20-prev_charge)/max(charge-prev_charge,1e-12),0,1)
            cap_minT=previous_minT+cap_fraction*(current_minT-previous_minT)
            charge=20.0
            break
        last_j=j
    else:
        first_fail_cell=int(np.argmin(meanT))+1
    return {'success':False,'reason':reason,'failure_cell':first_fail_cell,
            'time_s':min(float(ii*dt),max_time),'charge_C_cm2':charge,'max_j_A_cm2':max_j,
            'min_voltage_V':min_v,'max_ice_fraction':max_ice,'min_temp_C':cap_minT if reason=='累计电荷上限' else float(np.min(meanT)-273.15),
            'end_cell_ice':cell_ice.copy(),'history':pd.DataFrame(rows) if record else None}


def _bounds(kind):
    if kind=='恒流策略':return [(0.005,0.5)]
    if kind=='线性升载':return [(0.005,0.4),(0.45,0.5),(5,50)]
    return [(0.005,0.4),(0.10,0.80),(0.10,1.0),(2,45),(2,45)]


def _score(res):
    if res['success']:
        return float(res['time_s'])+0.002*float(res['charge_C_cm2'])
    return 210+5*max(0,-res['min_temp_C'])+700*max(0,0.30-res['min_voltage_V'])+80*max(0,res['max_ice_fraction']-0.99)


def _search(ctx,kind,T0,seed,quick=False,starting=None):


    from scipy.stats import qmc
    bounds=np.asarray(_bounds(kind),dtype=float)
    low,high=bounds[:,0],bounds[:,1]
    cache={}
    def evaluate(z):
        z=np.clip(np.asarray(z,dtype=float),low,high)
        key=tuple(np.round(z,8))
        if key not in cache:
            result=_simulate_stack(ctx,kind,z,T0)
            cache[key]=(_score(result),bool(result['success']))
        return cache[key]
    candidates=[]
    if starting is not None:candidates.append(np.clip(np.asarray(starting,dtype=float),low,high))
    if kind=='恒流策略':
        candidates.extend(np.array([j]) for j in np.linspace(low[0],high[0],9 if quick else 17))
    else:
        sampler=qmc.Sobol(d=len(low),scramble=True,seed=seed)
        unit=sampler.random_base2(m=3 if quick else 4)
        candidates.extend(low+unit*(high-low))
        candidates.append((low+high)/2)
    best=min(candidates,key=lambda z:(not evaluate(z)[1],evaluate(z)[0]))
    score,feasible=evaluate(best)
    step=(high-low)*(0.22 if quick else 0.28)
    for _ in range(1 if quick else 3):
        improved=False
        for axis in range(len(low)):
            for direction in (-1,1):
                trial=best.copy();trial[axis]+=direction*step[axis]
                trial=np.clip(trial,low,high)
                value,ok=evaluate(trial)
                if (not ok,value)<(not feasible,score):
                    best,score,feasible=trial,value,ok
                    improved=True
        step*=0.5 if not improved else 0.7
    return best,score


def _params_text(kind,x,ts):
    if kind=='恒流策略':return f'j0={x[0]:.4f} A·cm⁻²'
    if kind=='线性升载':return f'jmin={x[0]:.4f}, k={(x[1]-x[0])/x[2]:.5f} A·cm⁻²·s⁻¹, t1={x[2]:.1f} s'
    j1=x[0];j2=j1+x[1]*(0.5-j1);j3=j2+x[2]*(0.5-j2)
    return (f'j1={j1:.4f}, j2={j2:.4f}, j3={j3:.4f} A·cm⁻²; '
            f't1={x[3]:.1f}, t2={x[4]:.1f}, t3={max(0,ts-x[3]-x[4]):.1f} s')


def compute_results(logger):
    p1,p2=get_input_paths();par=_number_map(pd.read_excel(p1,sheet_name='参数清单'))
    theta=_calibrated_theta(p2);full=_grid(par);coarse=_coarse_grid(full)
    search_ctx=_context(par,coarse,theta,1.0)
    final_ctx=_context(par,full,theta,0.1)
    logger.emit(f'五片×{len(full["dx"])}空间网格；搜索用五层简化网格，最终用27格和0.1 s回算；片间导热导纳={search_ctx["G_inter"]:.1f} W·m⁻²·K⁻¹；端部等效换热导纳={final_ctx["G_end"]:.1f} W·m⁻²·K⁻¹', 'DETAIL')
    families=['恒流策略','线性升载','分段阶梯加载']
    best={}; summary=[]; trajectories=[]
    for kk,kind in enumerate(families):
        x,_=_search(search_ctx,kind,-10,20260924+kk)
        res=_simulate_stack(final_ctx,kind,x,-10,record=True)
        if not res['success']:

            logger.warning(f'{kind}粗网格候选未通过完整网格复算，启用完整网格局部搜索')
            def obj(z):return _score(_simulate_stack(final_ctx,kind,z,-10))
            local=minimize(obj,x,method='Nelder-Mead',options={'maxiter':25,'xatol':0.002,'fatol':0.01})
            xx=np.clip(local.x,[b[0] for b in _bounds(kind)],[b[1] for b in _bounds(kind)])
            rr=_simulate_stack(final_ctx,kind,xx,-10,record=True)
            if _score(rr)<_score(res):x,res=xx,rr
            if kind!='恒流策略' and not res['success']:
                global_fit=differential_evolution(obj,_bounds(kind),seed=20264000+kk,
                                                  popsize=3,maxiter=4,polish=False,
                                                  workers=1,updating='immediate')
                rr=_simulate_stack(final_ctx,kind,global_fit.x,-10,record=True)
                if _score(rr)<_score(res):x,res=global_fit.x,rr
        best[kind]=x
        if res['history'] is not None:
            tr=res['history'].copy();tr.insert(0,'加载策略',kind);trajectories.append(tr)
        summary.append({'加载策略':kind,'最优加载参数':_params_text(kind,x,res['time_s']),
                        '启动时间/s':round(res['time_s'],2) if res['success'] else np.nan,
                        '累计电荷量 /C·cm⁻²':round(res['charge_C_cm2'],3),
                        '最大电流密度/A·cm⁻²':round(res['max_j_A_cm2'],4),
                        '最低电压/V':round(res['min_voltage_V'],4),
                        '最大冰体积分数':round(res['max_ice_fraction'],5),
                        '启动结果':'成功' if res['success'] else '失败：'+res['reason'],
                        '失败关键电池':res['failure_cell'] if not res['success'] else ''})
        logger.core(f'{kind}：'+(f'成功，t={res["time_s"]:.2f} s，q={res["charge_C_cm2"]:.2f} C·cm⁻²，Vmin={res["min_voltage_V"]:.3f} V' if res['success'] else f'失败，{res["reason"]}'))
    table=pd.DataFrame(summary)
    success=table[table['启动结果']=='成功']
    if success.empty:
        logger.warning('−10℃三类策略均未通过完整模型；最低初温不能定义为≤−10℃，需报告不可行')

    scan=[]; scan_best={}; coarse_temps=np.arange(-40,-9,5,dtype=float)
    for T0 in coarse_temps:
        picks=[]
        for kk,kind in enumerate(families):
            x,_=_search(search_ctx,kind,float(T0),20261000+int(T0)+kk,quick=True,starting=best[kind])
            r=_simulate_stack(final_ctx,kind,x,float(T0))
            picks.append((kind,x,r))
        feasible=[p for p in picks if p[2]['success']]
        if feasible:
            chosen=min(feasible,key=lambda p:p[2]['time_s'])
        else:
            chosen=min(picks,key=lambda p:_score(p[2]))
        kind,x,r=chosen;scan_best[float(T0)]=(kind,x,r)
        scan.append({'初始温度/℃':float(T0),'可行':'是' if r['success'] else '否',
                     '最佳策略':kind,'启动时间/s':round(r['time_s'],2) if r['success'] else np.nan,
                     '累计电荷量/C·cm⁻²':round(r['charge_C_cm2'],3),
                     '最低电压/V':round(r['min_voltage_V'],4),
                     '最大冰体积分数':round(r['max_ice_fraction'],5),
                     '最低单片温度/℃':round(r['min_temp_C'],3),
                     '主导失败原因':'' if r['success'] else r['reason'],
                     '关键电池':'' if r['success'] else r['failure_cell']})
    scan=pd.DataFrame(scan)
    logger.status('初温粗扫描完成：'+', '.join(f'{r[0]:.0f}℃{r[1]}' for r in scan[['初始温度/℃','可行']].itertuples(index=False,name=None)))
    feasible_t=scan.loc[scan['可行']=='是','初始温度/℃'].to_numpy()
    critical=None; failure_detail=None; critical_history=None
    if len(feasible_t):
        lowest=float(np.min(feasible_t))
        below=scan.loc[scan['初始温度/℃']<lowest,'初始温度/℃'].to_numpy()
        if len(below):
            low=float(max(below));high=lowest

            coarse_monotone=all(scan.loc[scan['初始温度/℃']>=lowest,'可行']=='是')
            if coarse_monotone:
                for tt in np.arange(low+1,high,1,dtype=float):
                    picks=[]
                    for kk,kind in enumerate(families):
                        anchor=scan_best[high][1]
                        start=anchor if scan_best[high][0]==kind else best[kind]
                        x,_=_search(search_ctx,kind,float(tt),20262000+int(tt)+kk,quick=True,starting=start)
                        r=_simulate_stack(final_ctx,kind,x,float(tt))
                        picks.append((kind,x,r))
                    feasible=[z for z in picks if z[2]['success']]
                    chosen=min(feasible,key=lambda z:z[2]['time_s']) if feasible else min(picks,key=lambda z:_score(z[2]))
                    scan_best[float(tt)]=chosen
                    if chosen[2]['success']:high=min(high,float(tt))
                    else:low=float(tt)

                for tt0 in np.arange(low+0.1,high,0.1):
                    tt=round(float(tt0),1)
                    picks=[]
                    for strategy in families:
                        anchor=scan_best[high][1] if scan_best[high][0]==strategy else best[strategy]
                        trial=_simulate_stack(final_ctx,strategy,anchor,tt)
                        picks.append((strategy,anchor,trial))
                    feasible=[z for z in picks if z[2]['success']]
                    chosen=min(feasible,key=lambda z:z[2]['time_s']) if feasible else min(picks,key=lambda z:_score(z[2]))
                    scan_best[tt]=chosen
                    if chosen[2]['success']:high=min(high,tt)
                    else:low=max(low,tt)
                lowest=high
            else:
                logger.warning('初温可行性非单调，保留粗扫描可行区间，不进行单区间二分')
        kind,x,r=scan_best[lowest]
        verification_ctx=_context(par,full,theta,0.05)
        verified=_simulate_stack(verification_ctx,kind,x,lowest)
        if not verified['success']:
            logger.warning('0.05 s时间步未复现临界可行解，临界温度上调0.1℃直至验证通过')
            for _ in range(10):
                lowest=round(lowest+0.1,1)
                kind,x,_=scan_best.get(lowest,(kind,x,None))
                verified=_simulate_stack(verification_ctx,kind,x,lowest)
                if verified['success']:break
            if not verified['success']:raise ValueError('临界初温在细时间步下无法验证')
            r=_simulate_stack(final_ctx,kind,x,lowest)
        critical={'最低初始温度/℃':lowest,'温度分辨率/℃':0.1 if len(below) else 5.0,
                  '加载策略':kind,'加载参数':_params_text(kind,x,verified['time_s']),
                  '启动时间/s':verified['time_s'],'累计电荷量/C·cm⁻²':verified['charge_C_cm2'],
                  '最低电压/V':verified['min_voltage_V'],'最大冰体积分数':verified['max_ice_fraction'],
                  '复核时间步/s':0.05,'复核累计电荷量/C·cm⁻²':verified['charge_C_cm2']}
        rr=_simulate_stack(verification_ctx,kind,x,lowest,record=True)
        critical_history=rr['history']
        fail_T=round(lowest-critical['温度分辨率/℃'],1)
        fail_kind,fail_x,_=scan_best.get(fail_T,(kind,x,None))
        fail=_simulate_stack(verification_ctx,fail_kind,fail_x,fail_T,record=True)
        if fail['success']:
            raise ValueError(f'临界值未被细时间步证实：{fail_T:.1f}℃仍可启动，应继续向低温搜索')
        key_cell=fail['failure_cell']
        if fail['history'] is not None and len(fail['history']):
            last=fail['history'].iloc[-1]
            if abs(last['电池1温度/℃']-last['电池5温度/℃'])<1e-5 and key_cell in (1,5):
                key_cell='第1、5片'
        failure_detail={'初温/℃':fail_T,'加载策略':fail_kind,'原因':fail['reason'],
                        '关键电池':key_cell,'最低电压/V':fail['min_voltage_V'],
                        '最大冰体积分数':fail['max_ice_fraction'],
                        '最终最低单片温度/℃':fail['min_temp_C'],
                        '累计电荷量/C·cm⁻²':fail['charge_C_cm2']}
        if fail['history'] is not None:
            tr=fail['history'].copy();tr.insert(0,'加载策略',f'临界低温失败{fail_T:.0f}℃');trajectories.append(tr)
        logger.core(f'最低可行初温≈{lowest:.1f}℃（分辨率{critical["温度分辨率/℃"]:.1f}℃，0.05s复核）；策略={kind}；低一个温度步长结果：{fail["reason"]}，关键电池{key_cell}')
    else:
        logger.warning('−40～−10℃粗扫描未发现可行初温；不伪造T0')
    return {'table':table,'scan':scan,'critical':critical,'failure':failure_detail,
            'trajectories':pd.concat(trajectories,ignore_index=True) if trajectories else pd.DataFrame(),
            'critical_history':critical_history,'best':best,
            'G_inter':search_ctx['G_inter'],'G_end':search_ctx['G_end']}


def validate_computation(results,logger):
    table=results['table']
    if len(table)!=3 or set(table['加载策略'])!={'恒流策略','线性升载','分段阶梯加载'}:
        raise ValueError('表3策略行缺失')
    for _,r in table.iterrows():
        if r['最大电流密度/A·cm⁻²']>0.5001 or r['最低电压/V']<0.30-1e-3 and r['启动结果']=='成功':
            raise ValueError('成功方案违反电流或电压约束')
        if r['启动结果']=='成功' and r['累计电荷量 /C·cm⁻²']>20.001:
            raise ValueError('成功方案违反电荷上限')
    logger.status('计算验收通过：三类策略均已回算，成功方案满足电流、电荷与单片电压约束')


def get_figure_contract():
    return {
        'evidence':'three optimized current curves and stack limiting temperature/voltage/ice; feasibility across initial temperatures and critical cell',
        'layout':'new evidence-led Chinese composite figures at 600 DPI; normalized safety margins; no old-image reuse',
    }


def get_render_dependency_names():
    return ['matplotlib','numpy']


def render_figures(results, output_dir, logger):
    return render_question('Q2', results, output_dir, logger)

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
    return _q2(results, output_dir, logger)

def _q2(r, out, log):
    table = r['table']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    labels = table['加载策略'].tolist()
    colors = [TEAL, ORANGE, PLUM]
    for i, row in table.iterrows():
        ax[0].scatter(row['累计电荷量 /C·cm⁻²'], row['启动时间/s'], s=120, color=colors[i], label=row['加载策略'])
        ax[0].annotate(row['加载策略'], (row['累计电荷量 /C·cm⁻²'], row['启动时间/s']), xytext=(7, 6), textcoords='offset points', fontsize=8)
    width = np.arange(len(labels))
    voltage_margin = (table['最低电压/V'] - 0.3) / 0.3
    ice_margin = (0.99 - table['最大冰体积分数']) / 0.99
    ax[1].bar(width - 0.18, voltage_margin, width=0.36, color=TEAL, label='电压相对裕量')
    ax[1].bar(width + 0.18, ice_margin, width=0.36, color=ORANGE, label='冰相相对裕量')
    ax[1].set_xticks(width, labels, rotation=12)
    ax[0].set(xlabel='累计电荷 / C/cm2', ylabel='启动时间 / s')
    ax[1].set(ylabel='相对安全裕量')
    _panel(ax[0], 'a  启动时间与电荷的联合选择')
    _panel(ax[1], 'b  三类策略的安全余量')
    ax[1].legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    paths = [_save(fig, out, 'Q2', 'Q2_加载决策与安全裕量.png')]
    tr = r['trajectories']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for label, color in zip(labels, colors):
        d = tr[tr['加载策略'] == label]
        ax[0].plot(d['时间/s'], d['电池1温度/℃'], color=color, lw=1.6, label=label)
        ax[1].plot(d['时间/s'], d['累计电荷量/C_cm2'], color=color, lw=1.6, label=label)
    ax[0].axhline(0, color=INK, lw=0.8, ls='--')
    ax[1].axhline(20, color=INK, lw=0.8, ls='--', label='电荷上限')
    ax[0].set(xlabel='时间 / s', ylabel='端片温度 / ℃')
    ax[1].set(xlabel='时间 / s', ylabel='累计电荷 / C/cm2')
    _panel(ax[0], 'a  限制端片的升温轨迹')
    _panel(ax[1], 'b  电荷预算如何被消耗')
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    paths.append(_save(fig, out, 'Q2', 'Q2_端片与电荷约束轨迹.png'))
    if r.get('critical') and r.get('critical_history') is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
        scan = r['scan']
        feasible = scan['可行'] == '是'
        ax[0].scatter(scan.loc[~feasible, '初始温度/℃'], np.zeros((~feasible).sum()), color=ORANGE, marker='x', s=60, label='不可行')
        ax[0].scatter(scan.loc[feasible, '初始温度/℃'], np.zeros(feasible.sum()), color=TEAL, s=75, label='可行')
        threshold = r['critical']['最低初始温度/℃']
        ax[0].axvline(threshold, color=INK, ls='--', lw=1)
        ax[0].annotate(f'细化边界 {threshold:g}℃', (threshold, 0), xytext=(6, 22), textcoords='offset points', fontsize=9)
        ax[0].set(xlabel='初始温度 / ℃', yticks=[])
        hist = r['critical_history']
        ax[1].plot(hist['时间/s'], hist['最低温度/℃'], color=TEAL, lw=1.8)
        ax[1].axhline(0, color=INK, ls='--', lw=0.8)
        ax[1].set(xlabel='时间 / s', ylabel='五片最低温度 / ℃')
        _panel(ax[0], 'a  可自启动初温的判定边界')
        _panel(ax[1], 'b  临界初温下的限制片轨迹')
        ax[0].legend(frameon=False)
        paths.append(_save(fig, out, 'Q2', 'Q2_临界初温与限制片.png'))
    log.status(f'Q2重设计证据图生成：{len(paths)}张600 DPI PNG')
    return paths


def export_tables(results,output_dir,logger):
    from pathlib import Path
    import pandas as pd
    out=Path(output_dir);files=[]
    for name,frame in [('Q2_表3策略比较.csv',results['table']),
                       ('Q2_初温扫描.csv',results['scan']),
                       ('Q2_全时段轨迹.csv',results['trajectories'])]:
        files.append(write_dataframe_csv(frame,out/name))
    critical=pd.DataFrame([results['critical']]) if results['critical'] else pd.DataFrame([{'最低初始温度/℃':'本扫描范围未发现可行解'}])
    failure=pd.DataFrame([results['failure']]) if results['failure'] else pd.DataFrame([{'原因':'未形成可用的低温失效诊断'}])
    files.append(write_dataframe_csv(critical,out/'Q2_最低初温.csv'))
    files.append(write_dataframe_csv(failure,out/'Q2_关键电池诊断.csv'))
    if results['critical_history'] is not None:
        files.append(write_dataframe_csv(results['critical_history'],out/'Q2_临界工况轨迹.csv'))
    table=results['table'].drop(columns=['失败关键电池']).copy()
    critical_display=critical.round(4)
    failure_display=failure.round(4)
    note='成功判据：五片平均温度同时>0℃、所有局部冰体积分数<0.99、全过程单片电压≥0.30V；累计电荷量≤20C·cm⁻²，电流密度≤0.5A·cm⁻²。'
    md=write_markdown_tables([
        ('表3 不同加载策略下电堆冷启动性能优化结果对比',table,note),
        ('电堆最低自冷启动初温',critical_display,'温度分辨率见表内；最低初温由扫描与临界区间复算得到。'),
        ('低于临界初温的关键单片与失效原因',failure_display,'低于临界初温0.1℃，采用低温扫描候选策略以0.05秒时间步复核。'),
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
