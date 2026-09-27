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


QUESTION_ID = "Q4"
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

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve, eigh_tridiagonal
from scipy.optimize import differential_evolution, minimize


def _attachment(name):
    root = Path(__file__).resolve().parent
    candidates = [root/name, root/'B题'/name, root.parent/name, root.parent/'B题'/name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'请将 {name} 放在脚本目录或其上一级目录。')


def get_input_paths():
    root=Path(__file__).resolve().parent
    return [_attachment('附件1.xlsx'),root/'Q1'/'Q1_模型参数.csv',
            root/'Q3'/'Q3_各片能耗分配.csv',root/'Q3'/'Q3_约束及端部指标.csv']


def get_analysis_contract():
    return {
        'question':'Q4','pre_cooling':'seven-node endplate–five-cell heat network, exact matrix exponential',
        'cooling_initial_C':25,'cooling_ambient_C':-30,'cooling_minutes':list(range(10,101,10)),
        'startup':'Q3 five-cell 27-volume water–ice–thermal–voltage model, initial fields from pre-cooling',
        'current':'j=min(0.005t,0.3) A/cm2','electric_charge_cap_C_cm2':20,
        'controls':'rolling short-horizon temperature/voltage prediction from five measured cell states and sampled rates; projected independent 0-1 W/cm2 heater commands',
        'outcome':'five mean T>0 C, all local ice fractions<0.99, path V>=0.30 V, qcharge<=20',
        'search':'constant multi-start; rolling predictive controller parameter search with Q3 constant policy feasible anchor; full-grid fine-step verification',
        'seed':20260924,
        'final_selected':_FINAL_SELECTED,
        'final_method':'energy-peak-exposure multiobjective DE/Nelder-Mead plus cap scan, lambda=0.2',
    }


def get_compute_dependency_names():
    return ['numpy','pandas','scipy','openpyxl']


def _pre_cooling(ctx,minutes,par):


    g=ctx['grid'];dx=[];conduct=[];heatcap=[];component=[];material=[];local=[]
    def append(count,width,k,cp,part,name,offset=-1):
        for i in range(count):
            dx.append(width/count);conduct.append(k);heatcap.append(cp)
            component.append(part);material.append(name);local.append(offset+i if offset>=0 else -1)
    plate=float(par['端板厚度']);bp_a=float(par['阳极双极板厚度'])
    bp_c=float(par['阴极双极板厚度'])
    cp_end=float(par['密度'])*float(str(par['比热容']).replace('\xa0','').strip())
    k_end=float(par['导热系数']);cp_bp=1980*766
    append(10,plate,k_end,cp_end,0,'左端板')
    for cell in range(1,6):
        append(2,bp_a,95.,cp_bp,cell,'阳极双极板')
        for i in range(len(g['dx'])):
            dx.append(float(g['dx'][i]));conduct.append(float(g['k'][i]))
            heatcap.append(float(g['cp'][i]));component.append(cell)
            material.append(str(g['layer'][i]));local.append(i)
        append(2,bp_c,95.,cp_bp,cell,'阴极双极板')
    append(10,plate,k_end,cp_end,6,'右端板')
    dx=np.asarray(dx);conduct=np.asarray(conduct);C=dx*np.asarray(heatcap)
    faces=1/(dx[:-1]/(2*conduct[:-1])+dx[1:]/(2*conduct[1:]))
    boundary_left=1/(dx[0]/(2*conduct[0])+1/ctx['h'])
    boundary_right=1/(dx[-1]/(2*conduct[-1])+1/ctx['h'])
    diagonal=np.zeros(len(dx));diagonal[:-1]-=faces/C[:-1]
    diagonal[1:]-=faces/C[1:]
    diagonal[0]-=boundary_left/C[0];diagonal[-1]-=boundary_right/C[-1]
    off=faces/np.sqrt(C[:-1]*C[1:])
    values,vectors=eigh_tridiagonal(diagonal,off)
    coeff=vectors.T@(55*np.sqrt(C))
    x=np.cumsum(dx)-dx/2;component=np.asarray(component);local=np.asarray(local)
    material=np.asarray(material)
    summaries=[];profiles=[]
    for minute in minutes:
        T=-30+(vectors@(coeff*np.exp(values*60*float(minute))))/np.sqrt(C)
        for i in range(len(dx)):
            profiles.append({'预冷时间/min':float(minute),'位置编号':int(component[i]),
                             '部件':'左端板' if component[i]==0 else ('右端板' if component[i]==6 else f'第{component[i]}片单电池'),
                             '材料层':material[i],'单片网格编号':int(local[i]),
                             '叠层坐标/mm':float(1000*x[i]),'网格厚度/m':float(dx[i]),
                             '温度/℃':float(T[i])})
        for part in range(7):
            mask=(component==part)&((local>=0) if 1<=part<=5 else (local<0))
            summaries.append({'预冷时间/min':float(minute),'位置编号':part,
                              '部件':'左端板' if part==0 else ('右端板' if part==6 else f'第{part}片单电池'),
                              '叠层坐标/mm':float(1000*np.average(x[mask],weights=dx[mask])),
                              '温度/℃':float(np.average(T[mask],weights=dx[mask]))})
    return pd.DataFrame(summaries),pd.DataFrame(profiles)


def _initial_from_cooling(frame,field,minute):
    d=frame[np.isclose(frame['预冷时间/min'],minute)].sort_values('位置编号')
    T=d['温度/℃'].to_numpy()
    f=field[np.isclose(field['预冷时间/min'],minute)&(field['单片网格编号']>=0)]
    profiles=np.asarray([f[f['位置编号']==k].sort_values('单片网格编号')['温度/℃'].to_numpy()
                         for k in range(1,6)])
    if len(T)!=7 or profiles.shape!=(5,27):raise ValueError(f'{minute} min预冷温度场缺失')
    return profiles,np.asarray([T[0],T[6]])


def _mirror(z):
    return np.asarray([z[0],z[1],z[2],z[1],z[0]],dtype=float)


def _heater_command(mode,param,t,T,V,dT,dV,projected_heater_rate):
    if mode=='恒功率策略':
        q=_mirror(param[:3]) if t<param[3] else np.zeros(5)
        return np.clip(q,0,1)


    base=_mirror(param[:3]);scale,t_on,t_off,shift,cold_gain,slow_gain,volt_gain,drop_gain=param[3:]
    T=np.asarray(T);V=np.asarray(V);dT=np.asarray(dT);dV=np.asarray(dV)
    horizon=float(np.clip(6+shift/15,4,12))
    remaining=max(t_off-t,horizon)
    target_rate=np.maximum(-T,0)/remaining

    q_predict=np.clip((target_rate-dT)/projected_heater_rate,0,1)
    q=np.maximum(scale*base,10*cold_gain*q_predict)
    voltage_forecast=V+horizon*dV
    voltage_floor=np.clip((0.35-voltage_forecast)/0.5,0,1)
    q=np.maximum(q,volt_gain*voltage_floor)
    q+=slow_gain*np.maximum(0.15-dT,0)
    q+=drop_gain*np.maximum(-dV-0.002,0)
    q=np.clip(q,0,1)
    if t<t_on or t>=t_off:q[:]=0
    if np.all(T>=0):q[:]=0
    return q


def _simulate(ctx,mode,param,initial_cells,initial_ends,record=False,max_time=120):
    p=ctx['par'];g=ctx['grid'];theta=ctx['theta'];dt=ctx['dt'];dx=g['dx'];n=len(dx)
    initial_cells=np.asarray(initial_cells,dtype=float)
    if initial_cells.shape==(5,n):
        initial_profile=initial_cells
    elif initial_cells.shape==(5,27) and n==5:
        cuts=((0,8),(8,11),(11,15),(15,19),(19,27))
        initial_profile=np.column_stack([np.mean(initial_cells[:,lo:hi],axis=1) for lo,hi in cuts])
    elif initial_cells.shape==(5,):
        initial_profile=np.repeat(initial_cells[:,None],n,axis=1)
    else:
        raise ValueError('预冷初始温度场维度不正确')
    initial_means=np.average(initial_profile,axis=1,weights=dx)
    temp=273.15+initial_profile.copy()
    end=273.15+np.asarray(initial_ends,dtype=float).copy();ambient=243.15
    mobile=np.zeros((5,n));ice=np.zeros((5,n))
    mobile[:,g['mem']]=float(p['初始膜含水量'])*2150*0.018/(float(p['膜当量质量'])/1000)
    polar=0.;charge=0.;prev_j=0.;prev_T=initial_means.copy()
    prev_V=np.ones(5);energy=np.zeros(5);minV=np.inf;maxIce=0.;maxDT=0.
    rows=[];reason='120s内未启动';peak_ice=np.zeros(5)
    heater_rate=1e4/float(np.sum(ctx['cp']*dx))

    def electro(j):
        loss=polar+theta[4]*np.exp(-charge/theta[5]);V=[];icefrac=[]
        for k in range(5):
            v,aux=_voltage(j,temp[k],mobile[k],ice[k],g,p,loss,theta[0])
            V.append(v);icefrac.append(float(np.max(aux['ice_vol'])))
        return np.asarray(V),np.asarray(icefrac)

    def snapshot(t,j,q,V,icefrac,T):
        row={'时间/s':t,'电流密度/A_cm2':j,'辅助累计能耗/J':float(np.sum(energy)),
             '累计电荷量/C_cm2':charge,'五片最低温度/℃':float(np.min(T)),
             '五片温差/℃':float(np.ptp(T)),'最低电压/V':float(np.min(V)),
             '最大冰体积分数':float(np.max(icefrac))}
        for k in range(5):
            row[f'第{k+1}片温度/℃']=float(T[k]);row[f'第{k+1}片电压/V']=float(V[k])
            row[f'第{k+1}片最大冰体积分数']=float(icefrac[k]);row[f'第{k+1}片加热功率/W_cm2']=float(q[k])
            row[f'第{k+1}片累计加热能耗/J']=float(energy[k])
        rows.append(row)

    V,icefrac=electro(0);prev_V=V.copy();minV=float(np.min(V))
    if record:snapshot(0,0,np.zeros(5),V,icefrac,prev_T)
    for step in range(1,int(np.ceil(max_time/dt))+1):
        t0=(step-1)*dt;t=step*dt;prev_charge=charge
        j=min(0.005*t0,0.3)
        polar=polar*np.exp(-dt/theta[3])+theta[2]*np.exp(-charge/theta[6])*max(j-prev_j,0)
        charge+=0.0025*(min(t,60.0)**2-min(t0,60.0)**2)
        charge+=0.3*max(t-max(t0,60.0),0.0)
        meanT=np.average(temp,axis=1,weights=dx)
        TC=meanT-273.15
        V,icefrac=electro(j)
        dT=(TC-prev_T)/dt;dV=(V-prev_V)/dt
        power=_heater_command(mode,param,t0,TC,V,dT,dV,heater_rate)
        energy+=25*power*dt
        minV=min(minV,float(np.min(V)));maxIce=max(maxIce,float(np.max(icefrac)))
        peak_ice=np.maximum(peak_ice,icefrac)
        source=np.zeros(n);source[g['ccl']]=0.018*(j*1e4)/(2*96485*float(p['阴极 CL 厚度']))
        freeze=np.zeros((5,n));melt=np.zeros((5,n))
        for k in range(5):
            mobile[k]=np.maximum(lu_solve(ctx['water_lu'],mobile[k]+dt*source),0)
            sat=np.asarray([_sat_water(float(tt),max(float(ee),1e-9)) if pp else 0.
                            for tt,ee,pp in zip(temp[k],g['eps'],g['porous'])])
            free=np.maximum(mobile[k]-sat,0)
            freeze[k]=np.minimum(free*(1-np.exp(-0.6*dt)),np.maximum(g['eps']*920-ice[k],0))
            freeze[k,~g['porous']]=0;freeze[k,temp[k]>=273.15]=0
            mobile[k]-=freeze[k];ice[k]+=freeze[k]
            melt[k]=np.where(temp[k]>=273.15,ice[k]*(1-np.exp(-0.6*dt)),0)
            mobile[k]+=melt[k];ice[k]-=melt[k]
        inter=ctx['G_inter']*(meanT[:-1]-meanT[1:])
        flux=np.zeros(5);flux[:-1]-=inter;flux[1:]+=inter
        left=ctx['G_end']*(end[0]-meanT[0]);right=ctx['G_end']*(end[1]-meanT[-1])
        flux[0]+=left-ctx['h']*(meanT[0]-ambient)
        flux[-1]+=right-ctx['h']*(meanT[-1]-ambient)
        end[0]+=dt*(-left-ctx['h']*(end[0]-ambient))/ctx['C_end']
        end[1]+=dt*(-right-ctx['h']*(end[1]-ambient))/ctx['C_end']
        for k in range(5):
            heater=np.zeros(n);heater[0]=power[k]*1e4/(2*dx[0]);heater[-1]=power[k]*1e4/(2*dx[-1])
            qgen=j*1e4*(1.48-V[k])/g['length']
            rhs=temp[k]+dt*(qgen+flux[k]/g['length']+heater+
                             float(p['水冻结潜热'])*(freeze[k]-melt[k])/dt)/ctx['cp']
            temp[k]=lu_solve(ctx['heat_lu'],rhs)
        newT=np.average(temp,axis=1,weights=dx)-273.15
        maxDT=max(maxDT,float(np.ptp(newT)))
        if record:snapshot(t,j,power,V,icefrac,newT)
        if minV<0.30:reason='电压下限';break
        if maxIce>=0.99:reason='严重冰堵';break
        if float(np.min(newT))>0 and j>0:
            frac=np.clip(-float(np.min(TC))/max(float(np.min(newT)-np.min(TC)),1e-12),0,1)
            ts=t0+dt*frac
            qsuccess=prev_charge+0.0025*(min(ts,60.0)**2-min(t0,60.0)**2)
            qsuccess+=0.3*max(ts-max(t0,60.0),0.0)
            if qsuccess<=20+1e-9:

                if np.min(initial_means)>0:ts=t;qsuccess=charge
                return {'success':True,'reason':'成功','time_s':ts,'total_energy_J':float(np.sum(energy)),
                        'energy_cells_J':energy.copy(),'max_delta_T_C':maxDT,'min_voltage_V':minV,
                        'max_ice_fraction':maxIce,'peak_cell_ice':peak_ice.copy(),
                        'charge_C_cm2':qsuccess,'initial_delta_T_C':float(np.ptp(initial_means)),
                        'trajectory':pd.DataFrame(rows) if record else None}
        if charge>20:
            reason='累计电荷上限';charge=20.;break
        prev_j=j;prev_T=TC;prev_V=V
    return {'success':False,'reason':reason,'time_s':t,'total_energy_J':float(np.sum(energy)),
            'energy_cells_J':energy.copy(),'max_delta_T_C':maxDT,'min_voltage_V':minV,
            'max_ice_fraction':maxIce,'peak_cell_ice':peak_ice.copy(),'charge_C_cm2':charge,
            'initial_delta_T_C':float(np.ptp(initial_means)),'final_minT_C':float(np.min(newT)),
            'trajectory':pd.DataFrame(rows) if record else None}


def _score(result):
    if result['success']:
        return result['total_energy_J']+0.02*result['time_s']+0.01*result['max_delta_T_C']
    return 10000+result['total_energy_J']+100*max(0,-result.get('final_minT_C',-30))+1000*max(0,0.3-result['min_voltage_V'])


def _constant_opt(ctx,cells,ends,seed,anchor):
    zero=np.zeros(4);rz=_simulate(ctx,'恒功率策略',zero,cells,ends)
    if rz['success']:return zero
    bounds=[(0,1)]*3+[(2,130)]
    candidates=[anchor.copy(),np.asarray([0.8,0.8,0.8,75.0])]
    for scale in [0.10,.2,.4,.6,.8,1.0,1.2]:
        for th in [10,20,30,45,60]:
            candidates.append(np.r_[np.minimum(anchor[:3]*scale,1),th])
    obj=lambda z:_score(_simulate(ctx,'恒功率策略',z,cells,ends))
    best=min(candidates,key=obj)
    de=differential_evolution(obj,bounds,seed=seed,popsize=4,maxiter=4,polish=False,
                              workers=1,updating='immediate',x0=best)
    candidates.append(de.x)
    return min(candidates,key=obj)


def _dynamic_opt(ctx,cells,ends,seed,constant,base):
    zero=np.asarray([0,0,0,0,0,1,-10,0,0,0,0],dtype=float)
    rz=_simulate(ctx,'动态功率策略',zero,cells,ends)
    if rz['success']:return zero

    def make(z):
        return np.r_[base,z[0],z[1],z[2],30.0,z[3],z[4],.02,.05]
    bounds=[(0,2),(0,85),(15,97),(0,.08),(0,.08)]
    candidates=[np.asarray([1.,0.,min(constant[3],97.),0,0]),
                np.asarray([.5,20.,75.,.005,.005])]
    for scale in [.4,.5,.6,.8,1.,1.5,2.]:
        for on,off in [(0,30),(15,65),(20,75),(30,80),(40,96),(60,80),(70,85)]:
            candidates.append(np.asarray([scale,on,off,.005,.005]))
    obj=lambda z:_score(_simulate(ctx,'动态功率策略',make(z),cells,ends))
    best=min(candidates,key=obj)
    de=differential_evolution(obj,bounds,seed=seed,popsize=4,maxiter=7,polish=False,
                              workers=1,updating='immediate',x0=best)
    candidates.append(de.x)
    return make(min(candidates,key=obj))


def _verify(ctx,mode,param,cells,ends):
    result=_simulate(ctx,mode,param,cells,ends)
    if result['success']:return np.asarray(param),result
    options=[]
    if mode=='恒功率策略':
        for scale in [1.01,1.03,1.06,1.10]:
            for plus in [.2,.5,1,2]:
                z=np.r_[np.minimum(np.asarray(param[:3])*scale,1),min(param[3]+plus,130)]
                r=_simulate(ctx,mode,z,cells,ends)
                if r['success']:options.append((_score(r),z,r))
    else:
        for scale in [1.01,1.03,1.06,1.10]:
            for plus in [.2,.5,1,2]:
                z=np.asarray(param).copy();z[3]=min(z[3]*scale,2);z[5]=min(z[5]+plus,97)
                r=_simulate(ctx,mode,z,cells,ends)
                if r['success']:options.append((_score(r),z,r))
    if not options:raise RuntimeError(f'{mode}的候选在细步长下不可行：{result["reason"]}')
    _,z,r=min(options,key=lambda v:v[0]);return z,r


def compute_results(logger):
    paths=get_input_paths();par=_number_map(pd.read_excel(paths[0],sheet_name='参数清单'))
    theta=_calibrated_theta(paths[1]);g=_grid(par)
    coarse=_context(par,_coarse_grid(g),theta,1.0)
    full=_context(par,g,theta,.2);verify=_context(par,g,theta,.05)
    cooling,cooling_field=_pre_cooling(full,range(0,101,10),par)
    q3alloc=pd.read_csv(paths[2]);q3m=pd.read_csv(paths[3])
    q3=q3alloc[q3alloc['辅助策略']=='恒定功率协同启动'].sort_values('单片编号')
    qbase=q3['功率密度/W_cm2'].to_numpy();th=float(q3m.loc[q3m['策略']=='恒定功率协同启动','加热时间/s'].iloc[0])
    anchor=np.r_[qbase[:3],th]
    states={'工况1':(np.full((5,len(g['dx'])),-30.),np.full(2,-30.)),
            '工况2':_initial_from_cooling(cooling,cooling_field,20),
            '工况3':_initial_from_cooling(cooling,cooling_field,40)}
    rows=[];tracks=[];control=[];selected={}
    for i,(label,(cells,ends)) in enumerate(states.items()):
        means=np.average(cells,axis=1,weights=g['dx'])
        logger.status(f'{label}预冷初温：端片{means[0]:.2f}℃，中片{means[2]:.2f}℃，片间温差{np.ptp(means):.3f}℃')
        const=np.asarray(_PAPER_BASELINES[f'{label}_恒功率策略']['param'])
        choice=_FINAL_SELECTED[i]
        dyn=np.asarray(choice['param'])
        for mode,param,cap,dt in [('恒功率策略',const,1.,.05),
                                  ('动态功率策略',dyn,choice['design_cap'],.0125)]:
            rr=_opt_run(i,mode,param,cap,dt,original=True,record=True)
            if not _opt_feasible(rr):raise ValueError(f'{label}{mode}回算约束核验失败')
            selected[(label,mode)]=(param,rr)
            tr=rr['trajectory'];tr.insert(0,'控制策略',mode);tr.insert(0,'工况',label);tracks.append(tr)
            if mode=='恒功率策略':
                description='q='+str(tuple(round(float(v),3) for v in _mirror(param[:3])))+f'，最迟{param[3]:.1f}s，达标即停'
            else:
                description=(f'反馈律：基准{tuple(round(float(v),3) for v in _mirror(param[:3]))}；'
                             f'{param[4]:.1f}—{param[5]:.1f}s，状态调节')
            rows.append({'工况':label,'控制策略':mode,'功率控制策略/W·cm⁻²':description,
                         '启动时间/s':round(rr['time_s'],2),'辅助加热总能耗/J':round(rr['total_energy_J'],2),
                         '最大温差/℃':round(rr['max_delta_T_C'],4),'最低单片电压/V':round(rr['min_voltage_V'],4),
                         '最大冰体积分数':round(rr['max_ice_fraction'],5),'启动结果':'成功'})
            for k in range(5):
                control.append({'工况':label,'控制策略':mode,'单片编号':k+1,
                                '累计辅助加热能耗/J':rr['energy_cells_J'][k],
                                '最大冰体积分数':rr['peak_cell_ice'][k]})
            logger.core(f'{label} {mode}：E={rr["total_energy_J"]:.2f}J，t={rr["time_s"]:.2f}s，'
                        f'ΔTmax={rr["max_delta_T_C"]:.3f}℃，Vmin={rr["min_voltage_V"]:.3f}V，冰={rr["max_ice_fraction"]:.4f}')
    global _opt__STATS
    _opt__STATS = None
    scan=[];scantracks=[]
    for minute in range(10,101,10):
        cells,ends=_initial_from_cooling(cooling,cooling_field,minute)
        means=np.average(cells,axis=1,weights=g['dx'])
        z=_constant_opt(coarse,cells,ends,20262024+minute,anchor)
        z,r=_verify(verify,'恒功率策略',z,cells,ends)
        scan.append({'预冷时间/min':minute,'端片初温/℃':means[0],'中片初温/℃':means[2],
                     '初始片间温差/℃':float(np.ptp(means)),'加热功率分配/W_cm2':str(tuple(round(float(v),4) for v in _mirror(z[:3]))),
                     '加热持续时间/s':z[3],'启动时间/s':r['time_s'],'辅助能耗/J':r['total_energy_J'],
                     '最大温差/℃':r['max_delta_T_C'],'最低电压/V':r['min_voltage_V'],
                     '最大冰体积分数':r['max_ice_fraction'],'累计电荷/C_cm2':r['charge_C_cm2'],'启动结果':'成功' if r['success'] else r['reason']})
        logger.status(f'{minute}min预冷扫描：端片{means[0]:.2f}℃；恒功率辅助能耗{r["total_energy_J"]:.1f}J；{r["reason"]}')
    return {'table':pd.DataFrame(rows),'cooling':cooling,'cooling_field':cooling_field,'scan':pd.DataFrame(scan),
            'tracks':pd.concat(tracks,ignore_index=True),'cell_metrics':pd.DataFrame(control),
            'selected':selected,'states':states}


def validate_computation(results,logger):
    table=results['table']
    if len(table)!=6:raise ValueError('三工况六策略表不完整')
    if (table['启动结果']!='成功').any():raise ValueError('对比表含不可行策略')
    for (_,mode),(param,r) in results['selected'].items():
        if not r['success'] or r['min_voltage_V']<.30-1e-8 or r['max_ice_fraction']>=.99 or r['charge_C_cm2']>20+1e-8:
            raise ValueError('成功判据或电荷预算未满足')
        if mode=='恒功率策略' and (np.any(param[:3]<0) or np.any(param[:3]>1)):raise ValueError('功率上限越界')
    if len(results['cooling'])!=77 or len(results['scan'])!=10:raise ValueError('预冷扫描缺失')
    if (results['scan']['启动结果']!='成功').any():raise ValueError('预冷扫描含未成功工况')
    logger.status('计算验收通过：三工况六策略与10—100min预冷扫描均已完成，约束满足')
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


_opt_HERE = Path(__file__).resolve().parent / 'Q4'
_opt_MODES = ['恒功率策略', '动态功率策略']
_opt_WEIGHTS = [.05, .2, .8]
_opt_NAMES = {.05:'节能优先', .2:'均衡', .8:'功率余量优先'}
_opt_TARGET = 4.97
_opt_DEADLINE = 60 + (20 - 9) / .3
_opt__STATS = None
_opt__raw = _heater_command
_PAPER_BASELINES = {'工况1_恒功率策略': {'param': [0.8374655312848379,
                         0.1717356282854362,
                         0.12962869727935605,
                         102.29440567533042]},
 '工况2_恒功率策略': {'param': [0.21869111131261718,
                         0.6289226526219183,
                         0.11907694680254868,
                         14.213068576747514]},
 '工况3_恒功率策略': {'param': [0.6511810248714609,
                         0.037676872727140265,
                         0.03713070956469222,
                         121.24338702734634]}}
_FINAL_SELECTED = [{'case': 1,
  'mode': '动态功率策略',
  'weight': 0.2,
  'dt': 0.0125,
  'param': [0.865400412968799,
            0.3317125774758345,
            0.07942204036147629,
            1.0,
            16.435133503294928,
            97.0,
            30.0,
            0.02520764715600564,
            0.0545313314782269,
            0.02,
            0.05],
  'design_cap': 0.865400412968799,
  'trajectory_file': '工况1_动态功率策略_均衡_dt0.0125.csv'},
 {'case': 2,
  'mode': '动态功率策略',
  'weight': 0.2,
  'dt': 0.0125,
  'param': [0.9866243996819202,
            3.428817051890268e-05,
            1.731421902151535e-05,
            1.0,
            84.97306104258475,
            95.04256356165533,
            30.0,
            3.730244425731512e-05,
            2.2651213144584248e-05,
            0.02,
            0.05],
  'design_cap': 1.0,
  'trajectory_file': '工况2_动态功率策略_节能优先_dt0.0125.csv'},
 {'case': 3,
  'mode': '动态功率策略',
  'weight': 0.2,
  'dt': 0.0125,
  'param': [0.94,
            0.5138802462065085,
            0.07168268714465156,
            1.0,
            67.03125,
            97.0,
            30.0,
            0.0,
            0.0,
            0.0,
            0.0],
  'design_cap': 0.94,
  'trajectory_file': '工况3_动态功率策略_均衡_dt0.0125.csv'}]
_opt_BASE = [{'case': 1,
  'mode': '恒功率策略',
  'param': [0.999855095159851, 0.050176283035575725, 0.00023278202932249358, 96.6666666667],
  'cap': 1.0},
 {'case': 1,
  'mode': '动态功率策略',
  'param': [0.9999486162118711,
            0.3662701268927495,
            0.09285820288185787,
            1.0,
            30.251162666230226,
            96.65434538965363,
            30.0,
            0.024448029363034234,
            0.05188397397399571,
            0.02,
            0.05],
  'cap': 1.0},
 {'case': 2,
  'mode': '恒功率策略',
  'param': [0.09472145919034679, 4.246942297967653e-07, 2.6914316579396528e-05, 96.6666666667],
  'cap': 1.0},
 {'case': 2,
  'mode': '动态功率策略',
  'param': [0.9866243996819202,
            3.428817051890268e-05,
            1.731421902151535e-05,
            1.0,
            84.97306104258475,
            95.04256356165533,
            30.0,
            3.730244425731512e-05,
            2.2651213144584248e-05,
            0.02,
            0.05],
  'cap': 1.0},
 {'case': 3,
  'mode': '恒功率策略',
  'param': [0.6457435562486997, 8.136823534676937e-05, 8.569236738562085e-05, 96.6666666667],
  'cap': 1.0},
 {'case': 3,
  'mode': '动态功率策略',
  'param': [0.99906909306265,
            0.5749181092053166,
            0.08019704057547675,
            1.0,
            70.53247822875366,
            96.92723320646856,
            30.0,
            0.047793709645517504,
            0.011481518057640817,
            0.02,
            0.05],
  'cap': 1.0}]
_PRIOR_RESULTS = {'工况3_恒功率策略_复核.json': [{'time_s': 96.64991643317992, 'total_energy_J': 3121.1560017517722},
                       {'time_s': 96.64066898213524, 'total_energy_J': 3121.156001751656},
                       {'time_s': 96.63681145170882, 'total_energy_J': 3120.7523343949306}],
 '工况3_动态功率策略_复核.json': [{'time_s': 96.5962232610129, 'total_energy_J': 2102.3645716567935},
                        {'time_s': 96.59334643714791, 'total_energy_J': 2102.3569864081983},
                        {'time_s': 96.58251975713787, 'total_energy_J': 2102.3526309654853}],
 '工况2_动态功率策略_复核.json': [{'time_s': 87.14999760244513, 'total_energy_J': 106.0669265525855},
                        {'time_s': 87.13559987597122, 'total_energy_J': 107.3002483714579},
                        {'time_s': 87.13837840642394, 'total_energy_J': 107.30024527020095}],
 '工况1_恒功率策略_复核.json': [{'time_s': 96.64996930963078, 'total_energy_J': 5074.8390947077605},
                       {'time_s': 96.64439720321336, 'total_energy_J': 5074.8390947074695},
                       {'time_s': 96.64023281052235, 'total_energy_J': 5074.839094707324}],
 '工况2_恒功率策略_复核.json': [{'time_s': 96.64755415008773, 'total_energy_J': 457.8085355896623},
                       {'time_s': 96.65042721023592, 'total_energy_J': 457.92695476596884},
                       {'time_s': 96.6213418079628, 'total_energy_J': 457.69011641336317}],
 '工况1_动态功率策略_复核.json': [{'time_s': 96.46164546784357, 'total_energy_J': 4676.074315804812},
                        {'time_s': 96.4382828985449, 'total_energy_J': 4674.280824433179},
                        {'time_s': 96.42766623127015, 'total_energy_J': 4674.273383951697}]}


def _search_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))

@lru_cache(None)
def _search_inputs():
    paths = get_input_paths()
    par = _number_map(pd.read_excel(paths[0], sheet_name='参数清单'))
    theta = _calibrated_theta(paths[1])
    g = _grid(par)
    ctx = _context(par, g, theta, 0.2)
    cooling, field = _pre_cooling(ctx, [0, 20, 40], par)
    states = [(np.full((5, 27), -30.0), np.full(2, -30.0))] + [_initial_from_cooling(cooling, field, m) for m in [20, 40]]
    return (par, theta, g, states, cooling, field)

@lru_cache(None)
def _search_context(dt):
    p, th, g, *_ = _search_inputs()
    return _context(p, g, th, dt)

def _search_sat(T, eps):
    tc = T - 273.15
    ps = np.where(tc >= 0, 611.21 * np.exp((18.678 - tc / 234.5) * tc / (257.14 + tc)), 611.15 * np.exp((23.036 - tc / 333.7) * tc / (279.82 + tc)))
    return eps * 0.018 * ps / (8.314 * T)

def _search_volts(j, T, mob, ice, polar, charge, c):
    g = c['grid']
    p = c['par']
    th = c['theta']
    dx = g['dx']
    F = 96485.0
    R = 8.314
    liq = np.where(g['porous'], np.maximum(mob - _search_sat(T, g['eps']), 0), 0)
    gas = np.maximum(g['eps'] - liq / 990 - ice / 920, 1e-05)
    pore = ice / 920 / np.maximum(g['eps'], 1e-08)
    area = np.average(np.maximum(1 - pore[:, g['ccl']], 1e-05) ** float(p['阴极冰覆盖活性面积指数']), axis=1, weights=dx[g['ccl']])
    diff = 2.2e-05 * (T / 298.15) ** 1.75 * gas ** 1.5
    resistance = np.sum(dx[g['cath']] / np.maximum(diff[:, g['cath']], 1e-12), axis=1)
    mean = np.average(T, axis=1, weights=dx)
    cath = np.average(T[:, g['cath']], axis=1, weights=dx[g['cath']])
    pressure = float(p['初始压力及操作压力'])
    jlim = 4 * F * 0.21 * pressure / (R * cath) / np.maximum(resistance, 1e-09)
    J = j * 10000.0
    po2 = np.maximum(0.21 * (1 - np.minimum(J / np.maximum(jlim, 1e-06), 0.999)), 1e-06) * pressure
    rev = 1.229 - 0.00085 * (mean - 298.15) + R * mean / (2 * F) * np.log(pressure / 101325 * (po2 / 101325) ** 0.5)
    j0 = 10 ** th[0] * np.exp(-67000 / R * (1 / mean - 1 / 298.15))
    act = R * mean / (0.5 * F) * np.arcsinh(J / np.maximum(2 * j0 * area, 1e-12))
    mem = np.average(mob[:, g['mem']], axis=1, weights=dx[g['mem']])
    lam = np.clip(mem * float(p['膜当量质量']) / 1000 / (2150 * 0.018), float(p['初始膜含水量']), 22)
    kappa = np.maximum((0.5139 * lam - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / mean)), 0.02)
    ohm = J * (float(p['质子交换膜厚度']) / kappa + 1e-06)
    con = -R * mean / (4 * F) * np.log(np.maximum(1 - J / np.maximum(jlim, 1e-06), 1e-06))
    return rev - act - ohm - con - polar - th[4] * np.exp(-charge / th[5])

def _search_simulate(case, mode, z, dt=0.2):

    c = _search_context(dt)
    p = c['par']
    g = c['grid']
    th = c['theta']
    dx = g['dx']
    cells, ends = _search_inputs()[3][case]
    T = cells.copy() + 273.15
    end = ends.copy() + 273.15
    mob = np.zeros_like(T)
    ice = np.zeros_like(T)
    mob[:, g['mem']] = float(p['初始膜含水量']) * 2150 * 0.018 / (float(p['膜当量质量']) / 1000)
    polar = charge = prev_j = 0.0
    prev_T = np.average(cells, axis=1, weights=dx)
    prev_V = _search_volts(0, T, mob, ice, polar, charge, c)
    energy = np.zeros(5)
    minV = prev_V.min()
    maxIce = maxDT = 0.0
    peak = np.zeros(5)
    heater_rate = 10000.0 / np.sum(c['cp'] * dx)
    success = False
    reason = '累计电荷上限'
    for step in range(1, int(np.ceil(120 / dt)) + 1):
        t0 = (step - 1) * dt
        t = step * dt
        prev_charge = charge
        j = min(0.005 * t0, 0.3)
        polar = polar * np.exp(-dt / th[3]) + th[2] * np.exp(-charge / th[6]) * max(j - prev_j, 0.0)
        charge += 0.0025 * (min(t, 60) ** 2 - min(t0, 60) ** 2) + 0.3 * max(t - max(t0, 60), 0)
        meanT = np.average(T, axis=1, weights=dx)
        TC = meanT - 273.15
        V = _search_volts(j, T, mob, ice, polar, charge, c)
        icefrac = np.max(ice / 920, axis=1)
        power = _heater_command(mode, z, t0, TC, V, (TC - prev_T) / dt, (V - prev_V) / dt, heater_rate)
        energy += 25 * power * dt
        minV = min(minV, V.min())
        maxIce = max(maxIce, icefrac.max())
        peak = np.maximum(peak, icefrac)
        source = np.zeros(len(dx))
        source[g['ccl']] = 0.018 * j * 10000.0 / (2 * 96485 * float(p['阴极 CL 厚度']))
        mob = np.maximum(lu_solve(c['water_lu'], (mob + dt * source).T, check_finite=False).T, 0)
        freeze = np.minimum(np.maximum(mob - _search_sat(T, g['eps']), 0) * (1 - np.exp(-0.6 * dt)), np.maximum(g['eps'] * 920 - ice, 0))
        freeze[:, ~g['porous']] = 0
        freeze[T >= 273.15] = 0
        mob -= freeze
        ice += freeze
        melt = np.where(T >= 273.15, ice * (1 - np.exp(-0.6 * dt)), 0)
        mob += melt
        ice -= melt
        inter = c['G_inter'] * (meanT[:-1] - meanT[1:])
        flux = np.zeros(5)
        flux[:-1] -= inter
        flux[1:] += inter
        left = c['G_end'] * (end[0] - meanT[0])
        right = c['G_end'] * (end[1] - meanT[-1])
        flux[0] += left - c['h'] * (meanT[0] - 243.15)
        flux[-1] += right - c['h'] * (meanT[-1] - 243.15)
        end += dt * (np.array([-left, -right]) - c['h'] * (end - 243.15)) / c['C_end']
        heater = np.zeros_like(T)
        heater[:, 0] = power * 10000.0 / (2 * dx[0])
        heater[:, -1] = power * 10000.0 / (2 * dx[-1])
        rhs = T + dt * (j * 10000.0 * (1.48 - V[:, None]) / g['length'] + flux[:, None] / g['length'] + heater + float(p['水冻结潜热']) * (freeze - melt) / dt) / c['cp']
        T = lu_solve(c['heat_lu'], rhs.T, check_finite=False).T
        newT = np.average(T, axis=1, weights=dx) - 273.15
        maxDT = max(maxDT, np.ptp(newT))
        if minV < 0.3:
            reason = '电压下限'
            break
        if maxIce >= 0.99:
            reason = '严重冰堵'
            break
        if newT.min() > 0 and j > 0:
            frac = np.clip(-TC.min() / max(newT.min() - TC.min(), 1e-12), 0, 1)
            ts = t0 + dt * frac
            qs = prev_charge + 0.0025 * (min(ts, 60) ** 2 - min(t0, 60) ** 2) + 0.3 * max(ts - max(t0, 60), 0)
            if qs <= 20 + 1e-09:
                success = True
                t = ts
                charge = qs
                reason = '成功'
                break
        if charge > 20:
            charge = 20.0
            break
        prev_j = j
        prev_T = TC
        prev_V = V
    return dict(success=success, reason=reason, time_s=t, total_energy_J=energy.sum(), energy_cells_J=energy, max_delta_T_C=maxDT, min_voltage_V=minV, max_ice_fraction=maxIce, charge_C_cm2=charge, final_minT_C=newT.min())

def _search_original(case, mode, z, dt, record=False):
    cells, ends = _search_inputs()[3][case]
    return _simulate(_search_context(dt), mode, np.array(z), cells, ends, record=record)


def _opt_tracked_command(*args):
    if _opt__STATS is None:
        return _opt__raw(*args)
    q = np.clip(_opt__raw(*args), 0, _opt__STATS['cap'])
    _opt__STATS['peak'] = np.maximum(_opt__STATS['peak'], q)
    _opt__STATS['tau95'] += (q >= 0.95 - 1e-12) * _opt__STATS['dt']
    _opt__STATS['soft'] += np.clip((q - 0.85) / 0.15, 0, 1) ** 2 * _opt__STATS['dt']
    _opt__STATS['heating'] += (q > 1e-10) * _opt__STATS['dt']
    return q

def _opt_old_param(case, mode):
    return np.array(next((r['param'] for r in _opt_BASE if r['case'] == case + 1 and r['mode'] == mode)))

@lru_cache(None)
def _opt_old_result(case, mode):
    return _PRIOR_RESULTS[f'工况{case + 1}_{mode}_复核.json'][-1]

@lru_cache(None)
def _opt_energy_ref(case):
    return _opt_old_result(case, _opt_MODES[1])['total_energy_J']

def _opt_run(case, mode, param, cap=1.0, dt=0.2, original=False, record=False):
    global _opt__STATS
    _opt__STATS = dict(cap=cap, dt=dt, peak=np.zeros(5), tau95=np.zeros(5), soft=np.zeros(5), heating=np.zeros(5))
    r = _search_original(case, mode, np.asarray(param), dt, record=record) if original else _search_simulate(case, mode, np.asarray(param), dt)
    r.update(actual_peak=float(_opt__STATS['peak'].max()), peak_cells=_opt__STATS['peak'].copy(), high_duration95=float(_opt__STATS['tau95'].max()), high_duration95_cells=_opt__STATS['tau95'].copy(), high_exposure=float(_opt__STATS['soft'].max()), high_exposure_cells=_opt__STATS['soft'].copy(), heating_duration_cells=_opt__STATS['heating'].copy(), design_cap=float(cap))
    return r

def _opt_feasible(r, limit=5.0):
    return r['success'] and r['max_delta_T_C'] <= limit + 1e-10 and (r['actual_peak'] <= 1 + 1e-12) and (r['charge_C_cm2'] <= 20 + 1e-09) and (r['min_voltage_V'] >= 0.3) and (r['max_ice_fraction'] < 0.99)

def _opt_score(r, case, w, search_phase=False):
    if not _opt_feasible(r, _opt_TARGET if search_phase else 5.0):
        violation = max(0.0, r['max_delta_T_C'] - (_opt_TARGET if search_phase else 5.0)) + max(0.0, -r.get('final_minT_C', 0))
        violation += 100 * max(0.0, 0.3 - r['min_voltage_V']) + 100 * max(0.0, r['max_ice_fraction'] - 0.99)
        return 10000 + 1000 * violation + 0.0001 * r['total_energy_J']
    return r['total_energy_J'] / _opt_energy_ref(case) + w * r['actual_peak'] ** 2 + w * r['high_exposure'] / _opt_DEADLINE

def _opt_pack(mode, z, cap=1.0):
    return np.asarray(z) if mode == _opt_MODES[0] else np.r_[cap, np.asarray(z[:3]) * z[3] / max(cap, 1e-10), z[4:6], z[7:9]]

def _opt_unpack(mode, p):
    if mode == _opt_MODES[0]:
        return (np.asarray(p), 1.0)
    return (np.r_[p[1:4] * p[0], 1.0, p[4:6], 30.0, p[6:8], 0.02, 0.05], float(p[0]))

def _opt_test():
    checks = []
    for c in range(3):
        for m in _opt_MODES:
            z = _opt_old_param(c, m)
            r = _opt_run(c, m, z, dt=0.05, original=True, record=True)
            tr = r.pop('trajectory')
            q = tr[[f'第{k}片加热功率/W_cm2' for k in range(1, 6)]].to_numpy()[1:]
            ds = np.diff(tr['时间/s'].to_numpy())
            errors = [abs(r['actual_peak'] - q.max()), abs(r['high_duration95'] - np.max(((q >= 0.95 - 1e-12) * ds[:, None]).sum(axis=0))), abs(r['high_exposure'] - np.max((np.clip((q - 0.85) / 0.15, 0, 1) ** 2 * ds[:, None]).sum(axis=0)))]
            assert max(errors) < 1e-08, errors
            old = _PRIOR_RESULTS[f'工况{c + 1}_{m}_复核.json'][0]
            assert abs(r['total_energy_J'] - old['total_energy_J']) < 1e-06
            checks.append(dict(case=c + 1, mode=m, errors=errors))
            print('INSTRUMENT PASS', c + 1, m, flush=True)
    _search_write(_opt_HERE / '目标统计等价核对.json', checks)

def _opt_search_one(case, mode):
    start = time.perf_counter()
    hist = []
    cache = {}
    observed = []
    bounds = np.array([[0, 1]] * 3 + [[0, _opt_DEADLINE]]) if mode == _opt_MODES[0] else np.array([[0, 1]] * 4 + [[0, 85], [15, 97], [0, 0.08], [0, 0.08]])
    lo = bounds[:, 0]
    span = bounds[:, 1] - lo
    n = len(bounds)
    zold = _opt_old_param(case, mode)
    seeds = [_opt_pack(mode, zold)]
    if mode == _opt_MODES[0]:
        for a in [0.72, 0.8, 0.85, 0.9, 0.95] if case == 0 else [0.09, 0.1, 0.12] if case == 1 else [0.55, 0.6, 0.65]:
            for bb, cc in [(0.3, 0.1), (0.23, 0.1), (0.15, 0.1)] if case == 0 else [(0, 0), (0.04, 0.04)]:
                seeds.append(np.array([a, bb, cc, _opt_DEADLINE]))
    else:
        ts = _opt_old_result(case, mode)['time_s']
        on = zold[4]
        for cap in [0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 1.0]:
            for extra in [0.0, 2.0, 5.0]:
                zz = zold.copy()
                zz[:3] = np.minimum(zz[:3] * cap, cap)
                zz[4] = max(0.0, ts - (ts - on) / cap - extra)
                zz[5] = 97.0
                seeds.append(_opt_pack(mode, zz, cap))
        const = json.loads((_opt_HERE / f'工况{case + 1}_{_opt_MODES[0]}_搜索.json').read_text())
        for item in const['candidates']:
            z = np.array(item['param'])
            seeds.append(_opt_pack(mode, np.r_[z[:3], 1.0, 0.0, z[3], 30.0, 0, 0, 0, 0], 1.0))

    def evaluate(p, dt=0.2):
        key = (dt, *np.round(p, 11))
        if key in cache:
            return cache[key]
        z, cap = _opt_unpack(mode, p)
        r = _opt_run(case, mode, z, cap, dt)
        rec = dict(p=np.asarray(p), param=z, cap=cap, dt=dt, result=r)
        cache[key] = r
        hist.append(rec)
        if _opt_feasible(r):
            observed.append(rec)
        return r
    starts = [np.clip(s, lo, lo + span) for s in seeds]
    for w in _opt_WEIGHTS:
        best = min(starts, key=lambda p: _opt_score(evaluate(p), case, w, True))

        def obj(x):
            return _opt_score(evaluate(lo + x * span), case, w, True)
        de = differential_evolution(obj, [(0, 1)] * n, x0=(best - lo) / span, seed=280927 + case * 10 + int(w * 100), popsize=4, maxiter=8, polish=False, tol=0.0001)
        nm = minimize(obj, de.x, method='Nelder-Mead', bounds=[(0, 1)] * n, options={'maxfev': 100, 'xatol': 0.0001, 'fatol': 1e-05})
        starts.extend([lo + de.x * span, lo + nm.x * span])
        print('SEARCH', case + 1, mode, w, len(hist), obj(nm.x), flush=True)
    proposals = [_opt_pack(mode, zold)]
    seen = set()
    for w in _opt_WEIGHTS:
        count = 0
        for rec in sorted(observed, key=lambda v: _opt_score(v['result'], case, w)):
            key = tuple(np.round(rec['p'], 6))
            if key in seen:
                continue
            seen.add(key)
            proposals.append(rec['p'])
            count += 1
            if count >= 3:
                break
    for p in proposals:
        evaluate(p, 0.05)
    for w in _opt_WEIGHTS:
        allowed = [h for h in hist if h['dt'] == 0.05 and _opt_feasible(h['result'])]
        if not allowed:
            continue
        anchor = min(allowed, key=lambda h: _opt_score(h['result'], case, w))['p']
        minimize(lambda x: _opt_score(evaluate(lo + x * span, 0.05), case, w, True), np.clip((anchor - lo) / span, 0, 1), method='Nelder-Mead', bounds=[(0, 1)] * n, options={'maxfev': 85, 'xatol': 0.0001, 'fatol': 1e-05})
    verified = []
    seen = set()
    todo = [_opt_pack(mode, zold)]
    for w in _opt_WEIGHTS:
        valid = [r for r in hist if r['dt'] == 0.05 and _opt_feasible(r['result'])]
        todo.extend([r['p'] for r in sorted(valid, key=lambda h: _opt_score(h['result'], case, w))[:4]])
    for p in todo:
        key = tuple(np.round(p, 8))
        if key in seen:
            continue
        seen.add(key)
        z, cap = _opt_unpack(mode, p)
        rs = [evaluate(p, dt) for dt in [0.05, 0.025, 0.0125]]
        if all((_opt_feasible(r) for r in rs)):
            verified.append(dict(param=z, cap=cap, p=np.asarray(p), results=rs))
    assert verified, (case, mode)
    selected = {str(w): min(verified, key=lambda x: _opt_score(x['results'][-1], case, w)) for w in _opt_WEIGHTS}
    pareto = []
    for a in verified:
        ar = a['results'][-1]
        av = np.array([ar['total_energy_J'], ar['actual_peak'], ar['high_exposure']])
        if not any((np.all(np.array([x['results'][-1]['total_energy_J'], x['results'][-1]['actual_peak'], x['results'][-1]['high_exposure']]) <= av + 1e-09) and np.any(np.array([x['results'][-1]['total_energy_J'], x['results'][-1]['actual_peak'], x['results'][-1]['high_exposure']]) < av - 1e-09) for x in verified)):
            pareto.append(a)
    result = dict(case=case + 1, mode=mode, weights=_opt_WEIGHTS, selected=selected, candidates=pareto, energy_reference=_opt_energy_ref(case), evaluations=len(hist), seconds=time.perf_counter() - start)
    _search_write(_opt_HERE / f'工况{case + 1}_{mode}_搜索.json', result)
    _search_write(_opt_HERE / f'工况{case + 1}_{mode}_搜索历史.json', hist)
    print('DONE', case + 1, mode, {w: (s['results'][-1]['total_energy_J'], s['results'][-1]['actual_peak'], s['results'][-1]['high_duration95']) for w, s in selected.items()}, flush=True)
    return result

def _opt_search():
    assert (_opt_HERE / '目标统计等价核对.json').exists()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(_opt_search_one, c, _opt_MODES[0]) for c in range(3)]):
            f.result()
        for f in as_completed([ex.submit(_opt_search_one, c, _opt_MODES[1]) for c in range(3)]):
            f.result()

def _opt_verify_one(case, mode):
    d = json.loads((_opt_HERE / f'工况{case + 1}_{mode}_搜索.json').read_text())
    rows = []
    cache = {}
    for w in _opt_WEIGHTS:
        choice = d['selected'][str(w)]
        z = np.array(choice['param'])
        cap = choice['cap']
        for dt in [0.05, 0.025, 0.0125]:
            key = (tuple(z), cap, dt)
            if key not in cache:
                r = _opt_run(case, mode, z, cap, dt, original=True, record=True)
                tr = r.pop('trajectory')
                assert _opt_feasible(r), (case, mode, w, dt, r)
                q = tr[[f'第{k}片加热功率/W_cm2' for k in range(1, 6)]].to_numpy()
                t = tr['时间/s'].to_numpy()
                ds = np.diff(t)
                e = 25 * (q[1:] * ds[:, None]).sum(axis=0)
                assert np.max(abs(e - r['energy_cells_J'])) < 1e-06
                assert q.max() <= cap + 1e-12 and q.min() >= 0
                assert np.max(abs(tr['累计电荷量/C_cm2'] - (0.0025 * np.minimum(t, 60) ** 2 + 0.3 * np.maximum(t - 60, 0)))) < 1e-08
                assert tr['五片温差/℃'].max() <= 5 + 1e-10
                assert abs(r['high_duration95'] - np.max(((q[1:] >= 0.95 - 1e-12) * ds[:, None]).sum(axis=0))) < 1e-08
                assert abs(r['high_exposure'] - np.max((np.clip((q[1:] - 0.85) / 0.15, 0, 1) ** 2 * ds[:, None]).sum(axis=0))) < 1e-08
                fast = _opt_run(case, mode, z, cap, dt)
                assert max((abs(fast[k] - r[k]) for k in ['total_energy_J', 'time_s', 'max_delta_T_C', 'actual_peak', 'high_duration95', 'high_exposure'])) < 1e-06
                filename = f'工况{case + 1}_{mode}_{_opt_NAMES[w]}_dt{dt}.csv'
                tr.to_csv(_opt_HERE / filename, index=False, encoding='utf-8-sig')
                r['trajectory_file'] = filename
                cache[key] = r
            r = dict(cache[key])
            r.update(case=case + 1, mode=mode, weight=w, label=_opt_NAMES[w], dt=dt, param=z, objective=_opt_score(r, case, w))
            rows.append(r)
    baseline = _opt_run(case, mode, _opt_old_param(case, mode), 1.0, 0.0125, original=True, record=False)
    baseline.pop('trajectory', None)
    for w in _opt_WEIGHTS:
        new = next((r for r in rows if r['weight'] == w and r['dt'] == 0.0125))
        assert new['objective'] <= _opt_score(baseline, case, w) + 1e-06, (case, mode, w, new['objective'], _opt_score(baseline, case, w))
    _search_write(_opt_HERE / f'工况{case + 1}_{mode}_旧参数复核.json', baseline)
    _search_write(_opt_HERE / f'工况{case + 1}_{mode}_复核.json', rows)
    print('VERIFIED', case + 1, mode, flush=True)
    return rows

def _opt_verify():
    with ProcessPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(_opt_verify_one, c, m) for c in range(3) for m in _opt_MODES]):
            f.result()

def _opt_scan_case(case):
    mode = _opt_MODES[1]
    old = _opt_old_param(case, mode)
    rat = old[1:3] / old[0]
    const = json.loads((_opt_HERE / f'工况{case + 1}_{_opt_MODES[0]}_搜索.json').read_text())['selected']['0.2']['param']
    cr = np.array(const[1:3]) / const[0]
    ratios = [rat * x for x in [1.0, 0.95, 0.9, 0.8, 0.6]] + [cr]
    hist = []
    seeds = []

    def eval(cap, rr, on, dt=0.1):
        z = np.r_[cap, cap * rr, 1.0, on, 97.0, 30.0, 0.0, 0.0, 0.0, 0.0]
        r = _opt_run(case, mode, z, cap, dt)
        hist.append(dict(param=z, cap=cap, dt=dt, result=r))
        return (z, r)
    for cap in [0.8, 0.85, 0.9, 0.94]:
        for rr in ratios:
            grid = [0.0, 20.0, 40.0, 60.0, 75.0, 85.0]
            passed = []
            for on in grid:
                z, r = eval(cap, rr, on)
                if _opt_feasible(r, _opt_TARGET):
                    passed.append((on, z, r))
            if not passed:
                continue
            low = max((x[0] for x in passed))
            higher = [t for t in grid if t > low]
            if higher:
                high = min(higher)
                for _ in range(9):
                    mid = (low + high) / 2
                    z, r = eval(cap, rr, mid)
                    if _opt_feasible(r, _opt_TARGET):
                        low = mid
                        passed.append((mid, z, r))
                    else:
                        high = mid
            for w in _opt_WEIGHTS:
                _, z, r = min(passed, key=lambda x: _opt_score(x[2], case, w))
                seeds.append((z, cap))
        print('CAP SCAN', case + 1, cap, len(hist), flush=True)
    feasible_points = []
    seen = set()
    for z, cap in seeds:
        key = (tuple(np.round(z, 9)), cap)
        if key in seen:
            continue
        seen.add(key)
        rs = [_opt_run(case, mode, z, cap, dt) for dt in [0.05, 0.025, 0.0125]]
        if all((_opt_feasible(r) for r in rs)):
            feasible_points.append(dict(param=z, cap=cap, p=_opt_pack(mode, z, cap), results=rs))
    _search_write(_opt_HERE / f'工况{case + 1}_峰值分档候选.json', feasible_points)
    _search_write(_opt_HERE / f'工况{case + 1}_峰值分档搜索历史.json', hist)
    print('CAP DONE', case + 1, len(feasible_points), flush=True)

def _opt_merge():
    for case in [0, 2]:
        path = _opt_HERE / f'工况{case + 1}_{_opt_MODES[1]}_搜索.json'
        d = json.loads(path.read_text())
        extra = json.loads((_opt_HERE / f'工况{case + 1}_峰值分档候选.json').read_text())
        const = json.loads((_opt_HERE / f'工况{case + 1}_{_opt_MODES[0]}_搜索.json').read_text())
        for item in const['selected'].values():
            q = np.array(item['param'])
            z = np.r_[q[:3], 1.0, 0.0, q[3], 30.0, 0.0, 0.0, 0.0, 0.0]
            cap = float(max(q[:3]))
            rs = [_opt_run(case, _opt_MODES[1], z, cap, dt) for dt in [0.05, 0.025, 0.0125]]
            if all((_opt_feasible(r) for r in rs)):
                extra.append(dict(param=z, cap=cap, p=_opt_pack(_opt_MODES[1], z, cap), results=rs, note='恒功率嵌入动态控制族，四项反馈增益均为0'))
        pool = d['candidates'] + list(d['selected'].values()) + extra
        d['before_cap_scan'] = d['selected']
        d['supplemental_candidates'] = extra
        nd = []
        for a in pool:
            ar = a['results'][-1]
            av = np.array([ar['total_energy_J'], ar['actual_peak'], ar['high_exposure']])
            if not any((np.all(np.array([x['results'][-1]['total_energy_J'], x['results'][-1]['actual_peak'], x['results'][-1]['high_exposure']]) <= av + 1e-09) and np.any(np.array([x['results'][-1]['total_energy_J'], x['results'][-1]['actual_peak'], x['results'][-1]['high_exposure']]) < av - 1e-09) for x in pool)):
                nd.append(a)
        d['candidates'] = nd
        for w in _opt_WEIGHTS:
            d['selected'][str(w)] = min(pool, key=lambda x: _opt_score(x['results'][-1], case, w))
        _search_write(path, d)
        print('MERGED', case + 1, {w: (s['results'][-1]['total_energy_J'], s['results'][-1]['actual_peak']) for w, s in d['selected'].items()}, flush=True)

_heater_command = _opt_tracked_command


def get_figure_contract():
    return {
      'evidence':'7-node pre-cooling temperature field from 10–100 min; three-case constant/dynamic power and thermal response; cooling duration versus auxiliary energy, startup time, temperature spread and ice',
      'layout':'new evidence-led Chinese composite figures at 600 DPI; no old-image reuse'
    }


def get_render_dependency_names():
    return ['matplotlib','numpy']


def render_figures(results, output_dir, logger):
    return render_question('Q4', results, output_dir, logger)

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
    return _q4(results, output_dir, logger)

def _q4(r, out, log):
    cooling = r['cooling_field']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for minute, color in [(20, TEAL), (40, ORANGE)]:
        d = cooling[cooling['预冷时间/min'] == minute]
        ax[0].plot(d['叠层坐标/mm'], d['温度/℃'], color=color, lw=1.6, label=f'{minute} min')
    scan = r['scan']
    ax[1].plot(scan['预冷时间/min'], scan['辅助能耗/J'], marker='o', color=PLUM, lw=1.6)
    ax[0].set(xlabel='叠层位置 / mm', ylabel='预冷末温度 / ℃')
    ax[1].set(xlabel='预冷时间 / min', ylabel='恒功率辅助能耗 / J')
    _panel(ax[0], 'a  端板至膜电极的初温剖面')
    _panel(ax[1], 'b  预冷进程带来的能耗变化')
    ax[0].legend(frameon=False)
    paths = [_save(fig, out, 'Q4', 'Q4_预冷剖面与能耗路径.png')]
    table = r['table']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for mode, color, marker in [('恒功率策略', GREY, 's'), ('动态功率策略', TEAL, 'o')]:
        d = table[table['控制策略'] == mode]
        ax[0].scatter(d['启动时间/s'], d['辅助加热总能耗/J'], color=color, marker=marker, s=80, label=mode)
        for _, row in d.iterrows():
            ax[0].annotate(row['工况'], (row['启动时间/s'], row['辅助加热总能耗/J']), xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax[1].plot(d['工况'], d['最大冰体积分数'], marker=marker, color=color, lw=1.5, label=mode)
    ax[0].set(xlabel='启动时间 / s', ylabel='辅助能耗 / J')
    ax[1].set(xlabel='预冷工况', ylabel='最大局部冰体积分数')
    _panel(ax[0], 'a  三种初态的时间–能耗权衡')
    _panel(ax[1], 'b  节能策略的冰风险代价')
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    paths.append(_save(fig, out, 'Q4', 'Q4_三工况控制权衡.png'))
    log.status('Q4重设计证据图生成：2张600 DPI PNG')
    return paths


def export_tables(results,output_dir,logger):
    from pathlib import Path
    out=Path(output_dir);files=[]
    table=results['table'].copy()
    parameters=[]
    for case in ('工况1','工况2','工况3'):
        p,_=results['selected'][(case,'动态功率策略')]
        base=[float(p[0]),float(p[1]),float(p[2]),float(p[1]),float(p[0])]
        parameters.append({'工况':case,'基准功率密度q1/W_cm2':base[0],
                           '基准功率密度q2/W_cm2':base[1],'基准功率密度q3/W_cm2':base[2],
                           '基准功率密度q4/W_cm2':base[3],'基准功率密度q5/W_cm2':base[4],
                           '设计功率上限/W_cm2':float(_FINAL_SELECTED[int(case[-1])-1]['design_cap']),
                           '基准倍率s':float(p[3]),'开启时刻/s':float(p[4]),
                           '最迟停止时刻/s':float(p[5]),'预测窗口调节量/℃':float(p[6]),
                           '预测补热增益':float(p[7]),'温升不足增益':float(p[8]),
                           '电压预测保护增益':float(p[9]),'电压下降增益':float(p[10])})
        table.loc[(table['工况']==case)&(table['控制策略']=='动态功率策略'),'功率控制策略/W·cm⁻²']=(
            (f'延迟恒功率：{p[4]:.1f}—{p[5]:.1f}s，反馈增益为0' if np.all(p[7:11]==0) else f'滚动预测：基准×{p[3]:.3f}；{p[4]:.1f}—{p[5]:.1f}s；逐片T、dT/dt、V、dV/dt反馈'))
    feedback=pd.DataFrame(parameters)
    for name,frame in [('Q4_表4三工况策略比较.csv',table),
                       ('Q4_预冷温度场.csv',results['cooling']),
                       ('Q4_预冷全厚度温度场.csv',results['cooling_field']),
                       ('Q4_预冷10至100分钟扫描.csv',results['scan']),
                       ('Q4_控制时序.csv',results['tracks']),
                       ('Q4_分片指标.csv',results['cell_metrics']),
                       ('Q4_反馈控制参数.csv',feedback)]:
        files.append(write_dataframe_csv(frame,out/name))
    note='初始均为25℃，环境−30℃；工况1直接均匀−30℃，工况2/3分别取20/40 min预冷末温度场。电流曲线为0.005 A·cm⁻²·s⁻¹升载60 s后保持0.3；沿用20 C·cm⁻²电荷预算。动态律只读取实时温度、电压及其变化率，冰量仅作模型安全核验。'
    scan=results['scan'][['预冷时间/min','端片初温/℃','中片初温/℃','初始片间温差/℃',
                           '启动时间/s','辅助能耗/J','最低电压/V','最大冰体积分数','启动结果']].round(3)
    f_small=feedback[['工况','基准倍率s','开启时刻/s','最迟停止时刻/s','预测窗口调节量/℃',
                      '预测补热增益','温升不足增益','电压预测保护增益','电压下降增益']].round(4)
    formula=('滚动窗口 H=clip(6+预测窗口调节量/15,4,12) s，剩余时间 R=max(t关−t,H)；'
             '目标升温率=max(−T,0)/R，热源增益G=10000/Σ(cpᵢΔxᵢ)，预测功率q̂=clip[(目标升温率−dT/dt)/G,0,1]。'
             '在开启窗口内，逐片功率为 clip{max(sq₀,k,10g₁q̂,g₃clip[(0.35−V−H·dV/dt)/0.5,0,1])'
             '+g₂(0.15−dT/dt)₊+g₄(−dV/dt−0.002)₊,0,设计功率上限}；达到目标后立即关断。'
             '冰相仅作机理模型安全核验，不作传感输入。')
    files.append(write_markdown_tables([
        ('表4 不同预冷工况下辅助加热控制策略优化结果对比',table,note),
        ('预冷10—100 min温度场及恒功率冷启动结果',scan,
         '温度场来自端板—双极板—膜电极全厚度有限体积模型。计算所得40 min片间温差小于20 min，与题述“更明显”不一致；未人为修改物性使其吻合。'),
        ('动态功率反馈律参数',f_small,formula)
    ],out/'表格输出.md'))
    logger.status('已导出7个CSV与表格输出.md')
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='第四问最终控制方案：复算或重新优化')
    parser.add_argument('--search', action='store_true', help='重新执行多目标搜索、峰值分档补搜和三步长核验')
    args = parser.parse_args()
    if args.search:
        _opt_HERE.mkdir(parents=True, exist_ok=True)
        _opt_test()
        _opt_search()
        for case in [0, 2]:
            _opt_scan_case(case)
        _opt_merge()
        _opt_verify()
        for case in range(3):
            selected = json.loads((_opt_HERE/f'工况{case+1}_动态功率策略_搜索.json').read_text())['selected']['0.2']
            _FINAL_SELECTED[case].update(param=selected['param'], design_cap=selected['cap'])
        global _opt__STATS
        _opt__STATS = None
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
