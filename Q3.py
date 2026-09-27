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


QUESTION_ID = "Q3"
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
from scipy.linalg import lu_factor, lu_solve
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
    return [_attachment('附件1.xlsx'),root/'Q1'/'Q1_模型参数.csv',root/'Q2'/'Q2_表3策略比较.csv']


def get_analysis_contract():
    return {
        'question':'Q3',
        'physics':'Q2 five-cell finite-volume model plus five independent BP-equivalent boundary heater sources',
        'initial_and_ambient_C':-30.0,'heater_area_cm2':25.0,'heater_bounds_W_cm2':[0,1],
        'preheat':'zero current to all five cells above zero, then Q2 optimized rising linear current',
        'coheat':'j=min(0.005*t,0.3) A/cm2 from t=0; constant individual powers until optimized off-time',
        'success':'all cell mean T>0 C, all local ice volume fractions<0.99, all cell V>=0.3 V throughout',
        'charge_cap_C_cm2':20.0,
        'charge_cap_reason':'carry Q2 operating constraint to avoid an auxiliary-energy zero solution that starts only after the Q2 charge budget',
        'objective':'minimum total auxiliary energy 25*sum(q_k)*heating duration J',
        'search':'mirror-symmetric five powers represented by three independent levels plus common duration; mesh-adaptive direct search; full 27-grid/0.2s optimization and 0.05s verification',
        'seed':20260924,
    }


def get_compute_dependency_names():
    return ['numpy','pandas','scipy','openpyxl']


def _simulate_aux(ctx,mode,powers,heat_time,pre_jmin=0.1535,pre_k=0.06823,
                  record=False,max_time=180):
    p=ctx['par']; g=ctx['grid']; theta=ctx['theta']; dt=ctx['dt']; dx=g['dx']; n=len(dx)
    q=np.asarray(powers,dtype=float)
    if q.shape!=(5,) or np.any(q<0) or np.any(q>1):
        raise ValueError('五片加热功率密度须处于0—1 W/cm2')
    temp=np.full((5,n),243.15); mobile=np.zeros((5,n)); ice=np.zeros((5,n))
    mobile[:,g['mem']]=float(p['初始膜含水量'])*2150*0.018/(float(p['膜当量质量'])/1000)
    end=np.full(2,243.15);ambient=243.15
    polar=0.; charge=0.;last_j=0.;min_v=np.inf;max_ice=0.;reason='时限内未成功'
    rows=[]; used_heat=0.;preheat_ready=False;preheat_minT=np.nan
    max_end_ice=0.;max_mid_ice=0.

    def snapshot(t,j,volts,icecell,meanT,qeff):
        row={'时间/s':float(t),'电流密度/A_cm2':float(j),'最低单片温度/℃':float(np.min(meanT)-273.15),
             '最低单片电压/V':float(np.min(volts)),'最大局部冰体积分数':float(np.max(icecell)),
             '辅助累计能耗/J':25*float(np.sum(q))*used_heat,'加热开关':float(qeff),
             '左端板温度/℃':float(end[0]-273.15),'右端板温度/℃':float(end[1]-273.15)}
        for k in range(5):
            row[f'第{k+1}片温度/℃']=float(meanT[k]-273.15)
            row[f'第{k+1}片电压/V']=float(volts[k])
            row[f'第{k+1}片最大冰体积分数']=float(icecell[k])
            row[f'第{k+1}片加热功率密度/W_cm2']=float(q[k]*qeff)
        rows.append(row)

    def electro(j):
        loss=polar+theta[4]*np.exp(-charge/theta[5]);vv=[];ii=[]
        for k in range(5):
            v,a=_voltage(j,temp[k],mobile[k],ice[k],g,p,loss,theta[0]);vv.append(v);ii.append(a)
        return np.asarray(vv),np.asarray([float(np.max(a['ice_vol'])) for a in ii])

    vv,ic=electro(0);meanT=np.average(temp,axis=1,weights=dx)
    if record:snapshot(0,0,vv,ic,meanT,0)
    for step in range(1,int(np.ceil(max_time/dt))+1):
        t0=(step-1)*dt;t=step*dt
        prev_charge=charge;prev_minT=float(np.min(meanT)-273.15)
        qeff=np.clip((heat_time-t0)/dt,0,1)
        if mode=='纯预热启动':
            j=0. if t0<heat_time else min(pre_jmin+pre_k*(t0-heat_time),0.5)
        else:
            j=min(0.005*t0,0.3)
        polar=polar*np.exp(-dt/theta[3])+theta[2]*np.exp(-charge/theta[6])*max(j-last_j,0)

        if mode=='纯预热启动':
            charge+=j*dt
        else:
            charge+=0.0025*(min(t,60.0)**2-min(t0,60.0)**2)
            charge+=0.3*max(t-max(t0,60.0),0.0)
        used_heat+=qeff*dt
        source=np.zeros(n)
        source[g['ccl']]=0.018*(j*1e4)/(2*96485*float(p['阴极 CL 厚度']))
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
        vv,ic=electro(j)
        min_v=min(min_v,float(np.min(vv)));max_ice=max(max_ice,float(np.max(ic)))
        max_end_ice=max(max_end_ice,float(max(ic[0],ic[4])))
        max_mid_ice=max(max_mid_ice,float(np.max(ic[1:4])))
        meanT=np.average(temp,axis=1,weights=dx)
        inter=ctx['G_inter']*(meanT[:-1]-meanT[1:])
        flux=np.zeros(5);flux[:-1]-=inter;flux[1:]+=inter
        left=ctx['G_end']*(end[0]-meanT[0]);right=ctx['G_end']*(end[1]-meanT[-1])
        flux[0]+=left-ctx['h']*(meanT[0]-ambient)
        flux[-1]+=right-ctx['h']*(meanT[-1]-ambient)
        end[0]+=dt*(-left-ctx['h']*(end[0]-ambient))/ctx['C_end']
        end[1]+=dt*(-right-ctx['h']*(end[1]-ambient))/ctx['C_end']
        for k in range(5):
            qgen=j*1e4*(1.48-vv[k])/g['length']
            heater=np.zeros(n)

            heater[0]=qeff*q[k]*1e4/(2*dx[0]);heater[-1]=qeff*q[k]*1e4/(2*dx[-1])
            rhs=temp[k]+dt*(qgen+flux[k]/g['length']+heater+
                             float(p['水冻结潜热'])*(freeze[k]-melt[k])/dt)/ctx['cp']
            temp[k]=lu_solve(ctx['heat_lu'],rhs)
        meanT=np.average(temp,axis=1,weights=dx);minT=float(np.min(meanT)-273.15)
        if mode=='纯预热启动' and not preheat_ready and t>=heat_time-1e-9:
            preheat_ready=minT>0;preheat_minT=minT
            if not preheat_ready:reason='预热结束时仍有单片低于0℃';break
        if record:snapshot(t,j,vv,ic,meanT,qeff)
        if mode!='纯预热启动' or t>heat_time:
            if min_v<0.30:reason='单片电压低于0.30 V';break
            if max_ice>=0.99:reason='局部冰体积分数达到0.99';break
            if minT>0 and j>0:
                frac=1.0 if mode=='纯预热启动' else np.clip(-prev_minT/max(minT-prev_minT,1e-12),0,1)
                ts=t0+dt*frac
                if mode=='纯预热启动':
                    qsuccess=prev_charge+j*dt*frac
                else:
                    ts_clip=t0+dt*frac
                    qsuccess=prev_charge+0.0025*(min(ts_clip,60.0)**2-min(t0,60.0)**2)
                    qsuccess+=0.3*max(ts_clip-max(t0,60.0),0.0)
                if qsuccess<=20+1e-9:
                    true_heat=min(float(heat_time),ts)
                    return {'success':True,'reason':'成功','time_s':ts,'heating_s':true_heat,
                            'q':q,'energies_J':25*q*true_heat,'total_energy_J':25*float(np.sum(q))*true_heat,
                            'min_voltage_V':min_v,'max_ice_fraction':max_ice,
                            'max_end_ice':max_end_ice,'max_mid_ice':max_mid_ice,
                            'final_minT_C':minT,'preheat_minT_C':preheat_minT,
                            'charge_C_cm2':qsuccess,'trajectory':pd.DataFrame(rows) if record else None}
        if charge>20:
            reason='累计电荷达到20 C·cm⁻²';charge=20.
            capfrac=np.clip((20-prev_charge)/max(j*dt,1e-12),0,1)
            minT=prev_minT+capfrac*(minT-prev_minT)
            break
        last_j=j
    return {'success':False,'reason':reason,'time_s':t,'heating_s':used_heat,'q':q,
            'energies_J':25*q*used_heat,'total_energy_J':25*float(np.sum(q))*used_heat,
            'min_voltage_V':min_v,'max_ice_fraction':max_ice,
            'max_end_ice':max_end_ice,'max_mid_ice':max_mid_ice,
            'final_minT_C':minT,'preheat_minT_C':preheat_minT,
            'charge_C_cm2':charge,'trajectory':pd.DataFrame(rows) if record else None}


def _objective(z,ctx,mode,pre_jmin,pre_k):
    r=_simulate_aux(ctx,mode,z[:5],z[5],pre_jmin,pre_k)
    if r['success']:return r['total_energy_J']+0.001*r['time_s']
    deficit=max(0,-r['final_minT_C'])
    return 10000+r['total_energy_J']+50*deficit+1000*max(0,0.30-r['min_voltage_V'])


def _mirror(z):
    return np.asarray([z[0],z[1],z[2],z[1],z[0],z[3]],dtype=float)


def _search(ctx,mode,seed,pre_jmin,pre_k,starting=None,maxiter=6):


    low=np.array([0.,0.,0.,4.]);high=np.array([1.,1.,1.,130.])
    obj=lambda z:_objective(_mirror(np.clip(z,low,high)),ctx,mode,pre_jmin,pre_k)
    seeds=[]
    if starting is not None:seeds.append(np.asarray([starting[0],starting[1],starting[2],starting[5]]))
    if mode=='纯预热启动':seeds.append(np.asarray([.219,.545,.141,66.0]))
    for power in [0.25,0.5,0.8,1.0]:
        for duration in [60.,90.,120.]:seeds.append(np.asarray([power]*3+[duration]))
    best=min(seeds,key=obj);value=obj(best)
    mesh=np.array([.2,.2,.2,20.])
    for iteration in range(6):
        candidates=[]
        for axis in range(4):
            for direction in (-1.,1.):
                z=best.copy();z[axis]+=direction*mesh[axis]
                z=np.clip(z,low,high)
                candidates.append((obj(z),z))
        new_value,new_point=min(candidates,key=lambda item:item[0])
        if new_value+1e-7<value:
            value,best=new_value,new_point
            mesh=np.minimum(mesh*1.15,[.3,.3,.3,30.])
        else:
            mesh*=0.5
        if np.max(mesh[:3])<.006 and mesh[3]<.6:break
    return _mirror(best)


def compute_results(logger):
    inputs=get_input_paths();par=_number_map(pd.read_excel(inputs[0],sheet_name='参数清单'))
    theta=_calibrated_theta(inputs[1]);q2=pd.read_csv(inputs[2])
    import re
    linear=str(q2.loc[q2['加载策略']=='线性升载','最优加载参数'].iloc[0])
    pre_jmin=float(re.search(r'jmin=([0-9.]+)',linear).group(1))
    pre_k=float(re.search(r'k=([0-9.]+)',linear).group(1))
    full=_grid(par);coarse=_coarse_grid(full)
    search_ctx=_context(par,coarse,theta,1.0)
    final_ctx=_context(par,full,theta,0.2)
    verify_ctx=_context(par,full,theta,0.05)
    modes=['纯预热启动','恒定功率协同启动'];results={};records=[];summary=[];alloc=[]
    for idx,mode in enumerate(modes):

        baseline=_simulate_aux(search_ctx,mode,[0.8]*5,90,pre_jmin,pre_k)
        logger.status(f'{mode}快速可行性：{baseline["reason"]}，加热={baseline["heating_s"]:.1f}s')
        x=_search(search_ctx,mode,20260924+idx,pre_jmin,pre_k,maxiter=7)
        trial=_simulate_aux(final_ctx,mode,x[:5],x[5],pre_jmin,pre_k)
        if not trial['success']:
            logger.warning(f'{mode}粗网格解未通过完整模型，启动完整网格局部修正')
            bounds=[(0,1)]*3+[(4,130)]
            low=np.asarray([b[0] for b in bounds]);high=np.asarray([b[1] for b in bounds])
            obj4=lambda z:_objective(_mirror(np.clip(z,low,high)),final_ctx,mode,pre_jmin,pre_k)
            z0=np.asarray([x[0],x[1],x[2],x[5]])
            local=minimize(obj4,z0,method='Nelder-Mead',bounds=bounds,
                           options={'maxiter':70,'xatol':0.002,'fatol':0.1})
            candidates=[_mirror(np.clip(local.x,low,high)),_mirror(z0),
                        np.asarray([0.8]*5+[90]),np.asarray([1.0]*5+[120])]
            if mode=='纯预热启动':candidates.append(_mirror([.219,.545,.141,66.0]))
            x=min(candidates,key=lambda z:_objective(z,final_ctx,mode,pre_jmin,pre_k))
            trial=_simulate_aux(final_ctx,mode,x[:5],x[5],pre_jmin,pre_k)
        if not trial['success']:
            raise RuntimeError(f'{mode}未找到满足成功判据的可行解；{trial["reason"]}')

        x_sym=np.asarray([x[0],x[1],x[2],x[5]])
        local=minimize(lambda z:_objective(_mirror(z),final_ctx,mode,pre_jmin,pre_k),x_sym,method='SLSQP',
                       bounds=[(0,1)]*3+[(4,130)],options={'maxiter':25,'ftol':0.05,'disp':False})
        if _objective(_mirror(local.x),final_ctx,mode,pre_jmin,pre_k)<_objective(x,final_ctx,mode,pre_jmin,pre_k):x=_mirror(local.x)
        verified=_simulate_aux(verify_ctx,mode,x[:5],x[5],pre_jmin,pre_k)
        if not verified['success']:
            logger.warning(f'{mode}在0.05 s步长下未通过，搜索邻域中能耗最低的可行修正')
            candidates=[]
            for scale in [1.0,1.01,1.03]:
                for extra in [0.2,0.5,1.0]:
                    z=np.r_[np.minimum(x[:5]*scale,1.0),min(x[5]+extra,130.0)]
                    vr=_simulate_aux(verify_ctx,mode,z[:5],z[5],pre_jmin,pre_k)
                    if vr['success']:candidates.append((vr['total_energy_J'],z))
            if not candidates:
                z=np.asarray([0.8]*5+[90.0]);vr=_simulate_aux(verify_ctx,mode,z[:5],z[5],pre_jmin,pre_k)
                if vr['success']:candidates.append((vr['total_energy_J'],z))
            if not candidates:raise RuntimeError(f'{mode}的粗步长解无法在0.05 s步长下修复')
            x=min(candidates,key=lambda z:z[0])[1]
        r=_simulate_aux(verify_ctx,mode,x[:5],x[5],pre_jmin,pre_k,record=True)
        results[mode]=r
        tr=r['trajectory'];tr.insert(0,'辅助策略',mode);records.append(tr)
        qtext='('+', '.join(f'{v:.3f}' for v in r['q'])+')'
        etext='('+', '.join(f'{v:.1f}' for v in r['energies_J'])+')'
        summary.append({'辅助冷启动策略':mode,'加热功率密度分配 /W·cm⁻²':qtext,
                        '加热持续时间 /s':round(r['heating_s'],2),'各片辅助加热能耗/J':etext,
                        '总辅助加热能耗/J':round(r['total_energy_J'],2),
                        '总启动时间/s':round(r['time_s'],2),'最大冰体积分数':round(r['max_ice_fraction'],5),
                        '启动结果':'成功' if r['success'] else '失败：'+r['reason']})
        for k in range(5):
            alloc.append({'辅助策略':mode,'单片编号':k+1,'功率密度/W_cm2':r['q'][k],
                          '辅助能耗/J':r['energies_J'][k],
                          '角色':'端部' if k in (0,4) else '中间'})
        logger.core(f'{mode}：Eaux={r["total_energy_J"]:.2f} J，t_s={r["time_s"]:.2f} s，'
                    f't_h={r["heating_s"]:.2f} s，Vmin={r["min_voltage_V"]:.3f} V，'
                    f'εice,max={r["max_ice_fraction"]:.4f}')
    logger.core('协同相对预热：能耗变化='+f'{100*(results[modes[1]]["total_energy_J"]/results[modes[0]]["total_energy_J"]-1):.1f}%'+
                '，启动时间变化='+f'{100*(results[modes[1]]["time_s"]/results[modes[0]]["time_s"]-1):.1f}%')
    logger.emit('BP加热源投影至两侧相邻控制体；Q2等效端部换热边界继续沿用。优化为多起点启发式，最小值指搜索得到的可行最小值。','DETAIL')
    return {'table':pd.DataFrame(summary),'allocation':pd.DataFrame(alloc),
            'trajectories':pd.concat(records,ignore_index=True),'metrics':results,
            'preheat_current':{'jmin':pre_jmin,'k':pre_k}}


def validate_computation(results,logger):
    if len(results['table'])!=2:raise ValueError('表4策略数不等于2')
    for mode,r in results['metrics'].items():
        if not r['success']:raise ValueError(f'{mode}未启动成功')
        if np.min(r['q'])<-1e-9 or np.max(r['q'])>1+1e-9:raise ValueError('电热丝功率越界')
        if abs(r['q'][0]-r['q'][4])>1e-7 or abs(r['q'][1]-r['q'][3])>1e-7:raise ValueError('对称工况功率分配未保持镜像对称')
        if r['min_voltage_V']<0.30-1e-8 or r['max_ice_fraction']>=0.99:raise ValueError('安全条件未满足')
        if r['charge_C_cm2']>20+1e-8:raise ValueError('累计电荷越界')
        if abs(np.sum(r['energies_J'])-r['total_energy_J'])>1e-7:raise ValueError('能耗不守恒')
        if mode=='纯预热启动' and not r['preheat_minT_C']>0:raise ValueError('纯预热截止温度未超过0℃')
    logger.status('计算验收通过：两策略成功，五片功率、全程电压与冰阈值及能耗积分均满足')


def get_figure_contract():
    return {
      'evidence':'two optimized strategies: five heater powers, cell temperatures, voltage, ice, energy trajectories and allocation',
      'layout':'new evidence-led Chinese composite figures at 600 DPI; no old-image reuse'
    }


def get_render_dependency_names():
    return ['matplotlib','numpy']


def render_figures(results, output_dir, logger):
    return render_question('Q3', results, output_dir, logger)

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
    return _q3(results, output_dir, logger)

def _q3(r, out, log):
    table = r['table']
    alloc = r['allocation']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    modes = table['辅助冷启动策略'].tolist()
    for i, row in table.iterrows():
        ax[0].scatter(row['总启动时间/s'], row['总辅助加热能耗/J'], color=[TEAL, ORANGE][i], s=150, label=row['辅助冷启动策略'])
        ax[0].annotate(row['辅助冷启动策略'], (row['总启动时间/s'], row['总辅助加热能耗/J']), xytext=(8, 5), textcoords='offset points', fontsize=8)
    matrix = np.vstack([alloc[alloc['辅助策略'] == mode].sort_values('单片编号')['辅助能耗/J'].to_numpy() for mode in modes])
    image = ax[1].imshow(matrix, aspect='auto', cmap='YlOrBr', vmin=0)
    ax[1].set_xticks(range(5), [f'第{i}片' for i in range(1, 6)])
    ax[1].set_yticks(range(2), modes)
    fig.colorbar(image, ax=ax[1], label='单片辅助能耗 / J')
    ax[0].set(xlabel='总启动时间 / s', ylabel='辅助总能耗 / J')
    _panel(ax[0], 'a  节能与启动速度的取舍')
    ax[1].set_title('b  热量集中在哪些单片', loc='left', weight='bold', pad=10)
    paths = [_save(fig, out, 'Q3', 'Q3_时间能耗与分片热量.png')]
    tr = r['trajectories']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for mode, color in zip(modes, [TEAL, ORANGE]):
        d = tr[tr['辅助策略'] == mode]
        ax[0].plot(d['时间/s'], d['最低单片温度/℃'], lw=1.7, color=color, label=mode)
        ax[1].plot(d['时间/s'], d['最大局部冰体积分数'], lw=1.7, color=color, label=mode)
    ax[0].axhline(0, color=INK, lw=0.8, ls='--')
    ax[0].set(xlabel='时间 / s', ylabel='最低单片温度 / ℃')
    ax[1].set(xlabel='时间 / s', ylabel='最大局部冰体积分数')
    _panel(ax[0], 'a  达到全堆过零的时间路径')
    _panel(ax[1], 'b  两种加热时序的结冰代价')
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    paths.append(_save(fig, out, 'Q3', 'Q3_升温路径与冰风险.png'))
    log.status('Q3重设计证据图生成：2张600 DPI PNG')
    return paths


def export_tables(results,output_dir,logger):
    from pathlib import Path
    import pandas as pd
    out=Path(output_dir);files=[]
    for name,frame in [('Q3_表4策略比较.csv',results['table']),
                       ('Q3_各片能耗分配.csv',results['allocation']),
                       ('Q3_全时段轨迹.csv',results['trajectories'])]:
        files.append(write_dataframe_csv(frame,out/name))
    metrics=[]
    for mode,r in results['metrics'].items():
        metrics.append({'策略':mode,'加热时间/s':r['heating_s'],'总启动时间/s':r['time_s'],
                        '总辅助能耗/J':r['total_energy_J'],'最低单片电压/V':r['min_voltage_V'],
                        '最大局部冰体积分数':r['max_ice_fraction'],
                        '端片最大冰体积分数':r['max_end_ice'],
                        '中片最大冰体积分数':r['max_mid_ice'],
                        '累计电荷量/C_cm2':r['charge_C_cm2']})
    metrics=pd.DataFrame(metrics)
    files.append(write_dataframe_csv(metrics,out/'Q3_约束及端部指标.csv'))
    note='A=25 cm²；各片功率密度限于0—1 W·cm⁻²。协同启动采用题给60 s升载至0.3 A·cm⁻²；纯预热截止后采用问题2优化的线性升载。沿用问题2的20 C·cm⁻²电荷预算，并检验温度、冰和全过程电压条件。'
    files.append(write_markdown_tables([
        ('表4 不同辅助冷启动策略的优化结果及启动性能对比',results['table'],note),
        ('端部冰堵与约束指标',metrics.round(4),'最大冰体积分数为局部冰体积/对应网格总体积；优化值为多起点启发式搜索得到的可行最小值。')
    ],out/'表格输出.md'))
    logger.status('已导出4个CSV及表格输出.md')
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
