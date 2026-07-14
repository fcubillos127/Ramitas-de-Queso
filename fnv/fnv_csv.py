# fnv/fnv_csv.py
from __future__ import annotations
from typing import Tuple, Dict, List, Optional
import re
import numpy as np
import pandas as pd
from pathlib import Path
from .fnv_store import FNVData, FNVBlock

# ---------------- helpers internos ----------------
def _block_to_dataframe(blk: FNVBlock) -> pd.DataFrame:
    rows = []
    r = blk.r
    for (kind, chan), arr in blk.data.items():
        if arr is None:
            continue
        Nn, Nf, Nr = arr.shape
        for i, n in enumerate(blk.n_list):
            for j, f in enumerate(blk.freq_list):
                vec = arr[i, j, :]
                df_cur = pd.DataFrame({
                    "kind": kind,
                    "chan": chan,
                    "n": int(n),
                    "freq": float(f),
                    "r": r,
                    "real": np.real(vec),
                    "imag": np.imag(vec),
                })
                rows.append(df_cur)
    if not rows:
        return pd.DataFrame(columns=["kind","chan","n","freq","r","real","imag"])
    return pd.concat(rows, ignore_index=True)

def _dataframe_to_block(df: pd.DataFrame) -> FNVBlock:
    df = df.copy()
    df["n"] = df["n"].astype(int)
    df["freq"] = df["freq"].astype(float)
    df["r"] = df["r"].astype(float)
    df["real"] = df["real"].astype(float)
    df["imag"] = df["imag"].astype(float)

    r_sorted = np.sort(df["r"].unique())
    n_list   = sorted(df["n"].unique().tolist())
    freq_list= sorted(df["freq"].unique().tolist())

    blk = FNVBlock(r=r_sorted, n_list=n_list, freq_list=freq_list)
    blk.ensure_arrays(n_list, freq_list)

    for kind, chan in [("F","b"), ("F","c"), ("V","b"), ("V","c")]:
        sub = df[(df["kind"] == kind) & (df["chan"] == chan)]
        if sub.empty:
            continue
        for (n, f), g in sub.groupby(["n", "freq"], sort=False):
            s = pd.Series(g["real"].values + 1j*g["imag"].values, index=g["r"].values)
            aligned = s.reindex(r_sorted)
            i = blk.n_list.index(int(n))
            j = blk.freq_list.index(float(f))
            blk.data[(kind, chan)][i, j, :] = aligned.values.astype(complex)
    return blk

# ---------------- API pública: CSV único (largo) ----------------
def export_block_csv(blk: FNVBlock, csv_path: str) -> str:
    df = _block_to_dataframe(blk)
    pd.DataFrame(df).to_csv(csv_path, index=False)
    return csv_path

def export_default_block_csv(fnv_data: FNVData, csv_path: str) -> str:
    if fnv_data.default_key is None:
        raise ValueError("FNVData no tiene default_key.")
    blk = fnv_data.blocks[fnv_data.default_key]
    return export_block_csv(blk, csv_path)

def load_block_from_csv(csv_path: str) -> FNVBlock:
    df = pd.read_csv(csv_path)
    return _dataframe_to_block(df)

def load_fnv_from_csv(csv_path: str) -> FNVData:
    blk = load_block_from_csv(csv_path)
    fnv = FNVData()
    import hashlib
    key = (blk.r.size, hashlib.sha1(np.asarray(blk.r, dtype=float).tobytes()).hexdigest())
    fnv.blocks[key] = blk
    fnv.default_key = key
    return fnv

# ---------------- API pública: CSVs separados por canal y modo n ----------------
def export_block_csv_split(
    blk: FNVBlock,
    out_dir: str | Path,
    name_template: str = "{kind}{chan}_n{n}.csv",
    float_fmt: Optional[str] = None,
) -> List[Path]:
    """
    Exporta un CSV por cada (kind, chan, n).
    Columnas: freq, r, real, imag  (sin metadatos redundantes).
    Nombre: name_template con placeholders {kind},{chan},{n}.
    Devuelve la lista de rutas.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for (kind, chan), arr in blk.data.items():
        if arr is None:
            continue
        # arr: (Nn, Nf, Nr)
        for i, n in enumerate(blk.n_list):
            # Construimos DataFrame largo (freq, r)
            rows = []
            for j, f in enumerate(blk.freq_list):
                vec = arr[i, j, :]
                df_cur = pd.DataFrame({
                    "freq": float(f),
                    "r": blk.r,
                    "real": np.real(vec),
                    "imag": np.imag(vec),
                })
                rows.append(df_cur)
            df_n = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["freq","r","real","imag"])

            fname = name_template.format(kind=kind, chan=chan, n=int(n))
            fpath = outp / fname
            df_n.to_csv(fpath, index=False, float_format=float_fmt)
            paths.append(fpath)
    return paths

def export_default_split_csv(
    fnv_data: FNVData,
    out_dir: str | Path,
    name_template: str = "{kind}{chan}_n{n}.csv",
    float_fmt: Optional[str] = None,
) -> List[Path]:
    if fnv_data.default_key is None:
        raise ValueError("FNVData no tiene default_key.")
    blk = fnv_data.blocks[fnv_data.default_key]
    return export_block_csv_split(blk, out_dir, name_template=name_template, float_fmt=float_fmt)

def load_block_from_csv_split(
    in_dir: str | Path,
    name_pattern: str = r"(?P<kind>[FV])(?P<chan>[bc])_n(?P<n>\d+)\.csv",
) -> FNVBlock:
    """
    Reconstruye un FNVBlock desde múltiples CSVs (cada uno: columnas freq,r,real,imag).
    Extrae (kind,chan,n) desde el nombre de archivo con name_pattern.
    """
    in_path = Path(in_dir)
    regex = re.compile(name_pattern)

    # Acumulamos filas como en el formato "largo"
    rows = []
    for p in sorted(in_path.glob("*.csv")):
        m = regex.match(p.name)
        if not m:
            continue
        kind = m.group("kind")
        chan = m.group("chan")
        n = int(m.group("n"))
        df = pd.read_csv(p)
        # añadimos columnas meta para rearmar el bloque
        df["kind"] = kind
        df["chan"] = chan
        df["n"] = n
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No se encontraron CSVs válidos en {in_path} con patrón {name_pattern}")

    df_all = pd.concat(rows, ignore_index=True)
    # Reordenamos columnas para _dataframe_to_block
    df_all = df_all[["kind","chan","n","freq","r","real","imag"]]
    return _dataframe_to_block(df_all)

def load_fnv_from_csv_split(
    in_dir: str | Path,
    name_pattern: str = r"(?P<kind>[FV])(?P<chan>[bc])_n(?P<n>\d+)\.csv",
) -> FNVData:
    blk = load_block_from_csv_split(in_dir, name_pattern=name_pattern)
    fnv = FNVData()
    import hashlib
    key = (blk.r.size, hashlib.sha1(np.asarray(blk.r, dtype=float).tobytes()).hexdigest())
    fnv.blocks[key] = blk
    fnv.default_key = key
    return fnv
