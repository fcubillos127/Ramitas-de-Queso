from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Literal, Optional
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
from .fnv_core import compute_series, Kind, Chan

def _r_key(r: np.ndarray) -> Tuple[int, str]:
    arr = np.asarray(r, dtype=float)
    h = hashlib.sha1(arr.tobytes()).hexdigest()
    return (arr.size, h)

@dataclass
class FNVBlock:
    r: np.ndarray
    n_list: List[int] = field(default_factory=list)
    freq_list: List[float] = field(default_factory=list)
    data: Dict[Tuple[Kind, Chan], np.ndarray] = field(default_factory=dict)  # (Nn, Nf, Nr)

    def ensure_arrays(self, n_list: Iterable[int], freq_list: Iterable[float]):
        new_ns = sorted(set(self.n_list).union(n_list))
        new_fs = sorted(set(self.freq_list).union(freq_list))
        Nn, Nf, Nr = len(new_ns), len(new_fs), self.r.size

        def grow(arr: Optional[np.ndarray]) -> np.ndarray:
            out = np.full((Nn, Nf, Nr), np.nan + 1j*np.nan, dtype=complex)
            if arr is not None:
                old_n_idx = {n:i for i, n in enumerate(self.n_list)}
                old_f_idx = {f:i for i, f in enumerate(self.freq_list)}
                new_n_idx = {n:i for i, n in enumerate(new_ns)}
                new_f_idx = {f:i for i, f in enumerate(new_fs)}
                for n, i_old in old_n_idx.items():
                    for f, j_old in old_f_idx.items():
                        out[new_n_idx[n], new_f_idx[f], :] = arr[i_old, j_old, :]
            return out

        for key in [("F","b"), ("F","c"), ("V","b"), ("V","c")]:
            self.data[key] = grow(self.data.get(key))

        self.n_list = new_ns
        self.freq_list = new_fs

    def has_series(self, kind: Kind, chan: Chan, n: int, f: float) -> bool:
        i = self.n_list.index(n)
        j = self.freq_list.index(f)
        vec = self.data[(kind, chan)][i, j, :]
        return np.all(np.isfinite(vec.real)) and np.all(np.isfinite(vec.imag))

    def set_series(self, kind: Kind, chan: Chan, n: int, f: float, vec: np.ndarray):
        i = self.n_list.index(n)
        j = self.freq_list.index(f)
        self.data[(kind, chan)][i, j, :] = vec

@dataclass
class FNVData:
    blocks: Dict[Tuple[int, str], FNVBlock] = field(default_factory=dict)
    default_key: Optional[Tuple[int, str]] = None

    def get_block(self, r: np.ndarray) -> Optional[FNVBlock]:
        key = _r_key(r)
        return self.blocks.get(key, None)

    def get_or_create_block(self, r: np.ndarray) -> FNVBlock:
        key = _r_key(r)
        if key not in self.blocks:
            self.blocks[key] = FNVBlock(r=np.asarray(r, dtype=float))
            if self.default_key is None:
                self.default_key = key
        return self.blocks[key]

    def set_default(self, r: np.ndarray):
        self.default_key = _r_key(r)

def build_fnv_grid(red, n_list: Iterable[int], freq_list: Iterable[float], r: np.ndarray) -> FNVData:
    if getattr(red, "fnv_data", None) is None:
        red.fnv_data = FNVData()

    block = red.fnv_data.get_or_create_block(r)
    block.ensure_arrays(n_list, freq_list)

    for n in block.n_list:
        for f in block.freq_list:
            for kind, chan in [("F","b"), ("F","c"), ("V","b"), ("V","c")]:
                if (n in n_list) and (f in freq_list) and not block.has_series(kind, chan, n, f):
                    vec = compute_series(red, kind, chan, n, f, r1=red.r1, r2=red.r2, r=block.r)
                    block.set_series(kind, chan, n, f, vec)

    red.fnv_data.set_default(r)
    return red.fnv_data

def save_fnv(fnv_data: FNVData, path: str):
    to_save = {}
    meta = {"blocks": []}
    for idx, (k, blk) in enumerate(fnv_data.blocks.items()):
        prefix = f"b{idx}"
        meta["blocks"].append({
            "key": list(k),
            "n_list": blk.n_list,
            "freq_list": blk.freq_list,
            "r_len": int(blk.r.size),
            "prefix": prefix
        })
        to_save[f"{prefix}_r"] = blk.r.astype(float)
        for (kind, chan), arr in blk.data.items():
            to_save[f"{prefix}_{kind}{chan}"] = arr
    meta["default_key"] = list(fnv_data.default_key) if fnv_data.default_key else None
    to_save["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **to_save)

def load_fnv(path: str) -> FNVData:
    npz = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(npz["__meta__"].tolist()).decode("utf-8"))
    fnv = FNVData()
    for entry in meta["blocks"]:
        prefix = entry["prefix"]
        r = np.array(npz[f"{prefix}_r"], dtype=float)
        blk = FNVBlock(r=r, n_list=list(entry["n_list"]), freq_list=list(entry["freq_list"]))
        for key in [("F","b"), ("F","c"), ("V","b"), ("V","c")]:
            kind, chan = key
            blk.data[key] = np.array(npz[f"{prefix}_{kind}{chan}"])
        fnv.blocks[tuple(entry["key"])] = blk
    fnv.default_key = tuple(meta["default_key"]) if meta["default_key"] else None
    return fnv
