from copy import deepcopy
from pathlib import Path
import yaml

def deep_merge(base,child):
    out=deepcopy(base)
    for k,v in child.items():
        if k in out and isinstance(out[k],dict) and isinstance(v,dict): out[k]=deep_merge(out[k],v)
        else: out[k]=deepcopy(v)
    return out

def load_config(path):
    path=Path(path).resolve()
    with path.open('r',encoding='utf-8') as f: cfg=yaml.safe_load(f) or {}
    base=cfg.pop('_base_',None)
    if base: cfg=deep_merge(load_config(path.parent/base),cfg)
    return cfg
