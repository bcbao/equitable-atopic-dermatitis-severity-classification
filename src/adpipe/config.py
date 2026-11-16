from __future__ import annotations

import copy

import yaml


def _parse(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def load_config(path: str, variant: str | None = None, backbone: str | None = None, overrides=None):
    cfg = yaml.safe_load(open(path))
    variants = cfg.pop("variants", {})
    cfg = copy.deepcopy(cfg)
    if variant:
        if variant not in variants:
            raise KeyError(f"variant '{variant}' not in config; available: {list(variants)}")
        cfg.update(variants[variant] or {})
        cfg["variant"] = variant
    if backbone:
        cfg["backbone"] = backbone
    for item in overrides or []:
        k, v = item.split("=", 1)
        cfg[k] = _parse(v)
    return cfg
