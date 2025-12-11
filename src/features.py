#features.py
from embedding_loaders import (
    load_delta_per_residue,
    load_delta_pooled,
    load_raw_mut_per_residue, 
    load_raw_mut_pooled, 
    load_raw_wt_pooled, 
    load_raw_wt_per_residue)
from pathlib import Path


def delta_residue_feature(rec, dir: Path, layer:int):
    delta = load_delta_per_residue(rec, dir, layer)
    return delta[rec.position-1]

def delta_pooled_feature(rec, dir: Path, layer:int):
    delta = load_delta_pooled(rec, dir, layer)
    return delta

def raw_wt_pooled_feature(rec, dir: Path, layer:int):
    return load_raw_wt_pooled(rec, dir, layer)

def raw_mut_pooled_feature(rec, dir: Path, layer:int):
    return load_raw_mut_pooled(rec, dir, layer)

def raw_wt_residue_feature(rec, dir: Path, layer:int):
    return load_raw_wt_per_residue[rec.position-1]

def raw_mut_residue_feature(rec, dir:Path, lyaer:int):
    return load_raw_mut_per_residue[rec.position-1]