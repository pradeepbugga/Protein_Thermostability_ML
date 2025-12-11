#embedding_loaders.py

from pathlib import Path
import torch

def load_raw_wt_per_residue(rec, dir: Path, layer:int):
    path =  dir / "raw_embeddings" / f"layer_{layer}" / "wt" / "per_residue" / f"{rec.embed_key}.pt"
    return torch.load(path)["embedding"]

def load_raw_mut_per_residue(rec, dir: Path, layer:int):
    path =  dir / "raw_embeddings" / f"layer_{layer}" / "mut" / "per_residue" / f"{rec.embed_key}.pt"
    return torch.load(path)["embedding"]

def load_raw_wt_pooled(rec, dir: Path, layer:int):
    path =  dir / "raw_embeddings" / f"layer_{layer}" / "wt" / "pooled" / f"{rec.embed_key}.pt"
    return torch.load(path)["embedding"]

def load_raw_mut_pooled(rec, dir: Path, layer:int):
    path =  dir / "raw_embeddings" / f"layer_{layer}" / "mut" / "pooled" / f"{rec.embed_key}.pt"
    return torch.load(path)["embedding"]

def load_delta_pooled(rec, dir: Path, layer:int):
    path =  dir / "delta_embeddings" / f"layer_{layer}" / "pooled" f"{rec.embed_key}.pt"
    return torch.load(path)["embedding"]

def load_delta_per_residue(rec, dir: Path, layer:int):
    path = dir / "delta_embeddings" / f"layer_{layer}" / "per_residue" / f"{rec.embed_key}.pt"
    return torch.load(path)["embedding"]
