#embedding_loaders.py

from pathlib import Path
import torch

#for per_residue, make sure to trim the BOS and EOS tokens (ESM-2 artifacts)

def load_raw_wt_per_residue(rec, dir: Path, layer:int):
    path =  dir / "raw_embeddings" / f"layer_{layer}" / "wt" / "per_residue" / f"{rec.embed_key}.pt"
    d = torch.load(path)
    if d["embedding"].shape[0] == len(d['sequence']) +2:
        return d["embedding"][1:-1]
    elif d["embedding"].shape[0] == len(d['sequence']):
        return d["embedding"]
    else:
        raise ValueError(
            f"Unexpected embedding length {d["embedding"].shape[0]} for sequence length {len(d["sequence"])}"
        )

def load_raw_mut_per_residue(rec, dir: Path, layer:int):
    path =  dir / "raw_embeddings" / f"layer_{layer}" / "mut" / "per_residue" / f"{rec.embed_key}.pt"
    d = torch.load(path)
    if d["embedding"].shape[0] == len(d['sequence']) +2:
        return d["embedding"][1:-1]
    elif d["embedding"].shape[0] == len(d['sequence']):
        return d["embedding"]
    else:
        raise ValueError(
            f"Unexpected embedding length {d["embedding"].shape[0]} for sequence length {len(d["sequence"])}"
        )

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
    d = torch.load(path) 
    if len(d['mut_sequence']) != len(d['wt_sequence']): 
        raise ValueError (f"Mut sequence length {len(d['mut_sequence'])} not equal to wt sequence length {len(d['wt_sequence'])}") 
    else: 
        if d["embedding"].shape[0] == len(d['wt_sequence']) +2: 
            return d["embedding"][1:-1] 
        elif d["embedding"].shape[0] == len(d['wt_sequence']): 
            return d["embedding"] 
        else: 
            raise ValueError( f"Unexpected embedding length {d["embedding"].shape[0]} for sequence length {len(d["wt_sequence"])}" )