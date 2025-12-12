#sample_inference.py
#this script performs inference on sample TdT embeddings using our trained regression model

import torch
from pathlib import Path
import numpy as np
from models.mlp import mlp
import pandas as pd

embed_folder = Path("./data/sample_embeddings/delta_embeddings/per_residue")
models_dir = Path("./results/models/mlp_lopo/layer_33")


#check dimension (should be 2560 for 3B param embeddings)
sample_file = next(embed_folder.glob("*.pt"))
sample = torch.load(sample_file)
L = sample["delta_embedding"].shape[0]
dim = sample["delta_embedding"].shape[1]


def extract_delta(path):
    d = torch.load(path)
    emb = clean_embedding(d)

    #parse TdT_A#B into A#B
    mutation = Path(path).stem.split('_')[1]
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    pos = int(mutation[1:-1])

    delta = emb[pos-1]
    return delta.unsqueeze(0)
   
#handle BOS/EOS tokens
def clean_embedding(d):
    if d["delta_embedding"].shape[0] == len(d['sequence']) +2: 
        return d["delta_embedding"][1:-1] 
    elif d["delta_embedding"].shape[0] == len(d['sequence']): 
        return d["delta_embedding"] 
    else: 
        raise ValueError( f"Unexpected embedding length {d["delta_embedding"].shape[0]} for sequence length {len(d["sequence"])}" )

def load_all_models(models_dir, dim):
    models = []
    for fold_dir in sorted(models_dir.glob("fold_*")):
        model_path = fold_dir / "model.pt"
        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_path}")
        else:
            print("Found model_path")
        
        model = mlp(dim)
        print(model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        models.append(model)
    return models

models = load_all_models(models_dir, dim)
results = {}
for f in embed_folder.glob("*.pt"):
    name = f.stem
    x = extract_delta(f)

    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(float(m(x)))
    preds = np.array(preds)

    results[name] = {"mean": preds.mean(), "std": preds.std(), "all_folds": preds.tolist()}

results_path = Path("./results/models/mlp_lopo/layer_33/inference/sample_inference_results.json")
results_path.parent.mkdir(parents=True, exist_ok=True)
pd.Series(results).to_json(results_path)
print(f"Results saved to {results_path}")
