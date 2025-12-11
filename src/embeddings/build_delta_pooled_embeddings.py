#build_delta_pooled_embeddings.py
#this script computes pooled delta-embeddings between mutant and wild-type sequences
import torch
from pathlib import Path
from tqdm import tqdm


layer = 33  # ESM-2 layer to extract embeddings from
embed_dir = Path(f"./data/embeddings/raw_embeddings/layer_{layer}")
output_dir = Path(f"./data/embeddings/delta_embeddings/layer_{layer}/pooled")
output_dir.mkdir(parents=True, exist_ok=True)

#define function to load embeddings
def load_embeddings (file_path):
    d = torch.load(file_path)
    return d['name'], d['sequence'], d['embedding']

def save_delta(output_dir, delta):
    torch.save({
        'embedding': delta}, output_dir)

if __name__ == "__main__":
    wt_pooled_dir = embed_dir / "wt" / "pooled"
    mut_pooled_dir = embed_dir / "mut" / "pooled"
    
    for wt_file in tqdm(list(wt_pooled_dir.glob("*.pt"))):
        stem = wt_file.stem
        mut_file = mut_pooled_dir / f"{stem}.pt"

        if not mut_file.exists():
            print(f"Mutant file {mut_file} does not exist, skipping.")
            continue

        wt_name, wt_seq, wt_pooled = load_embeddings(wt_file)
        mut_name, mut_seq, mut_pooled = load_embeddings(mut_file)
        
        print(f"Mutant: {mut_name}, length: {len(mut_seq)}, embedding dim: {mut_dim}")
        
        if wt_pooled.shape!= mut_pooled.shape:
            print(f"Dimension mismatch for {mut_name}, skipping.")
            continue

        delta = mut_pooled - wt_pooled

        #save delta embedding tensors
        outpath = output_dir / f"{stem}.pt"
        save_delta(outpath, delta)


