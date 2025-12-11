#build_delta_per_residue_embeddings.py
#this script computes pooled delta-embeddings between mutant and wild-type sequences
import torch
from pathlib import Path
from tqdm import tqdm


layer = 33  # ESM-2 layer to extract embeddings from
embed_dir = Path(f"./data/embeddings/raw_embeddings/layer_{layer}")
output_dir = Path(f"./data/embeddings/delta_embeddings/layer_{layer}/per_residue")
output_dir.mkdir(parents=True, exist_ok=True)

#define function to load embeddings
def load_embeddings (file_path):
    d = torch.load(file_path)
    return d['name'], d['sequence'], d['embedding']

def save_delta(output_dir, delta):
    torch.save({
        'embedding': delta}, output_dir)

if __name__ == "__main__":
    wt_per_residue_dir = embed_dir / "wt" / "per_residue"
    mut_per_residue_dir = embed_dir / "mut" / "per_residue"
    
    for wt_file in tqdm(list(wt_per_residue_dir.glob("*.pt"))):
        stem = wt_file.stem
        mut_file = mut_per_residue_dir / f"{stem}.pt"

        if not mut_file.exists():
            print(f"Mutant file {mut_file} does not exist, skipping.")
            continue

        wt_name, wt_seq, wt_per_residue = load_embeddings(wt_file)
        mut_name, mut_seq, mut_per_residue = load_embeddings(mut_file)
        
        print(f"Mutant: {mut_name}, length: {len(mut_seq)}, embedding sja[e]: {mut_per_residue.shape}")
        
        if wt_per_residue.shape!= mut_per_residue.shape:
            print(f"Dimension mismatch for {mut_name}, skipping.")
            continue

        delta = mut_per_residue - wt_per_residue

        #save delta embedding tensors
        outpath = output_dir / f"{stem}.pt"
        save_delta(outpath, base_name, wt_seq, mut_seq, delta)


