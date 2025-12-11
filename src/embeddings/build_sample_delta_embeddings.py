#build_sample_delta_embeddings.py
#this script computes delta-embeddings between mutant and wild-type TdT (sample protein) sequences
import torch
from pathlib import Path
import pandas as pd

embed_dir = Path("./data/sample_embeddings/raw_embeddings")
output_dir = Path("./data/sample_embeddings/delta_embeddings/per_residue")
output_dir.mkdir(parents=True, exist_ok=True)

#define function to load embeddings
def load_embeddings (file_path):
    d = torch.load(file_path)
    
    return d['name'], d['sequence'], d['per residue_embeddings']

if __name__ == "__main__":
    embed_files = list(embed_dir.glob("*.pt"))

    wt_file = None
    for f in embed_files:
        if "TdT_wt" in f.stem:
            wt_file = f
            print(f"Found WT file: {wt_file.name}")
            break
    wt_name, wt_seq, wt_per_residue = load_embeddings(wt_file)
    wt_dim = wt_per_residue.shape[0]

    print(f"WT: {wt_name}, length: {len(wt_seq)}, embedding dim: {wt_dim}")

    #Prepare data frame to hold delta-embedidngs
    rows = []
    for f in embed_files:
        mut_name, mut_seq, mut_per_residue = load_embeddings(f)
        
        #skip WT
        if mut_name == wt_name:
            continue

        delta = mut_per_residue - wt_per_residue
        
        #save delta embedding tensors
        tensor_out =  output_dir / f"{mut_name}.pt"
        torch.save({'name': mut_name, 'sequence': mut_seq, 'delta_embedding': delta}, tensor_out)

       

