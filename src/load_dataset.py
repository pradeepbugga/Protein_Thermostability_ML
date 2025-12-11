#load_dataset.py


import torch
import pandas as pd
from pathlib import Path
import numpy as np

def parse_filename(filename):
    """Parse filename to extract index, protein_id, mutation."""
    stem = filename.stem
    index = int(stem.split("_")[0])
    name = "_".join(stem.split("_")[1:])
    protein_id = name.rsplit("_",1)[0]
    mutation = name.rsplit("_",1)[1]    
    wt_aa = mutation[0]
    position = int(''.join(filter(str.isdigit, mutation)))
    mut_aa = mutation[-1]
    return index, protein_id, mutation, wt_aa, position, mut_aa



def load_fireprot_dataset(embed_folder, input_csv):
    #load all embeddings and ddG values
    embed_folder = Path(embed_folder)
    df = pd.read_csv(input_csv)
    files = list(embed_folder.glob("*.pt"))
    X_list = []
    y_list = []

    for f in files:
        index, protein_id, mutation, wt_aa, position, mut_aa = parse_filename(f)

        row = df.loc[index]

        ddG = row['DDG']
        if ddG is None or pd.isna(ddG):
            print(f"Skipping index {index} due to missing DDG value.")
            continue
        
        if index not in mapping:
            mapping[index] = () #initialize a tuple 
        
        
        
    for index in sorted(mapping.keys()):
        delta_embed_path, ddG, position = mapping[index]

        delta_embedding = torch.load(delta_embed_path)['delta_embedding']
        #this is a LxD tensor where L is sequence length, D is embedding dimension
        #get the embedding for the mutated residue only
        residue_embedding = delta_embedding[position-1,:]

        X_list.append(residue_embedding)
        y_list.append(ddG)
    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)
    return X, y    