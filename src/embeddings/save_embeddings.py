#save_embeddings.py

import torch
from pathlib import Path

def save_embedding(output_dir, index, name, sequence, embedding): 
    path = Path(output_dir) / f"{index}_{name}.pt"
    torch.save({
        'name': name, 
        'sequence': sequence,
        "embedding": embedding        
    }, path)
    return path
