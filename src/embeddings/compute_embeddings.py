#compute_embeddings.py
#this script computes ESM-2 embeddings for wild-type and mutant sequences in our cleaned dataset
import esm
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from embed_sequence import embed_sequence
from esm_model_loader import load_esm_model
from save_embeddings import save_embeddings


layer = 33  # ESM-2 layer to extract embeddings from
embed_dir_wt_per_residue = Path(f"./data/embeddings/raw_embeddings/layer_{layer}/wt/per_residue")
embed_dir_mut_per_residue = Path(f"./data/embeddings/raw_embeddings/layer_{layer}/mut/per_residue")
embed_dir_wt_pooled = Path(f"./data/embeddings/raw_embeddings/layer_{layer}/wt/pooled")
embed_dir_mut_pooled = Path(f"./data/embeddings/raw_embeddings/layer_{layer}/mut/pooled")
embed_dir_wt_per_residue.mkdir(parents=True, exist_ok=True)
embed_dir_mut_per_residue.mkdir(parents=True, exist_ok=True)
embed_dir_wt_pooled.mkdir(parents=True, exist_ok=True)
embed_dir_mut_pooled.mkdir(parents=True, exist_ok=True)

input_csv = "./data/fireprot_ddG_sequences_clean.csv"

if __name__ == "__main__":

    df=pd.read_csv(input_csv)

    model, alphabet = load_esm_model('esm2_t36_3B_UR50D')

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        esm_wt_sequence = row['ESM_WT_SEQUENCE']
        protein_id = row['PROTEIN']
        esm_mut_sequence = row['ESM_MUT_SEQUENCE']
        mutation = row['ALIGNED_MUTATION']

        name = f'{protein_id}_{mutation}'
        safe_name = name.replace('/', '_')

        print(f"Embedding wt sequence {safe_name} at {index}...")
        per_res_wt, pooled_wt = embed_sequence(esm_wt_sequence, model, alphabet=alphabet, layer = layer,  device='cuda')

        print(f"Embedding mut sequence {safe_name} at {index}...")
        per_res_mut, pooled__mut = embed_sequence(esm_mut_sequence, model, alphabet, layer = layer, device='cuda')

        # Save embeddings
        
        save_embeddings(embed_dir_wt_per_residue, index, safe_name, esm_wt_sequence, per_res_wt)
        save_embeddings(embed_dir_wt_pooled, index, safe_name, esm_wt_sequence, pooled_wt)
        save_embeddings(embed_dir_mut_per_residue, index, safe_name, esm_mut_sequence, per_res_mut)
        save_embeddings(embed_dir_mut_pooled, index, safe_name, esm_mut_sequence, pooled__mut)
       

   
