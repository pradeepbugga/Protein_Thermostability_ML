#train_mlp_lopo.py
#this script trains an MLP model using Leave-One-Protein-Out cross-validation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
import numpy as np
import torch.optim as optim
from models.mlp import mlp
from training.loops import train, validate
from build_records import build_records
from features import delta_residue_feature 
from dataset_builder import build_X, build_y, ddg_target
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from protein_grouping import group_by_protein

#Hyperparameters
BATCH_SIZE = 256
learning_rate = 0.001
epochs = 30
input_csv = Path("./data/fireprot_ddG_sequences_clean.csv")
embed_folder = Path("./data/embeddings")
layer = 33
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    #build mutation records
    records = build_records(
        raw_embed_folder = embed_folder,
        csv_path = input_csv,
        layer = layer     
    )

    print(f"Built {len(records)} records.")

    #build delta per residue
    feature_fn = lambda rec: delta_residue_feature(rec, dir = embed_folder, layer=layer )

    #build feature matrix
    X = build_X(records, feature_fn)
    print(f"Built feature matrix with shape: {X.shape}")

    #build targets
    y = build_y(records, ddg_target)

    #build dataset    
    dataset = TensorDataset(X, y)

    #build groups
    protein_ids = [rec.protein_id for rec in records]
    groups, protein_counts = group_by_protein(protein_ids, min_mutations=10)

    gkf = GroupKFold(n_splits= len(np.unique(groups)))
    fold_results = []
    dim = X.shape[1]

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n==========================")
        print(f"Starting fold {fold+1}...")
        print(f"Validation proteins: {np.unique(np.array(protein_ids)[val_idx])}")
        print("==========================")

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler = torch.utils.data.SubsetRandomSampler(train_idx)) 
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler = torch.utils.data.SubsetRandomSampler(val_idx))

        #initialize model, criterion, optimizer
             
        model = mlp(dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #training loop
        print("Starting training...")
        for epoch in range(1,epochs+1):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        rmse = float(np.sqrt(val_loss))

        #save model
        output_path = Path("./results/models/mlp_lopo") / f"fold_{fold+1}" / "model.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Saved model for fold {fold+1} to {output_path}")

        fold_output = {
            'fold': fold+1,
            'validation_proteins': np.unique(np.array(protein_ids)[val_idx]).tolist(),
            'rmse': rmse
        }

        #save fold results
        results_path = Path("./results/models/mlp_lopo") / f"fold_{fold+1}"/"results.json"
        pd.Series(fold_output).to_json(results_path)
        print(f"Fold results saved to {results_path}")
