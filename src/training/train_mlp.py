#train_mlp.py
#this script trains a multi-layer perceptron (MLP) regression model on ddG and ESM-2 delta embeddings

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

    n = len(dataset)
    idx = torch.randperm(n)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler = torch.utils.data.SubsetRandomSampler(train_idx)) 
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler = torch.utils.data.SubsetRandomSampler(val_idx))

    #initialize model, criterion, optimizer
    
    dim = X.shape[1]
    model = mlp(dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #training loop
    print("Starting training...")
    for epoch in range(1,epochs+1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    #save model
    output_path = Path("./results/models/mlp_random_split/model.pt")
    Path("./results/models/mlp_random_split").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")  
