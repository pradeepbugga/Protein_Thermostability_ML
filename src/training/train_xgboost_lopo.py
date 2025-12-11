#train_xgboost_lopo.py
#this script trains an XGBoost model using Leave-One-Protein-Out cross-validation

import pandas as pd
from pathlib import Path
import numpy as np
from build_records import build_records
from features import delta_residue_feature 
from dataset_builder import build_X, build_y, ddg_target
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from protein_grouping import group_by_protein

from xgboost import XGBRegressor


input_csv = Path("./data/fireprot_ddG_sequences_clean.csv")
embed_folder = Path("./data/embeddings")
layer = 33

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

    #convert X,y to numpy for xgboost
    X = X.cpu().numpy()
    y = y.cpu().numpy()

    #build groups
    protein_ids = [rec.protein_id for rec in records]
    groups, protein_counts = group_by_protein(protein_ids, min_mutations=10)

    gkf = GroupKFold(n_splits= len(np.unique(groups)))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n==========================")
        print(f"Starting fold {fold+1}...")
        print(f"Validation proteins: {np.unique(np.array(protein_ids)[val_idx])}")
        print("==========================")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]   

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample = 0.8,
            colsample_bytree = 0.8,
            objective='reg:squarederror',
            tree_method='hist',
            device = 'cuda'
        )

         #training loop
        print("Starting training...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)

        #save model
        output_path = Path("./results/models/xgboost_lopo") / f"fold_{fold+1}" / "model.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(output_path)
        print(f"Model saved to {output_path}") 

        fold_output = {
            'fold': fold+1,
            'validation_proteins': np.unique(np.array(protein_ids)[val_idx]).tolist(),
            'rmse': rmse
        }

        #save fold results
        results_path = Path("./results/models/xgboost_lopo") / f"fold_{fold+1}"/"results.json"
        pd.Series(fold_output).to_json(results_path)
        print(f"Fold results saved to {results_path}")
