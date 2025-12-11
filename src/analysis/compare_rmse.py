#visualize_rmse.py
#this script visualizes RMSE between different models

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd

model1_json_dir = "./results/models/mlp_lopo"
model2_json_dir = "./results/models/xgboost_lopo/"
plot_dir1 = model1_json_dir / "plots"
plot_dir2 = model2_json_dir / "plots"
plot_dir1.mkdir(parents=True, exist_ok=True)
plot_dir2.mkdir(parents=True, exist_ok=True)

def json_to_df(json_dir, n_folds, modelname):
    pd_series = []
    rmse_scores = []
    for fold in range(1, n_folds+1):
        results_path = f"{json_dir}/fold_{fold}/results.json"
        df = pd.read_json(results_path, typ='series')
        pd_series.append(df)
    full_df = pd.concat(pd_series, axis=1)
    full_df2 = full_df.T
    full_df2['model'] = f"{modelname}"
    return full_df2

df1 = json_to_df(model1_json_dir, 65, "MLP LOPO")
df2 = json_to_df(model2_json_dir, 65, "XGBoost LOPO")

combined_df = pd.concat([df1, df2], ignore_index=True)

#graph fold and rsme
plt.figure(figsize = (20,6), layout = 'constrained')
sns.barplot(x='fold', y='rmse', data=combined_df, hue='model')
plt.title("RMSE Across Folds")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.xticks(rotation=90, fontsize = 8)
plt.savefig(f"{plot_dir1}/comparison_with_{df2['model'].iloc[0]}.png", bbox_inches = 'tight')
plt.savefig(f"{plot_dir2}/comparison_with_{df1['model'].iloc[0]}.png", bbox_inches = 'tight')