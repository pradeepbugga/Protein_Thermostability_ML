#visualize_rmse.py
#this script visualizes RMSE across folds

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd

model_json_dir = "./results/models/mlp_lopo"
plot_dir = model1_json_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

def json_to_df(json_dir, n_folds, modelname):
    pd_series = []
    rmse_scores = []
    for fold in range(1, n_folds+1):
        results_path = f"{json_dir}/fold_{fold}/results.json"
        df = pd.read_json(results_path, typ='series')
        pd_series.append(df)
    full_df = pd.concat(pd_series, axis=1)
    full_df2 = full_df.T
    return full_df2

df = json_to_df(model_json_dir, 65)

#graph fold and rsme
plt.figure(figsize = (20,6), layout = 'constrained')
sns.barplot(x='fold', y='rmse', data=df1)
plt.title("RMSE Across Folds")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.xticks(rotation=90, fontsize = 8)
plt.savefig(f"{plot_dir}/rmse_across_folds.png", bbox_inches = 'tight')