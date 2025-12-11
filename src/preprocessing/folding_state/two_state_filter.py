#two_state_filter.py
import pandas as pd

fireprot_csv = "./data/fireprotdb_ddg_with_folding_label.csv"

df = pd.read_csv(fireprot_csv)
df2 = df[df['FOLDING_MODEL'] == 'two-state']
df2.to_csv("./data/fireprotdb_ddg_two_state_only.csv", index=False)