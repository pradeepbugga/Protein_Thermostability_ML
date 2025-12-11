#ingest_fireprot_db.py
#this script ingests the FireProtDB dataset CSV and cleans it (removes emtpy ddG entries) for downstream use

import pandas as pd

fireprot_csv = "./data/fireprotdb_20251015-164116.csv"
chunk_size = 10000
chunks = []

#use chunks because the file is large

df = pd.read_csv(fireprot_csv, chunksize=chunk_size)

for chunk in df:
    print("Processing new chunk")
    chunk_cleaned = chunk.dropna(subset=['DDG'])
    chunks.append(chunk_cleaned)


ddg_df = pd.concat(chunks, ignore_index=True)

ddg_df.to_csv("./data/fireprotdb_ddg_cleaned.csv", index=False)







