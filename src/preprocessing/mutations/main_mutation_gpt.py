#main_mutation_gpt.py
import multiprocessing as mp
import pandas as pd
import asyncio
import subprocess
from pathlib import Path
import numpy as np

NUM_WORKERS = 21
#adjust as needed

csv_path = "./data/fireprotdb_ddg_two_state_only.csv"

def create_shards (dois, num_workers):
    shards = np.array_split(np.array(dois), num_workers
    )
    #make shards dir
    shards_dir = Path("./data/shards")
    shards_dir.mkdir(exist_ok=True)
    for i, shard in enumerate(shards):
        shard_path = shards_dir / f"shard_{i}.txt"
        with open(shard_path, "w") as f:
            f.write("\n".join(shard))

def worker(shard_path):
    subprocess.run(["python", "worker_mutation_gpt.py", "--shard", str(shard_path)])

if __name__ == "__main__":
    df = pd.read_csv(csv_path)
    unique_dois = df['PUBLICATION_DOI'].dropna().unique().tolist()

    create_shards(unique_dois, NUM_WORKERS)

    workers = []
    for shard_id in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(shard_id,))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    print("All workers completed.")