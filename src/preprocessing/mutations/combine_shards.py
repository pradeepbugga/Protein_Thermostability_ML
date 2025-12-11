#combine_shards.py
#this script combines all the extracted mutation json shard files into one combined_extracted_mutations.json file

import json
from pathlib import Path

all_results = {}

#combine all json shard files into one
for f in Path("./data/").glob("*_extracted_shard_*.json"):
    with open(f, "r") as infile:
        shard_data = json.load(infile)
        all_results.update(shard_data)
        print(f"Loaded {len(shard_data)} entries from {f}")
print(len(all_results), "total entries combined.")

with open("./data/combined_extracted_mutations.json", "w") as outfile:
    json.dump(all_results, outfile, indent=4)