#protein_sequence_mapping.py
#this script generates a mapping file from UniProt IDs to sequences

import pandas as pd
import requests
import time
import json

IDS_FILE = "./data/protein_ID_list.txt"
JSON_OUTPUT = "./data/protein_ID_mapping.json"

def fetch_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch data for UniProt ID {uniprot_id}")

    lines = response.text.strip().split("\n")
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    return sequence

if __name__ == "__main__":
    with open(IDS_FILE, "r") as f:
        uniprot_ids = [line.strip() for line in f.readlines()]

    id_sequence_map = {}
    for uid in uniprot_ids:
        try:
            sequence = fetch_sequence(uid)
            id_sequence_map[uid] = sequence
            print(f"Fetched sequence for {uid}")
            time.sleep(1)  # To avoid overwhelming the server
        except ValueError as e:
            print(e)

    # Save mapping to JSON
    with open(JSON_OUTPUT, "w") as f:
        json.dump(id_sequence_map, f, indent=4)

    print(f"Saved UniProt ID to sequence mapping to {JSON_OUTPUT}")