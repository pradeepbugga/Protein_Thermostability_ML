#build_records.py
#this script populates MutationRecord objects from a data source

from pathlib import Path
import pandas as pd
from mutation_record import MutationRecord

def parse_filepath (filepath):
    stem = filepath.stem

    index = int(stem.split('_')[0])
    name = "_".join(stem.split("_")[1:])

    protein_id = name.rsplit("_",1)[0]
    mutation = name.rsplit("_",1)[1]
    wt_aa = mutation[0]
    position = int(''.join(filter(str.isdigit, mutation)))
    mut_aa = mutation[-1]

    return index, protein_id, wt_aa, position, mut_aa, mutation

def build_records (raw_embed_folder: Path, csv_path: Path, layer:int):
    df = pd.read_csv(csv_path)

    raw_wt_folder = raw_embed_folder / f"layer_{layer}" / "wt" / "per_residue"
    files = list(raw_wt_folder.glob("*.pt"))

    records = []
    for f in files:
        index, protein_id, wt_aa, position, mut_aa, mutation = parse_filepath(f)
        row = df.loc[index]
        ddG = row["DDG"]

        record = MutationRecord(
            index = index,
            protein_id = protein_id,
            mutation = mutation,
            wt_residue = wt_aa,
            position = position,
            mut_residue = mut_aa,
            ddG = ddG,
            embed_key = f"{index}_{protein_id}_{mutation}"
        )
        records.append(record)

    return sorted(records, key=lambda r: r.index)


