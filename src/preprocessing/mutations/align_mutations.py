#align_mutations.py
#align mutations from csv and json results
#this is needed because the fireprot mutation positions may be offset from the extracted mutations from the pdfs

import pandas as pd
import json
from pathlib import Path
from collections import Counter
import re

fireprot_csv = "./data/fireprotdb_ddg_two_state_only.csv"
json_results_path = "./data/combined_extracted_mutations.json"

#iterate through each protein, doi combination

df = pd.read_csv(fireprot_csv)
with open (json_results_path, "r") as f:
    all_mutations = json.load(f)
def parse_mut(m):
    try:
        pos = int(m[1:-1])
    except ValueError:
        matches = re.findall(r'\d+(?=\D)', m[1:-1])
        if matches:
            pos = int(matches[0])
        else:
            pos = 0
    return m[0], pos, m[-1]

def compute_best_offset(fp_muts, pdf_muts):
    offsets = []
    for fp in fp_muts:
        fp_wt, fp_pos, fp_mut = parse_mut(fp)
    
    
        for pdf in pdf_muts:
            pdf_wt, pdf_pos, pdf_mut = parse_mut(pdf)
            if fp_wt == pdf_wt and fp_mut == pdf_mut:
                offsets.append(pdf_pos - fp_pos)
    if not offsets:
        return None

    offset_counts = Counter(offsets)
    best_offset, _ = offset_counts.most_common(1)[0]
    return best_offset

def align_mutation(fp_mut, pdf_muts, offset):
    fp_wt, fp_pos, fp_mut = parse_mut(fp_mut)
    aligned_pos = fp_pos + offset

    for pdf in pdf_muts:
        wt, pos, mut = parse_mut(pdf)
        if pos == aligned_pos and wt == fp_wt and mut == fp_mut:
            return pdf
    return "NO MATCH"

aligned = []
for (protein, doi), group in df.groupby(['PROTEIN', 'PUBLICATION_DOI']):
    fp_muts = group['SUBSTITUTION'].tolist()
    pdf_muts = all_mutations.get(doi, [])

    best_offset = compute_best_offset(fp_muts, pdf_muts)
    
    for index, row in group.iterrows():
        fp_mut = row['SUBSTITUTION']
        if best_offset is None:
            df.at[index, 'ALIGNED_MUTATION'] = "NO OFFSET FOUND"
            continue
        
        aligned_mut = align_mutation(fp_mut, pdf_muts, best_offset)
        df.at[index, 'ALIGNED_MUTATION'] = aligned_mut
        df.at[index, 'MUTATION_OFFSET'] = best_offset


df.to_csv("./data/fireprotdb_ddg_aligned_mutations.csv", index=False)
