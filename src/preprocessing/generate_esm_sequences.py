#generate_esm_sequences.py
#this script generates the sequences needed for ESM-2 
#manually added N- and C- termini to the wild-type sequences are combined with the mutations to create full ESM input sequences

import pandas as pd

input_csv = "./data/fireprotdb_ddg_aligned_mutations_wt_sequence_2.csv"
output_csv = "./data/fireprot_ddG_sequences_clean.csv

df = pd.read_csv(input_csv)

df = df.dropna(subset=['WT_SEQUENCE'])

for index, row in df.iterrows():
    wt_sequence = row['WT_SEQUENCE']
    n_terminus = row['N-TERMINI']
    c_terminus = row['C-TERMINI']
    mutation = row['ALIGNED_MUTATION']  # e.g., "A123C"
    if mutation == "NO OFFSET FOUND" or mutation == "" or pd.isna(mutation):
        print(f"Skipping index {index} due to invalid mutation: {mutation}")
        continue
    wt_aa = mutation[0]
    position = int(''.join(filter(str.isdigit, mutation)))
    mut_aa = mutation[-1]

    #check position
    if position < 1 or position > len(wt_sequence):
        print(f"Warning: Position {position} out of bounds for protein ID {row['PROTEIN']} with sequence length {len(wt_sequence)} at index {index}.") 

    #check wt amino acid matches
    if wt_sequence[position-1] != wt_aa:
        print(f"Warning: WT amino acid at position {position} does not match for protein ID {row['PROTEIN']} and mutation {mutation}.")
        print(f"Expected: {wt_aa}, Found: {wt_sequence[position-1]}")
        break

    # Generate ESM input sequence
    if pd.isna(n_terminus):
        n_terminus = ""
    if pd.isna(c_terminus):
        c_terminus = ""

    esm_wt_sequence = str(n_terminus).strip() + wt_sequence + str(c_terminus).strip()
    esm_mut_sequence = str(n_terminus).strip() + wt_sequence[:position-1] + mut_aa + wt_sequence[position:] + str(c_terminus).strip()
    df.at[index, 'ESM_WT_SEQUENCE'] = esm_wt_sequence
    df.at[index, 'ESM_MUT_SEQUENCE'] = esm_mut_sequence
    

df.to_csv(output_csv, index=False)
