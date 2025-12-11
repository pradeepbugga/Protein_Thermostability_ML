#fix_mayo_paper.py
#this script fixes the aligned mutations for the Mayo paper in the FireProt dataset
#the Mayo paper has a large number of entries, so we will edit programmatically

import pandas as pd

fireprot_csv = "./data/fireprotdb_ddg_aligned_mutations_manual_check.csv"

df = pd.read_csv(fireprot_csv)

#fix entries corresponding to the Mayo paper 
mayo_doi = "10.1073/pnas.1903888116"

#fix entries that are "NO OFFSET FOUND" by manually assigning aligned mutations based on known offset of -226
for index, row in df.iterrows():
    if row['PUBLICATION_DOI'] == mayo_doi and row['ALIGNED_MUTATION'] == 'NO OFFSET FOUND':
        mutation = row['SUBSTITUTION']
        wt_aa = mutation[0]
        position = int(''.join(filter(str.isdigit, mutation)))
        mut_aa = mutation[-1]
        
        #apply offset of -226
        aligned_position = position - 226
        aligned_mutation = f"{wt_aa}{aligned_position}{mut_aa}"
        
        df.at[index, 'ALIGNED_MUTATION'] = aligned_mutation
        df.at[index, 'MUTATION_OFFSET'] = -226
        print(f"Fixed Mayo paper mutation {mutation} to aligned mutation {aligned_mutation}")

df.to_csv("./data/fireprotdb_ddg_aligned_mutations_manual_check_fixed_mayo.csv", index=False)