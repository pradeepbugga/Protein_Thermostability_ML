#initial_filter_db.py
#this script filters the FireProt database to only single mutants
#and removes the MegaScale dataset (which uses an indirect assay for ddG measurement)

import pandas as pd
import re 

fireprot_csv = "./data/fireprotdb_ddg_cleaned.csv"

df = pd.read_csv(fireprot_csv)

print("Original DF is of shape",df.shape)

#drop columns that are only NaN
df = df.dropna(axis=1, how='all')

#delete entries where 'insertion' or 'deletion' columsn are not missing
df2 = df[df['INSERTION'].isna() & df['DELETION'].isna()]
df2 = df2.drop(columns=['INSERTION', 'DELETION'], errors='ignore')


#remove rows with 'SOURCE_DATASET' equal to 'MegaScale'
df3 = df2[df2['SOURCE_DATASET'] != 'MegaScale']
df3 = df3[['UNIPROTKB', 'SEQUENCE_LENGTH', 'SUBSTITUTION', 'DDG', 'PROTEIN', 'EXPERIMENT_ID', 'PUBLICATION_DOI', 'PUBLICATION_YEAR', 'PUBLICATION_PMID']]

#remove multi mutant 
single_regex = re.compile(r"^[A-Z][0-9]+[A-Z]$")

def is_single_mutant(substitution):
    if pd.isna(substitution):
        return False
    return bool(single_regex.match(substitution.strip()))

df_single = df3[df3['SUBSTITUTION'].apply(is_single_mutant)]

print("Original DF is of shape",df.shape)
print("Filtered DF is of shape",df_single.shape)

df_single.to_csv("./data/fireprotdb_ddg_single_pub.csv", index=False)


