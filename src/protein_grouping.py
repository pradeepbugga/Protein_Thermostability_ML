#protein_grouping.py
import numpy as np

def group_by_protein(protein_ids, min_mutations = 10):


    unique,counts = np.unique(protein_ids, return_counts=True)
    protein_counts = dict(zip(unique,counts))

      
    large_proteins = [p for p,c in protein_counts.items() if c>=min_mutations]
    print(f"Proteins with at least {min_mutations} samples: {large_proteins}")
    small_proteins = [p for p,c in protein_counts.items() if c< min_mutations]
    print(f"Proteins with less than {min_mutations} samples: {small_proteins}")

    group_map = {}
    for i,p in enumerate(large_proteins):
        group_map[p] = i
    
    small_group_id = len(large_proteins)
    for p in small_proteins:
        group_map[p] = small_group_id

    groups = np.array([group_map[p] for p in protein_ids])
    return groups, protein_counts