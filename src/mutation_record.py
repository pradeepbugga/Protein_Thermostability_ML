from dataclasses import dataclass
from pathlib import Path

@dataclass
class MutationRecord:
   index: int
   protein_id: str
   mutation: str
   position: int
   wt_residue: str
   mut_residue: str
   ddG : float
   embed_key: str


  