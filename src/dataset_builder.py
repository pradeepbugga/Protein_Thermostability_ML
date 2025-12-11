#dataset_builder.py

import torch
from mutation_record import MutationRecord

def ddg_target(rec):
    return rec.ddG

def build_X(records, feature_fn):
    return torch.stack(list(feature_fn(r) for r in records))


def build_y(records, target_fn):
    return torch.tensor([target_fn(r) for r in records], dtype=torch.float32)
    