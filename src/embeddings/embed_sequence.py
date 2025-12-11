#embed_sequence.py

import torch

def embed_sequence (seq, model, alphabet, layer =33, device="cpu"):
    batch_converter = alphabet.get_batch_converter()
    data = [("seq", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)

    per_residue = results["representations"][layer].squeeze(0).cpu()
    pooled = per_residue.mean(0)
    return per_residue, pooled