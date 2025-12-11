#esm_model_loader.py
import torch
import esm

def load_esm_model(model_name="esm2_t36_3B_UR50D", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading ESM-2 model...")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device).eval()
    return model, alphabet
