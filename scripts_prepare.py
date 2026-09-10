"""Precompute GPT-2 residual-stream activations + MiniLM embeddings for the toy NLA.

Run once; the notebook loads the .npz so nothing slow happens on stage.
"""
import numpy as np, torch, warnings, sys
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
LAYERS = [6, 8, 12]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
gpt2 = AutoModelForCausalLM.from_pretrained(MODEL).to(DEV).eval()
layers = [l for l in LAYERS if l <= gpt2.config.n_layer]
enc = SentenceTransformer("all-MiniLM-L6-v2", device=DEV)

sents = open("data/corpus.txt").read().split("\n")
print(f"{MODEL}: {gpt2.config.n_layer} layers d={gpt2.config.n_embd} | {len(sents)} sentences | layers {layers}")

X = enc.encode(sents, batch_size=256, convert_to_numpy=True, show_progress_bar=False)

acc = {f"mean_{l}": [] for l in layers}   # mean-pooled matches MiniLM; last-token fits worse
with torch.no_grad():
    for i in range(0, len(sents), 96):
        b = tok(sents[i:i+96], return_tensors="pt", padding=True,
                truncation=True, max_length=48).to(DEV)
        hs = gpt2(**b, output_hidden_states=True).hidden_states
        m = b["attention_mask"].unsqueeze(-1).float()
        for l in layers:
            h = hs[l].float()
            acc[f"mean_{l}"].append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        if i % 2400 == 0:
            print(f"  {i}/{len(sents)}", flush=True)

out = {k: np.concatenate(v).astype(np.float16) for k, v in acc.items()}
out["X"] = X.astype(np.float32)
np.savez_compressed(f"data/acts_{MODEL}.npz", **out)
print("saved", f"data/acts_{MODEL}.npz")
for l in layers:
    print(f"  layer {l}: mean|h| = "
          f"{np.linalg.norm(out[f'mean_{l}'].astype(np.float32),axis=1).mean():.1f}")
