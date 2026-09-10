# How to Lobotomize an LLM

Talk materials for **PyAtl, September 2026**. Sequel to
[demystify-attention](https://github.com/tchoffman/demystify-attention) (PyAtl, February 2026).

Last talk: *it's "just predicting the next word" — but doing that well requires real
internal modelling.* This talk: **if there's real internal modelling in there, can we read
it? And can we edit it?**

- `lobotomy.ipynb` — the talk notebook (Parts 0–7)
- `RUN_OF_SHOW.md` — chronological punch list: every cell in order, talk vs run, clock marks
- `TALK_NOTES.md` — speaker notes, the argument, numbers cheat-sheet, pre-flight, anticipated Qs
- `build_notebook.py` — generates `lobotomy.ipynb`; edit here and re-run
- `scripts_prepare.py` — pre-computes GPT-2 activations for the toy NLA (Part 6)
- `scripts_train_gan.py` — pre-trains the MNIST DCGAN (Part 2)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision transformers sentence-transformers scikit-learn \
            matplotlib datasets jupyter ipykernel

# register this venv as a kernel, or the notebook will run against the wrong interpreter
python -m ipykernel install --user --name lobotomy --display-name "Python (lobotomize-llm)"

python scripts_train_gan.py            # ~8 min, writes models/mnist_dcgan_g.pt
python scripts_prepare.py gpt2         # ~90 s,  writes data/acts_gpt2.npz
jupyter notebook lobotomy.ipynb        # kernel: Python (lobotomize-llm)
```

`data/corpus.txt` (12k WikiText-2 sentences) is committed; `data/acts_gpt2.npz` (80 MB) is
generated. The two pre-compute steps exist so nothing slow happens on stage.

Verify everything runs before the talk:

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=lobotomy --output executed.ipynb lobotomy.ipynb
```

Then set `HF_HUB_OFFLINE=1` on the day so flaky conference wifi can't stall a
`from_pretrained` call mid-demo.

## The arc

| Part | Beat | Demo |
|---|---|---|
| 0 | The blackmail story you've heard — and the number under it | — |
| 1 | Compression: autoencoders | MNIST autoencoder, live train |
| 2 | Latent spaces have directions | DCGAN latent walk + latent arithmetic |
| 3 | **Run the model backwards** | gradient ascent on the input: noise → "3" |
| 4 | The same trick on an LLM | ActAdd steering on GPT-2 |
| 5 | Lobotomy by scalpel | the refusal direction |
| 6 | **What if the bottleneck were English?** | toy NLA round-trip + audience injection |
| 7 | Back to the cold open | — |

## Three lobotomies

| | Method | Instrument |
|---|---|---|
| 1 | Ablate attention entirely *(last talk)* | sledgehammer |
| 2 | Project out the refusal direction | scalpel |
| 3 | Verbalize a thought, edit the text, re-encode | a text editor |

## Sources

- [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) — Anthropic, 2026
- [Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment) — Anthropic, 2025
- [Refusal in Language Models Is Mediated by a Single Direction](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf) — Arditi et al., NeurIPS 2024
- [Activation Addition](https://arxiv.org/abs/2308.10248) — Turner et al.

> **Note on Part 5.** The refusal-direction demo measures the *mechanism* — the model's
> probability of beginning a refusal, before and after ablation — and reports aggregate
> rates. It does not print ablated completions to harmful prompts. The point is how
> shallow the safety property is, and a collapsing bar chart makes that better than a wall
> of text would.
