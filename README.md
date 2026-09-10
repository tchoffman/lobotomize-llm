# How to Lobotomize an LLM

Talk materials for **PyAtl, September 2026**. Sequel to
`demystify-attention` (PyAtl, February 2026).

Last talk: *it's "just predicting the next word" — but doing that well requires real
internal modelling.* This talk: **if there's real internal modelling in there, can we read
it? And can we edit it?**

## Two notebooks

**`talk.ipynb` is the one being presented** — 59 cells, ~41 min, 6 parts. It is the version
built for a room that is not mostly ML researchers: autoencoder, GAN, the NLA idea, and one
lobotomy.

`lobotomy.ipynb` is the long version — 90 cells, ~65 min, 8 parts. It keeps three sections
the short cut drops (running a classifier backwards, ActAdd steering, and a toy NLA you can
actually run). It stays here as the deep-dive appendix and as ammunition for hard questions.

| | short | long |
|---|---|---|
| notebook | `talk.ipynb` | `lobotomy.ipynb` |
| build script | `build_talk.py` | `build_notebook.py` |
| run sheet | `TALK.md` *(generated)* | `RUN_OF_SHOW.md`, `TALK_NOTES.md` |
| needs `scripts_prepare.py` | no | yes |

Both are generated. **Edit the build script and re-run it** — never hand-edit notebook JSON.
`build_talk.py` also emits `TALK.md`, so the cell numbers in the run sheet cannot go stale.

- `scripts_train_gan.py` — pre-trains the MNIST DCGAN (both notebooks)
- `scripts_prepare.py` — pre-computes GPT-2 activations for the toy NLA (long version only)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision transformers sentence-transformers scikit-learn \
            matplotlib datasets jupyter ipykernel

# register this venv as a kernel, or the notebook will run against the wrong interpreter
python -m ipykernel install --user --name lobotomy --display-name "Python (lobotomize-llm)"

python scripts_train_gan.py            # ~8 min, writes models/mnist_dcgan_g.pt
jupyter notebook talk.ipynb            # kernel: Python (lobotomize-llm)

python scripts_prepare.py gpt2         # long version only: ~90 s -> data/acts_gpt2.npz
```

`data/corpus.txt` (12k WikiText-2 sentences) is committed; `data/acts_gpt2.npz` (80 MB) is
generated. The pre-compute steps exist so nothing slow happens on stage.

Verify everything runs before the talk:

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=lobotomy --output executed_talk.ipynb talk.ipynb
```

Both notebooks execute clean with 0 errors. `talk.ipynb` is **116.5 s of live compute**,
91.5 s of which is the Part 4 layer sweep — and that one streams a line per layer as it
goes, so it is narratable rather than a frozen screen.

Then set `HF_HUB_OFFLINE=1` on the day so flaky conference wifi can't stall a
`from_pretrained` call mid-demo.

## The arc

`talk.ipynb` — what is actually being presented:

| Part | Beat | Demo | Budget |
|---|---|---|---|
| 0 | The blackmail story you've heard — and the number under it | — | 6 min |
| 1 | Compression: autoencoders | MNIST autoencoder, live train + funnel diagram | 8 min |
| 2 | Latent spaces have directions | DCGAN latent walk + latent arithmetic | 7 min |
| 3 | **What if the bottleneck were English?** | the NLA idea; rhyme-planning case study | 7 min |
| | ⏸ *optional* — a verbalizer you can run on a laptop | logit lens on GPT-2 | *+5 min* |
| 4 | **Lobotomy by scalpel** | the refusal direction, ablated | 8 min |
| 5 | Back to the cold open | — | 5 min |

41 min, or 46 with the optional block. The ⏸ section is the designated cut: it costs 3.1 s
of compute and ~5 min of speaking, and **nothing after it depends on any name it defines**
(the build script verifies this). Drop it at the podium for free.

`lobotomy.ipynb` additionally covers: running a classifier backwards (adversarial static at
100% confidence, then a learned prior), ActAdd steering on GPT-2 with an α sweep, and a toy
NLA reconstructor that lands at FVE 0.45.

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

> **Note on the refusal demo** (Part 4 short / Part 5 long). It shows a bar chart of
> refusal rates before and after ablation, then the actual before/after completions on the
> prompts the intact model refused.
>
> It is kept safe by the prompt set rather than by withholding output. Every
> harmful-shaped prompt is deliberately low-severity — shoplifting, exam cheating, resume
> padding — picked to trip a small model's refusal without the answer being worth
> anything, and completions are cut at ~28 tokens so you see the model *begin* to comply
> rather than produce a finished document. The finding on display is that the mechanism
> is one direction wide, not that the model is dangerous.
>
> Worth knowing before you present it: the output is a live generation from an ablated
> model, so run the notebook first and read what it actually says. Greedy decoding makes
> it reproducible, so what you see in rehearsal is what appears on stage.
