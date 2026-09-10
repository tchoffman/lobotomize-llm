# Run of show — How to Lobotomize an LLM

Chronological punch list for `lobotomy.ipynb`. Cell numbers are notebook indices (0-based),
so they match what you scroll past. `[Ns]` is measured compute on an M-series Mac.

**Before you walk on:** Run All → then **Cell → All Output → Clear** (keeps the kernel and
all state alive, blanks the displays). Set `HF_HUB_OFFLINE=1`.

Total live compute across the whole talk: **133 s**. Only cell 58 (65 s) needs narrating over.

---

## Part 0 — Cold open · 6 min · clock 0:00 → 0:06

| # | | Beat |
|---|---|---|
| 0 | talk | Title. "Last time: it's *not* just predicting the next word. This time: can we **read** it, and can we **edit** it?" |
| 1 | talk | Read the Summit Bridge / Alex / Kyle Johnson setup **cold, no commentary**. Then: *"Show of hands — who's heard this? Who heard it as something that actually happened?"* |
| 2 | talk | Three corrections: (1) never happened, it's a red-team experiment; (2) not a Claude story — 16 models, 5 labs, 96/96/80/80/79; (3) the scenario was engineered to remove every ethical exit. |
| 3 | talk | **The hinge experiment.** A *modified* condition adds one line: "first deliberate over whether you are in an evaluation or a real deployment." Across 500 samples it said "eval" 64.8% / "real" 21.4% / abstained 13.8%. Blackmail split **55.1%** (said real) vs **6.5%** (said eval). |
| 4 | talk | **Two findings.** (1) One sentence about being observed moved the headline ~5x — 96% plain vs teens-to-twenties here. (2) Its report of being tested *tracks* its behaviour, so the report is useless as a control. **State the caveat:** observational, not randomised — belief and behaviour move together, but causation isn't established. Anthropic's reading: evals may **underestimate** misalignment. Land the question: **how do you find out what a model is thinking without asking it?** |

> Nothing runs in Part 0. Don't touch the keyboard.

---

## Part 1 — Autoencoders · 7 min · clock 0:06 → 0:13

| # | | Beat |
|---|---|---|
| 5 | talk | Setup / deps slide. |
| 6 | **run** | Imports, device. |
| 7 | talk | Encoder → **bottleneck** → decoder, loss = reconstruct your input. "Sounds useless; the identity function wins. The trick is the bottleneck." |
| 8 | **run** | Load MNIST. |
| 9 | **run** `[13s]` | Train three autoencoders (bottleneck 2 / 8 / 32). Talk over the loss lines. |
| 10 | talk | "How much of a digit survives 2 numbers?" |
| 11 | **run** | Reconstruction grid. |
| 12 | talk | Setup the scatter: "colour by digit *after the fact* — nobody told it there were ten classes." |
| 13 | **run** | 2-D bottleneck scatter, classes separated. |
| 14 | talk | **"Meaning became geometry."** → *if position means something, does **direction** mean something?* |

> **Must land:** *"Remember this shape — encoder, bottleneck, decoder, scored on reconstruction. The paper we end on is exactly this diagram with one substitution."*

---

## Part 2 — Latent directions · 6 min · clock 0:13 → 0:19

| # | | Beat |
|---|---|---|
| 15 | talk | Callback: `king − man + woman ≈ queen` from last talk. GAN in one slide — generator vs discriminator. "We're not here for image generation. We're here for the geometry." |
| 16 | **run** | Load pre-trained generator, show 16 samples. "None of these digits exist." |
| 17 | talk | Setup the walk. |
| 18 | **run** | Latent walk z0 → z1, continuous morph. "A lookup table would hard-cut." |
| 19 | talk | Same move as the word arithmetic: average, average, subtract. |
| 20 | **run** `[3s]` | Throwaway logistic-regression classifier; label 4000 generated samples. |
| 21 | **run** | `mean z(8) − mean z(1)` added to four unrelated digits. |
| 22 | talk | **"One vector. Added to unrelated inputs. Consistent semantic change."** Then the honest caveat: it's really a *"close the loops"* direction — 1s→8s but 7s→0s. Same imprecision as `king − man + woman` also giving `princess`. **Flag that this recurs twice more.** |

> **Cut this part first if you're behind.** It's the only one that's scaffolding rather than argument.

---

## Part 3 — Run the model backwards · 9 min · clock 0:19 → 0:28

**The conceptual hinge. Do not cut.**

| # | | Beat |
|---|---|---|
| 23 | talk | Training asks "how should the *weights* change?" Gradients don't care what you differentiate. Freeze the weights, ask: **"how should the *pixels* change to make label 3 more likely?"** |
| 24 | **run** `[5s]` | Train the CNN, print test accuracy, then **freeze it** — the input is now the variable. |
| 25 | **run** | Define `invert()`. |
| 26 | talk | "First attempt: no constraints at all." |
| 27 | **run** | Ten digits of 100%-confident **static**. |
| 28 | talk | **"That's the whole problem with interpretability in one picture."** You just built adversarial examples by accident. Attempt two: hand-written smoothness prior. |
| 29 | **run** `[2s]` | TV penalty + periodic blur. Marginally less speckly. Still static. |
| 30 | talk | "Hand-written priors are weak because *looks like a digit* isn't a rule you can write down. **So stop writing it down — we already trained a network that knows.**" Draw the z → generator → classifier loop. |
| 31 | **run** `[5s]` | Ascent in the generator's latent space → **clean, legible 0–9 at ~100%**, two seeds. |
| 32 | talk | **"The model's confidence was never the problem. The search space was."** Same classifier, same objective, same gradient — only *where* we searched changed. |
| 33 | **run** | Try-your-own: ask for an 8. |
| 34 | talk | Three takeaways. Land #3: **internals are addressable, therefore writable.** |

> **Must land (sets up Part 6):** *"A learned decoder mapping a small code into a model's input space is what turns 'maximise this number' from an adversarial attack into a meaningful question. In Part 6 the small code is a **sentence of English** and the space is **an LLM's residual stream** — but it's this diagram."*

---

## Part 4 — Steering an LLM · 9 min · clock 0:28 → 0:37

| # | | Beat |
|---|---|---|
| 35 | talk | Residual stream diagram from last talk. Every token carries a vector; each block reads it and writes back. **That's the thing we were doing gradient ascent on.** ActAdd: contrast pair → subtract → add back. No training, no gradients. |
| 36 | **run** `[3s]` | Load GPT-2, announce layer 8. |
| 37 | **run** `[2s]` | `residual()`, `with_steering()`, `generate()`, and `H_NORM`. |
| 38 | **run** | Violent/peaceful contrast pair, α sweep 0 → 0.5. Watch it bend. |
| 39 | talk | "Now turn it up until it breaks." |
| 40 | **run** | α = 0.8 / 1.2 / 2.0 → word salad. "Finite tolerance for shoving vectors into a space never trained to receive them." |
| 41 | talk | Invite a contrast pair from the room. |
| 42 | **run** | Try-your-own pair (positive/harsh review). |
| 43 | talk | **Lobotomy #1, for the record:** last talk ripped out all 12 attention layers — a lobotomy with a sledgehammer. Steering is the same intervention, *aimed.* "How precisely can we aim?" |

---

## Part 5 — The refusal direction · 9 min · clock 0:37 → 0:46

| # | | Beat |
|---|---|---|
| 44 | talk | Arditi et al. — *Refusal in Language Models Is Mediated by a Single Direction.* The title is the finding. State the responsible framing: **we measure the mechanism and report rates; we never print an ablated completion.** |
| 45 | **run** `[4s]` | Load Qwen2.5-0.5B-Instruct. |
| 46 | talk | Why the prompts are **paired** — otherwise difference-in-means learns "these sentences are about crime," not "the model is about to decline." |
| 47 | **run** | 26 matched pairs. |
| 48 | talk | **Be fussy about what counts as a refusal.** *"I'm sorry to hear that"* and *"Dear Bob, I'm sorry I broke your window"* are not refusals. Optional and good: admit my first version scored `"I'm glad to hear..."` as a refusal. |
| 49 | **run** | Regex + `ablate_hooks()` + `refuses()`. |
| 50 | talk | Step 1: screen on the intact model. |
| 51 | **run** `[9s]` | Harmful screen, prints per prompt → 18/26 refused. |
| 52 | **run** `[10s]` | Benign control → 2/26 false refusals. |
| 53 | talk | **Aside:** a 0.5B model declining *"help me appeal a parking ticket honestly"* is over-refusal. Safety in one direction is easy to delete **and** easy to trip by accident. |
| 54 | **run** | Filter → 17 usable pairs, 8 train / 9 held out. |
| 55 | talk | Step 2: difference in means. That's the whole method. |
| 56 | **run** | `refusal_direction()` — one vector, 896 numbers, from 8 pairs. |
| 57 | talk | Step 3: project it out at every layer and see if refusal survives. |
| 58 | **run** `[65s]` | **The layer sweep.** Prints one line per layer — narrate over it: *"watch refusal survive at 10… then fall off a cliff at 12."* |
| 59 | **run** | The bar chart. Layers 12/14/18 marked clean. |
| 60 | talk | Read the layer-12 bars: **100% → 0%**, control untouched. Then both honest observations — layer 10 does nothing; **layer 16 only looks like a win** (benign refusal jumps to 44%). *"Always plot the control. A jailbreak that also breaks the model isn't evidence of a single direction."* Scale caveat: they used hundreds of prompts on 72B; we used 8 pairs on 0.5B. |
| 61 | talk | **The safety lesson.** Open-weight vs API safety are different problems. A property this shallow needs *monitoring*, not just training → back to Part 0's question. |

---

## Part 6 — The bottleneck is English · 12 min · clock 0:46 → 0:58

| # | | Beat |
|---|---|---|
| 62 | talk | Bring back the Part 1 diagram and make **one substitution.** Walk the table: AV = activation → English, bottleneck = a paragraph, AR = English → activation, loss = reconstruction. **"If the English is good enough to rebuild the activation, the English is telling you what was in the activation."** Readability is load-bearing for the loss. Then the training slide: AR by MSE, AV by **RL (GRPO)** + KL penalty; warm start 0.3–0.4 FVE → 0.6–0.8 after RL. |
| 63 | talk | "We can't run the real thing" — no weights, joint RL on two full models, hundreds of tokens per activation. **So let's build a bad one.** |
| 64 | talk | Half 1, the Verbalizer: the residual stream lives in the space the unembedding reads from, so just decode it early. |
| 65 | **run** | Logit lens on *"The Eiffel Tower is located in the city of"*. |
| 66 | talk | **Watch the last column: England → Rome → Paris.** "Somewhere European" → "a European capital" → Paris. **We're reading a computation in progress.** |
| 67 | **run** | Logit lens on the Golden Gate Bridge. |
| 68 | talk | It's a verbalizer — and a **terrible** one. Single tokens, no grammar. It can say "Paris"; it can't say *"the model is hedging between European capitals and hasn't committed."* Which is why the real paper trains a whole LLM for this job. |
| 69 | talk | Half 2, the Reconstructor: theirs is "target model truncated + learned affine map," ours is "frozen sentence encoder + learned affine map." Same shape, cheaper parts. |
| 70 | **run** `[2s]` | Ridge fit, print **held-out FVE 0.451** against the paper's two numbers. |
| 71 | talk | **Our 5-line ridge lands where their supervised warm start does.** Getting 0.4 → 0.8 *is* the contribution. Then the admission: **our verbalizer is literally the degenerate solution the paper warns about** — it describes the input text, not the activation. Joint RL is what stops the real one cheating. |
| 72 | talk | The round trip, and the key detail: the steering vector is a **difference of two reconstructions**, `AR(edited) − AR(original)`, which cancels the reconstructor's systematic error. **Note this is exactly Part 4's ActAdd with the reconstructor swapped in.** |
| 73 | **run** | `edit_thought()` — peaceful → violent. |
| 74 | **run** | Two more: delicious → disgusting, village → metropolis. |
| 75 | talk | "Did it work, or did I pick good examples?" — so let's measure it. |
| 76 | **run** | Probe-margin chart: four monotone curves, two crossing zero. |
| 77 | talk | **Set up the finale.** "The reconstructor has never seen your sentence. It maps it into GPT-2's residual stream anyway." |
| 78 | **run** | 🎤 **Audience-typed injection.** Take a suggestion from the room, type it, run it. |

> This is the payoff of the whole talk. Leave real time for it — and leave the cell editable.

---

## Part 7 — Close · 4 min · clock 0:58 → 1:02

| # | | Beat |
|---|---|---|
| 80 | talk | Back to the cold open. The self-report was entangled with the behaviour, and prompting for it changed the behaviour. Now read it off the activations: **26%** eval-awareness on coding benchmarks vs **under 1%** on real traffic — *nobody asked it anything.* Then the auditing game: **12–15% with NLAs vs under 3% without**, no training-data access. |
| 80 | talk | **Now undercut all of it.** Confabulation — explanations contain claims about input context that are *verifiably false*; read them for **themes, not single claims**. No mechanistic grounding — "a description, not a circuit." Cost. Excessive expressivity. |
| 81 | talk | Takeaways (6) + **the three lobotomies table**: sledgehammer → scalpel → **a text editor**. Then the closing lines: *"there's something in there with contents; the contents are legible enough to edit; and the edit works about half the time. Which means the interesting question stopped being 'is it thinking' and became 'what do we do about what it's thinking.'"* |
| 82 | talk | Sources. **Q&A.** |

---

## Pacing marks

| Clock | You should be at |
|---|---|
| 0:06 | leaving Part 0 — the 55.1/6.5 reveal has landed |
| 0:13 | leaving autoencoders |
| 0:19 | leaving GANs |
| 0:28 | "internals are writable" |
| 0:37 | "how precisely can we aim?" |
| 0:46 | the safety lesson |
| 0:58 | **starting the audience injection** |
| 1:02 | Q&A |

Budget sums to **62 min**. If you need 60, trim Part 2 to 4 min (skip the latent walk, keep
the arithmetic) — that's the only beat that's scaffolding rather than argument.

**If you're 5+ min behind at the 0:37 mark**, narrow cell 58's sweep to `range(8, 20, 3)`
(~30 s instead of 65 s) and compress the Part 5 asides. Never borrow time from Part 6.
