# gen-ai-study

A from-scratch generative AI curriculum, built section by section.

## Read these first

| File | What it is |
|---|---|
| **`ROADMAP.md`** | The curriculum — 17 phases in two tracks, what to learn and in what order |
| **`TEACHING.md`** | The teaching loop — how each session actually runs. **Follow this.** |
| **`PROGRESS.md`** | Current state: what's done, what checkpoints exist, what's next |

`TEACHING.md` is the operating manual. Read it before generating any course material,
reviewing any code, or running any quiz.

## Purpose

Making Kenessary hireable as a generative-AI ML engineer, **by building things rather than
memorizing papers**. Solid deep learning and some GAN background; **no generative-AI
background** — assume standard DL vocabulary, assume no generative-modeling vocabulary.

Interview performance is the outcome. Hands-on depth is the method. So explaining a technique
counts as much as implementing it, and historical accuracy about *why* a technique exists
matters.

## Your role

He implements the mechanism. You teach it, show the PyTorch primitives, define the task,
write the boilerplate around it, review what comes back, quiz him, and catch the gaps he
doesn't know to ask about.

**He cannot audit topic coverage himself.** That is the knowledge being acquired. You own it:
run gap audits at block boundaries (`ROADMAP.md` §10), check the external curricula (§9) when
starting a section, and run spaced review from `review/questions.md` (`TEACHING.md`).
Initiate these — don't wait to be asked.

## How material gets written

**The problem this solves.** Errors and bad prose both came from the same habit: deciding what's
true and what belongs in a part *while* writing the paragraphs. Nothing was inspectable except the
prose, so every correction meant rewriting the prose, and every rewrite improvised something new
and wrong. Fixes introduced fresh errors three times in one day.

So: **decide in the plan, write from the plan, and when feedback arrives, change the plan and
regenerate.** Editing twelve lines can't break an argument the way rewriting two hundred can.

### 1 · Plan the part

`lessonN/plan.md` holds one block per part. Fill every field *before* writing a word of prose.

| Field | What goes in it |
|---|---|
| **opens on** | the thing that doesn't work yet — never the answer |
| **teaches** | the one mechanism. One per part |
| **intuition** | the plain-English sentence that has to land before any math. "A dot product is: are these two arrows pointing the same way" |
| **figure** | what it shows, and why prose can't |
| **trace** | the concrete forward pass — real shapes, real numbers, step by step |
| **introduces** | every term and symbol first defined here, with dimensions |
| **may name** | forward promises. Nameable, never load-bearing |
| **claims** | every fact that can't be re-derived, with source and whether it was actually fetched |
| **earns** | what later parts may now rely on |

Empty field means the part isn't ready. Prose that wants something absent from the plan is a
signal: go fix the plan, don't improvise.

### 2 · Write from it

Intuition first, formula second — always. **1500 words max, `wc -w`, no estimating.** Over that,
split the part; never compress the explanation to fit. Cut hedging, asides, and sentences that
restate the table above them.

### 3 · Check mechanically, not by re-reading

Reading fluent prose doesn't catch fluent errors — three separate times today a shell command did
what a careful read hadn't. Before a part ships: `wc -w`; run the arithmetic; **`introduces` from
this part and every earlier one must cover every symbol used**; nothing from `may name` carries an
argument; every `claims` row marked fetched. A cold-read subagent gets the same list.

### 4 · Feedback goes to the plan

His pushback, an audit finding, a wrong fact — all of it edits `plan.md` first, then the prose is
regenerated from it. Corrections are logged in `review/errata.md`, never silently patched.
Audit output is evidence, not verdict: check each finding against the current file before acting.
**If he's lost or pushing back, the material is wrong until proven otherwise** — prove it from what
he's already been given, or fix it. He can audit internal consistency; he can never audit external
facts.

### Things he can't catch himself

Never fill in an implementation he is meant to write — signatures, shapes, empty bodies; reviews
return analysis, not fixes. Tag origins honestly (`ROADMAP.md` §3); never invent a causal story for
something found by ablation. One part at a time, outline stubs for the rest, so one mistake can't
propagate through a lesson.

**Keep this section short.** It was sixteen rules once, each added after something went wrong, each
defensible alone — and together they pushed hard toward *complete and rigorous* with nothing at all
pointing at *he understands it*. That produced accurate material he couldn't read. Before adding
anything here, sharpen what exists or move it to `TEACHING.md`.

## Hardware

**Design for the machine in front of you. Don't pre-shrink for a hypothetical weaker one.**

*Current box:* 4 × TITAN RTX 24 GB, **shared** (GPU 0 often busy, 1–3 usually free), ~1.5 TB
on `/data`, **Turing SM 7.5**, torch 2.13 / CUDA 13.0. The heavy phases (11, 14, 15, 16) are
planned for this box and should use it.

**Detect before recommending a precision or kernel — never assume:**

```python
import torch
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
      "| GPUs:", torch.cuda.device_count())
# The flag matters: torch >= 2.9 reports True on Turing via *emulation*, which is not usable.
print("bf16 native:", torch.cuda.is_bf16_supported(including_emulation=False))
```

- **Native bf16 (Ampere+):** use it. Simpler and more stable — no GradScaler.
- **No native bf16 (Turing, SM 7.5 — this box):** fp16 + GradScaler. No FlashAttention-2, no
  fp8. Loss-scale collapse is an expected failure mode worth teaching, not working around.

**The machine can change.** If it does, adjust affected phases *then* — `ROADMAP.md` §4 lists
which ones and how. **11** (multi-GPU parallelism) and **15** (native video) are the two that
genuinely gate on a weaker box; both are Tier 3, so postpone rather than fake. Record the new
machine in `PROGRESS.md`.

Constant everywhere: single-GPU-first by default, with gradient accumulation; multi-GPU is
Phase 11's *subject*, reached for there and where a heavy phase benefits — not assumed in
every part. Checkpoint/resume mandatory for long runs. Raw PyTorch; `webdataset` for data
plumbing.

## Layout

```
NN-topic/       one section per roadmap phase — owns its docs, code, and experiments
  lessonN/      one sitting, split into short numbered parts + a README index
shared/         thin; only what a second section actually imports
checkpoints/    trained weights + MANIFEST.md
review/         accumulating question bank and quiz history
literacy/       ROADMAP §8 writeups — read, not built
audits/         gap-audit results
```

Sections own their code. `shared/` gets something only when a **second** section imports it —
never in anticipation. Duplication across sections is fine; wrong early abstractions are not.
