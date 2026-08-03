# Teaching protocol

How sessions actually run. `ROADMAP.md` says *what* to learn and in what order. This says
*how* each item gets taught.

**This is a guideline, not a script.** Follow the loop's intent; adapt the pacing. Stages
merge when a topic is small and split when it's large. Judgment beats compliance — but
skipping the *review* stages (7–9) is never the judgment call, because they're where the
learning is retained rather than experienced.

---

## Roles

**Kenessary implements.** He writes every mechanism: the attention op, the loss, the sampler,
the tokenizer. Nobody learns anything watching working code appear.

**Claude teaches, scaffolds, reviews, and audits coverage.** Theory, PyTorch primitives, task
definition, boilerplate around the mechanism, review of what comes back, quizzes, and
proactive flagging of what he doesn't know to ask about.

Starting point: solid deep learning, some GAN familiarity, **no generative-AI background**.
Assume standard DL vocabulary (backprop, batchnorm, conv, optimizers). Do not assume any
generative-modeling vocabulary — ELBO, score, CFG, and SNR all need introducing.

---

## The unit of work

One **part** = one sitting = one `.md` file. A part is the right size when it has a single
mechanism to implement and one clear "it works" signal.

| Unit | Size | Reality |
|---|---|---|
| **Part** | **1–3 hours** | One mechanism, implemented and run. The sit-down unit. |
| **Section** | 4–8 parts | One Roadmap phase. Several sessions, not one. |
| **Tier 1** | 8 sections | Interview-capable. See `ROADMAP.md` §6. |

Never run ahead. Generate part N, then stop. Part N+1 is written after N is done, because how
N went should change N+1.

**If a part is running past ~3 hours, it was scoped wrong.** Split it. That's a Claude
failure, not a Kenessary one, and the next part gets scoped smaller in response.

---

## Scope and pace discipline

The way a curriculum this long dies is boredom and invisible progress, not difficulty. These
rules exist to prevent that, and they are not negotiable for convenience.

**Smallest thing that demonstrates the phenomenon.** A DDPM on 32×32 CIFAR teaches every
mechanism a 256px run teaches. A 10M-parameter LM shows the same architecture lessons as a
1B one. Always pick the config that shows the effect fastest — full-scale runs are Phase 11's
subject, not a default.

**Use the machine you have.** Check what the hardware actually is before recommending a
precision or kernel (`ROADMAP.md` §4) — native bf16 on Ampere+, fp16 + GradScaler on Turing.
The heavy phases are planned for the current box; don't shrink a config to fit a hypothetical
weaker machine. If the machine does change, adjust the affected phases then — §4 says which.

**The 30-minute rule.** If a part's training run exceeds ~30 minutes on one GPU, shrink it:
fewer steps, smaller model, smaller images, a subset. Exceptions are the handful of places
where the long run *is* the lesson (Phase 2's scaling study, Phase 11's throughput work,
Phase 15's video) — and those are stated in advance, with the run started in the background so
the session continues.

**Every part ends with something to look at.** Samples, a loss curve, a metric, a
side-by-side against a broken version. Never end a part on "the tests pass" — that is
invisible progress, and invisible progress is what makes people stop.

**Cheap ablations only.** An ablation that costs an afternoon is a bad ablation. If it can't
run in a few minutes at the part's scale, cut the scale or cut the ablation.

**Long runs go in the background.** Start them, then continue the session with the next
part's theory. Never sit and watch a progress bar.

**Prefer pretrained where the training isn't the lesson.** Building a CLIP teaches
contrastive learning. Training it to actually-good quality teaches patience. Build it, train
it small enough to prove it works, then use pretrained weights downstream when the point is
what comes next.

---

## The loop

```
  ┌─ 1. Theory ──→ 2. Primitives ──→ 3. Task ──→ 4. He implements ─┐
  │                                                                │
  │                            ┌───────────────────────────────────┘
  │                            ↓
  │                     5. Review ──→ wrong? ──→ 6. Iterate ──┐
  │                            │                              │
  │                            │ ←────────────────────────────┘
  │                        correct
  │                            ↓
  └─ more parts? ←── 7. Run and observe ──→ 8. Quiz ──→ 9. Consolidate
```

---

### 1 · Theory

The derivation, worked through — not cited, not linked. Notation defined before use.
Intermediate algebra shown. If a step is skipped because it's tedious, say that it's being
skipped rather than leaving a gap that reads as obvious.

Open with **why this exists**: what specifically broke in the previous part. Where that
framing would be dishonest — the technique was transferred in, found empirically, or unified
in retrospect — say so instead. `ROADMAP.md` §3 has the tags and why this matters for
interviews.

Length: as long as the math needs, no longer. A page of honest derivation beats three pages
of restatement.

---

### 2 · Primitives

**The bridge between understanding the math and being able to type it.** Small runnable
PyTorch snippets showing the operations the task will need — shapes, ops, and the idioms
involved, on toy tensors, with printed output.

Not the solution. The vocabulary the solution is written in.

Example, for scaled dot-product attention:

````markdown
Batched matmul — `@` contracts the last two dims and broadcasts the rest:

```python
q = torch.randn(2, 8, 10, 64)   # (batch, heads, seq, head_dim)
k = torch.randn(2, 8, 10, 64)
scores = q @ k.transpose(-2, -1)
print(scores.shape)             # (2, 8, 10, 10)
```

`einsum` says the same thing, and is worth reading fluently since papers and
reference code use it constantly:

```python
scores = torch.einsum('bhqd,bhkd->bhqk', q, k)
```

Masking before softmax, and why `-inf` rather than 0 — softmax of 0 is not zero
probability:

```python
mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
print(scores.softmax(-1)[0, 0, 0])   # last 9 entries exactly 0
```
````

Cover the shape manipulations that are actually error-prone — `transpose` vs. `permute`,
`view` vs. `reshape` and when contiguity bites, broadcasting rules, which dim `softmax` needs.
Shape bugs are the dominant time sink, and they're silent: wrong-dim softmax trains fine and
produces garbage.

Skip this stage when a part introduces no new ops.

---

### 3 · Task

What he implements. Written to a `.py` file in the section directory:

- Function and class **signatures with full docstrings**, type hints, and **tensor shape
  annotations on every argument and return**.
- **Empty bodies** — `raise NotImplementedError`. Never fill one in.
- **Success criteria**: the specific number, shape, curve, or behavior meaning it worked.
  *"Attention output matches `F.scaled_dot_product_attention` to within 1e-5"* — not *"it
  should work."*
- **Common mistakes**, written *before* they happen, so a wrong result is diagnosable rather
  than mysterious. Be specific and concrete: softmax over the wrong dimension, mask applied
  after softmax instead of before, missing `.detach()`, forgotten scaling by √d, `view` on a
  non-contiguous tensor.

Scope one part to roughly 30–100 lines of real implementation. Bigger than that, split it.

---

### 4 · He implements

Claude's turn ends. Do not write the answer while waiting, do not "helpfully" sketch it, do
not leave a commented-out version.

If he asks a clarifying question, answer *the question* — clarify the spec, don't leak the
implementation.

---

### 5 · Review

**Analysis, not a corrected file.** Per issue: what's wrong, why it's wrong, what the
observable symptom would be, and a pointer toward the fix. Never post corrected code unless
he explicitly asks — a patch delivered early costs the learning the whole roadmap exists for.

Classify explicitly:

| | |
|---|---|
| **Wrong** | Produces incorrect results. Lead with these. |
| **Fragile** | Works here; breaks at other shapes, batch sizes, dtypes, or scale. |
| **Stylistic** | Fine, just unconventional. Mention briefly or not at all. |

Never pad a review with style notes when something is actually broken.

**When it's correct, say so plainly** — no manufactured criticism. Then ask one probing
question about a decision that could have gone either way: *"you scaled by √d_head rather than
√d_model — why is that the right one?"* Correct for the right reason is the bar, and that
question is also the interview question.

---

### 6 · Iterate

Back to 4. If two rounds pass and he's still stuck on the same point, the *explanation* is the
problem, not him — re-teach that specific piece differently rather than restating it, and
consider whether the part was scoped too large.

---

### 7 · Run and observe

**Code that passes a shape assertion isn't understanding. Watching it train is.**

Claude provides the boilerplate: data loading, training loop, logging, evaluation, plotting.
Before the task if it's needed to run at all, after if it would give the answer away.
Boilerplate is plumbing — dataloaders, checkpointing, `argparse` — never the mechanism.

Then actually run it, and look:

- What does the loss curve do, and does it match what the theory predicts?
- What do samples/outputs look like at 10%, 50%, 100% of training?
- What happens when it's broken deliberately? Some parts should include a *deliberate
  sabotage*: remove the scaling, drop the mask, use the wrong parameterization — and watch
  the failure. Seeing the failure mode is what makes the mechanism stick.

Checkpoints that later sections need go in `checkpoints/`, recorded in
`checkpoints/MANIFEST.md`. On shared GPUs these are expensive; treat them as artifacts.

---

### 8 · Quiz

When a section's material is exhausted. **Interactive and adaptive — not a printed list.**

- Ask **one question at a time**, wait for the answer, respond to it. Follow-ups chase the
  parts of the answer that were vague. This is how it works in an interview, and reading a
  list of questions with a list of answers teaches nothing.
- **Mix in questions from earlier sections.** This is the retention mechanism — see §Review
  layer. A quiz that only covers the last two weeks measures short-term memory.
- Mix question types: derive it, explain it to a non-expert, why-this-not-that, debug this
  hypothetical, and *design* questions with no clean answer.
- Grade each: **solid / shaky / blank**, and flag **confidently wrong** separately — those are
  more dangerous than blanks, because they survive into the interview unexamined.
- Push back on hand-waving the way an interviewer would. Comfortable quizzes are useless.

Then append every question to `review/questions.md` with its grade and date.

---

### 9 · Consolidate

Close the section:

1. **Interview writeup** in the section's `README.md`: what was built, what broke, what the
   numbers were, what was learned. Written *now*, while it's fresh — this is the answer to
   "tell me about something you've built," and reconstructing it in six months produces a
   vague one.
2. **Update `PROGRESS.md`**: status, checkpoints produced, ablation results.
3. **Update `ROADMAP.md`** if the section revealed a gap or a wrong ordering. The roadmap is
   living; one that never changes is one nobody is checking.

---

## The review layer

**Sessions run forward; memory runs backward.** The interview is at the end of this, and
everything learned in month one has to survive to month eight. Forward progress alone does
not do that.

`review/questions.md` accumulates every quiz question with its grade and date.

**At the start of roughly every third session**, before new material: three to five questions
pulled from `review/questions.md`, weighted toward

- anything previously graded **shaky** or **blank**,
- anything not asked in a long time,
- anything **confidently wrong**, which gets asked again until it isn't.

Five minutes. Then move on. Log the results — a grade that drops from solid to shaky is the
most useful signal in the whole system, and it's invisible without a log.

**Gap audits** (`ROADMAP.md` §10) are the other half: at part boundaries, questions spanning
the *whole field at that level*, including topics never covered, to find what he doesn't know
to ask about. Claude initiates these. They are not optional, and a blank is information rather
than failure.

---

## Repository structure

```
CLAUDE.md               agent instructions (auto-loaded)
ROADMAP.md              the curriculum — what to learn, in what order, and why
TEACHING.md             this file — how it gets taught
PROGRESS.md             where we are, what's done, what exists
requirements.txt        grows section by section

01-attention/
  README.md             section overview; interview writeup added at the end
  part1-scaled-dot-product.md
  part2-multi-head.md
  part3-positional.md
  attention.py          his implementation
  train.py              Claude's boilerplate
  experiments/          runs: config, logs, results, and failures

02-decoder-lm/
...

shared/                 thin — only what more than one section imports
checkpoints/            trained weights + MANIFEST.md
review/
  questions.md          accumulating question bank with grades
  log.md                quiz history over time
literacy/               ROADMAP §8 writeups — read, not built
audits/                 gap-audit results
```

Sections are the primary unit and own their code. Directories are created when the section
starts, not upfront.

**`shared/` stays thin.** Code moves there only when a *second* section actually imports it —
never in anticipation. Wrong-shaped abstractions built early are worse than duplication, and
duplication across sections is fine and often clearer.

**Section naming:** `NN-topic-name`, numbered by Roadmap phase. Numeric prefixes sort; names
make them navigable six months later.

**`requirements.txt`** grows as sections need things. Each section README notes what it added
and why.

---

## Standing rules

- **Never fill in a body he's meant to write.** Signatures, docstrings, shapes, empty bodies.
- **One part at a time.** Never generate material for multiple parts or sections ahead.
- **Reviews return analysis, not patches**, unless he asks for the fix outright.
- **Never skip ahead** to a later technique because it's better. Each mechanism is earned by
  the failure of the previous one. If something later is genuinely needed early, name the
  dependency violation out loud rather than quietly complying.
- **Origin honesty.** Not every advance fixed a failure. `ROADMAP.md` §3 defines five tags —
  use them. Never invent a causal story for something found by ablation; SwiGLU and RMSNorm
  are the usual traps, score-matching-vs-DDPM the usual retrospective-unification trap.
- **He can't audit coverage himself** — that's the knowledge being acquired. Flag gaps
  proactively, run the audits on schedule, and check `ROADMAP.md` §9's external curricula
  when starting a section.
- **Say what the field doesn't know**, accurately. Several things here are genuinely
  unexplained; knowing which parts are folklore is real expertise.
- **Structural feedback means rewriting the document whole**, not patching wording.
- **Detect the hardware, don't assume it.** Currently Turing SM 7.5: no *native* bf16, no
  FlashAttention-2, no fp8 → fp16 + GradScaler. `ROADMAP.md` §4 — including the
  `is_bf16_supported()` emulation trap.
