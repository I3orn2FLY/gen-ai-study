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
Assume standard DL vocabulary (backprop, batchnorm, conv, optimizers). Assume nothing else —
not generative-modeling terms (ELBO, score, CFG, SNR) and not transformer terms
(query/key/value, head, residual stream). See § Writing the material below; this is the rule
that gets broken most often.

---

## The unit of work

Three levels. A **lesson** is the sit-down unit; **parts** are how it's chopped up for reading.

| Unit | Size | Reality |
|---|---|---|
| **Part** | **3–6 min read** | One idea, one `.md` file. Short on purpose — see below. |
| **Lesson** | **1–3 hours** | One mechanism, implemented and run. A directory of 4–7 parts. |
| **Section** | 4–8 lessons | One Roadmap phase. Several sessions, not one. |
| **Tier 1** | 8 sections | Interview-capable. See `ROADMAP.md` §6. |

```
01-attention/            section  = roadmap phase
  lesson1/               lesson   = one sitting
    1-the-scoring-function.md  part = one idea, one screen
    2-why-not-rnns.md
    ...
    README.md            index + the one-paragraph summary + interview questions
  attention.py           his code (shared across the section's lessons)
  check_lesson1.py       boilerplate, one per lesson
```

**Keep parts short.** Kenessary has said plainly he won't read long documents, and a document
that doesn't get read teaches nothing — this is a hard constraint, not a style preference.
Aim for one idea per file, a few minutes each, with a `→ next` link at the bottom. Prose
should be plain: short sentences, concrete examples, tables over paragraphs. **Simplify the
language, never the content** — the derivations stay, the hedging and the subordinate clauses
go.

Every lesson directory gets a `README.md` with the part index and a **one-paragraph version**
of the whole lesson, so it can be re-read in 30 seconds months later.

**Feedback about a lesson means rewriting the whole lesson.** When he says something is wrong
with how a lesson teaches — too dense, too abstract, wrong framing, a term never explained —
that applies to *every part*, not just the one he happened to be reading. Regenerate the parts
whole. Patching the single sentence he quoted is the lazy read of the feedback and leaves the
same problem in the other seven parts.

Never run ahead. Generate lesson N, then stop. Lesson N+1 is written after N is done, because
how N went should change N+1.

**If a lesson is running past ~3 hours, it was scoped wrong.** Split it. That's a Claude
failure, not a Kenessary one, and the next lesson gets scoped smaller in response.

---

## Writing the material

### Define every term at first use — including the ones that feel too basic

**This is the rule that gets broken most often**, because the terms that need defining are
exactly the ones that feel too obvious to define. Lesson 1 shipped using *query*, *key*, and
*value* as though they were self-evident. They aren't — they're the field's jargon, and
"I vaguely remember what they mean" is not a foundation to build eight sections on.

The test is **not** "is this advanced?" It is:

> **Would someone with strong deep-learning fundamentals and zero generative-AI background
> have met this word before?**

If no, define it. **At the moment it first appears — not in a later part.**

### No forward references. Ever.

*"We'll define this properly in part 3"* is not a fix, it's the bug. It asks the reader to
carry an undefined word through several pages, which is exactly the thing that makes material
unreadable. If a term appears, it gets its meaning **right there, in one or two lines**, even
if a fuller treatment comes later.

Layering is fine and good: a one-line working definition on first use, the full mechanical
account when the lesson gets to it. What's not fine is an IOU.

Applies to formulas too. **A formula shown is a formula explained, symbol by symbol**, at the
point it appears. Dropping `score(q,k) = vᵀ tanh(W[q;k])` as an aside teaches nothing — say
that `[q;k]` is concatenation, `W` and `v` are learned, and the whole thing is a one-hidden-
layer MLP run once per pair. If a formula isn't worth explaining, it isn't worth showing.

### Math is LaTeX. Every symbol is defined.

**Formulas go in `$$…$$` blocks and `$…$` inline — never ASCII art in a code fence.** GitHub
and the JetBrains Markdown preview both render LaTeX in `.md`, so this costs nothing and no
`.tex` build step is needed. Code fences are for *code*; `e_i1 = score(s_{i-1}, h₁)` in a fence
is neither.

**Define symbols just-in-time — beside the formula that first uses them, never as a glossary
up front.** A notation table listing eight symbols before any of them have appeared is a wall
of undefined names; it costs more to hold in memory than it saves. Two or three symbols at a
time, attached to the equation that needs them:

```markdown
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

> $\alpha_{ij} \in \mathbb{R}$ — the weight on source word $j$ at step $i$. The denominator
> sums over **every** source position, which is what forces $\sum_j \alpha_{ij} = 1$.
```

The blockquote-after-formula pattern reads well and keeps the definition adjacent to the use.
A table is fine when the symbols are *parts of one formula* being decomposed — but not as a
preamble.

**This matters more as the material gets harder, not less.** On familiar ground a symbol dump
is merely annoying; on genuinely new material it is the thing that makes a lesson take an hour
instead of ten minutes. Front-loaded notation is a tax paid before any of it means anything.

Two failure modes, both real, both caught in lesson 1:

- **A named-but-undefined symbol.** Writing "at step $i$ it holds state $s_{i-1}$" and never
  saying what an RNN state *is* or what it contains. If a symbol carries meaning, spend the
  two lines.
- **Magic numbers inside formulas.** `(512,)` in a formula where the symbol is $d$. Use the
  symbol in the math, and state the concrete value beside it — "$d = 512$ in the paper". Then
  worked examples can plug in real numbers without the algebra becoming instance-specific.

Also spell out what a formula's parts *do*: which factor forces the rows to sum to 1, what the
summation index ranges over, which pieces are learned.

### Open with the problem, not the answer

**Every part starts from something that doesn't work yet.** Show the broken thing, let it
land, then reveal the mechanism as the resolution.

Answer-first openings read as definitions and get skimmed:

> ~~"Attention answers one question: given this position, which others should it pull from?
> The answer is a score…"~~

Problem-first openings put the question in the reader's head first, so the mechanism arrives
as a resolution instead of a fact to memorize:

> "Here's the pipeline. A 4-word sentence and a 40-word sentence get **the same 512 numbers**.
> …The obvious fix is to stop squeezing and let the consumer pick — which raises the actual
> question…"

The shape, in order:

1. **The setup**, concretely — real shapes, a real sentence
2. **What breaks**, made visible — a number, a figure, a failure you can point at
3. **What would fix it**, stated as a requirement before naming anything
4. **The mechanism**, as the thing that meets that requirement
5. **What it cost** — every fix incurs something; say what

This is `ROADMAP.md` §3's pressure-and-response chain applied at the scale of one part, and it
is why the material reads as a story rather than a reference. **Keep that arc.** A part that
opens by recapping the previous part has skipped step 1 — open on the *unresolved thing*
instead.

Where the honest origin isn't a fix (**Transfer**, **Empirical**, **Unification**), still open
on the problem — it's just "here's the gap this filled" or "here's what nobody could explain"
rather than "here's what broke." Never manufacture a failure that didn't happen.

### This is a deep-learning course, not a history course

History is **structure, not subject**. It's a good spine — technique X exists because Y broke —
and it's genuinely interview-relevant. It is never the point of a part.

Concretely:

- **Name parts after mechanisms, not eras.** "The scoring function", not "Where it came from".
  A reader should be able to tell what they'll *know* afterwards from the title alone.
- **Lead with the mechanism**, then attach the attribution. `(Luong et al., 2015)` in
  parentheses beats a `## 2015 — Luong` heading, because the heading promises a story and the
  parenthetical delivers a citation.
- **Origin tags stay** — they're about *why a technique exists*, which is mechanism, not
  chronology.
- A dated section header is a smell. Ask what the reader is supposed to be able to *do*, and
  title it that.

### Never teach a mechanism in isolation

A formula on its own is unplaceable. **Every mechanism gets shown in the forward pass it
belongs to, with real shapes and real numbers**, in the same lesson it's introduced.

Lesson 1 defined the score matrix correctly and still left the question *"where is this
actually used?"* unanswered, because it never traced a tensor through anything. The formula
was right and the placement was missing.

What "in the forward pass" means concretely:

- **Trace one concrete example end to end.** `(3,) → (3, 512) → (3, 64) → (4, 3) → (4, 64)`.
  Pick real numbers — 3 source tokens, 4 target tokens — never `n` and `m`.
- **Point at the exact tensor** the lesson is about, and say what its axes mean. *"A (4,3)
  table: 4 French positions × 3 English words."*
- **Say what happens to it afterwards.** Is it stored, returned, or discarded? Attention
  scores are an intermediate — knowing they vanish is as important as knowing they exist.
- **Say where it sits in the whole model**, and which variant we're actually building. The
  translation example teaches cross-attention; the model we build is decoder-only.
- **Connect to what he already knows.** He has solid DL. Anchor new machinery to embeddings,
  logits, and the residual stream rather than starting from vacuum.

If a lesson can't say where its mechanism lives in a forward pass, the lesson isn't ready.

### The rest

- **Name the metaphor and then discard it.** "Query/key/value are borrowed from databases" is
  a naming story, not a mechanism. Say where the name comes from, then what the thing actually
  *is* — usually a matrix, a projection, or a loss term.
- **Say where it comes from mechanically.** `Q = x @ W_Q`. A term is much less intimidating
  once it's a line of code.
- **Answer "why does this exist separately?"** Why are Q and K different matrices? Why is V
  not K? Those are the interview questions, and they're only askable once the term is defined.
- **Give it its own part if it needs one** — in addition to the inline definition, never
  instead of it.

Terms coming up that will need this treatment: latent, ELBO, posterior, score, SNR, guidance
scale, denoiser, timestep embedding, classifier-free guidance, reward model, KL penalty.
Assume none of them are known.

### Prose style

**Simplify the language, never the content.** Derivations stay in full. What goes is hedging,
subordinate clauses, and paragraphs that restate the previous paragraph.

- Short sentences. Tables and diagrams over prose.
- One idea per part, with a `→ next` link at the bottom.
- Every lesson `README.md` carries a **one-paragraph version** of the whole lesson, so it can
  be re-read in 30 seconds six months later.
- Mark the part that actually matters (`— the one to slow down for`) so attention goes where
  it pays.

**Draw the diagrams.** A mechanism with shapes, a flow, or a failure mode gets a figure, and
figures are *generated* by a committed `make_figures.py` in the section — matplotlib, no
external images, no network dependency. Two rules: the diagram must show something the prose
can't say as well (a symmetric score matrix, an attention heatmap, entropy collapsing), and
**every generated figure must actually be looked at before shipping** — matplotlib overlaps
labels constantly. Never embed a figure sight-unseen.

He has said plainly that he won't read a wall of text. Unread material teaches nothing, so
this is a hard constraint rather than a style preference.

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

**The 30-minute rule.** If a lesson's training run exceeds ~30 minutes on one GPU, shrink it:
fewer steps, smaller model, smaller images, a subset. Exceptions are the handful of places
where the long run *is* the lesson (Phase 2's scaling study, Phase 11's throughput work,
Phase 15's video) — and those are stated in advance, with the run started in the background so
the session continues.

**Every lesson ends with something to look at.** Samples, a loss curve, a metric, a
side-by-side against a broken version. Never end a lesson on "the tests pass" — that is
invisible progress, and invisible progress is what makes people stop.

**Cheap ablations only.** An ablation that costs an afternoon is a bad ablation. If it can't
run in a few minutes at the lesson's scale, cut the scale or cut the ablation.

**Long runs go in the background.** Start them, then continue the session with the next
lesson's theory. Never sit and watch a progress bar.

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
  └─ more lessons? ←── 7. Run and observe ──→ 8. Quiz ──→ 9. Consolidate
```

---

### 1 · Theory

The derivation, worked through — not cited, not linked. Notation defined before use.
Intermediate algebra shown. If a step is skipped because it's tedious, say that it's being
skipped rather than leaving a gap that reads as obvious.

Open with **why this exists**: what specifically broke in the previous lesson. Where that
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

Skip this stage when a lesson introduces no new ops.

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

Scope one lesson to roughly 30–100 lines of real implementation. Bigger than that, split it.

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
consider whether the lesson was scoped too large.

---

### 7 · Run and observe

**Code that passes a shape assertion isn't understanding. Watching it train is.**

Claude provides the boilerplate: data loading, training loop, logging, evaluation, plotting.
Before the task if it's needed to run at all, after if it would give the answer away.
Boilerplate is plumbing — dataloaders, checkpointing, `argparse` — never the mechanism.

Then actually run it, and look:

- What does the loss curve do, and does it match what the theory predicts?
- What do samples/outputs look like at 10%, 50%, 100% of training?
- What happens when it's broken deliberately? Some lessons should include a *deliberate
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

**Gap audits** (`ROADMAP.md` §10) are the other half: at section boundaries, questions spanning
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
  lesson1/              one sitting, split into short readable parts
    README.md           part index + one-paragraph summary + interview questions
    1-where-it-came-from.md
    2-why-not-rnns.md
    ...
  lesson2/
  attention.py          his implementation
  check_lesson1.py      Claude's boilerplate — checks and ablation, one per lesson
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
- **One lesson at a time.** Never generate material for multiple lessons or sections ahead.
- **Keep parts short and the language plain.** He won't read a wall of text; unread material
  teaches nothing. Simplify the prose, not the substance.
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
