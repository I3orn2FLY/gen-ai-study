# Progress

Current state of the curriculum. Updated at the end of every section (`TEACHING.md` stage 9).

**Next action:** Lesson 1 is now **ten** parts; **1–4 are written**, 5–10 are outline stubs.
Kenessary has read parts 1–2 in an older form and needs to re-read 1–4. Then part 5.

Fourteen errors found and fixed on 2026-08-10/11 — see `review/errata.md`. The pattern that
matters: errors 7–8 were introduced by the fix for 5–6, and 9b by the fix for 7. **A revision is
as likely to introduce an error as a first draft**, so the cold-read audit runs after every pass,
not once at the end.

Process added the same day: `lessonN/ledger.md` (claims + term order), a seeded
`review/questions.md`, and a cold-read audit pass. Rules in `CLAUDE.md` and `TEACHING.md`
§ Writing the material.

**Web access works from this session** even though the box itself has no network — paper
abstracts and arXiv HTML are fetchable, and errors 5 and 6 were settled that way. Use it to
verify claims rather than relying on recall; the symbols-over-numbers rule still stands, but
"no network" is not a reason to ship an unverified fact.

---

## Machine

Configs are designed for the machine actually in use — the heavy phases (11, 14, 15, 16) are
planned for this box. Log the machine whenever it changes; `ROADMAP.md` §4 lists which phases
need adjusting on a weaker one.

| From | Machine | GPUs | Storage | Precision | Notes |
|---|---|---|---|---|---|
| 2026-08-03 | Linux, torch 2.13.0 / CUDA 13.0 | 4 × TITAN RTX 24 GB, Turing SM 7.5, **shared** (GPU 0 usually busy) | ~1.5 TB `/data` | fp16 + GradScaler | No *native* bf16 — `is_bf16_supported()` returns True via emulation; pass `including_emulation=False`. No FlashAttention-2, no fp8. Triton works. |

**Machine change expected (noted 2026-08-03).** Kenessary is moving to another box and will
return here for the GPU-heavy phases. **Re-run the detection snippet in `ROADMAP.md` §4 before
recommending any precision or kernel** — in particular `is_bf16_supported(including_emulation=False)`,
since the default flag lies on Turing. Add a row above when it changes.

Lesson 1 is CPU-only, so nothing there is blocked by the move.

**Would gate on a weaker machine:** 11 (needs multiple GPUs) and 15 (native video) — both
Tier 3. Postpone rather than fake, and only if the machine actually changes.

---

## Build track

Tiers are stopping points, not skip lists — work in dependency order. See `ROADMAP.md` §6.

### Tier 1 — Core (interview-capable at the end of this)

| # | Section | Status | Lessons | Checkpoints |
|---|---|---|---|---|
| 01 | Attention and the transformer | **in progress** | 0 / 8 | — |
| 02 | The modern decoder LM | not started | 0 / 13 | — |
| 04 | ViT and CLIP | not started | 0 / 7 | — |
| 05 | Learned compression (VAE → VQ-VAE) | not started | 0 / 7 | — |
| 06 | Autoregressive text-to-image | not started | 0 / 5 | — |
| 07 | Diffusion, derived | not started | 0 / 11 | — |
| 08 | Latent diffusion | not started | 0 / 4 | — |
| 09 | Text-conditioned latent diffusion | not started | 0 / 7 | — |
| 12 | Adapting pretrained models | not started | 0 / 6 | — |

### Tier 2 — Competitive

| # | Section | Status | Lessons |
|---|---|---|---|
| 03 | Inference: making it run | not started | 0 / 7 |
| 10 | Diffusion transformers | not started | 0 / 6 |
| 17 | Alignment and preference optimization | not started | 0 / 6 |

### Tier 3 — Differentiating

| # | Section | Status | Lessons |
|---|---|---|---|
| 11 | Scale engineering | not started | 0 / 9 |
| 13 | Fast sampling and distillation | not started | 0 / 5 |
| 14 | Motion adapters | not started | 0 / 4 |
| 15 | Native video models | not started | 0 / 6 |
| 16 | Multimodal models | not started | 0 / 5 |

---

## Gap audits

Claude initiates these at block boundaries (`ROADMAP.md` §10). A blank is information.

| After phase | Status | Result |
|---|---|---|
| 3 | pending | — |
| 6 | pending | — |
| 10 | pending | — |
| 13 | pending | — |
| 15 | pending | — |
| 17 — full field | pending | — |

---

## Literacy track

Read and written up, not built. Each attaches to a Build-track phase — do it then, not at the
end. See `ROADMAP.md` §8.

| Topic | Attaches after | Status |
|---|---|---|
| 8.1 Prompt engineering | Phase 3 | not started |
| 8.2 RAG systems | Phase 4 | not started |
| 8.3 Agents and tool use | Phase 3 | not started |
| 8.4 Production serving | Phase 3 | not started |
| 8.5 System design for GenAI | Phase 9 | not started |
| 8.6 Safety and provenance | Phase 9 | not started |
| 8.7 Evaluation landscape | Phase 9 | not started |
| 8.8 Audio and speech generation | Phase 15 | not started |
| 8.9 3D and world models | Phase 15 | not started |
| 8.10 Commercial landscape | ongoing | not started |

---

## Checkpoints

Trained weights that later sections depend on. Full detail in `checkpoints/MANIFEST.md`.

*None yet.*

---

## Review

Spaced review runs at the start of roughly every third session (`TEACHING.md` § review layer).

- Question bank: `review/questions.md` — **20 questions** (01 lesson 1, parts 1–3), 3 marked ⚠
  as misconceptions he actually held. **None asked yet — a quiz is due before part 4.**
- Errata: `review/errata.md` — **6 corrections** (section 01 lesson 1, parts 1 and 3; none quizzed)
- Quiz history: `review/log.md` — **no quizzes run**
- Outstanding **shaky** or **confidently wrong** items: none

---

## Roadmap amendments

Changes made to `ROADMAP.md` after it was first written, and what prompted them. A roadmap
that never changes is one nobody is checking.

| Date | Change | Prompted by |
|---|---|---|
| 2026-07-29 | Initial version | — |
| 2026-08-03 | §4 rewritten: design for the current machine, adjust if it changes — replaces the "design against an 8 GB floor" framing that was shrinking every phase in advance | Kenessary: the heavy phases belong on this box |
| 2026-08-03 | Vocabulary: **section → lesson → part**. Roman-numeral groupings renamed Block I–VI to stop the collision. Lessons are directories of 3–6 min parts | Kenessary: docs too long/dense to actually read |
| 2026-08-03 | Rule added (`TEACHING.md` § Writing the material): define every term at first use, including ones that feel too basic. Lesson 1 gained a Query/key/value part | Kenessary: q/k/v were used without ever being explained |
| 2026-08-03 | Rule strengthened: **no forward references** — define inline where the term appears, never "explained later"; formulas explained symbol by symbol; history parts teach rather than just narrate | Kenessary: a deferral note is not an explanation |
| 2026-08-03 | Rules added: **never teach a mechanism in isolation** (trace a forward pass with real shapes) and **draw the diagrams** (generated by a committed `make_figures.py`, every figure eyeballed before shipping) | Kenessary: scores were explained but not placed; wanted illustrations |
| 2026-08-03 | Lesson 1 regenerated: forward pass moved to part 3 and anchored from part 1 onward; part 1 retitled *The scoring function*. Rules added — **mechanism is the subject, history is structure**, and **lesson feedback means rewriting every part** | Kenessary: scores still had no home; it's a DL course, not a history course |
| 2026-08-03 | Rule added: **open with the problem, not the answer** — setup → what breaks → what would fix it → mechanism → what it cost. Lesson 1 parts 1, 2, 5 re-opened accordingly | Kenessary: start with the problem; keep the story-like logic |
| 2026-08-03 | Rule added: **math is LaTeX in `.md`, every symbol defined with dimensions, no magic numbers in formulas**. Lesson 1 part 1 rewritten with math | Kenessary: formulas looked awful; `s_{i-1}` was never defined and 512 was used where $d$ belongs |
| 2026-08-03 | Rule refined: notation is **just-in-time** — two or three symbols beside the formula that uses them, never a glossary up front. Matters most on unfamiliar material | Kenessary: the notation table was overload before terms were introduced |
| 2026-08-03 | Rule added: **write one part at a time**; unwritten parts get outline stubs. A lesson drafted in one pass inherits the same mistake in every part | Kenessary: "if you generate whole lesson you suck" |
| 2026-08-10 | Lesson 1 split into **nine** parts — the scoring-function *choice* (additive vs dot product) became part 2 in its own right, since part 1 had grown past the length rule and the shape argument needs room | Kenessary asked for the additive formula's shapes; the honest answer didn't fit |
| 2026-08-10 | Rule added: **verify claims, don't ship recall** — symbols over numbers, shapes written before any cost claim, corrections logged in the new `review/errata.md` rather than patched silently | Four factual errors found in part 1 after it was marked written |
| 2026-08-11 | Rule sharpened: **a part may not use a mechanism it hasn't earned yet, even in a comparison table.** Part 3 compared "Recurrence $O(n)$" against "Attention $O(1)$", which is only true once the recurrence is deleted — the very thing the part was arguing for. Arguments must be derivable from the model in hand | Kenessary: "at that point it is still sequential for both encoder and decoder" |
| 2026-08-11 | Rule added: **credit a mechanism before criticising it.** Part 3 attacked recurrence for two sections before saying what recurrence was for; the three jobs it did now come first, and the closing section is explicitly the bill | Kenessary: "it was explained that rnns are bad and then it was said that is why they exist" |
| 2026-08-11 | Lesson 1 split again, nine parts → **ten**: part 3 had grown to 343 lines once the audit fixes landed, so *what deleting recurrence cost* became part 4 in its own right | The length ceiling; fixed structurally rather than by trimming, same as the earlier 8→9 split |
| 2026-08-11 | **Web access confirmed working** from the Claude Code session despite the box having no network. Claims are now fetched and checked, not recalled | Two part-3 claims settled by fetching the papers |
| 2026-08-03 | §4 bf16 detection corrected — `is_bf16_supported()` reports True on Turing via emulation; must pass `including_emulation=False` | Found by running the check on the actual machine |
