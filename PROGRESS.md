# Progress

Current state of the curriculum. Updated at the end of every section (`TEACHING.md` stage 9).

**Next action:** Section 01, lesson 1 — read `01-attention/lesson1/` (7 short parts, ~25 min),
then **implement `01-attention/attention.py`** (`causal_mask`,
`scaled_dot_product_attention`, `attention_entropy`) and run
`python 01-attention/check_lesson1.py`.

---

## Machine

Configs are designed for the machine actually in use — the heavy phases (11, 14, 15, 16) are
planned for this box. Log the machine whenever it changes; `ROADMAP.md` §4 lists which phases
need adjusting on a weaker one.

| From | Machine | GPUs | Storage | Precision | Notes |
|---|---|---|---|---|---|
| 2026-08-03 | Linux, torch 2.13.0 / CUDA 13.0 | 4 × TITAN RTX 24 GB, Turing SM 7.5, **shared** (GPU 0 usually busy) | ~1.5 TB `/data` | fp16 + GradScaler | No *native* bf16 — `is_bf16_supported()` returns True via emulation; pass `including_emulation=False`. No FlashAttention-2, no fp8. Triton works. |

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

- Question bank: `review/questions.md` — **0 questions**
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
| 2026-08-03 | §4 bf16 detection corrected — `is_bf16_supported()` reports True on Turing via emulation; must pass `including_emulation=False` | Found by running the check on the actual machine |
