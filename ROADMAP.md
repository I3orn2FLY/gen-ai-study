# Generative AI — From-Scratch Roadmap

A dependency-ordered curriculum for generative modeling, built as a single growing
repository. Ends at native video diffusion and unified multimodal generation.

---

## 1. Purpose

**This exists to make Kenessary hireable as a generative-AI ML engineer, by building things
rather than by memorizing papers.**

The bet: someone who has trained a VAE, watched it decode blurry, and fixed it with a
perceptual loss and a discriminator answers *"why does Stable Diffusion's autoencoder use a
discriminator?"* differently than someone who read the LDM paper. Hands-on depth is the
method. Interview performance is the outcome being optimized.

Starting point: solid deep learning, some GAN familiarity, **no generative-AI background**.

### The coverage problem

Kenessary cannot currently audit whether this roadmap covers the right topics — that is
precisely the knowledge being acquired. A curriculum invented from scratch would inherit
every blind spot of whoever wrote it, and the failure mode is silent: a question that is
obvious to a practitioner, met with a blank.

Three structural defenses, and they are the most important part of this document:

1. **Two tracks, not one** (§2). Building everything is impossible; being unable to *discuss*
   half the field is disqualifying. The Build track is what gets implemented. The Literacy
   track is what must be explainable without being built.
2. **Grounded in published curricula** (§9), not invention. Coverage is cross-mapped against
   Stanford CS236, Stanford CS336, MIT 6.S184, and others, so gaps are checkable against an
   external standard rather than trusted.
3. **Gap audits** (§10). At each part boundary, Claude generates interview questions spanning
   the *entire field at that level* — deliberately including topics not yet covered — grades
   the answers, and produces a written gap list. This is the mechanism for finding unknown
   unknowns. It is not optional.

---

## 2. Two tracks

| | **Build track** (§6) | **Literacy track** (§8) |
|---|---|---|
| What | Implemented from scratch, trained, ablated | Read, summarized, explainable — not built |
| Why | Depth that survives follow-up questions | Breadth that prevents blank stares |
| Size | 17 phases | ~10 topic areas |
| Output | Working code, trained checkpoints, ablation tables | Written notes + a worked example or two |
| Failure if skipped | You can only recite | You can only recite about *some things* and know *nothing* about others |

The Literacy track is not filler. RAG, agents, serving, and system design dominate real
generative-AI interview loops. They build less durable understanding per hour than
implementing a sampler, which is why they aren't the Build track — but "I've never looked at
that" is a worse answer than a competent overview.

Literacy topics are interleaved, not saved for the end. §8 states where each one attaches.

---

## 3. How techniques actually arrive

The most common pattern is pressure-and-response — X broke, so Y appeared. It's the best
pattern for retention and it covers most of §6. It is **not** universal. Forcing every
advance into that mold produces confidently wrong history, which is worse in an interview
than saying nothing.

Techniques carry an origin tag:

| Tag | Meaning | Example |
|---|---|---|
| **Fix** | Direct response to a named failure of what came before. | DDIM ← 1000 sequential steps is unusable. |
| **Transfer** | Imported from another domain; wasn't fixing a local problem. | ViT ← NLP transformers. |
| **Unification** | Reframes several existing things as one. Often *retrospective*. | The score/SDE view of diffusion. |
| **Scale unlock** | Not a fix — it simply didn't work until compute and data changed. | Most of what makes large models work. |
| **Empirical** | Found by sweeps; the theoretical story was written afterward. | SwiGLU, RMSNorm. |

Two of these are interview traps.

**Empirical.** *"Why SwiGLU?"* is honestly answered with **an ablation found it worked and
the justification came later**. Saying that is stronger than inventing a mechanism.

**Unification.** Score-based models and DDPM were developed largely *in parallel* and shown
to be the same object afterward. "DDPM was broken, so score matching fixed it" is a claim a
knowledgeable interviewer will correct.

### Structural principles

1. **A phase earns its place by what it enables** — usually by fixing a named failure of the
   previous phase, sometimes by transferring an idea in, sometimes by unifying several.
2. **No dead branches in the Build track.** Capsule networks enabled nothing downstream, so
   they never enter. See §11.
3. **Nothing built is throwaway.** Every phase produces a component surviving into the final
   video model. The adversarial machinery in Phase 5 is the same machinery in Phase 13.
4. **Obsolete architecture ≠ obsolete idea.** Normalizing flows lost, then mutated into flow
   matching, which is current SOTA.

---

## 4. Hardware

**The machine can change.** This curriculum is designed against a guaranteed floor, not
against whatever box is in front of it. Anything that only works on the current hardware is a
design flaw.

### The floor — design against this

| | Guaranteed minimum |
|---|---|
| GPU | **1 × 8 GB VRAM** |
| Storage | **200 GB** |

Every phase must have a configuration that runs here. Where a phase genuinely cannot, it is
marked **gated** below and gets postponed rather than faked.

The floor is a feature, not just a constraint: it enforces `TEACHING.md`'s "smallest thing
that demonstrates the phenomenon" rule, which is what keeps parts to a few hours. A DDPM on
32×32 CIFAR teaches every mechanism a 256px run does, and it fits in 8 GB.

### Current machine — a bonus, never an assumption

4 × NVIDIA TITAN RTX 24 GB (**shared** — assume 1–2 free), ~1.5 TB on `/data`,
**Turing SM 7.5**, torch 2.10 + CUDA 12.8.

Turing specifics that are **true here and may be false elsewhere**:

- No bf16, no FlashAttention-2, no fp8 → fp16 + GradScaler, with loss-scale collapse and
  attention overflow as expected failure modes. Taught in Phase 2, not worked around.
- `F.scaled_dot_product_attention` has a memory-efficient backend on SM 7.5, but not the
  flash backend.
- Triton works, so Phase 11's kernel work is viable.

### Detect, don't assume

**At the start of any session that will train something, check what the machine actually is**
before recommending a precision or a kernel:

```python
import torch
p = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)          # (7,5) Turing · (8,x) Ampere · (9,0) Hopper
print(p.name, f"{p.total_memory/1e9:.1f} GB", cap)
print("bf16:", torch.cuda.is_bf16_supported())     # False on Turing, True on Ampere+
print("GPUs:", torch.cuda.device_count())
```

Then adapt: **bf16 where supported** (simpler and more stable — no GradScaler, no loss-scale
collapse), fp16 + GradScaler where not. On Ampere+, the fp16 pathologies in Phase 2.10 and
Phase 11.5 become *historical* material rather than lived — still worth understanding, since
plenty of production systems run fp16, but read rather than debugged.

Record the machine in `PROGRESS.md` when it changes, so results stay comparable across boxes.

### Phase requirements at the floor

| Phase | 8 GB single GPU | What changes at the floor |
|---|---|---|
| 1 Attention | ✅ trivial | CPU is enough for most of it |
| 2 Decoder LM | ✅ | ~50M params instead of 150M; grad accumulation; scaling study uses smaller sizes |
| 3 Inference | ✅ | Fine — inference is the light one |
| 4 ViT & CLIP | ✅ | Small batches hurt contrastive learning — use gradient caching (already step 4.3) |
| 5 VAE / VQ-VAE | ✅ | 128px instead of 256px |
| 6 AR text-to-image | ✅ | Small token grid |
| 7 Diffusion | ✅ | CIFAR 32px — already the plan |
| 8 Latent diffusion | ✅ | **Latent caching becomes mandatory, not an optimization** |
| 9 Text-conditioned LDM | ✅ | Cached latents + cached text embeddings; small batch + accumulation |
| 10 Diffusion transformers | ✅ | Small DiT; the scaling comparison shrinks but still shows the trend |
| **11 Scale engineering** | ⚠️ **gated** | Multi-GPU parallelism cannot be taught on one GPU. Single-GPU parts (profiling, memory math, Triton kernel, data pipelines) still work; **DDP/FSDP postpone until multiple GPUs exist** |
| 12 Adapting pretrained | ⚠️ partial | SDXL LoRA does not fit comfortably in 8 GB. **Substitute SD1.5** + gradient checkpointing + 8-bit optimizer. Same mechanisms, smaller model |
| 13 Distillation | ✅ | Distill your own small model rather than SDXL |
| 14 Motion adapters | ⚠️ tight | SD1.5-based AnimateDiff at low resolution and few frames |
| **15 Native video** | ⚠️ **gated** | Capped even at 24 GB. At 8 GB, toy scale only — architecture and memory reasoning are the objective, samples are not |
| 16 Multimodal (VLM) | ✅ | Frozen vision encoder + small LM; LoRA rather than full fine-tune |
| 17 Alignment / DPO | ✅ | DPO holds policy **and** reference model — use LoRA-DPO so the reference is the base weights with adapters disabled |

**Two phases are genuinely gated: 11 and 15.** Both are Tier 3. If the machine drops to the
floor, work around them and return when hardware allows — that is exactly what the tiers in
§6 are for.

### Storage at 200 GB

Disk discipline stops being optional:

- **Pre-resize datasets on download.** Never keep full-resolution originals — a CC3M slice at
  256px is a fraction of the raw download.
- **Cache latents, then delete the images.** VAE latents are ~48× smaller than the RGB they
  encode. This is Phase 8.2, and at the floor it is a requirement rather than a speedup.
- **Budget the pretrained zoo.** SD1.5 ≈ 4 GB, SDXL ≈ 7 GB, T5-XXL ≈ 9 GB. Keep only what a
  current section needs.
- **Prune checkpoints.** Keep the best and the last per run, not every epoch. Record what was
  deleted in `checkpoints/MANIFEST.md`.
- Working target: **under 50 GB of datasets at any one time.**

### Constant across machines

- **Single-GPU-first everywhere.** Multi-GPU is Phase 11's *subject*, never an ambient
  assumption — that is what makes the curriculum portable.
- **Checkpoint and resume is mandatory** for anything long. Shared GPUs get preempted, and a
  machine you might migrate off of makes this doubly true.
- Where a phase is capped by hardware, it says so rather than pretending otherwise.

---


## 5. Repository layout

**Sections are the primary unit** and own their own docs, code, and experiments. Each section
maps to one phase.

```
CLAUDE.md               agent instructions (auto-loaded)
ROADMAP.md              this file — what to learn, in what order, and why
TEACHING.md             how it gets taught: the session loop, pacing, review
PROGRESS.md             current state, checkpoints, next action
requirements.txt        grows section by section

01-attention/
  README.md             overview; interview writeup added when the section closes
  part1-scaled-dot-product.md
  part2-multi-head.md
  attention.py          his implementation
  train.py              Claude's boilerplate
  experiments/          runs: config, logs, results, failures

02-decoder-lm/
...

shared/                 thin — only what a second section actually imports
checkpoints/            trained weights + MANIFEST.md
review/                 accumulating question bank + quiz history
literacy/               §8 writeups — read, not built
audits/                 §10 gap-audit results
```

Directories are created when a section starts, not upfront.

**`shared/` stays thin.** Code moves there only when a *second* section actually imports it —
never in anticipation. Duplication across sections is fine and often clearer; a wrong
abstraction built early is not. Trained checkpoints are the real cross-section dependency
(Phase 9 needs Phase 4's CLIP), which is what `checkpoints/MANIFEST.md` exists for.

**No standalone infrastructure section.** Training loops start as plain scripts and accrete
only when a real need appears.

### The from-scratch line

| Written by hand | Taken from libraries |
|---|---|
| Every model, layer, attention block | Dataloading, `webdataset`, tar sharding |
| Every loss, noise schedule, sampler | Logging, checkpoint plumbing |
| Tokenizer, LoRA, ControlNet injection | Distributed launchers (`torchrun`) |
| Metrics whose math teaches (FID, bits-per-byte) | Metric backbones (LPIPS weights, Inception) |

Raw PyTorch. Pretrained weights enter only after the equivalent has been built at small
scale — first at Phase 8's reality gate.

---

## 6. The Build track

Seventeen phases in six parts. Ordered by dependency, not by time.

Most phases open with the pressure that created them and the mechanism answering it. Where
that framing would be dishonest — transferred in, discovered independently, unified after the
fact — the phase says so using §3's tags.

### Tiers — the finish line is movable

Seventeen phases is a long commitment, and a curriculum where value only arrives at the end is
one that gets abandoned. So the phases are tiered by how much they move the interview needle.
**Interview-capability arrives at the end of Tier 1, not the end of the roadmap.**

| Tier | Phases | What it buys |
|---|---|---|
| **1 — Core** | **1, 2, 4, 5, 6, 7, 8, 9, 12** | The actual interview surface: transformers, LMs, CLIP, VAEs, diffusion, Stable Diffusion internals, LoRA/ControlNet. Finish this and you can hold a real conversation with a practitioner about how modern image generation works. |
| **2 — Competitive** | 3, 10, 17 | Inference/serving, DiT/MMDiT, and alignment. Heavily asked, and the difference between "understands the basics" and "current". |
| **3 — Differentiating** | 11, 13, 14, 15, 16 | Distributed training and kernels, distillation, video, multimodal. Few candidates have hands-on experience here — this is what makes you memorable rather than adequate. |

Tiers are about **stopping points, not skipping**. Work in dependency order; Tier 1 already
runs 1→2→4→…→12. If motivation or time runs out, stopping at a tier boundary leaves something
coherent instead of something half-finished.

Two notes on the tiering. **Phase 6** (autoregressive T2I) is Tier 1 despite being skippable
on paper, because it's what makes Phase 7's motivation *felt* rather than asserted — but it
can be run compressed if time is short. **Phase 3** (inference) is Tier 2 only because it
isn't a prerequisite for anything visual; by interview frequency alone it would be Tier 1, so
don't defer it indefinitely.

Pace: a **part** is 1–3 hours, a **section** is 4–8 parts. See `TEACHING.md` for the scope
discipline that keeps it there — the 30-minute training-run rule especially.

---

## Part I — Language foundations

*Front-loaded deliberately. The transformer is the shared substrate: the DiT, the CLIP text
encoder, the video backbone, and the VLM are all this same block under different conditioning.
Building it once here means Phase 10 is a reparameterization rather than a new architecture.
Also the highest-density interview material in the field.*

---

### Phase 1 — Attention and the transformer

> **Pressure.** Recurrence forces an O(n) path between distant positions and serializes
> computation along the sequence — it cannot exploit GPU parallelism and attenuates
> long-range gradient signal.
>
> **Response.** Attention: an O(1) path between any two positions, fully parallel over the
> sequence.
>
> **Cost incurred.** O(n²) compute and memory, and position must now be injected explicitly —
> attention alone is permutation-invariant.

RNNs are argued against analytically and never built, same rule as capsule networks.

**Steps**

1. Scaled dot-product attention from scratch. Why √d — dot-product variance grows with d, and
   without scaling softmax saturates and gradients vanish.
2. Why not recurrence: path length and parallelism, derived on paper.
3. Multi-head attention. What heads buy, head-dim tradeoffs, reshape mechanics.
4. Causal masking and the autoregressive factorization.
5. The full block: residual stream, MLP, LayerNorm. Why residuals make depth trainable.
6. Position: none → sinusoidal → learned → **RoPE**. Why absolute encodings fail to
   extrapolate past training length.
7. Encoder-decoder → encoder-only → decoder-only, and why decoder-only won for generation.
8. Attention complexity and the memory-bandwidth view — why FlashAttention is a *memory* IO
   optimization, not an approximation. *(Conceptual here; FA2 is unavailable on Turing.)*

**Deliverable.** `01-attention/transformer.py` — a GPT-2-style model trained on TinyShakespeare.

**Ablation.** Strip positional encoding and demonstrate provable permutation-invariance.
Strip residuals and show depth stops training.

---

### Phase 2 — The modern decoder LM

> **Pressure.** The 2017 block diverges when deep, spends parameters inefficiently, and its
> KV cache dominates inference memory.
>
> **Response.** A decade of architectural and optimization changes — each adopted only after
> its own ablation, and each tagged with how it actually arrived.

**Steps**

1. **Byte-level BPE from scratch.** Merge training, pretokenizer regex, special tokens,
   vocab-size tradeoffs. Why tokenizers cause arithmetic and multilingual failures.
2. Pre-norm vs. post-norm. *(**Fix**, with a real mechanism: the residual gradient path.)*
3. RMSNorm. *(**Empirical/efficiency** — it was not solving a failure. Don't claim it was.)*
4. SwiGLU and the 2/3-width convention. *(**Empirical** — the GLU-variants paper found it by
   ablation and says so. The canonical honest-description example.)*
5. RoPE in full; NTK-aware and YaRN context extension.
6. MHA → MQA → **GQA**. Do the KV-cache memory arithmetic first, architecture second.
7. **Mixture of Experts.** Routing, top-k gating, load-balancing loss, expert capacity, why
   MoE trades memory for compute. *(**Scale unlock.**)* Standard interview material — most
   frontier models are sparse now.
8. Optimizer and schedule: AdamW with decoupled weight decay, warmup, WSD vs. cosine,
   gradient clipping, depth-scaled init. Muon and modern alternatives, at least conceptually.
9. Stability: z-loss, QK-norm, logit soft-capping.
10. **Mixed precision on Turing.** fp16 + GradScaler, loss-scale collapse, which layers stay
    fp32, detecting overflow before it silently ruins a run.
11. **Checkpoint and resume.** First long run on shared GPUs; deterministic dataloader state.
12. **Scaling laws.** Train 4 sizes, fit L(N, D), derive your own compute-optimal token ratio.
    Kaplan vs. Chinchilla and why the answer changed. Cheap on text, and it is what sizes
    every model in every later phase.
13. **Evaluation.** Bits-per-byte, perplexity's pitfalls, benchmark contamination, why
    LM evaluation is genuinely unsolved.

**Data.** TinyStories → a FineWeb-Edu slice.

**Deliverable.** `02-decoder-lm/` and a trained ~50–150M dense model, plus a small MoE variant.

**Ablation.** Steps 2, 3, 4, 9 individually ablated at matched compute. **Report which ones
did not matter at your scale** — several won't, and knowing which is the point.

---

### Phase 3 — Inference: making it run

> **Pressure.** Phase 2 produces weights. Weights are not a system. Naive generation wastes
> almost all available compute, and inference is where most production cost lives.
>
> This is a separate skill from training and is disproportionately represented in interviews.

**Steps**

1. **KV cache** — the memory arithmetic, and why it grows linearly with context and batch.
2. Sampling: temperature, top-k, top-p, min-p, repetition penalties, and their failure modes.
3. **Prefill vs. decode** — compute-bound vs. memory-bandwidth-bound, and why they are
   scheduled differently. The single most useful mental model in LLM serving.
4. **Continuous batching** and **PagedAttention** — what problem each solves.
5. **Speculative decoding**: draft models, acceptance rates, when it fails to help.
6. **Quantization**: int8/int4, GPTQ vs. AWQ, weight-only vs. activation quantization,
   what degrades and what doesn't. *(Turing supports int8 well; int4 kernels are limited.)*
7. Structured output and constrained decoding.

**Deliverable.** `03-inference/` — your own batched inference server for the Phase 2 model,
with measured tokens/sec against a naive baseline.

**Literacy attachment.** §8.4 (production serving) reads best directly after this.

---

## Part II — Bridging to vision

---

### Phase 4 — ViT and CLIP

> Phase 2 understands text and nothing else. A useful image-generation interface takes free-
> form language, which requires a text representation aligned to images.
>
> **Needs no diffusion and no VAE** — only Phase 1's attention plus an image encoder.

**Steps**

1. **ViT:** patchify, CLS token vs. mean pooling, why ViT is more data-hungry than a CNN.
   *(**Transfer + Scale unlock**, not a Fix — CNNs were not failing at classification. The
   transformer was imported because it scales better with data and compute, and ViT actually
   underperforms ResNets below roughly ImageNet-21k scale. Both halves are worth saying.)*
2. **Contrastive learning:** InfoNCE, the symmetric image↔text loss, learned temperature.
3. Why contrastive learning is batch-size dependent; gradient caching for more negatives.
4. Train a small CLIP on a captioned subset (COCO / Flickr30k / a CC3M slice).
5. Eval: zero-shot classification, retrieval R@K — prove the space is actually aligned.
6. SigLIP and why the sigmoid loss removes the global-batch coupling. *(**Fix.**)*
7. **Embeddings and retrieval — the modeling half of RAG.** Text embedding models, hard-
   negative mining, ANN indexing, why cosine similarity is the wrong metric surprisingly
   often. The *systems* half of RAG is §8.2.

**Deliverable.** `04-vit-clip/` — a trained text encoder (used in Phase 9) and ViT (Phase 10).

**Ablation.** Learned vs. fixed temperature; batch-size sweep against retrieval R@K.

---

### Phase 5 — Learned compression: VAE → VQ-VAE

> **Pressure.** Images are not natural token sequences. At 256×256×3 a pixel sequence is
> ~196k long — intractable at O(n²) — and pixel-order factorization spends capacity on
> perceptually invisible detail. You need a short, perceptually dense representation.
>
> **Response.** Learn the compression.

**Steps**

1. Plain autoencoder → why its latent space has holes and cannot be sampled from.
2. **VAE:** full ELBO derivation, reparameterization trick, the KL term, posterior collapse,
   β-VAE.
3. Why an MSE-trained decoder is blurry — it reconstructs the posterior *mean*.
4. **LPIPS** perceptual loss.
5. **Patch discriminator**, hinge loss, LDM's adaptive λ. *Your GAN knowledge enters here as
   a working component, not as history.*
6. **VQ-VAE:** codebook, straight-through estimator, commitment loss, codebook collapse, EMA
   updates, and **FSQ** as the modern fix. *(FSQ is a **Fix** for a real failure.)*
7. Continuous KL-VAE vs. discrete VQ, and why **both** survive: KL feeds diffusion (Phase 8),
   VQ feeds token models (Phases 6 and 16).

**Eval.** rFID, PSNR, SSIM, LPIPS.

**Deliverable.** `05-compression/` — an f8 KL autoencoder and a VQ tokenizer.

---

### Phase 6 — Autoregressive text-to-image (tiny DALL·E 1)

> You now have a decoder LM (2), a text encoder (4), and an image tokenizer (5). Compose
> them and you have text-to-image with **no new machinery at all**.
>
> **What this phase exists to produce.** It works, and it hurts. Decoding is strictly
> sequential; VQ artifacts are baked into every sample; coherence degrades as the token grid
> grows. **These specific, felt failures are the motivation for Phase 7.**

Built rather than described. Arguing on paper that autoregression over images is limited is
far weaker than generating from one and watching where it breaks. The lineage is also alive —
it returns as Chameleon/Janus in Phase 16.

**Steps**

1. Sequence design: joint vocabulary vs. separate embeddings, positional handling across the
   modality boundary.
2. Train Phase 2's decoder on the joint sequence.
3. Sampling, and reranking candidates with Phase 4's CLIP — what DALL·E 1 actually did, and a
   clean demonstration of why guidance is needed at all.
4. **Masked generative alternatives** — MaskGIT/MUSE: parallel decoding over tokens.
   *(**Fix** for the sequential-decoding problem, from within the token paradigm. Worth
   knowing it exists and feeds some video models; build only if curious.)*
5. **Measure the pain.** Decode latency vs. resolution, FID, artifact taxonomy. This writeup
   *is* the motivation document for Phase 7.

**Deliverable.** A working text-to-image model. First real images from text, this early.

---

## Part III — Diffusion

---

### Phase 7 — Diffusion, derived

> **Pressure.** Phase 6 is inherently sequential and inherits its tokenizer's artifacts. GANs
> are the obvious alternative and are unstable and drop modes. You want stable training,
> parallel-over-sequence generation, and a likelihood-grounded objective.
>
> **Response.** Iterative denoising.

The longest phase and the spine of the repository. Each step is its own link.

**Steps**

1. Forward process; closed-form q(x_t | x_0); SNR as the real variable.
2. ELBO → the simplified ε-prediction objective. Full derivation, by hand.
3. UNet: residual blocks, timestep embedding, attention at low resolutions, group norm.
4. Ancestral sampling. *1000 sequential steps is unusable.*
5. → **DDIM.** Non-Markovian, deterministic at η=0, far fewer steps — and it reveals that
   sampling is solving an ODE. *(**Fix.**)*
6. **Score matching and the SDE view.** VP/VE SDEs, probability-flow ODE, Langevin dynamics,
   Fokker–Planck. *(**Unification**, and the tag matters. Score-based modeling developed
   largely in parallel with DDPM, not in response to it; the two were shown equivalent
   afterward. The "continuous-time generalization" framing is retrospective.)*
7. *Schedules and scalings were hand-tuned.* → **EDM.** Preconditioning
   (c_in / c_out / c_skip / c_noise), σ-sampling distribution, Karras schedule, Heun sampler.
8. *ε-prediction is badly weighted at SNR extremes.* → **v-prediction**, min-SNR-γ,
   zero-terminal-SNR.
9. *The noise→data path is curved, costing steps.* → **Flow matching / rectified flow.**
   Conditional flow matching, straight paths, what SD3 and Flux actually train.
   *(**Fix and Unification** at once — it straightens trajectories *and* subsumes diffusion
   as one probability path among many. It is also the surviving descendant of normalizing
   flows, which is why §11 lists flows as mutated rather than dead.)*
10. **Guidance.** Classifier guidance (derived) → *needs a separately trained noisy
    classifier* → **CFG** → guidance rescaling, scheduling, and CFG's known distortions.
11. Solver landscape: DPM-Solver++, UniPC — what higher-order solvers buy.

**Deliverable.** `07-diffusion/` with objectives and samplers strictly orthogonal.
Class-conditional CIFAR-10 / CelebA models.

**Ablation — the central one of the roadmap.** A full objective × sampler × step-count FID
grid. When you can explain every cell of that table, you understand diffusion.

---

### Phase 8 — Latent diffusion

> **Pressure.** Pixel-space diffusion above 256px is compute-prohibitive, and most of that
> compute goes into imperceptible high-frequency detail.
>
> **Response.** Run Phase 7 inside Phase 5's latent space.

**Steps**

1. Diffusion on latents; the latent scaling factor and why it exists.
2. **Latent caching** — precompute once, train many times. Critical on shared GPUs.
3. Class-conditional LDM at 256px.
4. **The reality gate.** Load pretrained SD1.5 VAE and UNet weights into *your own*
   implementation, match outputs numerically layer by layer, run inference. From-scratch work
   without an external check drifts into confidently wrong. First pretrained weights in the repo.

**Deliverable.** Your own LDM, plus a numerically verified reimplementation of SD1.5.

---

### Phase 9 — Text-conditioned latent diffusion

> **Pressure.** Class labels are a useless interface.

**Steps**

1. Cross-attention conditioning; pooled vs. full-sequence embeddings.
2. Wire in Phase 4's CLIP text encoder.
3. CLIP vs. T5 encoders, and why SD3 uses three.
4. Caption dropout — the training-side requirement for CFG.
5. Resolution handling: aspect-ratio bucketing, SDXL's size conditioning, and why naive
   center-cropping damages composition. *(Genuinely practical; commonly asked.)*
6. Train mini-SD on your captioned subset.
7. **Head-to-head against Phase 6** on identical data: FID, CLIPScore, latency, failure
   modes. The payoff for having built the autoregressive version.

**Deliverable.** A working mini-Stable-Diffusion and the comparison writeup.

---

### Phase 10 — Diffusion transformers

> **Pressure.** The UNet's convolutional inductive bias caps how it scales — it doesn't follow
> the transformer scaling curve Phase 2 measured.
>
> *(**Transfer + Scale unlock.** The UNet was not failing at the scales it was used at —
> SD1.5 is a UNet and works. DiT's argument is about the shape of the scaling curve, and it
> pays off only with enough compute. "The UNet was broken" is the wrong claim.)*

**Steps**

1. Patchify / unpatchify for latents.
2. Conditioning compared: in-context, cross-attention, adaLN, **adaLN-Zero**. Why
   zero-initialization matters.
3. **DiT.** Swap the backbone at matched parameter count and ablate against Phase 9.
4. 2D RoPE and QK-norm for stability at scale.
5. **MMDiT** (SD3): joint text/image streams, separate weights, shared attention — and why
   cross-attention's asymmetric treatment of the modalities was the limitation.
6. Scaling behavior: UNet vs. DiT, using Phase 2's methodology.

**Deliverable.** Interchangeable backbones behind one interface, plus a real ablation report.

---

## Part IV — Engineering

---

### Phase 11 — Scale engineering

> **Pressure.** Everything above is now bounded by throughput and memory rather than ideas.
> This is the part that separates an ML *engineer* from someone who has read the papers, and
> it is where interviews probe hardest for real experience.

**Steps**

1. DDP internals: gradient bucketing, compute/communication overlap.
2. **Parallelism taxonomy:** data, tensor, pipeline, sequence/context, expert. What each
   shards, what each costs in communication, when each is chosen. Standard interview question.
3. FSDP and ZeRO stages; sharding strategies; what breaks and how it presents.
4. Activation checkpointing, gradient accumulation, and the memory arithmetic to predict OOM
   before hitting it.
5. **fp16 on Turing, in depth.** Loss-scale dynamics, attention overflow, selective fp32.
   SDPA backend selection on SM 7.5.
6. Profiling: torch profiler, dataloader starvation, `channels_last`, `torch.compile`.
7. **GPU fundamentals and a Triton kernel.** Memory hierarchy, occupancy, arithmetic
   intensity, why fusion wins. Write one fused kernel and beat eager PyTorch on it. This is
   what makes "memory-bound vs. compute-bound" a real distinction rather than a phrase.
8. **Data at scale:** webdataset tar sharding, streaming, near-duplicate removal (pHash),
   aesthetic and NSFW filtering, **VLM recaptioning**, and why caption quality dominates
   text-to-image quality more than architecture does.
9. EMA weights, and why essentially every diffusion model needs them.

**Deliverable.** A measured scaling and throughput report across whatever GPUs you can get,
with the bottleneck identified at each configuration, plus one working Triton kernel.

---

### Phase 12 — Adapting pretrained models

> **Pressure.** You cannot pretrain a frontier model on four shared Turing cards. Full
> fine-tuning is memory-prohibitive — do the optimizer-state arithmetic and see.

**Steps**

1. Memory accounting for full fine-tuning: weights, gradients, Adam moments, activations.
2. **LoRA from scratch.** Why low-rank works, which modules to inject, α/rank scaling,
   merging. Then DoRA, and QLoRA's NF4 quantization.
3. Subject personalization: textual inversion; DreamBooth with prior preservation.
4. *You still can't control composition.* → **ControlNet**: zero-convolutions, trainable copy.
5. *And you can't condition on an image's style.* → **IP-Adapter**: decoupled cross-attention.
6. Inpainting, outpainting, and image-to-image via SDEdit — the practical workhorses.

**Deliverable.** A personalized, structurally controlled generation pipeline on SDXL.

---

### Phase 13 — Fast sampling and distillation

> **Pressure.** Even rectified flow wants 20–50 steps, and Phase 14 will multiply every step
> by the frame count. Step count becomes the dominant cost.

**Steps**

1. Why few-step sampling is hard: ODE trajectory curvature.
2. Progressive distillation.
3. Consistency models, LCM, consistency trajectory models.
4. Rectified-flow reflow — straightening the path Phase 7.9 introduced.
5. **Adversarial distillation:** SDXL-Turbo, LADD. *GAN machinery returns a second time,
   because it is the right tool — not for historical interest.*

**Deliverable.** A 1–4 step sampler for your own model, with an FID-vs-steps curve against
the Phase 7 baseline.

---

## Part V — Video

---

### Phase 14 — Motion adapters on a frozen T2I

> **Pressure.** Generating frames independently flickers — no temporal coherence, identity
> drifts between frames.
>
> **Response.** Temporal attention as an *adapter* on a frozen text-to-image model.

Placed here deliberately: AnimateDiff-style motion modules are structurally adapters, so they
belong beside Phase 12, not after everything. Moving pictures well before the heavy work.

**Steps**

1. Characterize the failure: flicker, identity drift, incoherent motion.
2. Temporal attention layers inserted into a frozen T2I UNet (AnimateDiff).
3. Motion LoRA; camera control.
4. Eval: temporal consistency metrics; introduce FVD.

**Deliverable.** Short animated clips from your Phase 12 pipeline.

---

### Phase 15 — Native video models

> **Pressure.** A motion adapter on a frozen image model is fundamentally limited — the image
> VAE wastes bits re-encoding temporally redundant content, and the frozen backbone has no
> native concept of time.

**Steps**

1. **Causal 3D VAE:** temporal compression, causal padding, first-frame handling.
2. Space-time attention: factorized vs. full 3D. Cost analysis first, architecture second.
3. **Video DiT with 3D RoPE** — the Latte / CogVideoX / Open-Sora lineage.
4. Image-to-video conditioning.
5. *Memory scales badly with frame count.* → chunked autoregressive long video, history
   conditioning, quality drift over time.
6. Eval: FVD, why FVD is a poor metric, and VBench.

**Hardware honesty.** 24 GB caps this hard — expect low resolution and few frames. The
objective is the architecture and the memory reasoning, not sample quality.

**Deliverable.** A small native text-to-video or image-to-video model, plus a written account
of exactly where the hardware ceiling bound you and what you'd change with more.

---

## Part VI — Convergence

---

### Phase 16 — Multimodal models

> **Pressure.** You now maintain two stacks — a language model and a visual generative model —
> sharing almost all their machinery. One model should do both.

**Steps**

1. **VLM** (Vision-Language Model — an LLM that takes images as input and emits text):
   frozen vision encoder + projector + LM, LLaVA-style. Training stages and what each fixes.
2. Resolution handling: tiling, AnyRes, and why naive downsampling destroys OCR ability.
3. **Unified autoregressive generation** over Phase 5's VQ tokens — Chameleon, Janus.
   *Returns to Phase 6's architecture, now with everything the intervening phases taught.*
4. Diffusion-head hybrids (Transfusion): autoregressive over text, diffusion over images,
   one backbone.
5. **Loop closure.** Use your VLM to recaption your Phase 15 video training data. The last
   phase feeds the second-to-last.

---

### Phase 17 — Alignment and preference optimization

> **Pressure.** Every model built so far imitates its training distribution. None of them do
> what a *user wants*. That gap is what alignment closes, and it is the last major piece of
> the standard interview surface.

Placed last because it applies to both stacks at once, and because Diffusion-DPO only makes
sense once both exist.

**Steps**

1. **SFT** and instruction tuning: data format, loss masking, why data quality dominates.
2. **Reward models** and the Bradley–Terry formulation.
3. **RLHF with PPO** — the full loop, and why it is finicky. Understand it even though you
   won't run it at scale.
4. **DPO** — *(**Fix**: removes the reward model and the RL loop entirely.)* Then GRPO and
   why reasoning models changed the picture.
5. **Diffusion-DPO** and aesthetic reward tuning — the same idea applied to image models.
   The two stacks converge here.
6. Evaluation of aligned models: preference benchmarks, LLM-as-judge and its biases.

**Deliverable.** DPO applied to your Phase 2 model, and to your Phase 9 image model.

---

## 7. Coverage map

Which phase makes which interview question answerable. Scan this when a topic feels missing.

| Topic | Where |
|---|---|
| Attention, transformers, positional encoding | 1 |
| Tokenization | 2.1 |
| Modern LM architecture, MoE, scaling laws | 2 |
| KV cache, batching, quantization, speculative decoding | 3 |
| ViT, CLIP, contrastive learning, embeddings | 4 |
| VAE, ELBO, VQ-VAE, perceptual + adversarial losses | 5 |
| Autoregressive image generation | 6 |
| Diffusion, score matching, SDE/ODE, flow matching, CFG | 7 |
| Latent diffusion, Stable Diffusion internals | 8, 9 |
| DiT, MMDiT | 10 |
| Distributed training, parallelism, GPU/kernels, data curation | 11 |
| LoRA, DreamBooth, ControlNet, IP-Adapter | 12 |
| Distillation, few-step sampling | 13 |
| Video generation | 14, 15 |
| VLMs, unified multimodal | 16 |
| SFT, RLHF, DPO | 17 |
| RAG, agents, prompting, serving, system design, safety | §8 |

---

## 8. The Literacy track

Read and write notes on; do not build. Each attaches to a Build-track phase — do it then,
not at the end. Output: a `notes/` writeup, plus a small worked example where marked.

**8.1 Prompt engineering** *(after Phase 3)* — few-shot, chain-of-thought, structured output,
prompt caching, why "prompt engineering is dead" is wrong for production systems. Two hours.
Previously excluded from this roadmap on the grounds that it builds no modeling skill; that
was correct and irrelevant, because it is asked.

**8.2 RAG systems** *(after Phase 4 — the modeling half is 4.7)* — chunking strategies,
hybrid search (BM25 + dense), rerankers, query rewriting, context-window management, and RAG
evaluation. *Worked example: build a small RAG over the papers you've read.* Dominates
applied-GenAI interviews.

**8.3 Agents and tool use** *(after Phase 3)* — function calling, ReAct, planning, MCP,
multi-agent patterns, failure modes and cost. *Worked example: a tool-calling loop over an
API.* Currently the most fashionable interview topic.

**8.4 Production serving** *(after Phase 3)* — vLLM/SGLang architecture, autoscaling, latency
vs. throughput SLAs, cost per token, caching. You built a toy version in Phase 3; this is how
real systems differ.

**8.5 System design for GenAI** *(after Phase 9, revisit after 15)* — "design an image
generation service", "design an LLM-powered search", "design a video pipeline". Practice
these out loud. Bounded by latency, cost, and safety, not by model quality.

**8.6 Safety and provenance** *(after Phase 9)* — NSFW filtering, watermarking, C2PA, model
cards, red-teaming, training-data copyright. Asked in nearly every image-generation interview.

**8.7 Evaluation landscape** *(after Phase 9)* — FID's known flaws, CLIPScore, GenEval,
HPSv2, VBench, HELM, LMArena, LLM-as-judge bias. You implement some metrics in the Build
track; this is knowing which are trusted and why.

**8.8 Audio and speech generation** *(after Phase 15)* — neural codecs, TTS, music generation.
A whole modality this roadmap otherwise ignores, and the codec/tokenizer ideas are Phase 5's.

**8.9 3D and world models** *(after Phase 15)* — NeRF, Gaussian splatting, text-to-3D,
video-as-world-model. Increasingly asked as video and 3D converge.

**8.10 The commercial landscape** *(ongoing)* — what Flux, SD3.5, Imagen, Veo, Sora, Midjourney
actually are; open vs. closed; licensing. *"What would you use for this and why?"* is a common
question with no theoretical component, and being unable to answer it reads as disengagement.

---

## 9. Reference curricula

External grounding. These exist so coverage is checkable against a published standard rather
than trusted to whoever wrote this file. **When starting a phase, check the corresponding
external material for topics this roadmap missed**, and add them.

| Source | Covers | Maps to |
|---|---|---|
| [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) | Probabilistic foundations: autoregressive models, VAEs, GANs, normalizing flows, energy-based and score-based models | Phases 5–7. The theory backbone; its [lecture notes](https://deepgenerativemodels.github.io/notes/) are the best written reference for the math |
| [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/) | Tokenization, architectures, MoE, GPUs/Triton, parallelism, scaling laws, inference, evaluation, data, alignment | Phases 1–3, 11, 17. **The closest existing analogue to this roadmap's language half** — assignments are strong self-tests |
| [MIT 6.S184 — Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/) | SDEs, Fokker–Planck, conditional/marginal probability paths, flow matching, score matching, guidance; builds a latent diffusion model | Phases 7–9. The most rigorous modern treatment of the flow-matching formulation |
| [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course) | Practical `diffusers`, fine-tuning, ControlNet | Phases 8–12, as the library counterpart to your from-scratch versions |
| [fast.ai Part 2 — Deep Learning Foundations to Stable Diffusion](https://course.fast.ai/Lessons/part2.html) | Stable Diffusion rebuilt from scratch | Phases 7–9. Closest in spirit to this roadmap's method |
| [Karpathy — Zero to Hero / nanoGPT](https://karpathy.ai/zero-to-hero.html) | Transformer and GPT from scratch, tokenizers | Phases 1–2 |
| [Lilian Weng's blog](https://lilianweng.github.io/) | Survey-quality explainers on diffusion, attention, alignment | Reference throughout |

---

## 10. Gap audits

**The defense against unknown unknowns. Do not skip these.**

At each part boundary — after Phases 3, 6, 10, 13, 15, 17 — Claude runs an audit:

1. Generates **20–30 interview questions** spanning the whole field at that level,
   deliberately including topics *outside* what has been built, drawn from §9's curricula and
   from what interviews actually ask.
2. Kenessary answers **without looking anything up**, in writing.
3. Claude grades each: **solid / shaky / blank**, and — critically — flags answers that are
   *confidently wrong*, which are more dangerous than blanks.
4. Produces `audits/partN-gaps.md`: what's missing, whether each gap is worth building,
   reading, or ignoring, and any roadmap amendments.
5. **The roadmap is amended.** This file is living. Audits that never change it aren't working.

A "blank" is information, not failure. It is the entire reason the audit exists.

---

## 11. Cross-cutting practice

Habits, not phases.

- **Every phase ends with a written derivation** in `docs/`. If it can't be written, it isn't
  understood.
- **Every phase ends with an ablation** that would have caught you being wrong.
- **Every phase ends with the questions it makes answerable** — three or four, answered from
  what you *observed*, not from the paper. *"Why does SD's autoencoder use a discriminator?"*
  answered with *"I trained one on MSE and it was blurry, because MSE recovers the posterior
  mean"* is the outcome this roadmap exists to produce.
- **Evaluation is built before the model it evaluates**, never after.
- **`experiments/` records negative results.** The run that failed and why is worth more than
  the run that worked, and "tell me about a time a training run failed" is a real question.
- **FID and bits-per-byte are implemented by hand.** Both are short, and both teach.
- **Say "I don't know" and "the field doesn't know" accurately.** Several things here are
  genuinely unexplained. Knowing which parts are understood and which are folklore is real
  expertise; overclaiming is how strong candidates lose credibility.

---

## 12. How sessions run

**This roadmap is the map, not the course. No course material is written until asked for.**

The session loop — theory, PyTorch primitives, task, implementation, review, run, quiz,
consolidate — plus pacing rules, the review layer, and all standing rules live in
**`TEACHING.md`**. Read it before generating material, reviewing code, or running a quiz.

Current state, checkpoints, and next action live in **`PROGRESS.md`**.

Two responsibilities that belong to Claude rather than Kenessary, because they are precisely
what he cannot yet see:

- **Gap audits** (§10) at part boundaries. Initiate them.
- **External curriculum checks** (§9) when starting a section — read the corresponding CS236 /
  CS336 / 6.S184 material for topics this roadmap missed, and say what's missing.

---


## 13. Not built

The Build track only admits mechanisms that enabled something downstream and survived.

| Not built | Reason |
|---|---|
| **Capsule networks** | Never enabled anything downstream; never scaled. The archetype for this table. |
| **RNN / LSTM / GRU** | Argued against analytically in Phase 1. Building them teaches only what attention replaced. |
| **PixelCNN / PixelRNN** | Autoregression over raw pixels — the dead end Phase 5's opening argument disposes of. |
| **RBMs, deep belief nets** | Historically pivotal, architecturally extinct. |
| **BERT / masked encoders** | Not the generative path. Know what they are; don't build one. |
| **StyleGAN-lineage T2I** | Lost general text-to-image to diffusion on mode coverage and training stability. |
| **unCLIP / DALL·E 2 prior** | Superseded by direct cross-attention conditioning. Read it, don't build it. |

**Two are mutations rather than deaths**, and both *are* built:

- **Normalizing flows** (RealNVP, Glow) lost on the cost of the invertibility constraint — but
  became continuous normalizing flows and then **flow matching**, which is Phase 7.9 and
  current SOTA.
- **GANs** lost standalone image generation — but survive as **Phase 5's adversarial
  reconstruction loss** and **Phase 13's adversarial distillation**. Existing GAN knowledge
  gets used twice, in both places where it remains the right tool.

**One is kept purely as a derivation step:** classifier guidance is obsolete in practice, but
CFG cannot be derived without it (Phase 7.10).

**Previously excluded, now in the Literacy track:** prompt engineering and RAG systems. The
original reasoning — that they build no durable modeling skill — was correct and beside the
point, because §1's purpose is interviews and both are asked constantly. They are read, not
built. §8.1, §8.2.
