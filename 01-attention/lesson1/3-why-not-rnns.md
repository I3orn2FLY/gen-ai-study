# 3 · Why not recurrence

*~4 min. Lesson 1 — **not written yet**.*

> **Placeholder.** Parts are written one at a time, after the previous one is right.
> Below is what this part will contain.

Part 1 solved the bottleneck and part 2 made scoring cheap — and the RNN is still sitting there.
So why delete it?

**Will cover**

- Two separate arguments people constantly conflate — and which one actually decided it
- **Path length**: information walks $O(|i-j|)$ steps in an RNN, jumps in $1$ with attention.
  Gradient through a path of length $n$ multiplies $n$ Jacobians → $\sigma^n$
- **Parallelism**: $O(n)$ sequential steps per layer vs $O(1)$. The real reason
- The complexity table, and the trap in it — attention is $O(n^2 d)$ vs recurrence $O(n d^2)$,
  so attention is **more** expensive past $n \approx d$. It won anyway, because the work is
  parallel
- The $n^2$ debt this creates, and where the roadmap pays it

*Figure ready: `figures/fig2-path-length.png`*
