# Work Log

**Author:** Shuvankar Biswas
**Repository:** [SKBiswas1998/MTJ-Inspired-Nonlinear-Neuron](https://github.com/SKBiswas1998/MTJ-Inspired-Nonlinear-Neuron) (fork of [Aref7792/MTJ-Inspired-Nonlinear-Neuron](https://github.com/Aref7792/MTJ-Inspired-Nonlinear-Neuron))
**Working branch:** `test_run`

Times are local (UTC−04:00). Newest session at the bottom.

---

## 2026-09-21 — Session 1: repo review and BLIF reproduction

### 14:56 — Repository check
- Local clone on `main`, clean, in sync with `origin/main` at `0f2b3a2`. Upstream also only has `main` at the same commit.
- Four experiments, all `784 → 128 → 64 → 10`, 25 time steps, 10 epochs, batch 64, seed 42:

  | Folder | Script | Neuron |
  |---|---|---|
  | `BLIF/` | `blif.py` | snnTorch Leaky LIF baseline (β = 0.9) |
  | `SMTJ/` | `smtj.py` | Soft MTJ (sigmoid-gated integrate/leak) |
  | `HMTJ/` | `hmtj.py` | Hard MTJ (pulse / no-pulse, straight-through estimator) |
  | `ternary_mtj/` | `thmtj.py` | Signed ternary MTJ (+1 / 0 / −1 spikes) — added in `0f2b3a2`, not yet in top-level README |

### ~15:00 — Architecture review: findings (not fixed yet)
1. **Ternary negative spikes can never fire.** `ternary_mtj/thmtj.py:55` sets `NEG_THRESHOLD = 1.6`, but the membrane is clamped to ±0.999 (`thmtj.py:614-616`). The ternary README states 0.90. The model is effectively binary.
2. **BLIF baseline is not like-for-like.** BLIF: SGD lr 0.01, loss on summed output membrane, no grad clipping. MTJ models: Adam lr 1e-3, loss on output spike counts, grad clip 1.0.
3. **Test-set model selection.** Every script picks the "best" checkpoint by test accuracy and reports it as "final"; there is no validation split.
4. **No reset after firing in MTJ neurons** (intentional per code comments). Leak near saturation is very slow, which explains the high MTJ firing rates (0.2–0.4 vs ~0.05–0.09 for BLIF).
5. Minor: SMTJ uses the sigmoid drive twice (current *and* pulse width); only 30 ps of each 100 ps step is integrated; z→J mapping is an assumption; `A(J)` table is non-monotonic at 7e11/8e11.
6. Docs: top-level README omits `ternary_mtj`; install line omits `matplotlib` (use `pip install -r requirements.txt`).

### ~15:03 — Branch
- Created branch `test_run` from `main` (`0f2b3a2`) and pushed it to `origin`. `git pull origin test_run` → already up to date.

### 15:05 — Training runs started (laptop CPU)
- Environment: Python 3.13.5, torch 2.10.0+cpu (CUDA not available; NVIDIA MX250 unused), 8 CPU threads.
- Launched all four unmodified scripts in parallel, 2 threads each.
- Measured speed: BLIF ≈ 4 min/epoch, MTJ models ≈ 14 min/epoch (~2.3 h each).
- Early observation (ternary): pulse rate 1.000 in every layer; layer-3 firing rate ≈ 0.015.

### ~15:16 — Scope reduced
- Stopped SMTJ, HMTJ and ternary runs; kept BLIF only. (No results recorded for those three.)
- Kaggle GPU discussed as the faster option (est. ~5–10 min for BLIF, ~15–30 min per MTJ model; needs `pip install snntorch`, Internet on, phone-verified account).

### 15:25 — BLIF run finished

| Epoch | Train loss | Train acc | Test acc | FR L1 | FR L2 | FR L3 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.9703 | 81.19% | 91.42% | 0.0800 | 0.0349 | 0.0460 |
| 2 | 0.3779 | 93.46% | 92.81% | 0.0797 | 0.0330 | 0.0434 |
| 3 | 0.2773 | 94.86% | 92.75% | 0.0805 | 0.0372 | 0.0520 |
| 4 | 0.2190 | 95.79% | 96.07% | 0.0848 | 0.0393 | 0.0571 |
| 5 | 0.1794 | 96.40% | 93.74% | 0.0853 | 0.0395 | 0.0640 |
| 6 | 0.1508 | 96.81% | 95.77% | 0.0864 | 0.0431 | 0.0679 |
| 7 | 0.1344 | 97.20% | 95.92% | 0.0836 | 0.0426 | 0.0716 |
| 8 | 0.1135 | 97.49% | 95.81% | 0.0834 | 0.0419 | 0.0703 |
| 9 | 0.0977 | 97.83% | **96.53%** | 0.0861 | 0.0440 | 0.0755 |
| 10 | 0.0911 | 98.01% | 95.45% | 0.0831 | 0.0447 | 0.0785 |

| Metric | README | This run | Δ |
|---|---:|---:|---:|
| Best test accuracy | 96.93% | 96.53% | −0.40 |
| Spike-count test accuracy | 97.12% | 96.87% | −0.25 |
| Last-epoch test accuracy | — | 95.45% | — |

Takeaways: reproduces within half a point (CPU vs GPU non-determinism); reported "final" accuracy is the reloaded best checkpoint; test accuracy is noisy under SGD lr 0.01.

### 15:27 — Commit `ddbb106` (pushed to `origin/test_run`)
- `results/blif_cpu_run_2026-09-21.log` — full training log
- `results/blif_cpu_run_2026-09-21.html` — per-epoch charts and README comparison
- `.gitignore` — ignores `data/`, `*.pth`, `__pycache__/`

### 15:45 — Session closed; this work log added

### Open to-do (not started)
- [ ] Fix ternary `NEG_THRESHOLD` (1.6 → 0.9) and re-run
- [ ] Make BLIF a fair baseline (Adam 1e-3, spike-count loss, grad clip 1.0)
- [ ] Add a validation split (e.g. 55k / 5k) for checkpoint selection
- [ ] Update top-level README (add `ternary_mtj`, fix install line)
- [ ] Run SMTJ, HMTJ and ternary models (Kaggle GPU or ~2.3 h each on CPU)

---

## 2026-09-24 — Session 2: BLIF rerun, plotting pipeline, HMTJ started

### 12:08 — All four runs launched, then narrowed to BLIF
- Launched `blif.py`, `smtj.py`, `hmtj.py`, `thmtj.py` in parallel on CPU, 2 threads each (8 cores).
- Sharing the CPU four ways put the MTJ models at ~35 min/epoch (≈6 h each), so SMTJ, HMTJ and
  ternary were stopped ~12 min in and their partial logs deleted. BLIF continued alone.
- Early ternary observation repeated from session 1: pulse rate 1.000 in all three layers,
  layer-3 firing rate ≈ 0.02 — consistent with the `NEG_THRESHOLD` bug (still unfixed).

### 12:25 — BLIF finished: bit-identical to the 2026-09-21 run

| Metric | 2026-09-21 | 2026-09-24 |
|---|---:|---:|
| Best test accuracy | 96.53% | 96.53% |
| Final test accuracy (best checkpoint) | 96.53% | 96.53% |
| Last-epoch test accuracy | 95.45% | 95.45% |
| Spike-count test accuracy | 96.87% | 96.87% |
| Final firing rates (L1/L2/L3) | 0.0861/0.0440/0.0755 | 0.0861/0.0440/0.0755 |

Every per-epoch figure matched to four decimals: **the pipeline is deterministic on CPU under
seed 42**, independent of thread count (the rerun used 2 threads, then 8 after the others stopped).
Useful as a fixed reference point before changing anything.

Curve shape: loss collapses by epoch 2 then flattens; test accuracy plateaus near 96% from
epoch 4 and oscillates ±1.5 points while train accuracy climbs to 98%; layer-3 firing rate rises
steadily 0.046 → 0.079. Confirms the session-1 finding that "best" is picked off SGD noise.

### 12:15 — Plotting pipeline added
- `results/make_plots.py` parses run logs (per-epoch TRAIN/TEST blocks plus the final summary)
  and writes PNGs to `results/plots/`. Safe to run mid-training; it plots completed epochs only.
- Outputs: `<model>_<date>.png` (loss / accuracy / per-layer firing rates), `comparison_<date>.png`,
  `best_accuracy_<date>.png`.
- Usage: `python results/make_plots.py [YYYY-MM-DD]` (defaults to today).

### 12:33 — Commit `a095e92`
- `results/blif_cpu_run_2026-09-24.log`, `results/make_plots.py`, three PNGs in `results/plots/`.
- Local only — **not pushed** to `origin/test_run`.

### 12:36 — HMTJ started, then stopped at user request
- Ran `hmtj.py` alone with 8 threads: ~12 min/epoch, so ≈2 h for 10 epochs.
- Stopped during epoch 2 (batch 900/938). Partial results, for reference:

  | Epoch | Train loss | Train acc | Test acc | FR L1 | FR L2 | FR L3 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | 0.5774 | 86.85% | 91.90% | 0.3070 | 0.3135 | 0.3087 |

- Pulse rates were genuinely sub-unity (0.34/0.39/0.46), unlike ternary's pinned 1.000.
- `hmtj.py` has no checkpoint-resume, so a restart begins from epoch 1.
  `results/hmtj_cpu_run_2026-09-24.log` committed in `ef0ee71` for reference.

### 12:47 — Session paused; branch pushed
- Committed the partial HMTJ log and pushed `test_run` to `origin`: `a095e92` (BLIF rerun,
  plotting script, images), `1f51014` (session work log), `ef0ee71` (HMTJ partial log).
- Nothing running at pause: no training processes, no background watchers.

### Progress: 1 of 4 models complete
- BLIF done (twice, identical results). HMTJ reached 1.5/10 epochs then stopped.
  SMTJ and ternary not started.
- Remaining ≈ 6 h of CPU time run sequentially; the Kaggle GPU route would cut that to
  well under an hour.

### Open to-do (carried forward)
- [ ] Run HMTJ to completion (~2 h, 8 threads), then SMTJ and ternary
- [ ] Fix ternary `NEG_THRESHOLD` (1.6 → 0.9) and re-run
- [ ] Make BLIF a fair baseline (Adam 1e-3, spike-count loss, grad clip 1.0)
- [ ] Add a validation split (e.g. 55k / 5k) for checkpoint selection
- [ ] Update top-level README (add `ternary_mtj`, fix install line)
- [x] Push `test_run` to `origin` — done 2026-09-24 12:47
