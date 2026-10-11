# med-dapt-qwen35-ablation

A **config-driven** DAPT (domain-adaptive pretraining) + DoRA ablation study for
`Qwen/Qwen3.5-4B-Base` on a medical corpus. Runs in Colab, writes its results to
Drive, and resumes across sessions.

The point of the layout: **every number lives in `config.json`, every behaviour
lives in `dapt_lib.py`, every step of the run lives in `steps/*.py`, and the
notebook is one line per step.** Changing an experiment means editing JSON — never
the notebook, and never the modules.

## Run it

Open the notebook in Colab:

```
https://colab.research.google.com/github/adi274903/med-dapt-qwen35-ablation/blob/main/unsloth_Med_Qwen3_5_4B_dapt_ABLATION.ipynb
```

The notebook's **preamble cell** is the only cell that holds real code — it is
what fetches the rest. It clones this repo (once per session, `--depth 1`), puts
it on `sys.path`, and defines

```python
step('05_verify_env')      # == %run -i steps/05_verify_env.py, in this namespace
```

Every other cell is one `step(...)` call, so the code the notebook runs is code
you can read, diff and version here. Re-running the preamble cell pulls the latest
commit instead of re-cloning, and each step re-reads its file, so a mid-session
`git pull` takes effect on the next cell.

**Code comes from GitHub; results go to Drive.** `ablation_root` in `config.json`
points at the Drive tree. That is deliberate: the session is disposable, the
results tree is not. Each arm writes `<root>/<arm_id>/status.json`, and the driver
skips arms whose `status.json` says `done`, so a dead session costs one cell
re-run, not a re-run of the grid.

## What's here

| path | role |
|---|---|
| `config.json` | every knob, with `_why` strings as the comments JSON cannot carry |
| `dapt_lib.py` | all logic: budget math, model build, data mix, trainer, `probe`, `run_grid`, keep/throw ruler, `phase0`, adapter checks |
| `steps/*.py` | the run itself, one file per notebook cell (table below) |
| `configs/ctx4096.json` | worked override example (adds a `B4k` arm at 4096 tokens) |
| `unsloth_Med_Qwen3_5_4B_dapt_ABLATION.ipynb` | the driver: preamble + one `step(...)` line per module |
| `selftest/` | stub `torch`/`datasets`/`unsloth` tree + harness that executes the *whole* module off-GPU |

### The steps

| step | file | what it does |
|---|---|---|
| — | preamble (in the notebook) | clone/pull this repo, `sys.path`, define `step()` |
| 01 | `steps/01_install.py` | pin the runtime: unsloth first, then the transformers v5 window, then the G4 wheels |
| 02 | `steps/02_colab_setup.py` | HF token, Drive mount, fresh compile cache |
| 03 | `steps/03_causal_conv1d.py` | the prebuilt GDN wheel from Drive (`--no-deps`) |
| 04 | `steps/04_open_run.py` | `run = open_run('config.json')` — the session state |
| 05 | `steps/05_verify_env.py` | the env gates: bf16, cut-cross-entropy, no QLoRA |
| 06 | `steps/06_budget.py` | the cost table for this grid |
| 07 | `steps/07_data.py` | build the interleaved mix + per-source holdout |
| 08 | `steps/08_probe.py` | measure s/step on this box (~5 min) |
| 09 | `steps/09_run_grid.py` | run the pending arms |
| 10 | `steps/10_tensorboard.py` | TensorBoard over `run.root` |
| 11 | `steps/11_ruler.py` | keep/throw ranking |
| 12 | `steps/12_verify_adapters.py` | validate the winner's adapters |
| 13 | `steps/13_phase0.py` | the does-DAPT-help curve (optional) |

Each file's docstring carries that step's "why". Steps 05-13 assume `run` exists
(step 04); the Colab-only steps (01-03) are plain `subprocess` pip installs so they
work as files instead of needing `%pip` magics.

## The study

| arm | what changes |
|---|---|
| `B` | baseline: DoRA r=32, alpha=64, LR 1e-4, decoupled embedding LR (0.1x) |
| `R16` | rank 16 (alpha 32) — can rank drop further? |
| `LO` | LoRA instead of DoRA, r=32 |
| `LR3e4` | LR 3e-4 |
| `EMB1` | no decoupled embedding LR (embed LR = LR) |
| `ADONLY` | `modules_to_save=None` — embed/head frozen, adapter only |

1392 steps and 45.6M tokens per arm at 5.0 h/arm; the grid is **30 h of a 45 h
ceiling** (~261 Colab units), ~3.2 sessions, with the remaining 10 h reserved for
two seed replicates of `B` (kept out of `grid` on purpose so a default run does
not spend them).

Two model facts drive the design and are asserted in `build_model()`, not assumed:

1. `tie_word_embeddings=True` on the 4B. PEFT's `modules_to_save` wraps both
   `embed_tokens` and `lm_head`, and both stay trainable — but **the tie is kept,
   measured, not assumed**: PEFT moves `lm_head` into `modules_to_tie` and points
   it at the embedding's trainable copy, so the pair moves as one 0.636B-param
   tensor out of 0.694B trainable. `ensure_weight_tying` was exposed as a baseline
   key to buy exactly that state; since it happens anyway, the key is a **no-op
   here**, and `build_model()` reports the tie state it actually got.
2. bf16 is mandatory (`qwen3_5` is in Unsloth's `FORCE_FLOAT32`) and QLoRA is
   discouraged for this model.

## Adding a variant

For a one-off experiment, do **not** edit `config.json`: add `configs/<name>.json`
with only the keys you change, and pass its name as `OVERRIDES` in the preamble
cell:

```jsonc
// configs/my-variant.json
{ "grid": [ ...the whole array, because arrays are replaced wholesale... ] }
```

Dicts deep-merge key-by-key; **arrays are replaced outright**, so a variant that
touches the grid restates it. To change context length, set an **arm's**
`max_length` — never `budget.seq_len`, which is the token-budget reference length
and would silently rescale tokens-per-arm. `check_config()` warns when the two
disagree.

## Off-GPU checks

These need no GPU and no torch (heavy imports are inside functions):

```bash
python3 dapt_lib.py budget      # the cost table
python3 dapt_lib.py config      # the fully-resolved config, defaults + overrides merged
python3 dapt_lib.py selftest    # config, budget math, status/CSV contract
python3 dapt_lib.py ruler       # scores whatever arms have finished
```

(`config` is the one to reach for first when a run behaves differently than the
JSON looks: it prints what the config *actually* resolved to.)

`selftest/` goes further: a stub `torch`/`datasets`/`unsloth` tree that lets the
entire module execute end-to-end — all three `build_model` paths (tie preserved,
tie broken, adapter-only), the trainer callbacks, `run_grid`'s budget
guards, `status.json` + `results.csv`, the ruler, `phase0` and the adapter
inspector — with no accelerator.

```bash
STUB_PEFT_FORWARD=1 python3 selftest/test_stub_run.py   # wrapper forwards .weight
STUB_PEFT_FORWARD=0 python3 selftest/test_stub_run.py   # older/different PEFT naming
```

**Before spending units:** run `probe()` (step 08 or `dapt_lib.probe`).
`budget.sec_per_step` ships as a **placeholder (12.0)** and a wrong value does not
error — it silently rescales the token budget, which is the one thing this design
must not do.
