# med-dapt-qwen35-ablation

A **config-driven** DAPT (domain-adaptive pretraining) + DoRA ablation study for
`Qwen/Qwen3.5-4B-Base` on a medical corpus. Runs in Colab, writes its results to
Drive, and resumes across sessions.

The point of the layout: **every number lives in a JSON file, every behaviour
lives in `dapt_lib.py`, and the notebook is drivers and prose only.** Changing an
experiment means adding a small JSON file — never editing the notebook, never
editing `base.json`.

## Run it

Open the notebook in Colab:

```
https://colab.research.google.com/github/adi274903/med-dapt-qwen35-ablation/blob/main/unsloth_Med_Qwen3_5_4B_dapt_ABLATION.ipynb
```

Cell 4 clones this repo (once per session, `--depth 1`) and puts it on
`sys.path`, so `import dapt_lib` resolves to the code here. Re-running the cell
pulls the latest commit instead of re-cloning.

**Code comes from GitHub; results go to Drive.** `ablation_root` in
`configs/base.json` points at the Drive tree. That is deliberate: the session is
disposable, the results tree is not. Each arm writes `<root>/<arm_id>/status.json`,
and the driver skips arms whose `status.json` says `done`, so a dead session
costs one cell re-run, not a re-run of the grid.

## What's here

| path | role |
|---|---|
| `dapt_lib.py` | all logic: budget math, model build, data mix, trainer, `probe`, `run_grid`, keep/throw ruler, `phase0`, adapter checks |
| `configs/base.json` | every knob, with `_why` strings as the comments JSON cannot carry |
| `configs/ctx4096.json` | worked override example (adds a `B4k` arm at 4096 tokens) |
| `unsloth_Med_Qwen3_5_4B_dapt_ABLATION.ipynb` | the run: 13 sections, one-line calls into `dapt_lib` |
| `selftest/` | stub `torch`/`datasets`/`unsloth` tree + harness that executes the *whole* module off-GPU |

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

1. `tie_word_embeddings=True` on the 4B. PEFT's `modules_to_save` deep-copies
   both `embed_tokens` and `lm_head`, so **the tie is broken** and both copies
   are trainable: 1.271B of the 1.329B trainable parameters. `ensure_weight_tying`
   is exposed as a baseline key (default `false`, i.e. the behaviour above);
   set it `true` and PEFT keeps **one** trainable tensor and wires the head to it.
   That changes *what is trainable*, so it belongs in the grid as its own arm.
2. bf16 is mandatory (`qwen3_5` is in Unsloth's `FORCE_FLOAT32`) and QLoRA is
   discouraged for this model.

## Adding a variant

Do **not** edit `configs/base.json`. Add `configs/<name>.json` with only the keys
you change, and pass its name as `OVERRIDES` in the bootstrap cell:

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
entire module execute end-to-end — all three `build_model` branches (tie broken /
`ensure_weight_tying` / adapter-only), the trainer callbacks, `run_grid`'s budget
guards, `status.json` + `results.csv`, the ruler, `phase0` and the adapter
inspector — with no accelerator.

```bash
STUB_PEFT_FORWARD=1 python3 selftest/test_stub_run.py   # wrapper forwards .weight
STUB_PEFT_FORWARD=0 python3 selftest/test_stub_run.py   # older/different PEFT naming
```

**Before spending units:** run `probe()` (notebook cell 8 or `dapt_lib.probe`).
`budget.sec_per_step` ships as a **placeholder (12.0)** and a wrong value does not
error — it silently rescales the token budget, which is the one thing this design
must not do.
