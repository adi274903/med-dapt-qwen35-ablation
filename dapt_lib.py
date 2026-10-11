"""
dapt_lib -- medical DAPT / DoRA ablation engine for Qwen3.5-4B-Base.

Every number lives in config.json (repo root); every piece of logic lives here.
The notebook only calls these functions, and every one of its cells is one line:
`step('<file>')`, which %runs steps/<file>.py from this repo.

WHY IT IS SHAPED THIS WAY
  * No heavy import at module level.  `import dapt_lib` must succeed before
    `pip install`, so torch / transformers / unsloth / datasets / pandas are
    imported inside the functions that need them.  The budget math and the
    ruler therefore run on a bare interpreter -- which is the only reason they
    can be tested without a GPU.
  * Config is read at CALL time, never captured at import time.  Cell order in
    Colab is a suggestion; a value captured at import is the classic silent bug.
  * The runtime contract is frozen.  The driver resumes across sessions off
    <root>/<arm_id>/status.json, so these names are an interface, not cosmetics:
        status.json : arm_id, phase, desc, cfg, done, wall_hours, steps_run,
                      tokens_run, tok_per_param, sec_per_step_amortised,
                      peak_vram_gib, best_metric, final_losses, base_losses,
                      domain_mean_final, guards,
                      cfg_hash, config_path, lib_mtime      (provenance, added)
        results.csv : status.json minus `cfg`, dict/list values JSON-encoded
        ranking.csv : arm, desc, domain_mean, wall_h, <each eval source>,
                      retention_delta, verdict
  * Every arm carries the resolved config and its sha256, so a results row is
    self-describing months later.

Typical use:  run = open_run("config.json"); print_budget(run)
              build_data(run); probe(run); run_grid(run); ruler(run)

CLI:  python3 dapt_lib.py budget|selftest|config [path] [overrides]
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys
import time

__all__ = [
    "Run", "open_run", "load_config", "deep_merge", "config_hash",
    "steps_for", "wall_hours", "derive", "build_arms", "check_config",
    "print_budget", "verify_env", "build_data", "probe", "run_grid",
    "run_one_arm", "ruler", "phase0", "verify_adapters", "selftest",
]

DEFAULT_CONFIG_CANDIDATES = (
    "config.json",                                     # repo root -- the file you edit
    "/content/med-dapt-qwen35-ablation/config.json",    # the Colab clone
    "configs/config.json",
)


# =====================================================================
# config
# =====================================================================
def deep_merge(base, over, _path=""):
    """Recursive merge: dicts merge, everything else is replaced by `over`."""
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v, "%s.%s" % (_path, k))
        else:
            out[k] = v
    return out


def _find_config(path=None):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError("config not found: %s" % path)
        return path
    env = os.environ.get("DAPT_CONFIG")
    if env:
        return _find_config(env)
    for cand in DEFAULT_CONFIG_CANDIDATES:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "no config found; looked at: %s (pass the path explicitly)"
        % ", ".join(DEFAULT_CONFIG_CANDIDATES))


def _read_json(p):
    with open(p) as fh:
        return json.load(fh)


def _resolve_override(overrides, base_path):
    """Find an override file: as given, then next to config.json, then in configs/.

    config.json sits at the repo root, so a bare name like "ctx4096.json" means
    configs/ctx4096.json -- that is where variants live.
    """
    d = os.path.dirname(os.path.abspath(base_path))
    for cand in (overrides,
                 os.path.join(d, overrides),
                 os.path.join(d, "configs", overrides),
                 os.path.join(d, "configs", os.path.basename(overrides))):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError("override config not found: %s" % overrides)


def load_config(path=None, overrides=None):
    """Load the config, then apply `overrides`.

    `overrides` may be a dict (deep-merged) or a path to a variant JSON file --
    that is the whole point of the split: a variant never forks the notebook.
    A variant file holds ONLY the keys it changes; it is not merged with the
    default config first.
    """
    p = _find_config(path)
    cfg = _read_json(p)
    if overrides:
        if isinstance(overrides, str):
            op = _resolve_override(overrides, p)
            over = _read_json(op)
            cfg["_overrides_path"] = op
        else:
            over = overrides
        cfg = deep_merge(cfg, over)
    cfg["_config_path"] = os.path.abspath(p)
    return cfg



def config_hash(cfg):
    """sha256[:12] of the config minus its own bookkeeping keys."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# =====================================================================
# budget math  (pure python -- no torch, no GPU)
# =====================================================================
def steps_for(tokens, effective_batch, seq_len):
    return max(1, int(tokens) // (effective_batch * seq_len))


def derive(cfg):
    """The four numbers everything else is derived from.

    Tokens per arm is held CONSTANT across arms, which is what makes arms
    comparable -- and why a longer-context arm gets proportionally fewer steps.
    """
    b = cfg["budget"]
    o = cfg["overheads"]
    overhead_h = (o["model_load_min"]
                  + (2 + o["evals_per_run"]) * o["eval_minutes"]) / 60.0
    train_h = max(0.0, b["wall_h_per_arm"] - overhead_h)
    steps = max(1, int(round(train_h * 3600.0 / b["sec_per_step"])))
    tokens = steps * b["effective_batch"] * b["seq_len"]
    return dict(overhead_h_per_run=overhead_h,
                train_h_per_run=train_h,
                steps_per_arm=steps,
                tokens_per_arm=tokens,
                text_params=b["text_params"])


def wall_hours(cfg, steps):
    return steps * cfg["budget"]["sec_per_step"] / 3600.0 + derive(cfg)["overhead_h_per_run"]


def build_arms(cfg):
    """baseline + per-arm overrides -> the full arm list, with steps and wall_h."""
    d = derive(cfg)
    arms = []
    for entry in cfg["grid"]:
        arm = dict(cfg["baseline"])
        arm.update(entry.get("overrides") or {})
        arm["arm_id"] = entry["id"]
        arm["desc"] = entry.get("desc", "")
        if entry.get("phase"):
            arm["phase"] = entry["phase"]
        if entry.get("force_save") is not None:
            arm["force_save"] = entry["force_save"]
        arm["steps"] = steps_for(d["tokens_per_arm"],
                                 cfg["budget"]["effective_batch"],
                                 arm["max_length"])
        arm["wall_h"] = wall_hours(cfg, arm["steps"])
        arms.append(arm)
    return arms


def check_config(cfg, arms=None):
    """Invariants that would otherwise fail silently mid-run. Returns warnings."""
    arms = build_arms(cfg) if arms is None else arms
    data, b = cfg["data"], cfg["budget"]
    warn = []
    assert abs(sum(data["mix"]) - 1.0) < 1e-8, "data.mix must sum to 1.0"
    assert len(data["mix"]) == len(data["sources"]), "data.mix length != sources"
    missing = set(data["guard_sources"]) - set(
        "".join(ch if ch.isalnum() else "_" for ch in s["name"]) for s in data["sources"])
    assert not missing, "guard sources not in sources: %s" % sorted(missing)
    assert cfg["baseline"]["lora_alpha"] == 2 * cfg["baseline"]["r"], (
        "baseline alpha != 2*r -- the effective scaling would differ from the"
        " config this grid was designed around")
    for a in arms:
        if a.get("use_dora") and a.get("use_rslora", False):
            warn.append("%s: rsLoRA is ignored/conflicting with DoRA" % a["arm_id"])
    total_h = sum(a["wall_h"] for a in arms)
    if total_h > b["total_budget_hours"]:
        warn.append("grid is %.2f h, over the %.2f h ceiling"
                    % (total_h, b["total_budget_hours"]))
    if b["sec_per_step"] <= 0:
        raise ValueError("budget.sec_per_step must be > 0")
    if cfg["baseline"]["max_length"] != b["seq_len"]:
        warn.append("budget.seq_len (%d) != baseline.max_length (%d): tokens/arm is"
                    " scaled by seq_len, so this changes the token budget by %.2fx."
                    " To change CONTEXT, set an arm's max_length -- do not touch"
                    " seq_len." % (b["seq_len"], cfg["baseline"]["max_length"],
                                   b["seq_len"] / cfg["baseline"]["max_length"]))
    off = sorted(set(a["max_length"] for a in arms) - {b["seq_len"]})
    if off:
        warn.append("arm max_length %s != seq_len %d: tokens/arm is still held"
                    " constant, so those arms get proportionally fewer steps, but"
                    " sec_per_step was measured at %d -- their wall-clock estimate"
                    " is an estimate, not a measurement (re-probe, or read the"
                    " Probes callback's s/step and re-run the budget cell)."
                    % (off, b["seq_len"], b["seq_len"]))
    return warn



# =====================================================================
# Run: state + paths + the runtime contract
# =====================================================================
class Run:
    """One config, one ablation root, and everything that is session state.

    Held as a single notebook variable so a Colab kernel restart costs you one
    re-run of the bootstrap cell, not a re-typed control panel.
    """

    def __init__(self, cfg, config_path=None):
        self.cfg = cfg
        self.config_path = config_path or cfg.get("_config_path")
        self.cfg_hash = config_hash(cfg)
        self.derived = derive(cfg)
        self.arms = build_arms(cfg)
        self.root = cfg["ablation_root"]
        self.streams = None          # set by build_data()
        self.eval_by_source = None

    # ---- paths -------------------------------------------------------
    def arm_dir(self, arm):
        return "%s/%s" % (arm.get("root", self.root), arm["arm_id"])

    def status_path(self, arm):
        return "%s/status.json" % self.arm_dir(arm)

    @property
    def results_csv(self):
        return "%s/results.csv" % self.root

    @property
    def ranking_csv(self):
        return "%s/ranking.csv" % self.root

    @property
    def run_config_path(self):
        return "%s/run_config.json" % self.root

    # ---- status ------------------------------------------------------
    def arm_status(self, arm):
        p = self.status_path(arm)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return None

    def arm_done(self, arm):
        s = self.arm_status(arm)
        return bool(s and s.get("done"))

    def done_ids(self):
        return [a["arm_id"] for a in self.arms if self.arm_done(a)]

    def pending(self):
        return [a for a in self.arms if not self.arm_done(a)]

    def arm_saves(self, arm):
        sv = self.cfg["saving"]
        return bool(sv["save_models"] or arm["arm_id"] in sv["save_arms"]
                    or arm.get("force_save", False))

    # ---- writes ------------------------------------------------------
    def ensure_root(self):
        os.makedirs(self.root, exist_ok=True)

    def write_run_config(self):
        """Freeze the resolved config next to the results, once per root."""
        self.ensure_root()
        if not os.path.exists(self.run_config_path):
            with open(self.run_config_path, "w") as fh:
                json.dump({k: v for k, v in self.cfg.items() if not k.startswith("_")},
                          fh, indent=2, sort_keys=True)
        return self.run_config_path

    def append_result(self, row):
        """Append to results.csv, widening the header if the schema grew.

        A ragged row would otherwise raise mid-GPU-run and cost the arm's
        results, so the older rows are padded with "" instead.
        """
        path = self.results_csv
        if not os.path.exists(path):
            with open(path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=list(row)).writeheader()
                csv.DictWriter(fh, fieldnames=list(row)).writerow(row)
            return
        with open(path, "r", newline="") as fh:
            rd = list(csv.DictReader(fh))
            header = list(rd[0].keys()) if rd else []
        missing = [k for k in row if k not in header]
        if missing:
            header = header + missing
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=header)
                w.writeheader()
                for r in rd:
                    w.writerow(r)
        with open(path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=header).writerow(row)

    # ---- data (lazy: built once, reused by every arm) -----------------
    def require_data(self, verbose=True):
        if self.streams is None:
            build_data(self, verbose=verbose)
        return self.streams, self.eval_by_source


def open_run(config_path=None, overrides=None):
    cfg = load_config(config_path, overrides)
    return Run(cfg, config_path=cfg["_config_path"])


# =====================================================================
# budget report  (prints what the old control-panel + grid cells printed)
# =====================================================================
def print_budget(run):
    cfg, d, b = run.cfg, run.derived, run.cfg["budget"]
    print("model                : %s" % cfg["model_id"])
    print("config               : %s  (hash %s)" % (run.config_path, run.cfg_hash))
    print("steps per arm        : %d" % d["steps_per_arm"])
    print("tokens per arm       : {:,.0f}  ({:.5f} tok/param)".format(
        d["tokens_per_arm"], d["tokens_per_arm"] / d["text_params"]))
    print("train h / arm        : %.2f  (%.0f%% of the arm's wall clock)"
          % (d["steps_per_arm"] * b["sec_per_step"] / 3600,
             100 * d["steps_per_arm"] * b["sec_per_step"] / 3600 / b["wall_h_per_arm"]))
    print("overhead h / arm     : %.2f  (reload %.1f min + %d evals x %.1f min)"
          % (d["overhead_h_per_run"], cfg["overheads"]["model_load_min"],
             2 + cfg["overheads"]["evals_per_run"], cfg["overheads"]["eval_minutes"]))
    print("wall h / arm         : %.2f" % b["wall_h_per_arm"])
    print("SAVE_MODELS          :", cfg["saving"]["save_models"],
          "| SAVE_ARMS:", cfg["saving"]["save_arms"])
    print("")
    print("%-9s %6s %7s %8s  %s" % ("arm", "steps", "wall h", "units", "description"))
    for a in run.arms:
        print("%-9s %6d %7.2f %8.1f  %s" % (a["arm_id"], a["steps"], a["wall_h"],
                                            a["wall_h"] * b["units_per_hour"], a["desc"]))
    total_h = sum(a["wall_h"] for a in run.arms)
    print("")
    print("GRID TOTAL : %.2f h  (%.0f%% of the %.1f h ceiling)"
          % (total_h, 100 * total_h / b["total_budget_hours"], b["total_budget_hours"]))
    print("COLAB UNITS: %.0f of %d  (%.0f%%)"
          % (total_h * b["units_per_hour"], b["unit_balance"],
             100 * total_h * b["units_per_hour"] / b["unit_balance"]))
    print("sessions   : ~%.1f at %.1f h each (the driver resumes across sessions)"
          % (total_h / b["session_hours"], b["session_hours"]))
    warns = check_config(cfg, run.arms)
    for w in warns:
        print("")
        print("!! %s" % w)
    if cfg["budget"]["sec_per_step"] == 12.0:
        print("")
        print("!! sec_per_step is still the PLACEHOLDER (12.0). Run probe() first:")
        print("   a wrong s/step does not error, it silently rescales the token")
        print("   budget -- the one thing this design must not do.")
    return run.arms


# =====================================================================
# environment gates  (imports torch/transformers internally)
# =====================================================================
def verify_env(run=None, stop_on_hard_fail=True):
    import importlib.util
    import transformers

    def show(label, val):
        print("  %-28s %s" % (label, val))

    print("=== Qwen3.5 gates ===")
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
    show("transformers", "%s   (qwen3_5 needs >= 5.2.0, < 5.18)" % transformers.__version__)
    hard_fail = False
    if tv < (5, 2):
        print("  !! STOP: transformers < 5.2 cannot instantiate the Qwen3.5 arch at all.")
        hard_fail = True
    elif tv >= (5, 18):
        print("  !! transformers is newer than Unsloth's tested ceiling (5.17.0).")
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available
        show("fla fast kernels", is_flash_linear_attention_available())
    except Exception as e:
        print("  fla probe failed:", e)
    try:
        from unsloth.models.loader import FORCE_FLOAT32
        show("FORCE_FLOAT32 qwen3_5", any("qwen3_5" in str(m) for m in FORCE_FLOAT32))
    except Exception as e:
        print("  FORCE_FLOAT32 unreadable:", e)
    print("  bf16 is MANDATORY on this arch: GDN layers give NaN grad norms in fp16.")
    print("  Do not switch to QLoRA: Unsloth advises against 4-bit for Qwen3.5.")

    print("=== unsloth engine flags ===")
    try:
        from unsloth.models import _utils as U
        for flag in ("HAS_FLASH_ATTENTION", "HAS_CUT_CROSS_ENTROPY", "SUPPORTS_BFLOAT16"):
            show(flag, getattr(U, flag, "n/a"))
    except Exception as e:
        print("  unsloth.models._utils unreadable:", e)

    print("=== packages ===")
    for pkg in (run.cfg["env"]["check_packages"] if run else
                ("flash_attn", "cut_cross_entropy", "fla", "xformers", "triton", "bitsandbytes")):
        try:
            ok = importlib.util.find_spec(pkg) is not None
        except Exception:
            ok = False
        show(pkg, ok)

    print("")
    print("Read this:")
    print("  HAS_CUT_CROSS_ENTROPY should be True. With a 248K vocab it computes the")
    print("  loss WITHOUT materialising [batch, seq, vocab] logits.")
    print("  HAS_FLASH_ATTENTION is NOT required here: Qwen3.5 is a hybrid, only 8 of")
    print("  its 32 layers are full attention, and Unsloth's own kernels are the fast")
    print("  path it documents for this model. grep the model-load output for")
    print('  "flash_attention_2 is not supported" and move on if you see it.')
    print("  First steps are slow by design: the Mamba/GDN Triton kernels compile at")
    print("  load. The probe discards those steps before timing.")
    if hard_fail and stop_on_hard_fail:
        raise RuntimeError("transformers too old for qwen3_5 -- re-run the install cell")
    return not hard_fail


# =====================================================================
# tokenizer / data
# =====================================================================
def get_text_tokenizer(tok):
    """FastModel returns a multimodal processor; the trainer wants the text one."""
    t = getattr(tok, "tokenizer", tok)
    t.padding_side = "right"
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    assert t.eos_token_id is not None
    return t


def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def _clean(v):
    return v.strip() if isinstance(v, str) else ""


def _fmt_medrag(ex):
    t = _clean(ex.get("contents"))
    if not t:
        t = "\n\n".join(p for p in (_clean(ex.get("title")), _clean(ex.get("content"))) if p)
    return {"text": t}


def _fmt_wikidoc(ex):
    q, a = _clean(ex.get("input")), _clean(ex.get("output"))
    return {"text": ("Question: " + q + "\n\nAnswer: " + a) if (q and a) else a}


def _fmt_patients(ex):
    return {"text": _clean(ex.get("description"))}


def _fmt_edu(ex):
    return {"text": _clean(ex.get("text"))}


def _fmt_math(ex):
    q, a = _clean(ex.get("question")), _clean(ex.get("answer"))
    return {"text": ("Question: " + q + "\n\nSolution: " + a) if (q and a) else ""}


FORMATTERS = {"medrag": _fmt_medrag, "patients": _fmt_patients, "wikidoc": _fmt_wikidoc,
              "edu": _fmt_edu, "math": _fmt_math}


def _prepare(ds, fmt, seed, shuffle_buffer):
    cols = ds.column_names
    if cols is None:
        first = next(iter(ds), None)
        if first is None:
            raise ValueError("empty source dataset")
        cols = list(first.keys())
    return (ds.map(fmt, remove_columns=cols)
              .filter(lambda ex: bool(ex["text"].strip()))
              .shuffle(seed=seed, buffer_size=shuffle_buffer))


def build_data(run, verbose=True):
    """ONE streaming mix + per-source holdout, built once and reused by every arm
    so all arms see the same data in the same order."""
    from datasets import load_dataset, Dataset, interleave_datasets

    d = run.cfg["data"]
    n_eval = run.cfg["overheads"]["eval_per_source"]
    streams, eval_by_source = [], {}
    for k, src in enumerate(d["sources"]):
        name, path, sub = src["name"], src["path"], src.get("config")
        ds = (load_dataset(path, sub, split="train", streaming=True)
              if sub else load_dataset(path, split="train", streaming=True))
        st = _prepare(ds, FORMATTERS[src["formatter"]], d["seed"] + k, d["shuffle_buffer"])

        held = list(st.take(n_eval))
        if len(held) != n_eval:
            raise ValueError("%s: wanted %d holdout rows, got %d" % (name, n_eval, len(held)))

        key = slug(name)
        eval_by_source[key] = Dataset.from_list(held)
        streams.append(st.skip(n_eval))
        if verbose:
            print("%-22s %-24s held out %d" % (name, key, len(held)))

    stream = interleave_datasets(streams, probabilities=d["mix"], seed=d["seed"],
                                 stopping_strategy="all_exhausted")
    assert set(d["guard_sources"]) <= set(eval_by_source), "guard sources missing from eval"
    if verbose:
        print("")
        print("mix        :", dict(zip([slug(s["name"]) for s in d["sources"]], d["mix"])))
        print("eval keys  :", list(eval_by_source))
        print("domain     :", d["domain_sources"])
        print("guards     :", d["guard_sources"])
    run.streams, run.eval_by_source = stream, eval_by_source
    return stream, eval_by_source


# =====================================================================
# model
# =====================================================================
def _trainable_weight(root, key):
    """The *trainable* weight for `key`, plus every name it appears under.

    Naming is not guaranteed across PEFT versions. With `modules_to_save` the
    module is replaced by a ModulesToSaveWrapper, so `...embed_tokens` resolves
    to the wrapper; PEFT's `AuxiliaryTrainingWrapper._getattr_wrapped` forwards
    `.weight` to the ACTIVE COPY, but that forwarding is an implementation
    detail. The frozen original also survives at
    `...embed_tokens.modules_to_save.original_module`. So match `key` anywhere
    in the dotted path and take the trainable one: the frozen original must
    never win, or the assertions below would lie.
    """
    cands = [(n, getattr(m, "weight", None)) for n, m in root.named_modules()
             if key in n.split(".")]
    cands = [(n, w) for n, w in cands if w is not None]
    for n, w in cands:
        if w.requires_grad:
            return n, w, [c[0] for c in cands]
    return None, None, [c[0] for c in cands]



def build_model(run, cfg):
    """FRESH base + FRESH adapters for EVERY arm.

    Reusing an adapted model across arms silently contaminates every arm after
    the first, so this is called per arm, never once.
    """
    import gc
    import torch
    from unsloth import FastModel

    gc.collect()
    torch.cuda.empty_cache()

    lora = run.cfg["lora"]
    want_ms = list(cfg.get("modules_to_save", lora["default_modules_to_save"]) or [])
    tying = bool(cfg.get("ensure_weight_tying", False))

    model, tok = FastModel.from_pretrained(
        model_name=run.cfg["model_id"],
        max_seq_length=cfg["max_length"],
        dtype=torch.bfloat16,          # mandatory on qwen3_5: fp16 -> NaN grads
        load_in_4bit=False,            # no QLoRA on Qwen3.5 (unsloth docs)
        load_in_8bit=False,
        full_finetuning=False,
        offload_embedding=False,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.use_cache = False

    peft_kw = dict(
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        target_modules=list(lora["target_modules"]),
        # embed/head can only be trained through modules_to_save on this path:
        # finetune_vision_layers=False enables scoping, and the finetune_*
        # filter silently drops them from target_modules.
        # On a TIED model never wrap only one of the pair: the unwrapped one
        # keeps pointing at the frozen original copy and silently stops moving.
        # `None` (the ADONLY arm) leaves both frozen and trains the adapter only.
        modules_to_save=(want_ms or None),
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=lora["bias"],
        use_dora=cfg["use_dora"],
        use_gradient_checkpointing=lora["gradient_checkpointing"],
        random_state=cfg["seed"],
        use_rslora=lora["use_rslora"],   # conflicts with DoRA
        loftq_config=None,
    )
    if tying:
        # PEFT's own opt-in: on a tied base it keeps ONE trainable
        # embedding/head copy and wires lm_head to it (peft#2777), instead of
        # deep-copying both and losing the tie.
        peft_kw["ensure_weight_tying"] = True
    try:
        model = FastModel.get_peft_model(model, **peft_kw)
    except TypeError as e:
        if not tying:
            raise
        raise TypeError(
            "lora.ensure_weight_tying is set, but this Unsloth build does not"
            " forward it to LoraConfig (%s). Either upgrade unsloth or set"
            " lora.ensure_weight_tying back to false -- silently training"
            " WITHOUT the tie is the failure this knob exists to avoid." % e)

    # ---- hard assertions: these are the silent-failure modes ----
    # PEFT rewrites `modules_to_save` IN PLACE, and on a TIED base it also
    # populates `modules_to_tie`, so the stored value is not the value we
    # passed: a leaf key comes back as the full dotted path PEFT resolved
    # (Qwen3.5 nests its text tower, so `embed_tokens` becomes
    # `model.language_model.embed_tokens`) and a module PEFT folds into the
    # tie moves to `modules_to_tie` and disappears from here. Compare leaf
    # NAMES, report the difference, and let the trainability checks below --
    # which look at the actual tensors -- decide what really happened.
    def _leaf(names):
        return set(k.split(".")[-1] for k in names if k)

    tied_names = set()
    for aname, pc in model.peft_config.items():
        assert pc.use_dora == cfg["use_dora"], "adapter %s: use_dora mismatch" % aname
        assert pc.r == cfg["r"], "adapter %s: rank mismatch" % aname
        got = set(pc.modules_to_save or [])
        tied_names |= set(getattr(pc, "modules_to_tie", None) or [])
        absent = _leaf(want_ms) - _leaf(got)
        if got != set(want_ms) or absent:
            print("  modules_to_save: requested %s -> PEFT stored %s%s"
                  % (want_ms, sorted(got),
                     ("  (moved to modules_to_tie: %s)" % sorted(tied_names))
                     if tied_names else
                     ("  (absent: %s)" % sorted(absent)) if absent else ""))
        assert not (_leaf(got) - _leaf(want_ms)), (
            "adapter %s: PEFT stored modules we never asked for: %s (requested %s)"
            % (aname, sorted(_leaf(got) - _leaf(want_ms)), want_ms))
    assert not getattr(model, "is_loaded_in_4bit", False)
    if tying and not tied_names:
        print("  WARNING: lora.ensure_weight_tying was set but PEFT reported no"
              " tied modules -- the tie is BROKEN, same as the default. Check"
              " the peft version before trusting these arms.")


    n_emb, w_emb, cand_emb = _trainable_weight(model, "embed_tokens")
    n_head, w_head, cand_head = _trainable_weight(model, "lm_head")
    print("  base tie_word_embeddings: %s" % getattr(model.config, "tie_word_embeddings", None))
    if cand_emb:
        print("  embed_tokens modules: %s" % cand_emb)
    if cand_head:
        print("  lm_head modules     : %s" % cand_head)
    if want_ms:
        if tying and tied_names:
            # ensure_weight_tying: lm_head is wired to the embed copy's tensor,
            # so there is one trainable tensor, not two.
            assert w_emb is not None and w_emb.requires_grad, (
                "embed_tokens not trainable -- the frozen original won")
            assert w_head is w_emb, (
                "ensure_weight_tying was applied but the embedding and head do"
                " not share a tensor -- the head would drift from the embedding")
            print("  embed_tokens/lm_head: ONE trainable tensor, TIE PRESERVED")
        else:
            # Tied base + modules_to_save: PEFT deep-copies BOTH modules, so the
            # tie is gone and the *trainable copies* must be the ones we found.
            assert w_emb is not None and w_emb.requires_grad, (
                "embed_tokens not trainable -- the frozen original won")
            assert w_head is not None and w_head.requires_grad, (
                "lm_head not trainable -- the frozen original won")
            assert w_emb is not w_head, (
                "modules_to_save did not untie embed_tokens/lm_head: the head still"
                " shares the embedding's tensor%s. PEFT is keeping the base tie"
                " (tie_word_embeddings=True) instead of deep-copying both, so there"
                " is ONE trainable tensor, not two. Either set"
                " lora.ensure_weight_tying=true to declare that as the intent, or"
                " drop lm_head from modules_to_save."
                % (" (modules_to_tie=%s)" % sorted(tied_names) if tied_names else ""))
            print("  embed_tokens/lm_head: trainable copies, TIE BROKEN (expected)")

    else:
        # Adapter-only: nothing up there may be trainable, and the tie must be
        # intact.  A broken tie here means a wrapper we did not ask for.
        assert w_emb is None and w_head is None, (
            "embed/head trainable while modules_to_save is off -- stray wrapper")
        e_w = model.get_input_embeddings().weight
        o_m = model.get_output_embeddings()
        print("  embed_tokens/lm_head: FROZEN (modules_to_save off)")
        print("  embed_tokens/lm_head still one tensor: %s  (expected True here)"
              % (o_m is not None and o_m.weight is e_w))

    trainable = set(n for n, p in model.named_parameters() if p.requires_grad)
    n_ms = sum(1 for n in trainable if "modules_to_save" in n)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tie_kept = bool(want_ms) and w_head is w_emb
    print("  trainable tensors: %d  (modules_to_save: %d, %.2fB params)"
          % (len(trainable), n_ms, n_par / 1e9))
    if not want_ms:
        print("  adapter-only: embed/head frozen, so grads + 8-bit optimizer state")
        print("  drop from ~%.2f GB to ~%.2f GB and a checkpoint is adapter-sized"
              % ((1.271e9 + n_par) * 4 / 1e9, n_par * 4 / 1e9))
        print("  (~%.2f GB)." % (n_par * 2 / 1e9))
    elif tie_kept:
        print("  the head shares the embedding tensor, so the pair moves as one")
        print("  (what pretraining did) and a checkpoint stays ~%.2f GB instead of"
              " ~%.2f GB." % (248320 * 2560 * 2 / 1e9, 2 * 248320 * 2560 * 2 / 1e9))
    else:
        print("  modules_to_save overwrote the tied embedding/head pair: the head no")
        print("  longer follows the embedding for free, and adapter checkpoints cost")
        print("  ~%.2f GB instead of ~%.2f GB."
              % (2 * 248320 * 2560 * 2 / 1e9, 248320 * 2560 * 2 / 1e9))


    return model, tok, get_text_tokenizer(tok)


# =====================================================================
# packing / collator / probes / trainer
# =====================================================================
def packed_stream_from(source, block_length, tok):
    """Fixed-length token blocks, so packing happens in OUR generator and TRL's
    packer stays switched off."""
    from datasets import IterableDataset

    def gen():
        buf = []
        eos = tok.eos_token_id
        for ex in source:
            s = ex.get("text")
            if not isinstance(s, str) or not s.strip():
                continue
            ids = tok(text=s, add_special_tokens=False, truncation=False,
                      return_attention_mask=False)["input_ids"]
            if not ids:
                continue
            if ids[-1] != eos:
                ids.append(eos)
            buf.extend(ids)
            off = 0
            while len(buf) - off >= block_length:
                yield {"input_ids": buf[off:off + block_length]}
                off += block_length
            if off:
                buf = buf[off:]
        if buf:
            yield {"input_ids": buf}

    return IterableDataset.from_generator(gen)


def make_collator(block_length, tok):
    def collate(examples):
        if "input_ids" in examples[0]:
            feats = [{"input_ids": e["input_ids"],
                      "attention_mask": [1] * len(e["input_ids"])} for e in examples]
            batch = tok.pad(feats, padding=True, return_tensors="pt")
        else:
            batch = tok(text=[e["text"] for e in examples], padding=True,
                        truncation=True, max_length=block_length, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
    return collate


def make_probes(model, cfg, tokens_per_step):
    """Callback: optimizer LR groups + throughput + peak VRAM + grad flow.

    Defined inside a function so this module imports without transformers.
    """
    import time as _time
    import torch
    from transformers import TrainerCallback

    class Probes(TrainerCallback):
        def __init__(self):
            self.cfg = cfg
            self.tok = tokens_per_step
            self.t0 = None
            self.watched = [(n, p) for n, p in model.named_parameters()
                            if p.requires_grad and ("embed_tokens" in n or "lm_head" in n)]
            self.grad_every = max(1, cfg["steps"] // 4)

        def on_train_begin(self, args, state, control, **kw):
            self.t0 = _time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            opt = kw.get("optimizer")
            if opt is not None:
                lrs = set()
                print("  [%s] optimizer param groups:" % self.cfg["arm_id"])
                for i, g in enumerate(opt.param_groups):
                    lrs.add(round(g["lr"], 12))
                    print("      group[%d] lr=%g n_params=%d" % (i, g["lr"], len(g["params"])))
                if len(lrs) < 2:
                    print("      !! single LR -- embedding_learning_rate is a NO-OP")

        def on_log(self, args, state, control, logs=None, **kw):
            if not state.global_step or self.t0 is None:
                return
            el = _time.perf_counter() - self.t0
            sps = el / state.global_step
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            eta = (self.cfg["steps"] - state.global_step) * sps / 60
            print("  [%s] step %4d/%d | %5.2f s/step | %7.0f tok/s | peak %5.2f GiB | ETA %5.1f min"
                  % (self.cfg["arm_id"], state.global_step, self.cfg["steps"],
                     sps, self.tok / sps, peak, eta))

        def on_step_end(self, args, state, control, **kw):
            if state.global_step % self.grad_every:
                return
            parts = []
            for n, p in self.watched[:3]:
                g = p.grad
                label = "embed_tokens" if "embed_tokens" in n else "lm_head"
                parts.append("%s=%s" % (label, "None  <-- NOT TRAINING" if g is None
                                        else "%.2e" % g.norm().item()))
            print("  [%s] gradflow @%d: %s" % (self.cfg["arm_id"], state.global_step,
                                               ", ".join(parts)))

    return Probes()


def make_trainer(run, model, tok, text_tok, cfg, packed, eval_ds):
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    t = run.cfg["trainer"]
    b = run.cfg["budget"]
    steps = cfg["steps"]
    eval_st = max(5, steps // run.cfg["overheads"]["evals_per_run"])
    save_st = eval_st
    emb_lr = cfg["learning_rate"] * cfg["embedding_lr_ratio"]

    # Short screening arms: keep checkpoints OFF Drive. modules_to_save writes
    # the full embed_tokens + lm_head (~2.5 GB per checkpoint on the 4B; the
    # ADONLY arm has neither, so its checkpoints are adapter-sized, ~0.12 GB) and
    # best-checkpoint selection is meaningless over a couple of hundred steps.
    # run.arm_saves(cfg) opts an individual arm in - required for any run that
    # will outlive one session, since without a checkpoint there is nothing to
    # resume from.
    saves = run.arm_saves(cfg)
    use_best = saves and steps >= t["best_model_min_steps"]
    if saves:
        out_dir, save_strategy = run.arm_dir(cfg), "steps"
    else:
        out_dir, save_strategy = "/content/ckpt/%s" % cfg["arm_id"], "no"

    kw = dict(
        output_dir=out_dir,
        report_to=t["report_to"],
        run_name=cfg["arm_id"],
        dataset_text_field="text",
        max_length=cfg["max_length"],
        max_steps=steps,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        fp16=False, bf16=True,             # fp16 is not an option on qwen3_5
        logging_steps=t["logging_steps"],
        eval_strategy="steps", eval_steps=eval_st,
        save_strategy=save_strategy, save_steps=save_st,
        save_total_limit=t["save_total_limit"],
        optim=t["optim"],
        gradient_checkpointing=True,
        max_grad_norm=t["max_grad_norm"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        lr_scheduler_kwargs={"num_cycles": t["lr_scheduler_num_cycles"]},
        learning_rate=cfg["learning_rate"],
        embedding_learning_rate=emb_lr,
        seed=cfg["seed"],
    )
    if use_best:
        # with a dict eval_dataset there is NO plain eval_loss -- HF prefixes
        # every key with the dataset name, so name exactly one dataset here
        kw.update(load_best_model_at_end=True,
                  metric_for_best_model="eval_%s_loss" % run.cfg["data"]["checkpoint_metric_source"],
                  greater_is_better=False)

    args = UnslothTrainingArguments(**kw)
    args.packing = False
    args.eval_packing = False
    args.padding_free = False
    args.dataset_kwargs = dict(args.dataset_kwargs or {}, skip_prepare_dataset=True)
    args.remove_unused_columns = False
    args.dataloader_num_workers = 0

    trainer = UnslothTrainer(
        model=model,
        args=args,
        train_dataset=packed,
        eval_dataset=eval_ds,
        processing_class=text_tok,
        data_collator=make_collator(cfg["max_length"], text_tok),
    )
    trainer.add_callback(make_probes(model, cfg,
                                     t["per_device_train_batch_size"]
                                     * t["gradient_accumulation_steps"]
                                     * cfg["max_length"]))
    return trainer, args


# =====================================================================
# probe
# =====================================================================
def probe(run):
    """Measure s/step on THIS box before committing the grid (~5 min).

    Skipping it does not error: a stale s/step silently rescales the token
    budget, and the token budget is the only thing that makes arms comparable.
    Runs the baseline for a few steps with eval/save disabled and discards the
    warmup (the Mamba/GDN Triton kernels compile at load).  It does NOT touch
    the config -- it reports the value to put in budget.sec_per_step.
    """
    import gc
    import torch
    from transformers import TrainerCallback

    cfg = dict(run.arms[0])                      # arm B, same config the grid uses
    cfg["steps"] = run.cfg["probe"]["steps"]
    warmup = run.cfg["probe"]["warmup"]
    b = run.cfg["budget"]

    class _Ticker(TrainerCallback):
        def __init__(self):
            self.t = []

        def on_step_end(self, args, state, control, **kw):
            self.t.append(time.perf_counter())

    model, tok, text_tok = build_model(run, cfg)
    packed = packed_stream_from(run.require_data()[0], cfg["max_length"], text_tok)
    trainer, args = make_trainer(run, model, tok, text_tok, cfg, packed, run.eval_by_source)
    args.eval_steps = 10 ** 9                    # we want pure training time
    args.save_steps = 10 ** 9
    tick = _Ticker()
    trainer.add_callback(tick)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    dts = [y - x for x, y in zip(tick.t, tick.t[1:])][warmup:]
    sec_per_step = (sum(dts) / len(dts)) if dts else wall / run.cfg["probe"]["steps"]
    tok_per_s = b["effective_batch"] * cfg["max_length"] / sec_per_step
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    d = run.derived
    at_probe = max(1, int(d["train_h_per_run"] * 3600 / sec_per_step))
    at_probe_tok = at_probe * b["effective_batch"] * cfg["max_length"]

    print("")
    print("=" * 76)
    print("PROBE RESULT  (arm %s, len=%d, eff batch=%d)"
          % (cfg["arm_id"], cfg["max_length"], b["effective_batch"]))
    print("  steps timed     : %d (+%d warmup discarded, total wall %.1f min)"
          % (len(dts), warmup, wall / 60))
    print("  SEC_PER_STEP    : %.1f        <-- set budget.sec_per_step, re-run the budget cell"
          % sec_per_step)
    print("  throughput      : %.0f tok/s  (%.2fM tok/h)" % (tok_per_s, tok_per_s * 3600 / 1e6))
    print("  peak VRAM       : %.1f GiB of 96    (headroom buys batch/context, not steps/h)"
          % peak)
    print("  at wall_h_per_arm=%.1f -> %d steps/arm = %.1fM tokens/arm (%.5f tok/param)"
          % (b["wall_h_per_arm"], at_probe, at_probe_tok / 1e6, at_probe_tok / d["text_params"]))
    print("  grid as configured (sec_per_step=%.1f): %d steps/arm, %.1fM tokens/arm, %.1f h total"
          % (b["sec_per_step"], d["steps_per_arm"], d["tokens_per_arm"] / 1e6,
             sum(a["wall_h"] for a in run.arms)))
    print("=" * 76)

    del model, trainer, tick
    gc.collect()
    torch.cuda.empty_cache()
    return sec_per_step


# =====================================================================
# run one arm
# =====================================================================
def run_one_arm(run, cfg):
    """Train one arm, evaluate, record. Fresh base + fresh adapters, always."""
    import gc
    import torch

    aid = cfg["arm_id"]
    print("")
    print("=" * 76)
    print("ARM %s : %s" % (aid, cfg["desc"]))
    print("  r=%d alpha=%d doRA=%s LR=%g embLR=%g drop=%g len=%d steps=%d"
          % (cfg["r"], cfg["lora_alpha"], cfg["use_dora"], cfg["learning_rate"],
             cfg["learning_rate"] * cfg["embedding_lr_ratio"],
             cfg["lora_dropout"], cfg["max_length"], cfg["steps"]))
    print("=" * 76)

    os.makedirs(run.arm_dir(cfg), exist_ok=True)      # status.json lives on Drive
    t_start = time.time()

    model, tok, text_tok = build_model(run, cfg)
    packed = packed_stream_from(run.require_data(verbose=False)[0], cfg["max_length"], text_tok)
    trainer, args = make_trainer(run, model, tok, text_tok, cfg, packed, run.eval_by_source)

    # step-0 held-out losses = THIS arm's own forgetting baseline
    with torch.no_grad():
        base = trainer.evaluate()
    base_losses = dict((k, float(v)) for k, v in base.items() if k.endswith("_loss"))
    print("  step-0 held-out:")
    for k in sorted(base_losses):
        print("     %-46s %.4f" % (k, base_losses[k]))

    trainer.train(resume_from_checkpoint=bool(run.arm_saves(cfg)))
    wall_h = (time.time() - t_start) / 3600

    final = trainer.evaluate()
    losses = dict((k, float(v)) for k, v in final.items() if k.endswith("_loss"))

    if run.arm_saves(cfg):
        out = "%s/final_adapter" % run.arm_dir(cfg)
        trainer.model.save_pretrained(out)
        tok.save_pretrained(out)
        print("  saved adapter -> %s" % out)

    data = run.cfg["data"]
    b = run.cfg["budget"]
    dom = [losses["eval_%s_loss" % s] for s in data["domain_sources"]
           if "eval_%s_loss" % s in losses]

    guards = {}
    for s in data["guard_sources"]:
        f = losses.get("eval_%s_loss" % s)
        bs = base_losses.get("eval_%s_loss" % s)
        guards[s] = dict(final=f, start=bs,
                         delta=(None if (f is None or bs is None) else f - bs))

    steps_run = int(trainer.state.global_step)
    tokens_run = steps_run * b["effective_batch"] * cfg["max_length"]
    status = dict(
        arm_id=aid,
        phase=cfg.get("phase", "grid"),
        desc=cfg["desc"],
        cfg=cfg,
        done=True,
        wall_hours=wall_h,
        steps_run=steps_run,
        tokens_run=tokens_run,
        tok_per_param=tokens_run / b["text_params"],
        sec_per_step_amortised=(None if not steps_run else wall_h * 3600 / steps_run),
        peak_vram_gib=float(torch.cuda.max_memory_allocated() / 1024 ** 3),
        best_metric=(None if getattr(trainer.state, "best_metric", None) is None
                     else float(trainer.state.best_metric)),
        final_losses=losses,
        base_losses=base_losses,
        domain_mean_final=(float(sum(dom) / len(dom)) if dom else None),
        guards=guards,
        # provenance: which config produced this row
        cfg_hash=run.cfg_hash,
        config_path=run.config_path,
        lib_mtime=_lib_mtime(),
    )
    with open(run.status_path(cfg), "w") as fh:
        json.dump(status, fh, indent=2, default=float)

    row = {}
    for k, v in status.items():
        if k == "cfg":
            continue
        row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    run.append_result(row)

    print("")
    print("  arm %s DONE in %.2f h | %d steps | %.1fM tok (%.5f tok/param) | domain_mean=%s"
          % (aid, wall_h, status["steps_run"], status["tokens_run"] / 1e6,
             status["tok_per_param"],
             "n/a" if status["domain_mean_final"] is None
             else "%.4f" % status["domain_mean_final"]))
    for s, v in guards.items():
        print("     guard %-12s %.4f -> %.4f (delta %+.4f)"
              % (s, v["start"], v["final"], v["delta"]))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return status


# =====================================================================
# driver
# =====================================================================
def run_grid(run):
    """Run pending arms until the grid budget or the session budget stops us.

    Safe to interrupt and re-run: completed arms are skipped, so a fresh
    session just continues the grid.
    """
    run.ensure_root()
    run.write_run_config()
    run.require_data(verbose=False)
    b = run.cfg["budget"]
    t_session = time.time()

    for cfg in run.pending():
        if run.arm_done(cfg):
            print("skip %s (already done)" % cfg["arm_id"])
            continue

        done_h = sum(a["wall_h"] for a in run.arms if run.arm_done(a))
        spent_h = (time.time() - t_session) / 3600

        if done_h + cfg["wall_h"] > b["total_budget_hours"]:
            print("")
            print("stopping: grid budget %.1f h would be exceeded (%.2f h done + %.2f h for %s)"
                  % (b["total_budget_hours"], done_h, cfg["wall_h"], cfg["arm_id"]))
            break
        if spent_h + cfg["wall_h"] > b["session_hours"]:
            print("")
            print("stopping: %.2f h used this session; %s needs %.2f h (session cap %.1f h)"
                  % (spent_h, cfg["arm_id"], cfg["wall_h"], b["session_hours"]))
            print("Re-run this cell in a fresh session to continue the grid.")
            break

        run_one_arm(run, cfg)

    done_h = sum(a["wall_h"] for a in run.arms if run.arm_done(a))
    print("")
    print("session wall clock : %.2f h" % ((time.time() - t_session) / 3600))
    print("grid progress      : %.2f / %.1f h  (%.0f%%)"
          % (done_h, b["total_budget_hours"],
             100 * done_h / b["total_budget_hours"]))
    print("arms done          : %s" % (run.done_ids() or "(none)"))
    return run.done_ids()


# =====================================================================
# the keep/throw ruler
# =====================================================================
RULER_COLS = ["arm", "desc", "domain_mean", "retention_delta", "verdict", "wall_h"]


def _ruler_rows(run):
    data = run.cfg["data"]
    tol = run.cfg["ruler"]["retention_tol"]
    rows = []
    for cfg in run.arms:
        s = run.arm_status(cfg)
        if not s or not s.get("done"):
            continue
        fin, base = s["final_losses"], s["base_losses"]
        r = dict(arm=s["arm_id"], desc=s["desc"],
                 domain_mean=s["domain_mean_final"], wall_h=s["wall_hours"])
        for src in data["domain_sources"] + data["guard_sources"]:
            r[src] = fin.get("eval_%s_loss" % src)
        deltas = []
        for g in data["guard_sources"]:
            f = fin.get("eval_%s_loss" % g)
            bs = base.get("eval_%s_loss" % g)
            if f is not None and bs is not None:
                deltas.append(f - bs)
        r["retention_delta"] = max(deltas) if deltas else None
        r["verdict"] = ("VETO (forgetting)"
                        if (r["retention_delta"] is not None
                            and r["retention_delta"] > tol)
                        else "keep")
        rows.append(r)
    return rows


def ruler(run):
    """verdict = best held-out DOMAIN loss, VETOED by a retention regression.

    Each arm is compared against its OWN step-0 baseline, so the veto does not
    depend on how the arm was configured.
    """
    data = run.cfg["data"]
    rows = _ruler_rows(run)
    cols = RULER_COLS + data["domain_sources"] + data["guard_sources"]
    written = run.ranking_csv
    if not rows:
        print("no completed arms yet")
        return None

    keys = list(rows[0].keys())
    with open(written, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ranked = sorted(rows, key=lambda r: (r["domain_mean"] is None, r["domain_mean"]))
    try:
        import pandas as pd
        df = pd.DataFrame(ranked)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 50)
        print(df[cols].to_string(index=False, float_format=lambda x: "%.4f" % x))
    except Exception:
        # no pandas? print the same table with the stdlib
        print("  ".join("%-10s" % c[:10] for c in cols[:6]) + "  " + "  ".join(
            "%-12s" % c[:12] for c in cols[6:]))
        for r in ranked:
            print("  ".join("%-10s" % ("%.4f" % r[c] if isinstance(r[c], float) else str(r[c])[:10])
                            for c in cols[:6]) + "  " + "  ".join(
                "%-12s" % ("%.4f" % r[c] if isinstance(r[c], float) else str(r[c])[:12])
                for c in cols[6:]))

    surv = [r for r in ranked if r["verdict"] == "keep"]
    if not surv:
        print("")
        print("!! every arm was vetoed -- nothing retained its general ability.")
        print("   Widen ruler.retention_tol or check the data mix / LR range.")
    print("")
    print("--- ranked survivors: lowest held-out domain loss first ---")
    for r in (surv or ranked)[:5]:
        print("    %-9s %-46s %8.4f  %+8.4f"
              % (r["arm"], r["desc"][:46], r["domain_mean"], r["retention_delta"]))
    print("")
    print("best arm: %s" % (surv or ranked)[0]["arm"])
    print("wrote %s" % written)
    return (surv or ranked)[0]["arm"]


# =====================================================================
# phase 0 (optional): the "does DAPT help" curve
# =====================================================================
def phase0(run):
    """Run ONLY the baseline at growing step counts and print held-out domain
    mean vs tokens: the bend in that curve answers "did the mix move this base",
    which the grid cannot answer (45M tokens = 0.010 tok/param; canonical DAPT
    is 1-7 tok/param, so this is a 1-3 step curve, not a from-scratch DAPT run).

    Cheaper alternative worth considering first: the same curve on
    Qwen3.5-0.8B-Base -- same architecture family and tokenizer, ~5.4x the
    tokens/param for the same wall clock, and the recipe ranking transfers.
    """
    ph = run.cfg["phase0"]
    if not ph.get("run", False):
        print("phase0.run is false in the config -- flip it to run this.")
        return None
    root = "%s/%s" % (os.path.dirname(run.root.rstrip("/")), ph["root_suffix"])
    out = []
    for s in ph["steps"]:
        c = dict(run.cfg["baseline"])
        c.update(arm_id="B%d" % s, desc="phase0 baseline @ %d steps" % s, steps=s,
                 root=root, phase="phase0", wall_h=wall_hours(run.cfg, s),
                 force_save=s >= ph["force_save_min_steps"])
        if run.arm_done(c):
            print("skip %s (already done)" % c["arm_id"])
        else:
            run_one_arm(run, c)
        out.append(c)

    print("")
    print("--- PHASE 0: held-out domain mean vs tokens (lower is better) ---")
    print("%8s %12s %12s %12s %14s" % ("steps", "tokens", "tok/param", "domain_mean",
                                       "max guard d"))
    for s in ph["steps"]:
        st = run.arm_status({"arm_id": "B%d" % s, "root": root})
        if not st or not st.get("done"):
            continue
        gd = [v["delta"] for v in st["guards"].values() if v["delta"] is not None]
        print("%8d %9.1fM %12.5f %12.4f %+14.4f"
              % (s, st["tokens_run"] / 1e6, st["tok_per_param"], st["domain_mean_final"],
                 max(gd) if gd else float("nan")))
    print("")
    print("Read it as: if domain_mean is still falling steeply at the last point,")
    print("the arm budget (not the arm choice) is the binding constraint.")
    return out


# =====================================================================
# validation (not part of the grid)
# =====================================================================
def verify_adapters(run, best):
    """Inspect a saved arm's state dict. modules_to_save writes the FULL
    embed_tokens + lm_head; on this tied base PEFT's wrapper unties the copies,
    so expect ~2.5 GB per checkpoint (2 x 636M params bf16). If `best` is ADONLY
    there is no modules_to_save in the state dict at all: the counts print 0 and
    the tie check prints False.
    """
    from safetensors.torch import load_file

    path = "%s/%s/final_adapter/adapter_model.safetensors" % (run.root, best)
    sd = load_file(path)
    print("adapter tensors     :", len(sd))
    print("modules_to_save     :", sum("modules_to_save" in k for k in sd))
    print("dora magnitude vecs :", sum("lora_magnitude_vector" in k for k in sd))
    e = [k for k in sd if "modules_to_save" in k and "embed_tokens" in k]
    h = [k for k in sd if "modules_to_save" in k and "lm_head" in k]
    print("tied in the saved adapter:", bool(e) and bool(h))
    if e and h:
        print("embed == head bitwise  :", bool((sd[e[0]] == sd[h[0]]).all()))
    return sd


# =====================================================================
# self-test: everything that can be checked without a GPU
# =====================================================================
def _lib_mtime():
    import datetime
    return datetime.datetime.fromtimestamp(os.path.getmtime(__file__)).strftime("%Y-%m-%d %H:%M:%S")


def selftest(config_path=None, overrides=None):
    """Pure-python checks. Fails loudly; prints PASS lines otherwise.

    Covers: config load + deep-merge of a variant, the derived budget, arm
    construction, the invariant checks, the status/results contract including
    header widening, the config hash, and the no-heavy-imports rule.
    """
    import shutil
    import tempfile

    ok = []
    cfg_path = config_path or _find_config(None)
    base = load_config(cfg_path)
    ok.append("config loads: %s" % cfg_path)

    # 1. deep merge: a variant changes one knob and nothing else
    variant = deep_merge(base, {"budget": {"wall_h_per_arm": 2.5}})
    assert variant["budget"]["wall_h_per_arm"] == 2.5
    assert variant["budget"]["total_budget_hours"] == base["budget"]["total_budget_hours"]
    assert variant["baseline"] == base["baseline"]
    ok.append("deep-merge: nested override touches one key only")

    # 2. derived budget is arithmetic, not magic
    d = derive(base)
    assert abs(d["overhead_h_per_run"]
               - (base["overheads"]["model_load_min"]
                  + (2 + base["overheads"]["evals_per_run"])
                  * base["overheads"]["eval_minutes"]) / 60) < 1e-12
    assert d["steps_per_arm"] == max(1, int(round(
        (base["budget"]["wall_h_per_arm"] - d["overhead_h_per_run"]) * 3600
        / base["budget"]["sec_per_step"])))
    assert d["tokens_per_arm"] == d["steps_per_arm"] * base["budget"]["effective_batch"] \
        * base["budget"]["seq_len"]
    ok.append("budget math: overhead %.4f h, %d steps, %.1fM tokens"
              % (d["overhead_h_per_run"], d["steps_per_arm"], d["tokens_per_arm"] / 1e6))

    # 3. arms: ids unique, overrides applied, alpha=2r on the baseline
    arms = build_arms(base)
    ids = [a["arm_id"] for a in arms]
    assert len(ids) == len(set(ids)), "duplicate arm ids: %s" % ids
    assert all(a["steps"] > 0 and a["wall_h"] > 0 for a in arms)
    for entry in base["grid"]:
        a = [x for x in arms if x["arm_id"] == entry["id"]][0]
        for k, v in (entry.get("overrides") or {}).items():
            assert a[k] == v, "%s: override %s not applied" % (entry["id"], k)
    ok.append("arms: %d built, overrides applied, ids unique (%s)" % (len(arms), ",".join(ids)))

    # 4. invariants
    warn = check_config(base, arms)
    ok.append("invariants: passed (%d warning(s): %s)" % (len(warn), warn or "none"))
    bad = deep_merge(base, {"data": {"mix": [0.5, 0.5] + [0.0] * (len(base["data"]["mix"]) - 2)}})
    try:
        check_config(bad)
        raise AssertionError("mix length mismatch was NOT caught")
    except AssertionError as e:
        if "mix" not in str(e):
            raise
    ok.append("invariants: a bad mix is rejected")

    # 5. status/results contract, including header widening
    tmp = tempfile.mkdtemp(prefix="dapt_selftest_")
    try:
        tcfg = deep_merge(base, {"ablation_root": tmp})
        run = Run(tcfg)
        arm = run.arms[0]
        run.append_result(dict(arm_id=arm["arm_id"], done=True, wall_hours=1.5))
        run.append_result(dict(arm_id="X", done=True, wall_hours=0.5, cfg_hash="deadbeef"))
        with open(run.results_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2 and "cfg_hash" in rows[1] and rows[0]["cfg_hash"] == ""
        ok.append("results.csv: header widened, old row padded, %d rows" % len(rows))

        st = dict(arm_id=arm["arm_id"], done=True, desc="x", cfg={}, wall_hours=1.0,
                  final_losses={"eval_a_loss": 1.0}, base_losses={"eval_a_loss": 1.2},
                  guards={}, domain_mean_final=1.0)
        os.makedirs(run.arm_dir(arm), exist_ok=True)
        with open(run.status_path(arm), "w") as fh:
            json.dump(st, fh)
        assert run.arm_done(arm) is True
        assert run.pending() == [a for a in run.arms if a["arm_id"] != arm["arm_id"]]
        ok.append("status.json: done-detection + pending list")

        run.write_run_config()
        assert os.path.exists(run.run_config_path)
        assert "_config_path" not in json.load(open(run.run_config_path))
        ok.append("run_config.json: written, private keys stripped")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 6. config hash: stable, and changes when the config changes
    h1, h2 = config_hash(base), config_hash(dict(base))
    h3 = config_hash(deep_merge(base, {"baseline": {"r": 64}}))
    assert h1 == h2 and h1 != h3 and len(h1) == 12
    ok.append("config hash: stable, distinct for a changed config (%s)" % h1)

    # 7. the module must import without torch/transformers/datasets
    heavy = [m for m in ("torch", "transformers", "unsloth", "datasets", "pandas")
             if m in sys.modules]
    assert not heavy, "module level import of %s -- the budget/ruler math must run" \
                      " without them" % heavy
    ok.append("imports: no heavy module pulled in at import time")

    # 8. wall_hours is monotone in steps (a sanity check on the overhead term)
    assert wall_hours(base, 100) < wall_hours(base, 200)
    ok.append("wall_hours: monotone in steps")

    for line in ok:
        print("PASS  %s" % line)
    print("\n%d checks passed" % len(ok))
    return ok


# =====================================================================
# CLI
# =====================================================================
def _main(argv):
    cmd = argv[1] if len(argv) > 1 else "budget"
    path = argv[2] if len(argv) > 2 else None
    over = argv[3] if len(argv) > 3 else None
    run = open_run(path, over)
    if cmd == "budget":
        print_budget(run)
    elif cmd == "selftest":
        selftest(path, over)
    elif cmd == "config":
        print(json.dumps({k: v for k, v in run.cfg.items() if not k.startswith("_")},
                         indent=2, sort_keys=True))
    elif cmd == "ruler":
        ruler(run)
    else:
        raise SystemExit("usage: dapt_lib.py [budget|selftest|config|ruler] [config] [overrides]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
