"""Fake unsloth: FastModel / trainers / arguments, plus a PEFT-faithful wrapper.

The point is the NAMING. PEFT's `modules_to_save` replaces the module with a
`ModulesToSaveWrapper`, so after `get_peft_model` the paths are

    m.embed_tokens                                 <- the wrapper
    m.embed_tokens.modules_to_save.default         <- the TRAINABLE deep copy
    m.embed_tokens.modules_to_save.original_module <- the frozen original

Set STUB_PEFT_FORWARD=1 to mimic PEFT versions whose wrapper forwards
`.weight` to the active copy; =0 for versions that do not. dapt_lib must reach
the trainable copy either way.
"""
import os
from types import SimpleNamespace

from torch import _Tensor

FORWARD = os.environ.get("STUB_PEFT_FORWARD", "0") == "1"


class _Original:
    """A module that only proves it was frozen."""

    def __init__(self, n):
        self.weight = _Tensor(n, rg=False)


class _Copy:
    """The deep copy PEFT makes trainable."""

    def __init__(self, n):
        self.weight = _Tensor(n, rg=True)


class _Wrapper:
    """PEFT ModulesToSaveWrapper: no weight of its own (unless it forwards)."""

    def __init__(self, original, copy):
        self.original_module = original
        self._copy = copy
        if FORWARD:                      # some PEFT versions forward .weight
            self.weight = copy.weight


class _FakeTokenizer:
    eos_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"
    eos_token_id = 248044
    pad_token_id = 248044
    padding_side = "right"

    def __call__(self, text=None, return_tensors=None, **kw):
        ids = [7, 8, 9, 10]
        if return_tensors:
            return SimpleNamespace(input_ids=[ids], attention_mask=[[1] * len(ids)])
        return SimpleNamespace(input_ids=ids, attention_mask=[1] * len(ids))

    def pad(self, feats, return_tensors=None, **kw):
        n = max(len(f["input_ids"]) for f in feats)
        ids = [list(f["input_ids"]) + [self.pad_token_id] * (n - len(f["input_ids"]))
               for f in feats]
        mask = [list(f["attention_mask"]) + [0] * (n - len(f["attention_mask"]))
                for f in feats]
        return SimpleNamespace(input_ids=_Tensor(sum(len(i) for i in ids)),
                               attention_mask=_Tensor(sum(len(m) for m in mask)))

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer_config.json"), "w") as fh:
            fh.write("{}")


class _FakeModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.config = SimpleNamespace(use_cache=True, tie_word_embeddings=True)
        self.is_loaded_in_4bit = False
        self.peft_config = {}
        self._mods = {"m.embed_tokens": _Original(636_000_000),
                      "m.lm_head": _Original(636_000_000)}
        for l in range(2):
            self._mods["m.layers.%d.q_proj" % l] = _Original(6_553_600)
        self._mods["m.lm_head"].weight = self._mods["m.embed_tokens"].weight  # tied
        self._marks = []

    # -- PEFT surface -----------------------------------------------------
    def get_input_embeddings(self):
        return self._mods["m.embed_tokens"]

    def get_output_embeddings(self):
        return self._mods["m.lm_head"]

    def named_modules(self):
        yield "", self
        for n, m in self._mods.items():
            yield n, m

    def named_parameters(self):
        seen = set()
        for n, m in self._mods.items():
            w = getattr(m, "weight", None)
            if w is None or id(w) in seen:
                continue          # torch dedupes shared tensors too
            seen.add(id(w))
            yield n, w

    def parameters(self):
        for _, w in self.named_parameters():
            yield w

    def adapt(self, **kw):
        """Stand in for FastModel.get_peft_model(model, **kw)."""
        mts = list(kw.get("modules_to_save") or [])
        tying = bool(kw.get("ensure_weight_tying"))
        if mts:
            emb_copy = _Copy(636_000_000)
            orig_emb = self._mods["m.embed_tokens"]
            orig_emb.weight.requires_grad = False
            self._mods["m.embed_tokens"] = _Wrapper(orig_emb, emb_copy)
            self._mods["m.embed_tokens.modules_to_save"] = SimpleNamespace()
            self._mods["m.embed_tokens.modules_to_save.default"] = emb_copy
            self._mods["m.embed_tokens.modules_to_save.original_module"] = orig_emb

            orig_head = self._mods["m.lm_head"]
            orig_head.weight.requires_grad = False
            # ensure_weight_tying: PEFT drops lm_head from modules_to_save and
            # wires it to the EMBED copy's tensor (peft#2777 fix).
            head_copy = emb_copy if tying else _Copy(636_000_000)
            self._mods["m.lm_head"] = _Wrapper(orig_head, head_copy)
            self._mods["m.lm_head.modules_to_save"] = SimpleNamespace()
            self._mods["m.lm_head.modules_to_save.default"] = head_copy
            self._mods["m.lm_head.modules_to_save.original_module"] = orig_head
        for l in range(2):
            self._mods["m.layers.%d.q_proj.lora_A.default" % l] = _Copy(6_553_600 // 64)
            self._mods["m.layers.%d.q_proj.lora_magnitude_vector.default" % l] = _Copy(2)
        pc = SimpleNamespace(use_dora=kw.get("use_dora"), r=kw.get("r"),
                             lora_alpha=kw.get("lora_alpha"),
                             target_modules=kw.get("target_modules"))
        if mts:
            pc.modules_to_save = (["embed_tokens"] if tying
                                  else [m.split(".")[-1] for m in mts])
            if tying:
                pc.modules_to_tie = ["lm_head.weight"]
        else:
            pc.modules_to_save = None
        self.peft_config = {"default": pc}
        if not mts:
            # nothing wrapped: the tie survives by construction
            self._mods["m.lm_head"].weight = self._mods["m.embed_tokens"].weight
        self._marks.append(kw)
        return self


    def save_pretrained(self, out, **kw):
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"stub")


def _losses(run_name, step, n_sources):
    """Deterministic pseudo-losses that differ per arm, so the ruler ranks."""
    seed = sum(ord(c) for c in str(run_name))
    jitter = (seed % 17) / 100.0
    domain = {}
    for i in range(n_sources):
        key = "s%d" % i
        # domain loss falls with steps
        domain[key] = 1.0 + jitter + 0.02 * i - 0.05 * min(step / 1000.0, 1.0)
    guard = {"FineWeb_Edu": 2.80 + jitter + 0.005 * min(step / 1000.0, 1.0),
             "Orca_Math": 1.50 + jitter}
    if "LR3e4" in str(run_name):        # one arm must trip the forgetting veto
        guard = {k: v + 0.09 for k, v in guard.items()}
    return domain, guard


class _FakeOptimizer:
    param_groups = [{"lr": 1e-5, "name": "embeddings", "params": [object()] * 2},
                    {"lr": 1e-4, "name": "others", "params": [object()] * 6}]



class _State:
    def __init__(self):
        self.global_step = 0
        self.best_metric = None


class _FakeTrainer:
    def __init__(self, model=None, args=None, train_dataset=None, eval_dataset=None,
                 processing_class=None, data_collator=None, **kw):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset or {}
        self.callbacks = []
        self.state = _State()
        self.saved = []

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def evaluate(self, *a, **k):
        # Trainer names the keys after the eval dataset dict keys, i.e.
        # "eval_MedRAG_textbooks_loss" -- exactly what dapt_lib looks up.
        if isinstance(self.eval_dataset, dict) and self.eval_dataset:
            keys = list(self.eval_dataset)
        else:
            keys = ["s%d" % i for i in range(len(self.eval_dataset or []))]
        n = min(self.state.global_step, 40)
        return {"eval_%s_loss" % key: 2.0 + 0.1 * i - 0.05 * (n / 40.0)
                for i, key in enumerate(keys)}

    def train(self, **kw):
        for cb in self.callbacks:
            cb.on_train_begin(self.args, self.state, None, optimizer=_FakeOptimizer())
        steps = int(getattr(self.args, "max_steps", 0) or 0)
        for i in range(1, steps + 1):
            self.state.global_step = i
            self.state.best_metric = 1.0 - i / (steps + 1.0)
            for _, w in self.model.named_parameters():
                if getattr(w, "requires_grad", False):
                    w.grad = _Tensor(0.001 * i)      # so gradflow can read a value
            for cb in self.callbacks:
                if self.args.logging_steps and i % self.args.logging_steps == 0:
                    cb.on_log(self.args, self.state, None, logs={"loss": 1.0 / i})
                cb.on_step_end(self.args, self.state, None)
        return SimpleNamespace(training_loss=0.42)


class FastModel:
    @staticmethod
    def from_pretrained(**kw):
        return _FakeModel(kw.get("model_name", "?")), _FakeTokenizer()

    @staticmethod
    def get_peft_model(model, **kw):
        return model.adapt(**kw)


class UnslothTrainingArguments:
    def __init__(self, **kw):
        self.dataset_kwargs = None
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return "UnslothTrainingArguments(run_name=%r, max_steps=%r)" % (
            getattr(self, "run_name", None), getattr(self, "max_steps", None))


def UnslothTrainer(**kw):
    return _FakeTrainer(**kw)


from . import models  # noqa: E402,F401
