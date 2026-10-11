"""Fake safetensors.torch: enough for verify_adapters to inspect a state dict."""
from torch import _Tensor


def load_file(path):
    t = lambda n: _Tensor(n)  # noqa: E731
    return {
        "m.embed_tokens.modules_to_save.default.weight": t(636_000_000),
        "m.lm_head.modules_to_save.default.weight": t(636_000_000),
        "m.layers.0.q_proj.lora_A.default.weight": t(102_400),
        "m.layers.0.q_proj.lora_magnitude_vector.default": t(512),
    }
