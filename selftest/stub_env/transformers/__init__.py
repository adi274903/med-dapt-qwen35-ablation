"""Fake transformers namespace (window gate + TrainerCallback)."""
from . import utils  # noqa: F401

__version__ = "5.3.0"


class TrainerCallback:
    def __init__(self, *a, **k):
        pass

    def on_train_begin(self, *a, **k):
        pass

    def on_log(self, *a, **k):
        pass

    def on_step_end(self, *a, **k):
        pass

    def on_evaluate(self, *a, **k):
        pass

    def on_save(self, *a, **k):
        pass

    def on_train_end(self, *a, **k):
        pass
