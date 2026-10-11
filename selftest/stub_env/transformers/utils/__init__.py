"""Fake transformers.utils package."""
import types

__all__ = ["import_utils"]

import_utils = types.ModuleType("transformers.utils.import_utils")


def is_flash_linear_attention_available():
    return False


import_utils.is_flash_linear_attention_available = is_flash_linear_attention_available
