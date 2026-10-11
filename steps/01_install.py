"""Step 01 -- install the runtime.

Run this BEFORE anything imports dapt_lib or torch. Colab's stock environment is
the wrong one: transformers 5.18.0 sits above Unsloth's tested ceiling, so the
qwen3_5 gates fail before the model ever loads -- that is exactly what
verify_env() reports when this step has not run.

Order matters. unsloth goes in FIRST so its own metadata pins
transformers/trl/peft/datasets, then the explicit line can only tighten.
Pins that are load-bearing for qwen3_5:
  transformers >= 5.2.0   (below this the qwen3_5 arch does not exist)
  transformers <= 5.17.0  (5.19.0 on PyPI is outside unsloth's range)
  datasets     <  5.0.0   (5.1.0 on PyPI is rejected by unsloth)
  trl          <= 1.13.0  (1.15.0 on PyPI is outside unsloth's range)
  peft         >= 0.18.0

The flash-attn wheel is the cu130 / torch 2.11 / cp313 build for the G4 image.
It installs LAST and non-fatally: if Colab's image moves, it is the most likely
thing to break, and that must not take the other installs down with it.
"""
import os
import subprocess
import sys


def _pip(*args, check=True):
    print('pip install %s' % ' '.join(args))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + list(args),
                   check=check)


_pip('-U', 'unsloth', 'unsloth_zoo')
_pip('-U', 'transformers>=5.2.0,<=5.17.0', 'peft>=0.18.0', 'trl>=0.18.2,<=1.13.0',
     'datasets>=3.4.1,<5.0.0', 'accelerate', 'bitsandbytes', 'ninja', 'packaging',
     'torchao')
# Unsloth vendors the flash-linear-attention kernels for the gated-deltanet
# layers; this install is belt-and-braces (and what the 27B run used). Step 05
# reports whether the fast kernels actually came up.
_pip('-U', 'flash-linear-attention')
_pip('cut-cross-entropy')
_pip('https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/'
     'download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp313-cp313-linux_x86_64.whl',
     check=False)
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
print('runtime installed -- step 05 gates it')
