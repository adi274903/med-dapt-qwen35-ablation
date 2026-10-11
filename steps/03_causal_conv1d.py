"""Step 03 -- the prebuilt causal_conv1d wheel.

The GDN (gated-deltanet) layers call into causal_conv1d. Building it from source
costs ~1 h, so the cu130 / torch 2.11 / cp313 wheel is shipped in Drive instead.
--no-deps: the wheel must not drag its own torch in. Needs step 02 (Drive).
"""
import subprocess
import sys

WHEEL = '/content/drive/MyDrive/wheels/causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl'
print('pip install --no-deps %s' % WHEEL)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', WHEEL],
               check=True)
print('causal_conv1d installed')
