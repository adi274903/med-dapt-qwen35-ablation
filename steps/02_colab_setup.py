"""Step 02 -- HF credentials, Drive, fresh compile cache.

Drive holds the results tree, so it is mounted before anything writes to it --
and before step 03, which installs a wheel that lives in MyDrive. The HF token
comes from the environment (Colab secrets or a .env beside the clone); it is
never written into the notebook.

The compile cache is wiped every session: unsloth caches compiled kernels, and a
stale one is the classic "it worked yesterday" failure.
"""
import os
import shutil

import huggingface_hub
from dotenv import load_dotenv
from google.colab import drive

load_dotenv()
huggingface_hub.login(token=os.getenv('HF_TOKEN'))

drive.mount('/content/drive')

shutil.rmtree('/content/unsloth_compiled_cache', ignore_errors=True)
print('drive mounted -- the ablation root comes from config.json (step 04)')
