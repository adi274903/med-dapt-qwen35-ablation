"""Step 04 -- open the run.

`run` is the only piece of session state. It carries the config, the resolved
budget, the arm list and the data streams, so a kernel restart costs you one
re-run of this step rather than a re-typed control panel.

The results go to Drive, because that is what has to survive the session --
run.root comes from `ablation_root` in config.json, and the driver resumes off
the status.json files it finds there. The CODE comes from the clone (preamble
cell); re-running that cell git-pulls, and importlib.reload below makes a lib
edit land without a restart.

CFG_PATH / OVERRIDES live in the preamble cell because they say where the repo
is, not what the run does. Every other knob is in config.json.
"""
import importlib
import os

import dapt_lib

importlib.reload(dapt_lib)      # an edit to dapt_lib.py lands without a restart

run = dapt_lib.open_run(CFG_PATH, overrides=OVERRIDES)
os.makedirs(run.root, exist_ok=True)        # the ablation tree, from the config

print('model     :', run.cfg['model_id'])
print('config    :', run.config_path, ' (hash %s)' % run.cfg_hash)
print('overrides :', run.cfg.get('_overrides_path', 'none'))
print('root      :', run.root)
print('arms      :', ', '.join(a['arm_id'] for a in run.arms))
print('')
print('nothing below this step needs editing: change config.json instead.')
