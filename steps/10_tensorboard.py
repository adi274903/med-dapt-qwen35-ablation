"""Step 10 -- TensorBoard over run.root.

run.root is the Qwen3.5-4B tree (from `ablation_root` in config.json), so these
logs never mix with the old 27B tree. The logdir is interpolated here in Python
and handed to the magic as a finished string, so it cannot reach tensorboard as
a literal "{run.root}".
"""
get_ipython().run_line_magic('load_ext', 'tensorboard')
print('tensorboard --logdir %s' % run.root)
get_ipython().run_line_magic('tensorboard', '--logdir %s' % run.root)
