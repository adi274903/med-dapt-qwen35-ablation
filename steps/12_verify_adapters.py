"""Step 12 -- validate the winner's adapters.

Deliberately NOT part of the grid: this is validation, not ablation, and it needs
the arm to have been checkpointed (`saving.save_models: true`, or the arm id
listed in `saving.save_arms`).

`modules_to_save` writes the FULL embed_tokens + lm_head. On this tied base
PEFT's wrapper unties the copies, so expect ~2.5 GB per checkpoint. If `best` is
ADONLY there is no modules_to_save in the state dict at all: the counts print 0
and the tie check prints False.
"""
_ = dapt_lib.verify_adapters(run, best)
