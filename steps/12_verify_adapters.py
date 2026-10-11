"""Step 12 -- validate the winner's adapters.

Deliberately NOT part of the grid: this is validation, not ablation, and it needs
the arm to have been checkpointed (`saving.save_models: true`, or the arm id
listed in `saving.save_arms`).

`modules_to_save` writes the FULL embed_tokens + lm_head. MEASURED: on this stack
the pair stays TIED (PEFT moves lm_head into modules_to_tie and shares the
embedding's copy), so expect ~1.4 GB per checkpoint -- the pair once plus the
adapter -- not the ~2.5 GB two independent copies would cost. If `best` is
ADONLY there is no modules_to_save in the state dict at all: the counts print 0
and the tie check prints False.
"""
_ = dapt_lib.verify_adapters(run, best)
