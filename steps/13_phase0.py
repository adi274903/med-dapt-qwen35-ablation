"""Step 13 -- Phase 0: the does-DAPT-help curve (optional).

The grid answers *which config ranks best*. It does not answer *did the medical
mix move this base*: a 2.99 h arm sees 45.6M tokens = 0.0105 tokens/param, and
canonical DAPT is 1-7 tok/param.

phase0() runs ONLY the baseline at growing step counts and prints held-out domain
mean vs tokens. 300 -> 8000 steps spans 9.8M -> 262M tokens (0.002 -> 0.060
tok/param) and ~1.4 -> 27.0 h of wall clock; the bend in that curve is the
answer, and it also tells you which grid budget is worth spending. Runs >= the
`saving`-forced step count checkpoint and can resume across sessions. It writes
to its own root, so it can never collide with the grid, and its rows carry
phase='phase0' in results.csv. Flip `phase0.run` to true in config.json.

Cheaper alternative worth considering first: the same curve on
Qwen3.5-0.8B-Base. Same architecture family, same tokenizer, ~5.4x the
tokens/param for the same wall clock, and the recipe ranking transfers.
"""
_ = dapt_lib.phase0(run)
