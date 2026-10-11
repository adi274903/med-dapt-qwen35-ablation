"""Step 08 -- measure s/step on THIS box before committing the grid (~5 min).

Skipping it does not error: a stale `sec_per_step` silently rescales the token
budget, and the token budget is the only thing that makes the arms comparable.

Paste the reported number into `budget.sec_per_step` in config.json, then re-run
the preamble cell (it git-pulls) and step 06 before the grid.
"""
sec_per_step = dapt_lib.probe(run)
