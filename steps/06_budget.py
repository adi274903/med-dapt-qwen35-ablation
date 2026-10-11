"""Step 06 -- the cost table.

Nothing to edit here: it is a readout of config.json. It prints the steps, the
tokens, the honest per-arm overhead and the whole grid's cost, and it shouts if
`budget.sec_per_step` is still the placeholder.
"""
arms = dapt_lib.print_budget(run)
