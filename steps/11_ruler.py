"""Step 11 -- the keep/throw ruler.

Scores every finished arm: best held-out domain loss, vetoed by a retention
regression against that arm's own step-0 baseline. Writes ranking.csv next to
results.csv. Returns the winning arm id as `best`, which step 12 uses.
"""
best = dapt_lib.ruler(run)
