"""Step 09 -- run the grid (the long one).

Pending arms only, a fresh base + fresh adapters per arm, stopping on either the
grid budget or the session budget. Interrupting is safe: re-run this step in a
fresh session and it continues from the status.json files in run.root.
"""
dapt_lib.run_grid(run)
