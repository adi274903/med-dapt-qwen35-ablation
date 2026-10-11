"""Step 07 -- build the data (once per session).

One streaming interleaved mix plus a per-source holdout, built once and reused
by every arm, so all arms see the same data in the same order. Source list, mix
fractions and holdout size live in config.json under `data`.
"""
dapt_lib.build_data(run)
