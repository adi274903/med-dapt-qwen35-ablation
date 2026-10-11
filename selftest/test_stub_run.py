"""Execute dapt_lib end-to-end against a stub unsloth/transformers/datasets.

Nothing here is a substitute for the real run: the model is fake, the losses are
fake. What it DOES prove is that every code path in the module actually runs --
build_model's two/three branches, make_trainer, the callbacks, run_one_arm's
status.json + results.csv, run_grid's budget guards, the ruler, phase0 and the
adapter inspector -- with no NameError, no wrong config key, and the assertions
holding under BOTH PEFT wrapper conventions.
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))      # selftest/
ROOT = os.path.dirname(HERE)                           # repo root
sys.path.insert(0, os.path.join(HERE, "stub_env"))     # torch/datasets/unsloth fakes
sys.path.insert(1, ROOT)                               # dapt_lib
os.chdir(ROOT)

import dapt_lib as d                                    # noqa: E402

TMP = tempfile.mkdtemp(prefix="dapt_stub_")
OVR = {"ablation_root": TMP,
       "budget": {"session_hours": 100.0},   # don't stop the grid early
       "saving": {"save_models": True},
       "probe": {"steps": 6, "warmup": 2}}

print("=" * 72)
print("1. env gates")
run = d.open_run("config.json", overrides=OVR)
d.verify_env(run)

print()
print("=" * 72)
print("2. data")
stream, eval_by_source = d.build_data(run)
assert set(eval_by_source) == set(run.cfg["data"]["domain_sources"] +
                                 run.cfg["data"]["guard_sources"])

print()
print("=" * 72)
print("3. build_model: THREE branches (default / ensure_weight_tying / modules_to_save=null)")
for label, over, expect in (
        ("default (tie broken)", {}, "TIE BROKEN"),
        ("ensure_weight_tying", {"ensure_weight_tying": True}, "TIE PRESERVED"),
        ("ADONLY (modules_to_save=null)", {"modules_to_save": None}, "FROZEN")):
    r2 = d.Run(d.load_config("config.json", d.deep_merge(OVR, over)))
    cfg = d.deep_merge(r2.arms[0], over)
    print("--- %s" % label)
    m, tok, text_tok = d.build_model(r2, cfg)
    print()

print("=" * 72)
print("4. probe")
probe_run = d.Run(d.load_config("config.json", OVR))
sec = d.probe(probe_run)
assert sec > 0
print("  probe returned %.4f s/step" % sec)

print()
print("=" * 72)
print("5. run_grid (all 6 arms)")
run = d.Run(d.load_config("config.json", OVR))
d.run_grid(run)
done = run.done_ids()
print("  arms done: %s" % sorted(done))
assert len(done) == len(run.arms), (sorted(done), [a["arm_id"] for a in run.arms])

print()
print("=" * 72)
print("6. status.json + results.csv contract")
st = json.load(open(run.status_path(run.arms[0])))
frozen = {"arm_id", "phase", "desc", "cfg", "done", "wall_hours", "steps_run",
          "tokens_run", "tok_per_param", "sec_per_step_amortised",
          "peak_vram_gib", "best_metric", "final_losses", "base_losses",
          "domain_mean_final", "guards"}
missing = frozen - set(st)
assert not missing, missing
print("  status.json: all %d frozen fields present" % len(frozen))
assert st.get("config_path") and st.get("cfg_hash"), "provenance not stamped"
assert os.path.exists(st["config_path"]), st["config_path"]
print("  stamped: config_path=%s cfg_hash=%s lib_mtime=%s"
      % (st["config_path"].split("/")[-1], st["cfg_hash"], st.get("lib_mtime")))
rows = open(run.results_csv).read().strip().split("\n")
assert len(rows) == len(run.arms) + 1, len(rows)
assert "cfg_hash" in rows[0], "results.csv header lacks the config hash"
print("  results.csv: %d rows (header + %d arms), header carries cfg_hash"
      % (len(rows), len(rows) - 1))
# the run's own frozen copy of the resolved config (so a later edit to
# config.json cannot silently re-label old results)
rc = json.load(open(run.run_config_path))
assert rc["model_id"] == run.cfg["model_id"] and len(rc["grid"]) == len(run.arms)
print("  run_config.json: %d grid arms frozen, model_id=%s"
      % (len(rc["grid"]), rc["model_id"]))

print()
print("=" * 72)
print("7. ruler")
best = d.ruler(run)
assert best in [a["arm_id"] for a in run.arms], best
print("  best = %s" % best)

print()
print("=" * 72)
print("8. verify_adapters on the winner")
d.verify_adapters(run, best)

print()
print("=" * 72)
print("9. phase0 (run=true, small ladder)")
p0 = d.deep_merge(OVR, {"phase0": {"run": True, "steps": [10, 20]}})
r3 = d.Run(d.load_config("config.json", p0))
d.phase0(r3)
txt = open(run.results_csv).read()
p0rows = [l for l in txt.strip().split("\n") if ",phase0," in l]
assert len(p0rows) == 2, p0rows
assert ",grid," in txt
rank = open(run.ranking_csv).read().strip().split("\n")
assert len(rank) == len(run.arms) + 1, len(rank)
print("  phase0 wrote %d rungs; the shared results.csv keeps both phases (%d rows)"
      % (len(p0rows), len(txt.strip().split("\n"))))
print("  ranking.csv still has only the %d grid arms -- rungs are never ranked"
      % (len(rank) - 1))
sfx = r3.cfg["phase0"]["root_suffix"]
p0_dir = "%s/%s" % (os.path.dirname(r3.root.rstrip("/")), sfx)
assert r3.arm_dir({"arm_id": "B10", "root": p0_dir}) == "%s/B10" % p0_dir
print("  phase0 arm dirs live outside the grid root: %s" % p0_dir)


print()
print("=" * 72)
print("10. resume: a second run_grid is a no-op")
n_before = len(open(run.results_csv).read().strip().split("\n"))
d.run_grid(run)
n_after = len(open(run.results_csv).read().strip().split("\n"))
assert n_before == n_after, (n_before, n_after)
print("  re-running run_grid added 0 rows (pending() is empty by arm_status)")

shutil.rmtree(TMP, ignore_errors=True)
print()
print("ALL STUB-RUN CHECKS PASS  (config %s)" % os.environ.get("DAPT_CFG", "config.json"))
