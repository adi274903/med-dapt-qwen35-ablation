import json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))   # selftest/
ROOT = os.path.dirname(HERE)                        # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import dapt_lib as d

base = d.load_config("configs/base.json")
print("=== variant deep-merge (ctx4096.json) ===")
v = d.load_config("configs/base.json", "ctx4096.json")
assert v["baseline"] == base["baseline"], "variant leaked into the baseline"
assert v["budget"] == base["budget"], "variant must not touch the budget"
assert v["_overrides_path"].endswith("ctx4096.json")
ids = [e["id"] for e in v["grid"]]
assert ids == ["B", "B4k", "R16", "LO", "EMB1"], ids
rv = d.Run(v)
b4k = [a for a in rv.arms if a["arm_id"] == "B4k"][0]
bb = [a for a in rv.arms if a["arm_id"] == "B"][0]
assert b4k["steps"] == bb["steps"] // 2, (b4k["steps"], bb["steps"])
assert b4k["steps"] * 16 * 4096 == bb["steps"] * 16 * 2048, "tokens/arm must be constant"
print("  baseline untouched | grid replaced wholesale (%s)" % ",".join(ids))
print("  B %d steps @2048 vs B4k %d steps @4096 -> same %.1fM tokens/arm"
      % (bb["steps"], b4k["steps"], bb["steps"] * 16 * 2048 / 1e6))
print("  warnings:", d.check_config(v, rv.arms))

print()
print("=== the seq_len guard ===")
bad = d.load_config("configs/base.json", {"budget": {"seq_len": 4096}})
w = [x for x in d.check_config(bad) if "seq_len" in x]
assert w, "seq_len drift was not caught"
print("  caught:", w[0][:120], "...")

print()
print("=== ruler on synthetic arms (no pandas installed -> stdlib fallback) ===")
tmp = tempfile.mkdtemp(prefix="dapt_ruler_")
run = d.Run(d.load_config("configs/base.json", {"ablation_root": tmp}))
for aid, dm, guard_delta in (("B", 1.10, +0.10), ("R16", 1.08, +0.01), ("LO", 1.12, -0.02)):
    arm = [a for a in run.arms if a["arm_id"] == aid][0]
    os.makedirs(run.arm_dir(arm), exist_ok=True)
    dom = {"eval_%s_loss" % s: dm + 0.01 * i
           for i, s in enumerate(base["data"]["domain_sources"])}
    fin = dict(dom)
    fin["eval_FineWeb_Edu_loss"] = 2.80 + guard_delta
    fin["eval_Orca_Math_loss"] = 1.50 + guard_delta
    base_l = dict(fin)
    base_l["eval_FineWeb_Edu_loss"] = 2.80
    base_l["eval_Orca_Math_loss"] = 1.50
    json.dump(dict(arm_id=aid, desc=arm["desc"], done=True, wall_hours=4.99,
                   steps_run=1392, tokens_run=45613056, tok_per_param=0.01049,
                   domain_mean_final=sum(dom.values()) / len(dom),
                   final_losses=fin, base_losses=base_l, guards={}),
              open(run.status_path(arm), "w"))
best = d.ruler(run)
assert best == "R16", best
assert os.path.exists(run.ranking_csv)
rows = open(run.ranking_csv).read().strip().split("\n")
assert len(rows) == 4
print()
print("  ranking.csv: %d rows (header + %d arms); best=%s; B vetoed by +0.10 guard rise"
      % (len(rows), len(rows) - 1, best))
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print("\nALL CHECKS PASS")
