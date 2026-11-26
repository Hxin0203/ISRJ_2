"""Read-only preflight for the server controller (server B, CPU only, no data written)."""
import inspect
import json
import py_compile
from pathlib import Path

ROOT = Path("D:/code/Xin/ISRJ_222")
out = {}
import paper2.data_1103 as d
out["generate_signature"] = str(inspect.signature(d.generate))
for name in ("controller", "sweep_suite", "rsdr_suite", "aggregate_suite", "equal_pfa_suite",
             "classical_suite", "external_suite"):
    py_compile.compile(str(ROOT / "tools/paper2_baseline_suite" / f"{name}.py"), doraise=True)
out["compile"] = "OK"
from tools.paper2_baseline_suite import sweep_suite
out["sweep_cells"] = len(sweep_suite.cells())
out["sweep_jobs"] = sweep_suite.jobs()
out["rsdr_assets"] = sorted(p.name for p in (ROOT / "outputs/rsdr_unified_20261115/assets").iterdir())
out["sweep_data_exists"] = (sweep_suite.DATA / "DATA_MANIFEST.json").is_file()
print(json.dumps(out, indent=1))
