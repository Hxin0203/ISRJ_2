"""Print SHA256 of the sources WP7 timing imports (run on both hosts and compare)."""
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
names = ["tools/vp_1103_recovp_hybrid.py", "paper2/network.py", "paper2/evaluation_1103_v4.py", "paper2/radar_eval.py",
         "tools/unet_ablation/models.py", "tools/paper2_baseline_suite/equal_pfa_suite.py",
         "tools/paper2_baseline_suite/external_suite.py", "tools/paper2_baseline_suite/rsdr_suite.py",
         "tools/paper2_baseline_suite/classical_suite.py", "tools/rsdr_adapter.py", "tools/rsdr_stream.py",
         "paper2/baselines_1103.py"]
names += sorted(str(p.relative_to(root)).replace("\\", "/") for p in (root / "tools/rsdr_vendor").rglob("*.py"))
names += sorted(str(p.relative_to(root)).replace("\\", "/") for p in (root / "tools/paper2_baseline_suite/cvednet_vendor").rglob("*.py"))
names += sorted(str(p.relative_to(root)).replace("\\", "/") for p in (root / "paper2/vendor/baselines_1103").rglob("*.py"))
for n in names:
    p = root / n
    print((hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else "MISSING"), n)
