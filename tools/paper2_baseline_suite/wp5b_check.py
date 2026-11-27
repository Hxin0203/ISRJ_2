"""Print SHA prefixes of the generator sources and the WP5b suite on this host."""
import hashlib
import py_compile

for p in ("paper2/data_1103.py", "paper2/vendor/generate_v2.py", "tools/paper2_baseline_suite/wp5b_suite.py"):
    print(hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], p)
py_compile.compile("tools/paper2_baseline_suite/wp5b_suite.py", doraise=True)
print("COMPILE_OK")
