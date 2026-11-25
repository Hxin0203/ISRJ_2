"""Versioned entry: preserves historical source hashes and uses the same timing kernel."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vp_1103_recovp_formal_latency import run
if __name__=='__main__':run(warmup=50,repeats=200)
