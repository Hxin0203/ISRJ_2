"""In-memory SSH inference; all persistent experiment files remain local.

Remote original assets are read-only. Python runs with -B, and observations and
outputs travel through pipes, without creating a remote experiment directory.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from paper2.io import ROOT
from tools.rsdr_adapter import RSDRNet


def server_source(device):
    source = (ROOT / "tools/rsdr_adapter.py").read_text(encoding="utf8")
    source = source.replace("from __future__ import annotations", "")
    source = source.replace("from paper2.io import ROOT, sha", "")
    source = source.replace("from tools.rsdr_vendor.", "from RGSSR_ISRJ.")
    source = source.replace("from RGSSR_ISRJ.gate_core import", "from gate_core import")
    source = source.replace('ASSETS = ROOT / "outputs/rsdr_unified_20261115/assets"',
        'ASSETS = Path("D:/code/Xin/RSDR_Net/ch4_parallel_20261105_r1/assets")')
    provenance = json.loads((ROOT / "tools/rsdr_vendor/PROVENANCE.json").read_text())
    before = '''import os,sys,hashlib,base64,io,json,traceback
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
PACKAGE=Path('D:/code/Xin/ISRJ_Last/RSDR32_lanes/lane1/src/repo')
GATE=Path('D:/code/Xin/RSDR_Net/ch4_review_20261104_r1/gate_bundle_r1')
sys.path[:0]=[str(PACKAGE),str(GATE)]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
'''
    before += f"provenance={provenance!r}\n"
    before += "for row in provenance.values():\n assert sha(row['remote'])==row['sha256']\n"
    after = f"\nmodel=RSDRNet(device={device!r},batch_pulses=256,verify_vendor=False)\n"
    after += '''print(json.dumps({'status':'READY','device':model.device,'torch':torch.__version__}),flush=True)
for line in sys.stdin:
 try:
  request=json.loads(line)
  if request.get('action')=='close':break
  with np.load(io.BytesIO(base64.b64decode(request['payload'])),allow_pickle=False) as z:x=z['x']
  indexes=request.get('state_indexes')
  if indexes is not None:
   full=np.zeros((len(x),4,8,1800),dtype=np.complex64);full[:,indexes]=x;x=full
  y=model.restore(x,state_indexes=indexes)
  out=io.BytesIO();np.save(out,y,allow_pickle=False)
  print(json.dumps({'status':'OK','payload':base64.b64encode(out.getvalue()).decode()}),flush=True)
 except Exception:
  print(json.dumps({'status':'ERROR','traceback':traceback.format_exc()}),flush=True)
  break
'''
    return before + source + after


class StreamRSDR:
    def __init__(self, device):
        self.device = device
        source = server_source(device)
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        dest = ROOT / "outputs/rsdr_unified_20261115"
        dest.mkdir(parents=True, exist_ok=True)
        self.error_stream = (dest / f"stream_{device.replace(':','_')}.stderr.log").open("ab")
        command = ('D:/code/Xin/ISRJ_Last/RSDR32_lanes/lane1/.venv/Scripts/python.exe '
                   '-B -u -c "import base64,sys;exec(base64.b64decode(sys.stdin.readline()))"')
        self.process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "rsdr-b", command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_stream,
            text=True, encoding="utf8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.process.stdin.write(base64.b64encode(source.encode()).decode() + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError(f"SSH inference failed to start on {device}; see local stderr")
        self.ready = json.loads(line)
        assert self.ready["status"] == "READY", self.ready

    def restore(self, states, state_indexes=None):
        stream = io.BytesIO()
        # Adapter casts real and imaginary inputs to float32 on both hosts.
        x = np.asarray(states, dtype=np.complex64)
        if state_indexes is not None:
            x = x[:, state_indexes]
        np.savez(stream, x=x)
        packet = dict(payload=base64.b64encode(stream.getvalue()).decode(), state_indexes=state_indexes)
        self.process.stdin.write(json.dumps(packet) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"SSH inference ended on {self.device}")
        response = json.loads(line)
        if response["status"] != "OK":
            raise RuntimeError(response)
        return np.load(io.BytesIO(base64.b64decode(response["payload"])), allow_pickle=False)

    def close(self):
        if getattr(self, "process", None):
            try:
                self.process.stdin.close()
                self.process.wait(timeout=20)
            finally:
                self.process.stdout.close()
                self.error_stream.close()


def make_backend(name):
    return RSDRNet(batch_pulses=256) if name == "local" else StreamRSDR(name.split("/", 1)[1])
