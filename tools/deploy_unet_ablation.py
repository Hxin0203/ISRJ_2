"""Explicit deployment and status retrieval for the dedicated server workspace.

SSH is used for execution; archives contain only enumerated study assets. Nothing
is written to the first-paper repositories or their Python environments.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json
from tools.unet_ablation.protocol import RUN_ID, BASE, EVIDENCE, ORIGINAL, SEEDS
from tools.unet_ablation.runtime import SERVER_ROOT

BOOTSTRAP = "D:/code/Xin/ISRJ_Last/RSDR32_lanes/lane1/.venv/Scripts/python.exe"
PYTHON = SERVER_ROOT + "/.venv/Scripts/python.exe"
LOCAL = ROOT / "outputs" / (RUN_ID + "_deployment")


def remote(source, *, timeout=180, bootstrap=False):
    program = BOOTSTRAP if bootstrap else PYTHON
    encoded = base64.b64encode(source.encode("utf8")).decode()
    command = program + ' -B -u -c "import base64,sys;sys.stdout.reconfigure(encoding=\'utf8\',errors=\'replace\');exec(base64.b64decode(\'' + encoded + '\'))"'
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "rsdr-b", command],
                            capture_output=True, timeout=timeout)
    out = result.stdout.decode("utf8", errors="replace")
    err = result.stderr.decode("utf8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"remote exit {result.returncode}: {out}\n{err}")
    return out


def prepare():
    LOCAL.mkdir(parents=True, exist_ok=True)
    from tools.unet_ablation.preflight import reference
    reference_path = reference()
    files = set(ROOT / name for name in ("tools/__init__.py", "tools/durable_io.py",
                "tools/aggregate_convergence_256_8seed.py", "tools/training_progress_256.py",
                "tools/start_unet_ablation.ps1", "tests/test_unet_ablation.py"))
    files.update((ROOT / "paper2").rglob("*.py"))
    files.update((ROOT / "tools/unet_ablation").glob("*.py"))
    files.add(reference_path)
    for role in ("train", "dev"):
        files.update(ROOT / "data/paper2_1103" / (role + suffix) for suffix in (".npz", ".receipt.json"))
    for seed in SEEDS:
        folder = ORIGINAL / f"seed_{seed}"
        files.update(folder / name for name in ("epoch_256.pt", "epoch_256.json", "TRAINING_COMPLETE.json"))
        files.update(folder.glob("dev_epoch_*_v4.json"))
        files.add(folder / "dev_epoch_256_v4.npz")
        for name in read(folder / "TRAINING_COMPLETE.json")["identity"]["source_sha256"]:
            files.add(ROOT / name)
    manifest = {p.relative_to(ROOT).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(files)}
    record = {"created_utc": now(), "server": "rsdr-b", "root": SERVER_ROOT, "files": manifest}
    manifest_path = LOCAL / "DEPLOYMENT_MANIFEST.json"
    atomic_json(manifest_path, record)
    archive = LOCAL / "payload.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(manifest_path, "DEPLOYMENT_MANIFEST.json")
    print(remote(f'''from pathlib import Path
import venv,site,subprocess,sys
root=Path({SERVER_ROOT!r})
root.mkdir(parents=True,exist_ok=True)
target=root/'.venv'
if not (target/'Scripts/python.exe').exists():
 venv.EnvBuilder(with_pip=False).create(target)
 pth=target/'Lib/site-packages/server_shared_dependencies.pth'
 pth.write_text('\\n'.join(site.getsitepackages())+'\\n',encoding='utf8')
print(root)
''', bootstrap=True))
    subprocess.run(["scp", "-q", str(archive), "rsdr-b:" + SERVER_ROOT + "/deployment_payload.zip"], check=True, timeout=600)
    print(remote(f'''from pathlib import Path
import zipfile,json,hashlib,os
root=Path({SERVER_ROOT!r})
archive=root/'deployment_payload.zip'
old=json.loads((root/'DEPLOYMENT_MANIFEST.json').read_text(encoding='utf8')) if (root/'DEPLOYMENT_MANIFEST.json').exists() else {{'files':{{}}}}
frozen=(root/'configs'/{(RUN_ID+'.json')!r}).exists()
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('DEPLOYMENT_MANIFEST.json'))
 for name,row in manifest['files'].items():
  target=(root/name).resolve()
  assert target.is_relative_to(root.resolve())
  raw=z.read(name)
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  if target.exists():
   current=hashlib.sha256(target.read_bytes()).hexdigest()
   if current!=row['sha256']:
    assert not frozen and name in old['files'] and current==old['files'][name]['sha256'], 'Existing server asset differs: '+name
    backup=root/'outputs/deployment_revisions'/current/name
    backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(target.read_bytes())
    tmp=target.with_name(target.name+'.deploy.tmp');tmp.write_bytes(raw);os.replace(tmp,target)
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   tmp=target.with_name(target.name+'.deploy.tmp')
   tmp.write_bytes(raw);os.replace(tmp,target)
 (root/'DEPLOYMENT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('VERIFIED_FILES',len(manifest['files']))
''', timeout=600))
    print(remote(f'''import os,subprocess,sys,json,platform
os.chdir({SERVER_ROOT!r})
os.environ.update(CUDA_VISIBLE_DEVICES='1',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
import torch,numpy,scipy,matplotlib,psutil,pytest
print(json.dumps(dict(host=platform.node(),python=sys.version,torch=torch.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())))
for args in (['-m','pytest','tests/test_unet_ablation.py','-q'],['-m','tools.unet_ablation.preflight'],['-m','tools.unet_ablation.smoke'],['-m','tools.unet_ablation.run','--prepare']):
 subprocess.run([sys.executable,*args],check=True)
''', timeout=300))
    fetch()


def start():
    # WMI owns the launcher, so ending SSH cannot end the long-running queue.
    # ExecutionPolicy is scoped to this known launcher process, not changed globally.
    source = f'''import subprocess,base64,json
script="""$ProgressPreference = 'SilentlyContinue'
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{{ShowWindow=[uint16]0}}
$argsForLaunch = @{{CommandLine='powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File {SERVER_ROOT}/tools/start_unet_ablation.ps1';CurrentDirectory='{SERVER_ROOT}';ProcessStartupInformation=$startup}}
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments $argsForLaunch
$result | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if ($result.ReturnValue -ne 0) {{exit 1}}
"""
p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(script.encode('utf-16le')).decode()],capture_output=True)
print(p.stdout.decode('utf8',errors='replace'))
if p.stderr:print(p.stderr.decode('utf8',errors='replace'))
assert p.returncode==0
'''
    print(remote(source))


def status():
    return remote(f'''import json,subprocess,platform,psutil
from pathlib import Path
root=Path({SERVER_ROOT!r});base=root/'outputs'/{RUN_ID!r}
out={{'checked_server_host':platform.node(),'files':{{}}}}
for name in ('QUEUE_STATUS.json','STATUS.json','EVAL_STATUS.json','RESOURCE_STATUS.json','CONTROLLER_FAILURE.json','PIPELINE_COMPLETE.json'):
 p=base/name
 if p.exists():
  row=json.loads(p.read_text(encoding='utf8'));row.pop('source_sha256',None);out['files'][name]=row
out['saved_epoch_counts']={{str(p.relative_to(base)):len(list(p.glob('training_epoch_*.json'))) for p in base.glob('*/seed_*')}}
out['processes']=[]
for p in psutil.process_iter(['pid','ppid','name','cmdline']):
 try:
  if any(arg in ('tools.unet_ablation.run','tools.unet_ablation.train') for arg in (p.info['cmdline'] or [])):
   out['processes'].append({{**p.info,'cwd':p.cwd(),'cuda_visible_devices':p.environ().get('CUDA_VISIBLE_DEVICES')}})
 except (psutil.AccessDenied,psutil.NoSuchProcess):pass
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(out))
''')


def fetch():
    """Fetch only lightweight evidence plus configuration; no raw or weights."""
    print(remote(f'''from pathlib import Path
import zipfile
root=Path({SERVER_ROOT!r});e=root/'evidence/outputs'/{RUN_ID!r};base=root/'outputs'/{RUN_ID!r}
files=list(e.rglob('*.json'))+list(base.glob('*.json'))+list(base.glob('*/seed_*/training_epoch_*.json'))
files+=list((root/'configs').glob({(RUN_ID+'.json')!r}))
files+=list(base.glob('*/ARCHITECTURE.json'))
files+=list((root/'evidence/outputs'/({RUN_ID!r}+'_engineering')).glob('*.json'))
files+=list((root/'data/unet_ablation_engineering').glob('DIAGNOSTIC*.json'))
with zipfile.ZipFile(root/'lightweight_results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in set(files):z.write(p,p.relative_to(root).as_posix())
print('LIGHTWEIGHT_FILES',len(set(files)))
'''))
    LOCAL.mkdir(parents=True, exist_ok=True)
    archive = LOCAL / "lightweight_results.zip"
    subprocess.run(["scp", "-q", "rsdr-b:" + SERVER_ROOT + "/lightweight_results.zip", str(archive)], check=True, timeout=120)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT.resolve()) or target.suffix != ".json":
                raise RuntimeError("unexpected lightweight result path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    raw = status()
    record = json.loads(raw)
    record["retrieved_utc"] = now()
    atomic_json(EVIDENCE / "SERVER_STATUS.json", record)
    print(json.dumps(record, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "start", "status", "fetch"))
    args = parser.parse_args()
    if args.action == "status":
        print(status())
    else:
        globals()[args.action]()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8", errors="replace")
    main()
