#!/usr/bin/env python3
"""Hash what the later stages read but the repository does not hold -> manifest.json.

The band-averaged datasets (data/, which build.py makes from the licensed DeepMIMO scenarios)
and the trained models (models/) are regenerable but not tracked; manifest.json records their
sha256, the dataset description and the environment that produced them.  build.py skips the
rebuild when every dataset matches."""
import hashlib, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = os.path.abspath(os.path.join(HERE, '..', 'sites'))


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


data = {f: {"sha256": sha(os.path.join(HERE, 'data', f)),
            "bytes": os.path.getsize(os.path.join(HERE, 'data', f))}
        for f in sorted(os.listdir(os.path.join(HERE, 'data')))}
models = {f: sha(os.path.join(HERE, 'models', f))
          for f in sorted(os.listdir(os.path.join(HERE, 'models')))}
try:
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                            capture_output=True, text=True, timeout=120).stdout
except Exception as e:
    freeze = f"unavailable: {e}"

m = {
  "built_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  "site_manifest_sha256": sha(os.path.join(SITES, 'site-manifest.json')),
  "environment": {
    "python": platform.python_version(), "platform": platform.platform(),
    "processor": platform.processor(),
    "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    "pip_freeze_lines": len(freeze.splitlines()),
  },
  "dataset_sha256": data, "model_sha256": models,
  "dataset": json.load(open(os.path.join(HERE, 'dataset.json'))),
}
blob = json.dumps(m, indent=1, sort_keys=True)
m['self_sha256'] = hashlib.sha256(blob.encode()).hexdigest()
out = os.path.join(HERE, 'manifest.json')
open(out, 'w').write(json.dumps(m, indent=1, sort_keys=True))
print(f"manifest {os.path.getsize(out)} bytes  sha256={m['self_sha256'][:16]}")
print(f"  {len(data)} datasets, {len(models)} model files")
