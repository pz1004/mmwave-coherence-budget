"""P3 acquisition: download the first eight numbered 28 GHz cities in the
frozen catalog order.  Records size and SHA-256 of every archive.  Site order
is the catalog's numerical order, frozen before any channel was decoded."""
import hashlib, json, os, sys, time
import deepmimo as dm

HERE = os.path.dirname(os.path.abspath(__file__))
dm.config.set('scenarios_folder', os.path.join(HERE, 'scenarios'))
ZIPS = os.path.join(HERE, 'zips')

SITES = ["city_0_newyork_28", "city_1_losangeles_28", "city_2_chicago_28",
         "city_3_houston_28", "city_4_phoenix_28", "city_5_philadelphia_28",
         "city_6_miami_28", "city_7_sandiego_28"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def tree_bytes(d):
    n = 0
    for root, _, fs in os.walk(d):
        for f in fs:
            n += os.path.getsize(os.path.join(root, f))
    return n

rec = []
for i, s in enumerate(SITES):
    out = os.path.join(HERE, 'scenarios', s)
    t0 = time.time()
    err = None
    if not os.path.isdir(out):
        try:
            dm.download(s, output_dir=ZIPS)
        except Exception as e:                      # keep failures, do not retry
            err = f"{type(e).__name__}: {e}"
    z = os.path.join(ZIPS, s + '_downloaded.zip')
    rec.append({"order": i, "scenario": s,
                "zip_bytes": os.path.getsize(z) if os.path.exists(z) else None,
                "zip_sha256": sha256(z) if os.path.exists(z) else None,
                "extracted_bytes": tree_bytes(out) if os.path.isdir(out) else None,
                "n_files": len(os.listdir(out)) if os.path.isdir(out) else None,
                "seconds": round(time.time() - t0, 1), "error": err})
    print(f"[{i}] {s}: {rec[-1]['zip_bytes']} B  {rec[-1]['seconds']}s  err={err}",
          file=sys.stderr, flush=True)

json.dump({"frozen_order": SITES, "records": rec},
          open(os.path.join(HERE, 'acquisition.json'), 'w'), indent=1)
print("TOTAL zip bytes:", sum(r['zip_bytes'] or 0 for r in rec), file=sys.stderr)
