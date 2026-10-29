#!/usr/bin/env python3
"""The experiment's protocol record, protocol.json: loader and checker.

protocol.json holds the protocol as frozen on 2026-09-15, before any dataset was built, any model
trained or any number seen (the observation law, the strata, the splits, the seeds, the grid of
periods, the method menu, the statistic and the primary criterion), together with its three
amendments, the changes made after results existed, the interpretation rules R1-R22 and the
analyses declared later, each before it ran.  Stage-4 scripts take their seed list from here.

`python3 protocol.py` checks three things and exits non-zero on any problem:

1. FROZEN CONTENT.  The frozen fields hash to self_sha256, the hash written at freeze time.
2. INPUTS.  Every file the frozen protocol hashed (the observation law, the scenario manifest and
   qualification, the budget's theory and DP) matches its hash, or the hash a later change records.
3. STORED FILES.  Every file a change or an analysis lists under files_sha256 matches, so a
   re-timing or a re-run cannot replace a stored result unnoticed.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'protocol.json')
RECORD = ('about', 'amendments', 'changes', 'rules', 'analyses')

# frozen field -> the file whose hash it records (paths relative to HERE)
INPUTS = {
    ('inherits', 'g1_observation_law'): 'g1.py',
    ('inherits', 'site_manifest'): '../sites/site-manifest.json',
    ('inherits', 'qualification'): '../sites/qualification.json',
    ('inherits', 'phase2_theory'): '../budget/theory.py',
    ('inherits', 'phase2_dp'): '../budget/dp.py',
    ('g2_observation', 'module_sha256'): 'obs.py',
}


def load():
    return json.load(open(PATH))


def sha(rel):
    return hashlib.sha256(open(os.path.join(HERE, rel), 'rb').read()).hexdigest()


def frozen_sha256(P):
    """sha256 of the frozen fields, serialized as at freeze time."""
    Q = {k: v for k, v in P.items() if k not in RECORD + ('self_sha256',)}
    return hashlib.sha256(json.dumps(Q, indent=1, sort_keys=True).encode()).hexdigest()


def seeds():
    """The seed list in force: the latest change that sets one, else the frozen protocol's.
    Every stage-4 script takes its default from here, so the list is written in one place."""
    P = load()
    for c in reversed(P.get('changes', [])):
        if 'seeds' in c:
            return tuple(c['seeds'])
    return tuple(P['seeds']['values'])


def seed_arg(argv=None):
    """`--seeds 17 29 43` on the command line overrides the protocol's list.
    Used to re-run the pre-registered three-seed analysis."""
    argv = sys.argv if argv is None else argv
    if '--seeds' in argv:
        i = argv.index('--seeds')
        vals = []
        for a in argv[i + 1:]:
            if a.startswith('--'):
                break
            vals.append(int(a))
        assert vals, "--seeds needs at least one value"
        return tuple(vals)
    return seeds()


def analysis(key, P=None):
    """The record of a later analysis, by key (e.g. 'calib_holm')."""
    P = load() if P is None else P
    return next(a for a in P['analyses'] if a['key'] == key)


def check(P=None):
    """Return (problems, notes) without exiting."""
    P = load() if P is None else P
    problems, notes = [], []

    # 1. the frozen content
    h = frozen_sha256(P)
    if h != P['self_sha256']:
        problems.append(f"frozen content hashes to {h[:16]}, not self_sha256 "
                        f"{P['self_sha256'][:16]}: a frozen field was edited")

    # 2. inputs the frozen protocol hashed; a later change may record a new hash
    later = {}
    for c in P.get('changes', []):
        for f, hf in c.get('inputs_sha256', {}).items():
            later[f] = (hf, c['date'])
    for (sec, key), f in INPUTS.items():
        now = sha(f)
        if now == P[sec][key]:
            continue
        if f not in later:
            problems.append(f"{f} changed since the freeze and no change records it")
        elif later[f][0] != now:
            problems.append(f"{f} differs from the hash recorded on {later[f][1]}")
        else:
            notes.append(f"{f}: as recorded by the change of {later[f][1]}")

    # 3. stored files; a later entry's hash supersedes an earlier one
    files = {}
    for e in P.get('changes', []) + P.get('analyses', []):
        for f, hf in e.get('files_sha256', {}).items():
            files[f] = hf
    for f, hf in sorted(files.items()):
        if not os.path.exists(os.path.join(HERE, f)):
            problems.append(f"{f} is listed but missing")
        elif sha(f) != hf:
            problems.append(f"{f} differs from its recorded hash")
    notes.append(f"{len(files)} stored file(s) verified")
    return problems, notes


def main():
    problems, notes = check()
    P = load()
    print(f"protocol {P['self_sha256'][:16]} (frozen {P['frozen_utc']}): "
          f"{len(P['amendments'])} amendments, {len(P['changes'])} changes, "
          f"{sum(r != 'convention' for r in P['rules'])} rules, {len(P['analyses'])} analyses")
    for n in notes:
        print(f"  ok  {n}")
    for p in problems:
        print(f"  !!  {p}")
    if problems:
        sys.exit(f"protocol: {len(problems)} problem(s)")
    print("  protocol verified")


if __name__ == '__main__':
    main()
