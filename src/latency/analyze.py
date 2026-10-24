#!/usr/bin/env python3
"""Turn measured per-decision charges into the coherence budget and the gate.

Accounting (G1/G3, unchanged):
  probe      20 us per measured beam (10 us pilot + 10 us feedback)
  compute    the measured p95 charge for the method's per-decision operation
  payload    whatever is left of the coherence interval T
  net rate   R = max(0, 1 - d/T) * log2(1 + gamma),  d = probe + compute

The published objective (e.g. arXiv:2201.04657 Eq. 34-35) sets d = probe.
Everything below is the difference that makes.
"""

import json
import os
import sys

PROBE_US = 20.0
N_BEAMS = 64
T_MS = {"vehicular 0.43 ms": 0.43, "2 ms": 2.0, "5 ms": 5.0,
        "pedestrian 7.7 ms": 7.7, "10 ms": 10.0}

# Representative refinement sweep width per method class, stated not measured.
K_PROBE = {
    "sweep_select": 64,
    "s1_r": 4, "s1_u": 8, "s1_q": 8,
    "graff_eig": 4, "graff_col": 4, "graff_aps": 4,
    "beam_mlp": 8, "beam_cnn": 8,
    "lstm_cached": 8, "lstm_seq": 8, "transformer_small": 8,
    "resnet18_vision": 4, "vit_b16_vision": 4,
    "earlyexit_1": 8, "earlyexit_2": 8, "earlyexit_3": 8,
}

LABEL = {
    "sweep_select": "Exhaustive 64-beam sweep",
    "s1_r": "S1 ranker + order (r)",
    "s1_u": "S1 mandatory stop (u_k)",
    "s1_q": "S1 stage decision, both value nets (q_k)",
    "graff_eig": "Graff et al. eigenvector net",
    "graff_col": "Graff et al. covariance net",
    "graff_aps": "Graff et al. APS conv net",
    "beam_mlp": "FC beam classifier",
    "beam_cnn": "CNN beam predictor",
    "lstm_cached": "LSTM tracker, cached step",
    "lstm_seq": "LSTM tracker, 8-step replay",
    "transformer_small": "Small transformer encoder",
    "resnet18_vision": "ResNet-18 vision-aided",
    "vit_b16_vision": "ViT-B/16 vision-aided",
    "earlyexit_1": "Anytime predictor, exit 1",
    "earlyexit_2": "Anytime predictor, exit 2",
    "earlyexit_3": "Anytime predictor, exit 3",
}

PROFILE_ORDER = ["cpu-1t-ecore", "cpu-1t-pcore", "cpu-8t-pcore", "gpu-resident"]
PROFILE_LABEL = {
    "cpu-1t-ecore": "E-core 1T (UE proxy)",
    "cpu-1t-pcore": "P-core 1T (G3 profile)",
    "cpu-8t-pcore": "P-core 8T (server)",
    "gpu-resident": "RTX 8000 (resident)",
}


def window(d_compute_us, k_probe, eta):
    """Coherence-time interval in which a method beats the exhaustive sweep.

    With payload fraction max(0, 1 - d/T) and spectral-efficiency ratio
    eta = log2(1+gamma_pred)/log2(1+gamma_max) <= 1:

        eta (1 - d_L/T) > (1 - d_S/T)   <=>   T (1 - eta) < d_S - eta d_L

    so the method beats the sweep on  d_L < T < (d_S - eta d_L)/(1 - eta),
    and nowhere at all if eta * d_L >= d_S.  The lower edge d_L contains the
    compute term; the published accounting omits it.
    """
    d_s = PROBE_US * N_BEAMS
    d_l = PROBE_US * k_probe + d_compute_us
    lo = d_l
    if eta >= 1.0:
        return lo, float("inf")
    num = d_s - eta * d_l
    if num <= 0:
        return None, None            # dominated at every coherence time
    hi = num / (1.0 - eta)
    return (lo, hi) if hi > lo else (None, None)


# The four profiles run_all.sh writes.  p_gpu_heavy_full.json, the vision models re-timed on the
# GPU with the full numbers of calls, is read by the later stages; it is not an atlas profile.
PROFILE_FILES = ("p_cpu1t_ecore.json", "p_cpu1t_pcore.json", "p_cpu8t_pcore.json", "p_gpu.json")


def main(d):
    missing = [f for f in PROFILE_FILES if not os.path.exists(os.path.join(d, f))]
    if missing:
        sys.exit(f"missing profile files: {', '.join(missing)} (run run_all.sh)")
    profs = {}
    for f in PROFILE_FILES:
        j = json.load(open(os.path.join(d, f)))
        profs[j["env"]["profile"]] = j

    charge = {}          # (component, profile) -> us
    meta = {}
    for pname, j in profs.items():
        for r in j["rows"]:
            charge[(r["component"], pname)] = r["charge_us"]
            meta[r["component"]] = (r["provenance"], r["note"], r["reduced_n"])

    comps = [c for c in LABEL if c in {k[0] for k in charge}]
    order = [p for p in PROFILE_ORDER if p in profs]

    # ---- atlas table ----
    print("\n" + "=" * 100)
    print("PHASE 1 LATENCY ATLAS  -  per-decision p95 charge, microseconds")
    print("=" * 100)
    head = f"{'component':<38}{'prov':<7}" + "".join(
        f"{PROFILE_LABEL[p]:>17}" for p in order)
    print(head)
    print("-" * len(head))
    for c in comps:
        prov = meta[c][0][:5]
        row = f"{LABEL[c]:<38}{prov:<7}"
        for p in order:
            v = charge.get((c, p))
            row += f"{v:>17,.2f}" if v is not None else f"{'-':>17}"
        print(row)

    # ---- coherence budget at the G3 profile ----
    ref = "cpu-1t-pcore" if "cpu-1t-pcore" in profs else order[0]
    print("\n" + "=" * 100)
    print(f"COHERENCE BUDGET  -  profile {PROFILE_LABEL[ref]}, probe {PROBE_US:.0f} us/beam")
    print("=" * 100)
    hdr = f"{'method':<38}{'K':>4}{'probe us':>10}{'compute us':>12}"
    for t in T_MS:
        hdr += f"{t.split()[0]+'ms':>11}"
    print(hdr)
    print(f"{'':<38}{'':>4}{'':>10}{'':>12}" + "".join(
        f"{'payload %':>11}" for _ in T_MS))
    print("-" * len(hdr))
    budget = {}
    for c in comps:
        tau = charge.get((c, ref))
        if tau is None:
            continue
        k = K_PROBE[c]
        pr = PROBE_US * k
        dtot = pr + tau
        row = f"{LABEL[c]:<38}{k:>4}{pr:>10,.0f}{tau:>12,.2f}"
        cells = {}
        for name, t in T_MS.items():
            frac = max(0.0, 1.0 - dtot / (t * 1000.0))
            cells[name] = frac
            row += f"{frac*100:>10.1f}%"
        budget[c] = {"K": k, "probe_us": pr, "compute_us": tau, "payload": cells}
        print(row)

    # ---- rho = compute / T ----
    print("\n" + "=" * 100)
    print(f"rho = compute / T   (fraction of the coherence interval spent deliberating)")
    print("=" * 100)
    hdr = f"{'method':<38}" + "".join(f"{t.split()[0]+'ms':>13}" for t in T_MS)
    print(hdr); print("-" * len(hdr))
    for c in comps:
        tau = charge.get((c, ref))
        if tau is None:
            continue
        row = f"{LABEL[c]:<38}"
        for t in T_MS.values():
            row += f"{tau/(t*1000.0):>12.3f} "
        print(row)

    # ---- operating window ----
    print("\n" + "=" * 100)
    print("OPERATING WINDOW in coherence time  -  where the method beats the sweep")
    print("  published accounting: lower edge = 20K us.  honest: lower edge = 20K + compute.")
    print("=" * 100)
    for eta in (0.90, 0.95, 0.99):
        print(f"\n  eta = {eta:.2f}  (spectral-efficiency ratio vs. the true best beam)")
        print(f"  {'method':<36}{'published lo':>14}{'honest lo':>12}"
              f"{'upper':>12}{'window shrink':>15}")
        print("  " + "-" * 89)
        for c in comps:
            if c in ("sweep_select", "s1_r", "s1_u"):
                continue
            tau = charge.get((c, ref))
            if tau is None:
                continue
            k = K_PROBE[c]
            lo_p, hi_p = window(0.0, k, eta)
            lo_h, hi_h = window(tau, k, eta)
            if lo_h is None:
                print(f"  {LABEL[c]:<36}{lo_p:>13,.0f}u{'DOMINATED':>12}"
                      f"{'-':>12}{'window closes':>15}")
                continue
            shrink = (hi_p - lo_p) - (hi_h - lo_h)
            print(f"  {LABEL[c]:<36}{lo_p:>13,.0f}u{lo_h:>11,.0f}u"
                  f"{hi_h/1000:>11,.1f}m{shrink:>14,.0f}u")

    # ---- gate ----
    print("\n" + "=" * 100)
    print("PHASE 1 GATE")
    print("  'the compute segment must be non-trivial for at least one credible")
    print("   published architecture at T_c <= 2 ms'")
    print("  stop rule: 'if every architecture lands under 50 us, the thesis is weak'")
    print("=" * 100)
    pub = [c for c in comps if c not in
           ("sweep_select", "s1_r", "s1_u", "s1_q",
            "earlyexit_1", "earlyexit_2", "earlyexit_3")]
    worst = max(pub, key=lambda c: charge.get((c, ref), 0))
    tau_w = charge[(worst, ref)]
    over50 = [c for c in pub if charge.get((c, ref), 0) >= 50.0]
    print(f"  published architectures measured : {len(pub)}")
    print(f"  at or above 50 us on {PROFILE_LABEL[ref]:<22}: {len(over50)}"
          f"  -> {', '.join(LABEL[c] for c in over50) or 'none'}")
    print(f"  largest compute charge           : {LABEL[worst]} at {tau_w:,.2f} us")
    print(f"  that is {tau_w/2000.0*100:.2f}% of a 2 ms coherence interval")
    verdict = "PASS" if over50 else "FAIL"
    print(f"\n  GATE: {verdict}")

    json.dump({"charge_us": {f"{c}|{p}": v for (c, p), v in charge.items()},
               "budget_ref_profile": ref, "budget": budget,
               "gate": {"verdict": verdict, "over_50us": over50,
                        "largest": worst, "largest_us": tau_w}},
              open(os.path.join(d, "atlas.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(d, 'atlas.json')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
