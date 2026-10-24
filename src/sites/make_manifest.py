"""Assemble the frozen Phase 3 site manifest: everything Phase 4 needs to
reproduce the data pipeline after this working directory is gone."""
import hashlib, json, os, platform, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1                       # the observation-law constants, not copies of them
L = lambda n: json.load(open(os.path.join(HERE, n)))

acq, qual, xc = L('acquisition.json'), L('qualification.json'), L('catalog_crosscheck.json')
fix = L('fixtures_city_2_chicago_28.json')

terms = {
 "resolved": "citation obligation, recorded; not a legal opinion",
 "license_page": {
   "url": "https://www.deepmimo.net/license",
   "module": "https://www.deepmimo.net/assets/License-s-b5tBZI.js",
   "module_sha256": "05b2e63ff7d84bf63520e2560405310668fa276fb9d22d284c2a19d357d4cea7",
   "note": "same module filename the 2026-09-13 screen recovered; page unchanged since",
   "states": "In order to use the DeepMIMO dataset/scripts or any (modified) part of "
             "them, please cite: Alkhateeb 2019 (DeepMIMO, ITA) and the ray tracer "
             "(Remcom Wireless InSite for these scenarios). 'Select a specific scenario "
             "above to see additional required citations.'"},
 "per_scenario": {
   "source": "https://www.deepmimo.net/api/scenarios (233 entries)",
   "finding": "all eight archives carry no publicationCitation, no license and no "
              "scenario-specific raytracer citation field, so no additional "
              "publication is required beyond the two general references",
   "required_citations": ["Alkhateeb 2019, DeepMIMO, Proc. ITA, San Diego, Feb. 2019",
                          "Remcom, Wireless InSite"]},
 "software_label_discrepancy": {
   "artifact": "deepmimo-4.0.5-py3-none-any.whl",
   "sha256": "4470cd17654eccfe8ec837e987ee89ad516ee94acfd7a86ee6066738df12e035",
   "METADATA": "License-Expression: GPL-2.0-or-later",
   "bundled_LICENSE": "Apache License 2.0 (11,338 bytes, no GPL text present)",
   "status": "unresolved on the exact artifact, exactly as the 2026-09-13 screen "
             "recorded; concerns the software, not the scenario data; neither label "
             "prohibits local research use, and nothing is redistributed"}}

env = {"python": sys.version.split()[0], "platform": platform.platform(),
       "deepmimo": "4.0.5",
       "wheel_sha256": "4470cd17654eccfe8ec837e987ee89ad516ee94acfd7a86ee6066738df12e035",
       "wheel_sha256_matches_2026_09_13_screen": True,
       "pip_freeze": open(os.path.join(HERE, 'requirements-pinned.txt')).read().split()}

man = {
 "phase": 3, "date": "2026-09-15",
 "gate": {"rule": "at least 4 sites decode and qualify",
          "qualified": qual["n_qualified"], "of": len(acq["frozen_order"]),
          "pass": qual["gate_pass"]},
 "environment": env,
 "terms": terms,
 "observation_law": {
   "source": "G1, reused verbatim (scratchpad/atlas/calibrate.py:41-42, pilot-design.md)",
   "P_tx_W": 1.0, "P_tx_dBm": 30.0, "bandwidth_Hz": 100e6, "noise_figure_dB": 7.0,
   "sigma_d2_W": g1.SIGMA_D2, "sigma_p2_W": g1.SIGMA_P2,
   "processing_gain": g1.PILOT_SYMBOLS, "n_subbands": g1.N_SUBBANDS,
   "processing_gain_note": ("the pilot carries BANDWIDTH * 10 us = "
                            "PILOT_SYMBOLS complex dimensions in total; "
                            "sigma_p2 is per sub-band, so the band average "
                            "over n_subbands recovers that gain exactly once"),
   "t_probe_us": 20.0, "codebook": "normalized 64-beam DFT, ||f_m|| = 1",
   "array": "64-element ULA along +y, half-wavelength spacing, isotropic elements",
   "channel": "h[n] = sum_p sqrt(P_p) exp(j phi_p) exp(j n psi_p), "
              "psi_p = pi sin(theta_p) sin(phi_p), theta from aod_el (polar), phi from aod_az",
   "gamma": "P_tx |h^T f|^2 / sigma_d2",
   "power_reference_resolved": "stored path power is dBW received for a 1 W isotropic "
                               "transmit, i.e. path gain in dB; verified by Friis on "
                               f"{[c for c in fix['checks'] if c['name'].startswith('F4')][0]['n_los']} "
                               "LoS links to a median offset of "
                               f"{[c for c in fix['checks'] if c['name'].startswith('F4')][0]['median_offset_dB']} dB. "
                               "30 dBm = 1 W = the dataset reference, so the scale factor is exactly 1."},
 "fixtures": fix,
 "catalog_crosscheck": xc,
 "acquisition": acq,
 "sites": [], "frozen_split": qual["frozen_split"]}

by_zip = {r["scenario"]: r for r in acq["records"]}
for s in qual["sites"]:
    t = [x for x in s["transmitters"] if x["tx_set_id"] == s["chosen_tx_set_id"]][0]
    man["sites"].append({
        "order": s["order"], "scenario": s["scenario"], "role": s.get("role"),
        "qualified": s["qualified"],
        "zip_sha256": by_zip[s["scenario"]]["zip_sha256"],
        "zip_bytes": by_zip[s["scenario"]]["zip_bytes"],
        "tx_set_id": s["chosen_tx_set_id"], "tx_index": s["chosen_tx_index"],
        "n_rx_total": t["n_rx_total"], "n_eligible": t["n_eligible"],
        "n_zero_path": t["n_zero_path"],
        "n_structurally_inconsistent": t["n_structurally_inconsistent"],
        "n_los": t["n_los"], "eligible_sha256": t["eligible_sha256"],
        "best_beam_snr_dB_pct": t["best_beam_snr_dB_pct"],
        "frac_best_beam_below_0dB": t["frac_best_beam_below_0dB"],
        "per_tx_eligible": {str(x["tx_set_id"]): x["n_eligible"] for x in s["transmitters"]},
        "decode_rel_err": s["decode_rel_err"], "band": s["band"]})

p = os.path.join(HERE, 'site-manifest.json')
json.dump(man, open(p, 'w'), indent=1)
print("wrote", p, os.path.getsize(p), "bytes")
print("sha256", hashlib.sha256(open(p,'rb').read()).hexdigest())
