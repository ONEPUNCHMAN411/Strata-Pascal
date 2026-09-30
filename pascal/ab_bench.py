"""A/B the Strata-Pascal changes on this machine: decode tok/s per variant, one engine start per variant.

    .venv/bin/python pascal/ab_bench.py [strata-*.json] [--variants base,fork,...] [--rounds 2]

Each variant is the installed config with some environment variables / engine flags changed; the rate is the median
over calibrate.py's three prompts (128 tokens each, temperature 0) after a warm-up, as calibrate measures it.  The
"base" variant switches off the fork's runtime-switchable changes (FP16 expert kernels, the Linux pin fix); the
compile-time ones (fused_gr one-launch tile, shared-memory grids) cannot be switched off, so "base" is not exactly
upstream.  Results go to stdout and pascal/ab_results-<time>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import calibrate as CAL  # noqa: E402

# name: (env overrides, flag overrides {flag: value or None to remove}, what it tests)
VARIANTS = {
    "base":     ({"STRATA_PASCAL_FP16": "0", "STRATA_PIN_LIMIT_GIB": "8"}, {}, "fork changes that can be switched off, off"),
    "pin":      ({"STRATA_PASCAL_FP16": "0"}, {}, "+ whole arena pinned (Linux multi-GPU fix)"),
    "fp16":     ({"STRATA_PIN_LIMIT_GIB": "8"}, {}, "+ FP16 expert kernels only"),
    "fork":     ({}, {}, "every fork change (the default)"),
    "devplan":  ({"STRATA_VERIFY_DEVICE_PLAN": "1"}, {}, "fork + all-resident layers planned on the GPU"),
    "spec6":    ({}, {"--spec": "6"}, "fork + verify windows up to 6 (+2 lookup)"),
    "spec8":    ({}, {"--spec": "8"}, "fork + verify windows up to 8"),
}


def newest_config() -> Path:
    cands = sorted(ROOT.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        sys.exit("no strata-*.json here: run ./setup.sh first")
    return cands[0]


def tokenizer(cfg: dict):
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tpath / "token_type.json").read_text()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma list of: " + ", ".join(VARIANTS))
    ap.add_argument("--rounds", type=int, default=2, help="measurements per variant (median taken)")
    a = ap.parse_args()
    cfg_path = Path(a.config) if a.config else newest_config()
    cfg = json.loads(cfg_path.read_text())
    from serve.server import StrataEngine, child_env
    ids_list = [CAL.chat_ids(tokenizer(cfg), p) for p in CAL.PROMPTS]
    base_args = list(cfg["args"])
    if isinstance(cfg.get("gpu"), list) and "--layer-split" not in base_args:
        base_args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    print(f"config {cfg_path.name}; {len(ids_list)} prompts x {CAL.MAX_NEW} tokens; {a.rounds} rounds per variant")
    results = {}
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            print(f"  unknown variant {name!r}, skipped")
            continue
        env_over, flag_over, what = VARIANTS[name]
        args = list(base_args)
        for f, v in flag_over.items():
            args = CAL.with_arg(args, f, v)
        env = child_env(cfg)
        env.update(env_over)
        print(f"[{name}] {what} ...", flush=True)
        t0 = time.time()
        eng = StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=env)
        try:
            s = CAL.Session(eng, ids_list)
            s.warm_up(1)
            rates = [s.rate() for _ in range(a.rounds)]
        except Exception as e:                          # a variant that fails to start or run is reported, not fatal
            print(f"  failed: {e}")
            results[name] = {"error": str(e)}
            continue
        finally:
            CAL.close(eng)
        results[name] = {"tok_s": round(statistics.median(rates), 2), "rates": [round(r, 2) for r in rates],
                         "seconds": round(time.time() - t0)}
        print(f"  {results[name]['tok_s']:.2f} tok/s  (rounds {results[name]['rates']})", flush=True)
    base = (results.get("base") or {}).get("tok_s")
    print("\nvariant    tok/s    vs base")
    for name, r in results.items():
        if "tok_s" in r:
            rel = f"{(r['tok_s'] / base - 1) * 100:+.1f}%" if base else "-"
            print(f"{name:<10} {r['tok_s']:>6.2f}   {rel}")
    out = ROOT / "pascal" / f"ab_results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"config": cfg_path.name, "results": results}, indent=1))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
