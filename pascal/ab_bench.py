"""A/B the Strata-Pascal changes on this machine: decode tok/s per variant, one engine start per variant.

    .venv/bin/python pascal/ab_bench.py [strata-*.json] [--variants base,fork,...] [--rounds 2]

Each variant is the installed config with some environment variables / engine flags changed; the rate is the median
over calibrate.py's three prompts (128 tokens each, temperature 0) after a warm-up, as calibrate measures it.  The
"base" variant switches off the fork's runtime-switchable changes (FP16 expert kernels, the Linux pin fix, freeing
other stages' dense matrices); the
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
    "base":     ({"STRATA_PASCAL_FP16": "0", "STRATA_PIN_LIMIT_GIB": "8", "STRATA_SPLIT_KEEP_DENSE": "1"}, {},
                 "fork changes that can be switched off, off"),
    "keepdense": ({"STRATA_SPLIT_KEEP_DENSE": "1"}, {}, "fork, but every card keeps all 48 layers' dense matrices"),
    "pin":      ({"STRATA_PASCAL_FP16": "0"}, {}, "+ whole arena pinned (Linux multi-GPU fix)"),
    "fp16":     ({"STRATA_PIN_LIMIT_GIB": "8"}, {}, "+ FP16 expert kernels only"),
    "fork":     ({}, {}, "every fork change (the default)"),
    "devplan":  ({"STRATA_VERIFY_DEVICE_PLAN": "1"}, {}, "fork + all-resident layers planned on the GPU"),
    "pciedma":  ({}, {"--pcie-mode": "dma"}, "fork + PCIe share copied by the copy engines instead of a copy kernel"),
    "pciedirect": ({}, {"--pcie-mode": "direct"}, "fork + PCIe share read in place by the expert kernels"),
    "pleram":   ({}, {"--ple-io": "ram"}, "fork + the 28.8 GB n-gram table held in RAM (no NVMe reads before a window)"),
    "smt":      ({}, {"--pool-workers": "22"}, "fork + CPU expert workers on hyperthreads too (22 instead of 11)"),
    "specsplit": ({}, {"--spec-split": True}, "fork + split windows: CPU experts of one half overlap the GPU's other half"),
    "spec6":    ({}, {"--spec": "6"}, "fork + verify windows up to 6 (+2 lookup)"),
    "spec8":    ({}, {"--spec": "8"}, "fork + verify windows up to 8"),
}


# the switchable options measured on top of the fork default (combined at the end when they win)
OPTIONS = ("devplan", "pciedma", "pciedirect", "pleram", "smt", "specsplit", "spec6", "spec8")


def set_flag(args: list[str], flag: str, value) -> list[str]:
    """`flag value`, or a bare boolean `flag` when value is True."""
    if value is True:
        return args if flag in args else args + [flag]
    return CAL.with_arg(args, flag, value)


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
    ap.add_argument("--out", help="folder for the results and one engine log per variant (default: pascal/)")
    ap.add_argument("--apply", action="store_true",
                    help="write the recommended options into the config (a .bak copy is kept) when they beat the default")
    ap.add_argument("--profile", action="store_true",
                    help="afterwards, one 512-token request with the engine's timing output on (profile.log)")
    a = ap.parse_args()
    cfg_path = Path(a.config) if a.config else newest_config()
    cfg = json.loads(cfg_path.read_text())
    out_dir = Path(a.out) if a.out else ROOT / "pascal"
    out_dir.mkdir(parents=True, exist_ok=True)
    from serve.server import StrataEngine, child_env
    ids_list = [CAL.chat_ids(tokenizer(cfg), p) for p in CAL.PROMPTS]
    base_args = list(cfg["args"])
    if isinstance(cfg.get("gpu"), list) and "--layer-split" not in base_args:
        base_args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    print(f"config {cfg_path.name}; {len(ids_list)} prompts x {CAL.MAX_NEW} tokens; {a.rounds} rounds per variant")
    results = {}

    def run_variant(name, env_over, flag_over, what):
        args = list(base_args)
        for f, v in flag_over.items():
            args = set_flag(args, f, v)
        env = child_env(cfg)
        env.update(env_over)
        print(f"[{name}] {what} ...", flush=True)
        t0 = time.time()
        log = str(out_dir / f"engine-{name}.log")
        eng = None
        try:
            eng = StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=log, env=env)
            s = CAL.Session(eng, ids_list)
            s.warm_up(1)
            rates = [s.rate() for _ in range(a.rounds)]
        except Exception as e:                          # a variant that fails to start or run is reported, not fatal
            print(f"  failed: {e}")
            results[name] = {"error": str(e)}
            return
        finally:
            if eng is not None:
                CAL.close(eng)
        results[name] = {"tok_s": round(statistics.median(rates), 2), "rates": [round(r, 2) for r in rates],
                         "seconds": round(time.time() - t0)}
        print(f"  {results[name]['tok_s']:.2f} tok/s  (rounds {results[name]['rates']})", flush=True)

    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            print(f"  unknown variant {name!r}, skipped")
            continue
        run_variant(name, *VARIANTS[name])

    # every "fork + X" option that beat the fork default by more than MIN_GAIN, together (the better of spec6/8)
    fork = (results.get("fork") or {}).get("tok_s")
    if fork:
        wins = [n for n in OPTIONS if (results.get(n) or {}).get("tok_s", 0) > fork * (1 + CAL.MIN_GAIN)]
        for x, y in (("spec6", "spec8"), ("pciedma", "pciedirect")):   # the same flag: keep the better one
            if x in wins and y in wins:
                wins.remove(x if results[x]["tok_s"] < results[y]["tok_s"] else y)
        if len(wins) >= 2:
            env_c, flags_c = {}, {}
            for n in wins:
                env_c.update(VARIANTS[n][0])
                flags_c.update(VARIANTS[n][1])
            run_variant("best", env_c, flags_c, "fork + " + " + ".join(wins))
        elif wins:
            env_c, flags_c = VARIANTS[wins[0]][0], VARIANTS[wins[0]][1]
        if wins:
            best = "best" if len(wins) >= 2 else wins[0]
            print(f"\nrecommended: {' + '.join(wins)} -> in {cfg_path.name}:")
            if flags_c:
                print("  \"args\": add " + ", ".join(f'"{k}"' if v is True else f'"{k}", "{v}"' for k, v in flags_c.items()))
            if env_c:
                print("  \"env\": " + json.dumps(env_c))
            best_rate = (results.get(best) or {}).get("tok_s") or 0
            results["recommended"] = {"options": wins, "flags": flags_c, "env": env_c, "tok_s": best_rate}
            if a.apply and best_rate > fork * (1 + CAL.MIN_GAIN):
                bak = cfg_path.with_name(cfg_path.name + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
                bak.write_text(cfg_path.read_text())
                new_cfg = json.loads(cfg_path.read_text())
                for k, v in flags_c.items():
                    new_cfg["args"] = set_flag(new_cfg["args"], k, v)
                new_cfg["env"] = {**(new_cfg.get("env") or {}), **env_c}
                cfg_path.write_text(json.dumps(new_cfg, indent=1))
                print(f"  applied to {cfg_path.name} (previous config saved as {bak.name})")
        else:
            print("\nrecommended: the defaults (no option beat them by more than 3%)")
    base = (results.get("base") or {}).get("tok_s")
    print("\nvariant    tok/s    vs base")
    for name, r in results.items():
        if "tok_s" in r and name != "recommended":
            rel = f"{(r['tok_s'] / base - 1) * 100:+.1f}%" if base else "-"
            print(f"{name:<10} {r['tok_s']:>6.2f}   {rel}")
    out = out_dir / f"ab_results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"config": cfg_path.name, "results": results}, indent=1))
    print(f"\nsaved {out}")
    if a.profile:
        profile(cfg, base_args, ids_list[0], out_dir, StrataEngine, child_env)
    return 0


def profile(cfg, args, ids, out_dir: Path, StrataEngine, child_env):
    """One long request with the engine's per-window timing on: where the time goes (see PERF_ANALYSIS.md)."""
    import threading
    env = child_env(cfg)
    env.update({"STRATA_DECODE_TIMING": "1", "STRATA_SPLIT_TIMING": "1", "STRATA_VERIFY_PROFILE": "1"})
    log = out_dir / "profile.log"
    print(f"[profile] 512 tokens with timing output -> {log.name} ...", flush=True)
    eng = None
    try:
        eng = StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=str(log), env=env)
        CAL.Session(eng, [ids]).warm_up(1)
        n = sum(1 for t in eng.generate(ids, 512, {"temperature": 0}, threading.Event()) if t is not None)
        ms = (eng.last or {}).get("decode_ms") or 0.0
        print(f"  {n} tokens, {n / (ms / 1000.0):.2f} tok/s" if ms else f"  {n} tokens", flush=True)
    except Exception as e:
        print(f"  profile failed: {e}")
    finally:
        if eng is not None:
            CAL.close(eng)


if __name__ == "__main__":
    sys.exit(main())
