"""One verification entry point; failed invariants return a nonzero status."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import unittest

from auditlib.data import verify_inputs, read_aggregate, order_summary, word_annotations, reconstruct_observations, analysis_frame, require, digest
from auditlib.consequences import analyze
from auditlib.incremental import analyze_incremental
from auditlib.scoring import score, load_cache
from auditlib.output import guarded, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true", help="Test implementation invariants only; not an empirical reproduction")
    for name in ("aggregate", "reading-times", "fixations", "pos", "gpt2", "gpt-neo", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--reuse-scores", action="store_true", help="Reuse hash-checked caches in output; explicitly not fresh model inference")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    require(unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(), "Synthetic tests failed")
    if args.synthetic_only:
        return
    required = ("aggregate", "reading_times", "fixations", "pos", "output")
    require(all(getattr(args, key) is not None for key in required), "Supply all input files and output directory")
    if not args.reuse_scores:
        require(args.gpt2 is not None and args.gpt_neo is not None, "Supply both pinned local model directories")
    paths = {key: getattr(args, key).resolve() for key in required if key != "output"}
    require(not args.output.is_symlink(), "Output directory must not be a symlink")
    output = args.output.resolve()
    # Never create outputs within or above any input/model directory.
    input_dirs = {path.parent for path in paths.values()}
    input_dirs |= {path.resolve() for path in (args.gpt2, args.gpt_neo) if path is not None}
    require(all(output != directory and not output.is_relative_to(directory) and not directory.is_relative_to(output) for directory in input_dirs),
            "Output directory overlaps an input/model directory")
    output.mkdir(parents=True, exist_ok=True)
    for leaf in ("results.json", "gpt2_scores.npz", "gpt2_scores.json", "gpt_neo_125m_scores.npz", "gpt_neo_125m_scores.json",
                 "incremental_fold_diagnostics.json", "incremental_fold_memberships.json"):
        guarded(output / leaf)
    started = time.time()
    hashes = verify_inputs(paths)
    raw = read_aggregate(paths["aggregate"])
    annotations = word_annotations(paths["reading_times"], paths["pos"])
    report = {"utc": datetime.now(timezone.utc).isoformat(), "scope": "FINITE_BENCHMARK_VERIFICATION_NOT_POPULATION_INFERENCE",
              "inputs_sha256": hashes, "python": platform.python_version(),
              "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "nltk", "torch", "transformers")},
              "order": order_summary(raw),
              "observations": reconstruct_observations(raw, paths["reading_times"], paths["fixations"]),
              "scores": {}, "analyses": []}
    frames = {}
    for name, directory in (("gpt2", args.gpt2), ("gpt_neo_125m", args.gpt_neo)):
        cache = output / (name + "_scores.npz")
        if args.reuse_scores:
            arrays, record = load_cache(cache, raw, name)
        else:
            require(not cache.exists(), "Existing score cache: use a new output directory or explicitly reuse scores")
            arrays, record = score(raw, directory, name, cache)
        report["scores"][name] = record
        frames[name] = analysis_frame(raw, arrays, name, annotations)
        report["analyses"].extend(analyze(frames[name], name))
    require(len(report["analyses"]) == 16, "Incomplete analysis design")
    require(all(run["mse_decrease_ms_squared"] > 0 for cell in report["analyses"] for run in cell["partition_runs"]), "Primary finite sign result differs")
    adverse = [cell["fixed_prediction_strata"][1]["mse_decrease_ms_squared"] for cell in report["analyses"] if cell["content_only"]]
    require(len(adverse) == 8 and all(value < 0 for value in adverse), "Required adverse stratum result differs")
    require(all(row["mse_decrease_ms_squared"] > 0 for cell in report["analyses"] for row in cell["deletion_refits"]), "Deletion result differs")
    incremental = analyze_incremental(frames, report["analyses"])
    rows = incremental["comparisons"]
    pattern = {"all_word_incremental_change_positive": sum(r["incremental_change"] > 0 for r in rows if not r["content_only"]),
               "content_incremental_change_negative": sum(r["incremental_change"] < 0 for r in rows if r["content_only"]),
               "negative_I_by_block": {b: sum(r["I"][b] < 0 for r in rows) for b in ("00", "10", "01", "11")}}
    require(pattern == {"all_word_incremental_change_positive": 160, "content_incremental_change_negative": 58,
                        "negative_I_by_block": {"00": 12, "10": 2, "01": 10, "11": 3}},
            "Declared finite incremental-value pattern differs")
    incremental["finite_pattern_counts"] = pattern
    artifacts = {}
    for key, filename in (("fold_diagnostics", "incremental_fold_diagnostics.json"),
                          ("fold_membership_hashes", "incremental_fold_memberships.json")):
        payload = incremental.pop(key)
        write_json(output / filename, payload)
        artifacts[key] = {"file": filename, "sha256": digest(output / filename), "records": len(payload)}
    incremental["diagnostic_artifacts"] = artifacts
    report["incremental_score"] = incremental
    report["elapsed_seconds"] = time.time() - started
    require(verify_inputs(paths) == hashes, "Input hashes changed during verification")
    report["input_hashes_unchanged_after_run"] = True
    report["status"] = "COMPUTATIONAL_CHECKS_PASSED"
    write_json(output / "results.json", report)
    print(json.dumps({"status": report["status"], "analyses": 16, "incremental_comparisons": len(rows),
                      "incremental_fits": incremental["unique_fits"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VERIFICATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
