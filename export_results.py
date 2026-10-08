"""Export complete English aggregate tables and original figures from a verified run."""
from pathlib import Path
import argparse
import csv
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from auditlib.data import require, digest
from auditlib.output import atomic_file, write_json

FILES=[]
OUT=None


def save_csv(name, rows):
    text=io.StringIO(newline="")
    writer=csv.DictWriter(text,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path=OUT/(name+".csv")
    payload=text.getvalue().encode()
    atomic_file(path,lambda stream:stream.write(payload))
    FILES.append({"file":path.name,"sha256":digest(path)})


def save_figure(fig,name,description):
    for ext in ("pdf","png","jpg"):
        path=OUT/(name+"."+ext)
        metadata={"metadata":{"Creator":"Matplotlib","Title":description}} if ext=="pdf" else {}
        atomic_file(path,lambda stream:fig.savefig(stream,format=ext,dpi=400,**metadata))
        FILES.append({"file":path.name,"sha256":digest(path),"description":description})
    plt.close(fig)


def english(data):
    require(len(data["analyses"]) == 16 and len(data["seeds"]) == 20, "Incomplete original result grid")
    rows, strata, deletions = [], [], []
    for a in data["analyses"]:
        identity = {k: a[k] for k in ("model", "content_only", "outcome", "source", "rows", "sentences")}
        runs = a["partition_runs"]
        first = runs[0]
        require(first["seed"] == 20260926 and len(runs) == 20, "Partition sequence differs")
        relative = np.array([r["relative_mse_decrease"] for r in runs]) * 100
        rmse = np.array([r["wrong_order_rmse_ms"] - r["correct_order_rmse_ms"] for r in runs])
        rows.append({**identity, "reference_wrong_rmse_ms": first["wrong_order_rmse_ms"],
                     "reference_keyed_rmse_ms": first["correct_order_rmse_ms"],
                     "reference_mse_decrease_ms2": first["mse_decrease_ms_squared"],
                     "relative_mse_decrease_min_percent": float(relative.min()),
                     "relative_mse_decrease_max_percent": float(relative.max()),
                     "rmse_decrease_min_ms": float(rmse.min()), "rmse_decrease_max_ms": float(rmse.max()),
                     "positive_partition_count": int((relative > 0).sum())})
        for s in a["fixed_prediction_strata"]:
            strata.append({**{k: identity[k] for k in ("model", "content_only", "outcome", "source")}, **s})
        for d in a["deletion_refits"]:
            deletions.append({**{k: identity[k] for k in ("model", "content_only", "outcome", "source")}, **d,
                              "folds_regenerated": True})
    save_csv("en_order_consequences", rows)
    save_csv("en_order_strata", strata)
    save_csv("en_order_deletions", deletions)
    canonical = [r for r in rows if r["source"] == "canonical"]
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.65), layout="constrained", sharey=True)
    labels = ["GPT-2 · all", "GPT-Neo · all", "GPT-2 · content", "GPT-Neo · content"]
    combinations = [(False, "gpt2"), (False, "gpt_neo_125m"), (True, "gpt2"), (True, "gpt_neo_125m")]
    for ax, outcome, title in zip(axes, ("RTfirstfix", "RTfirstpass"), ("First fixation", "First pass")):
        for i, (content, model) in enumerate(combinations):
            r = next(x for x in canonical if x["model"] == model and x["content_only"] == content and x["outcome"] == outcome)
            lo, hi = r["relative_mse_decrease_min_percent"], r["relative_mse_decrease_max_percent"]
            ax.plot([lo, hi], [3-i, 3-i], color="#17648a", lw=3)
            ax.plot([lo, hi], [3-i, 3-i], "|", color="#17648a", ms=8)
        ax.set_xlim(0, 25)
        ax.set_ylim(-.5, 3.5)
        ax.set_title(title)
        ax.set_xlabel("Overall MSE decrease (%)")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(4), labels[::-1])
    fig.supxlabel("Ranges over 20 sentence partitions; not confidence intervals.\nCanonical scores; joint position and predecessor correction.", fontsize=8)
    save_figure(fig, "en_partition_ranges", "Finite partition sensitivity of keyed-order prediction")

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.55), layout="constrained", sharey=True)
    for ax, outcome, title in zip(axes, ("RTfirstfix", "RTfirstpass"), ("First fixation", "First pass")):
        for i, model in enumerate(("gpt2", "gpt_neo_125m")):
            for flag, marker, offset, color in ((False, "o", .13, "#17648a"), (True, "s", -.13, "#ae471e")):
                r = next(x for x in strata if x["model"] == model and x["content_only"] and x["outcome"] == outcome and x["source"] == "canonical" and x["any_multitoken_current_or_predecessor"] == flag)
                ax.plot(r["mse_decrease_ms_squared"], 1-i+offset, marker=marker, color=color, ms=5,
                        label=("Unflagged (432)" if not flag else "Flagged (78)") if i == 0 else None)
        ax.axvline(0, color=".45", lw=.8)
        ax.set_title(title)
        ax.set_xlabel("MSE decrease (ms²)")
        ax.grid(axis="x", alpha=.2)
        ax.set_ylim(-.5, 1.6)
    axes[0].set_yticks([0, 1], ["GPT-Neo", "GPT-2"])
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.supxlabel("Canonical scores; content words; reference partition.\nFixed held-out predictions; negative values indicate higher loss after correction.", fontsize=8)
    save_figure(fig, "en_adverse_stratum", "Heterogeneous keyed-order consequences in content-word strata")
    return len(rows), len(strata), len(deletions)


def incremental(data):
    extension = data.get("incremental_score")
    require(extension is not None and extension["status"] == "FINITE_INCREMENTAL_COMPUTATION_CHECKED",
            "Rerun the current verifier to include the incremental-score extension")
    require(extension["unique_fits"] == 8000 and len(extension["comparisons"]) == 320
            and len(extension["baselines"]) == 320, "Incomplete incremental result grid")
    save_csv("en_incremental_baselines", extension["baselines"])
    blocks, changes, paths, groups = [], [], [], {}
    for row in extension["comparisons"]:
        identity = {k: row[k] for k in ("content_only", "outcome", "seed", "model", "source")}
        for block in ("00", "10", "01", "11"):
            blocks.append({**identity, "block": block, "N_mse_ms2": row["N"][block],
                           "F_mse_ms2": row["F"][block], "I_ms2": row["I"][block],
                           "I_over_N": row["I_over_N"][block],
                           "I_over_N_defined": row["I_over_N_defined"][block],
                           "N_rmse_ms": float(np.sqrt(row["N"][block])),
                           "F_rmse_ms": float(np.sqrt(row["F"][block]))})
        changes.append({**identity, "joint_full_gain_ms2": row["joint_full_gain"],
                        "joint_nuisance_gain_ms2": row["joint_nuisance_gain"],
                        "incremental_change_ms2": row["incremental_change"],
                        "identity_residual_ms2": row["identity_residual"]})
        paths.append({**identity, **{metric + "_" + key + "_ms2": value
                       for metric in ("N", "F", "I") for key, value in row[metric + "_paths"].items()}})
        key = tuple(identity[k] for k in ("content_only", "outcome", "model", "source"))
        groups.setdefault(key, []).append(row)
    summaries = []
    require(len(groups) == 16, "Incremental condition count differs")
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["seed"])
        require([r["seed"] for r in rows] == extension["seeds"], "Incremental partition sequence differs")
        values = np.array([r["incremental_change"] for r in rows])
        identity = dict(zip(("content_only", "outcome", "model", "source"), key))
        summaries.append({**identity, "reference_seed": rows[0]["seed"],
            "reference_incremental_change_ms2": float(values[0]),
            "mean_incremental_change_ms2": float(values.mean()),
            "min_incremental_change_ms2": float(values.min()),
            "max_incremental_change_ms2": float(values.max()),
            "negative_partition_count": int((values < 0).sum()),
            "zero_partition_count": int((values == 0).sum()),
            "positive_partition_count": int((values > 0).sum()),
            "reference_I00_ms2": rows[0]["I"]["00"], "reference_I11_ms2": rows[0]["I"]["11"],
            "mean_I00_ms2": float(np.mean([r["I"]["00"] for r in rows])),
            "mean_I11_ms2": float(np.mean([r["I"]["11"] for r in rows])),
            "mean_joint_full_gain_ms2": float(np.mean([r["joint_full_gain"] for r in rows])),
            "mean_joint_nuisance_gain_ms2": float(np.mean([r["joint_nuisance_gain"] for r in rows]))})
    save_csv("en_incremental_scores", blocks)
    save_csv("en_incremental_changes", changes)
    save_csv("en_incremental_paths", paths)
    save_csv("en_incremental_summary", summaries)
    return {"baselines": 320, "scores": len(blocks), "changes": len(changes),
            "paths": len(paths), "summary": len(summaries)}


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    data=json.loads(args.results.read_text())
    require(data["status"]=="COMPUTATIONAL_CHECKS_PASSED","Run verification first")
    require("incremental_score" in data,"Rerun the current verifier for the central extension")
    data["seeds"]=[r["seed"] for r in data["analyses"][0]["partition_runs"]]
    require(not args.output.is_symlink(),"Export directory must not be a symlink")
    OUT=args.output.resolve()
    require(args.results.resolve().parent!=OUT,"Keep exports separate from the run record")
    OUT.mkdir(parents=True,exist_ok=True)
    FILES.clear()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                        "axes.spines.right":False,"savefig.facecolor":"white","pdf.fonttype":42})
    counts=english(data)
    incremental_counts=incremental(data)
    write_json(OUT/"figure_manifest.json",{"result_sha256":digest(args.results),"table_row_counts":counts,
               "incremental_table_row_counts":incremental_counts,
               "files":FILES,"visual_status":"RENDERED_NOT_VISUALLY_CERTIFIED_BY_SCRIPT"})
    print(json.dumps({"files":len(FILES),"table_rows":counts,"incremental_table_rows":incremental_counts}))


if __name__=="__main__":
    main()
