"""Matched nuisance/full prediction under four position/predecessor constructions.

All statistics concern the fixed benchmark. Partitions are computational
sensitivities, not independent replications. No records or source text are exported.
"""
import hashlib

import numpy as np

from .consequences import OUTCOMES, SEEDS, design, split_groups
from .data import require

MODELS = ("gpt2", "gpt_neo_125m")
SOURCES = ("stored", "canonical")
BLOCKS = ((0, 0), (1, 0), (0, 1), (1, 1))
KEYS = {"00", "10", "01", "11"}
TOLERANCE_MSE = 1e-9
MASK_HASHES = {
    False: "a30e322237b6703c4055960a2698287df5a6f8cf7c7316e83ef06e58d6c56564",
    True: "6ba0957ec08992e4459493520392a19db5d0c536a4f58b0e9d8665f672afc2ce",
}


def occurrence_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()


def block_features(frame, source, position, predecessor):
    require(source in SOURCES and position in (0, 1) and predecessor in (0, 1),
            "Unknown score or order construction")
    result = frame.copy()
    result["position"] = frame["position" if position else "wrong_position"]
    result["surprisal"] = frame[source]
    prefix = "lag_" if predecessor else "wrong_lag_"
    for name in ("length", "frequency"):
        result["lag_" + name] = frame[prefix + name]
    result["lag_surprisal"] = frame[prefix + source]
    return result


def fit(train, test, response):
    require(train.ndim == test.ndim == 2 and response.ndim == 1,
            "Invalid fit dimensions")
    require(len(train) == len(response) and train.shape[1] == test.shape[1],
            "Fit shapes differ")
    require(len(train) > 0 and len(test) > 0 and train.shape[1] > 0,
            "Empty fit input")
    require(np.isfinite(train).all() and np.isfinite(test).all() and
            np.isfinite(response).all(), "Nonfinite fit input")
    coef, _, rank, singular = np.linalg.lstsq(train, response, rcond=None)
    require(rank == train.shape[1], "Rank-deficient fit")
    prediction = test @ coef
    require(np.isfinite(prediction).all(), "Nonfinite prediction")
    return prediction, {"rank": int(rank), "columns": train.shape[1],
                        "singular_value_min": float(singular[-1]),
                        "singular_value_max": float(singular[0])}


def pooled_mse(response, prediction):
    require(response.ndim == prediction.ndim == 1 and response.shape == prediction.shape,
            "Loss shapes differ")
    require(len(response) > 0 and np.isfinite(response).all() and
            np.isfinite(prediction).all(), "Empty or nonfinite loss input")
    return float(np.mean((response - prediction) ** 2))


def path_contrasts(values, loss=True):
    require(set(values) == KEYS and np.isfinite(list(values.values())).all(),
            "Incomplete or nonfinite block contrast")
    direction = 1 if loss else -1
    p0 = direction * (values["00"] - values["10"])
    p1 = direction * (values["01"] - values["11"])
    l0 = direction * (values["00"] - values["01"])
    l1 = direction * (values["10"] - values["11"])
    require(abs((p1 - p0) - (l1 - l0)) < TOLERANCE_MSE, "Path identity differs")
    return {"position_at_L0": p0, "position_at_L1": p1,
            "predecessor_at_P0": l0, "predecessor_at_P1": l1,
            "nonadditivity_position_L1_minus_L0": p1 - p0}


def decomposition(nuisance, full):
    require(set(nuisance) == set(full) == KEYS, "Incomplete four-block losses")
    values = np.asarray([*nuisance.values(), *full.values()], dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all(),
            "MSE must be finite and nonnegative")
    incremental = {k: nuisance[k] - full[k] for k in nuisance}
    joint = full["00"] - full["11"]
    nuisance_gain = nuisance["00"] - nuisance["11"]
    added_value_change = incremental["11"] - incremental["00"]
    residual = joint - nuisance_gain - added_value_change
    require(abs(residual) < TOLERANCE_MSE, "Decomposition identity differs")
    return {"N": nuisance, "F": full, "I": incremental,
            "I_over_N": {k: incremental[k] / nuisance[k] if nuisance[k] > 0 else None
                         for k in nuisance},
            "I_over_N_defined": {k: nuisance[k] > 0 for k in nuisance},
            "joint_full_gain": joint, "joint_nuisance_gain": nuisance_gain,
            "incremental_change": added_value_change, "identity_residual": residual,
            "N_paths": path_contrasts(nuisance), "F_paths": path_contrasts(full),
            "I_paths": path_contrasts(incremental, loss=False)}


def check_endpoints(full, previous):
    errors = [abs(full["00"] - previous["wrong_order_rmse_ms"] ** 2),
              abs(full["11"] - previous["correct_order_rmse_ms"] ** 2),
              abs(full["00"] - full["11"] - previous["mse_decrease_ms_squared"])]
    require(np.isfinite(errors).all() and max(errors) < TOLERANCE_MSE,
            "Original full-model endpoint changed")
    return max(errors)


def select_samples(frames):
    require(set(frames) == set(MODELS), "Both fixed checkpoints are required")
    needed = ["lag_canonical", "lag_length", "lag_frequency", "wrong_lag_canonical",
              "wrong_lag_length", "wrong_lag_frequency"]
    samples, profiles = {}, []
    for model in MODELS:
        frame = frames[model]
        require(frame.index.is_unique, "Duplicate occurrence indices")
        for content in (False, True):
            selected = frame.loc[frame.content].copy() if content else frame.copy()
            selected = selected.loc[np.isfinite(selected[needed]).all(axis=1)].copy()
            require(len(selected) == (510 if content else 1104), "Common-support count differs")
            require(selected.sent_id.nunique() == (195 if content else 204), "Sentence count differs")
            require(occurrence_digest(selected.index.to_numpy()) == MASK_HASHES[content],
                    "Exact common-support membership differs")
            require(np.isfinite(selected[[*OUTCOMES, "lag_stored", "wrong_lag_stored"]]).all().all(),
                    "Nonfinite response or stored lag on common support")
            samples[model, content] = selected
    for content in (False, True):
        a, b = (samples[model, content] for model in MODELS)
        require(a.index.equals(b.index) and np.array_equal(a.sent_id, b.sent_id),
                "Checkpoint samples differ")
        require(np.array_equal(a[list(OUTCOMES)], b[list(OUTCOMES)]), "Checkpoint responses differ")
        profiles.append({"content_only": content, "rows": len(a),
                         "sentences": int(a.sent_id.nunique()),
                         "occurrence_mask_sha256": occurrence_digest(a.index.to_numpy())})
    return samples, profiles


def analyze_incremental(frames, prior_analyses):
    """Return complete aggregate records and safe fold diagnostics.

    The original 16 endpoint analyses are required and checked for all20 seeds.
    No inherited/private module, private path, or expected private result file is
    needed by this public entry point.
    """
    samples, profiles = select_samples(frames)
    prior = {}
    require(len(prior_analyses) == 16, "Incomplete original endpoint analyses")
    for cell in prior_analyses:
        require([p["seed"] for p in cell["partition_runs"]] == list(SEEDS),
                "Original endpoint partition set differs")
        for partition in cell["partition_runs"]:
            key = (cell["model"], cell["content_only"], cell["outcome"], cell["source"], partition["seed"])
            require(key not in prior, "Duplicate original endpoint comparison")
            prior[key] = partition
    require(len(prior) == 320, "Original endpoint coverage differs")
    baselines, comparisons, diagnostics, fold_records = [], [], [], []
    maximum_prior_error = 0.0
    for content in (False, True):
        reference = samples[MODELS[0], content]
        for seed in SEEDS:
            splits = list(split_groups(reference.sent_id.to_numpy(), 5, seed))
            visits = np.zeros(len(reference), dtype=int)
            for fold, (tr, te) in enumerate(splits):
                visits[te] += 1
                fold_records.append({"content_only": content, "seed": seed, "fold": fold,
                    "train_rows": len(tr), "test_rows": len(te),
                    "train_occurrences_sha256": occurrence_digest(reference.index.to_numpy()[tr]),
                    "test_occurrences_sha256": occurrence_digest(reference.index.to_numpy()[te])})
            require(np.all(visits == 1), "Incomplete common held-out coverage")
            for outcome in OUTCOMES:
                y = reference[outcome].to_numpy()
                nuisance = {}
                full = {(m, c): {} for m in MODELS for c in SOURCES}
                for p, l in BLOCKS:
                    block = str(p) + str(l)
                    baseline_frame = block_features(reference, "canonical", p, l)
                    matrices = [design(baseline_frame.iloc[tr], baseline_frame.iloc[te]) for tr, te in splits]
                    prediction = np.full(len(reference), np.nan)
                    for fold, ((tr, te), (a, b)) in enumerate(zip(splits, matrices)):
                        predicted, diag = fit(a[:, :9], b[:, :9], y[tr])
                        prediction[te] = predicted
                        diagnostics.append({"kind": "nuisance", "content_only": content,
                            "outcome": outcome, "seed": seed, "block": block, "fold": fold,
                            "heldout_mse": pooled_mse(y[te], predicted), **diag})
                    nuisance[block] = pooled_mse(y, prediction)
                    baselines.append({"content_only": content, "outcome": outcome, "seed": seed,
                                      "block": block, "mse": nuisance[block]})
                    for model in MODELS:
                        for source in SOURCES:
                            f = block_features(samples[model, content], source, p, l)
                            prediction = np.full(len(reference), np.nan)
                            for fold, ((tr, te), (a0, b0)) in enumerate(zip(splits, matrices)):
                                a, b = design(f.iloc[tr], f.iloc[te])
                                require(np.array_equal(a[:, :9], a0[:, :9]) and
                                        np.array_equal(b[:, :9], b0[:, :9]), "Nuisance design differs")
                                predicted, diag = fit(a, b, y[tr])
                                prediction[te] = predicted
                                diagnostics.append({"kind": "full", "model": model, "source": source,
                                    "content_only": content, "outcome": outcome, "seed": seed,
                                    "block": block, "fold": fold,
                                    "heldout_mse": pooled_mse(y[te], predicted), **diag})
                            full[model, source][block] = pooled_mse(y, prediction)
                for (model, source), f in full.items():
                    error = check_endpoints(f, prior[model, content, outcome, source, seed])
                    maximum_prior_error = max(maximum_prior_error, error)
                    comparisons.append({"content_only": content, "outcome": outcome,
                        "seed": seed, "model": model, "source": source,
                        **decomposition(nuisance.copy(), f)})
            print(f"Incremental fits: content={content}, seed={seed}", flush=True)
    require(len(diagnostics) == 8000 and len(baselines) == len(comparisons) == 320
            and len(fold_records) == 200, "Declared extension grid differs")
    baseline_keys = {(r["content_only"], r["outcome"], r["seed"], r["block"]) for r in baselines}
    require(len(baseline_keys) == 320, "Baseline duplicated across model/score labels")
    require(sum(r["kind"] == "nuisance" for r in diagnostics) == 1600, "Baseline fit count differs")
    return {"status": "FINITE_INCREMENTAL_COMPUTATION_CHECKED", "seeds": list(SEEDS),
            "folds": 5, "samples": profiles, "baselines": baselines, "comparisons": comparisons,
            "unique_fits": 8000, "nuisance_fits": 1600, "full_fits": 6400,
            "maximum_prior_mse_difference": maximum_prior_error,
            "fold_diagnostics": diagnostics, "fold_membership_hashes": fold_records,
            "contrast_convention": {
                "loss_paths": "before minus after; positive means loss reduction",
                "I_paths": "after minus before; positive means incremental-value increase",
                "nonadditivity": "position contrast at L1 minus at L0; also predecessor contrast at P1 minus at P0",
                "I_over_N": "null and defined=false when baseline MSE is zero"},
            "limitations": ["Exploratory finite benchmark comparison", "Partitions are dependent sensitivities, not confidence intervals",
                            "Block contrasts are not causal effects or a unique allocation", "Negative I and incremental changes are retained"]}
