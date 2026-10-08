"""Fixed-form, sentence-held-out consequences on common observations."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .data import require

SEED = 20260926
SEEDS = tuple(SEED + i for i in range(20))
OUTCOMES = ("RTfirstfix", "RTfirstpass")


def split_groups(groups, k, seed):
    unique = np.sort(np.unique(groups))
    require(len(unique) >= k, "Too few groups")
    for train, test in KFold(k, shuffle=True, random_state=seed).split(unique):
        tr = np.flatnonzero(np.isin(groups, unique[train]))
        te = np.flatnonzero(np.isin(groups, unique[test]))
        require(not set(groups[tr]) & set(groups[te]), "Sentence leakage")
        yield tr, te


def design(train, test):
    cov = ["length", "frequency", "position", "lag_length", "lag_frequency"]
    a, b = train[cov].to_numpy(), test[cov].to_numpy()
    center, scale = a.mean(axis=0), a.std(axis=0)
    scale[scale < 1e-12] = 1.0
    a, b = (a - center) / scale, (b - center) / scale
    left, right = [np.ones(len(a)), *a.T], [np.ones(len(b)), *b.T]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        left.append(a[:, i] * a[:, j])
        right.append(b[:, i] * b[:, j])
    for name in ("surprisal", "lag_surprisal"):
        u, v = train[name].to_numpy(), test[name].to_numpy()
        center, scale = u.mean(), u.std()
        scale = scale if scale >= 1e-12 else 1.0
        left.append((u - center) / scale)
        right.append((v - center) / scale)
    a, b = np.column_stack(left), np.column_stack(right)
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite design")
    return a, b


def features(frame, source, correct):
    f = frame.assign(surprisal=frame[source]).copy()
    if not correct:
        f["position"] = frame.wrong_position
    prefix = "lag_" if correct else "wrong_lag_"
    for name in ("length", "frequency"):
        f["lag_" + name] = frame[prefix + name]
    f["lag_surprisal"] = frame[prefix + source]
    return f


def predict(frame, outcome, seed):
    prediction = np.full(len(frame), np.nan)
    for tr, te in split_groups(frame.sent_id.to_numpy(), 5, seed):
        a, b = design(frame.iloc[tr], frame.iloc[te])
        coef, _, rank, _ = np.linalg.lstsq(a, frame.iloc[tr][outcome].to_numpy(), rcond=None)
        require(rank == a.shape[1], "Rank-deficient OLS design")
        prediction[te] = b @ coef
    require(np.isfinite(prediction).all(), "Incomplete held-out predictions")
    return prediction


def losses(y, wrong, correct):
    a, b = (y - wrong) ** 2, (y - correct) ** 2
    return {"rows": len(y), "wrong_order_rmse_ms": float(np.sqrt(a.mean())),
            "correct_order_rmse_ms": float(np.sqrt(b.mean())),
            "mse_decrease_ms_squared": float(np.mean(a - b)),
            "relative_mse_decrease": float(1 - b.mean() / a.mean())}


def analyze(frame, model):
    result = []
    needed = ["lag_canonical", "lag_length", "lag_frequency", "wrong_lag_canonical", "wrong_lag_length", "wrong_lag_frequency"]
    for content in (False, True):
        selected = frame.loc[frame.content].copy() if content else frame.copy()
        selected = selected.loc[np.isfinite(selected[needed]).all(axis=1)].copy()
        require(len(selected) == (510 if content else 1104), "Lag-eligible sample differs")
        require(np.isin(selected[["multitoken", "correct_lag_affected", "wrong_lag_affected"]], [0, 1]).all(), "Unresolved flag")
        affected = (selected.multitoken.astype(bool) | selected.correct_lag_affected.astype(bool) | selected.wrong_lag_affected.astype(bool)).to_numpy()
        affected_sentences = selected.loc[affected, "sent_id"].unique()
        for outcome in OUTCOMES:
            for source in ("stored", "canonical"):
                wrong, correct = features(selected, source, False), features(selected, source, True)
                y, runs = selected[outcome].to_numpy(), []
                for seed in SEEDS:
                    p, q = predict(wrong, outcome, seed), predict(correct, outcome, seed)
                    runs.append({"seed": seed, **losses(y, p, q)})
                    if seed == SEED:
                        reference = (p, q)
                p, q = reference
                strata = [{"any_multitoken_current_or_predecessor": flag, **losses(y[affected == flag], p[affected == flag], q[affected == flag])} for flag in (False, True)]
                deletions = []
                for rule, keep in (("remove_affected_predictor_rows", ~affected),
                                   ("remove_sentences_with_affected_predictors", ~selected.sent_id.isin(affected_sentences).to_numpy())):
                    a, b = wrong.loc[keep], correct.loc[keep]
                    ap, bp = predict(a, outcome, SEED), predict(b, outcome, SEED)
                    deletions.append({"rule": rule, "sentences": int(a.sent_id.nunique()), **losses(y[keep], ap, bp)})
                pieces = pd.DataFrame({"sentence": selected.sent_id.to_numpy(), "difference": (y - p) ** 2 - (y - q) ** 2})
                sums = pieces.groupby("sentence").difference.agg(["sum", "size"])
                omit = (pieces.difference.sum() - sums["sum"]) / (len(y) - sums["size"])
                result.append({"model": model, "content_only": content, "outcome": outcome, "source": source,
                    "rows": len(selected), "sentences": int(selected.sent_id.nunique()),
                    "current_multitoken_n": int(selected.multitoken.sum()),
                    "true_predecessor_multitoken_n": int(selected.correct_lag_affected.sum()),
                    "row_predecessor_multitoken_n": int(selected.wrong_lag_affected.sum()),
                    "any_affected_n": int(affected.sum()), "partition_runs": runs,
                    "fixed_prediction_strata": strata, "deletion_refits": deletions,
                    "fixed_prediction_sentence_deletion_mse_decrease_range": [float(omit.min()), float(omit.max())]})
                print(f"Verified fits: {model}, content={content}, {outcome}, {source}", flush=True)
    return result
