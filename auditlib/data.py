"""Read authorized inputs in place; return summaries, never copied records."""
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer

INPUT_HASHES = {
    "aggregate": "db3ccd106220e1e3d7d64871b5f7c05a9d159ef4cb639f985b792d559686f7f6",
    "reading_times": "350dee6598b18271a0bdd8cd2fda0b7ca11b205d306f43ce3319bc2df61ab42e",
    "fixations": "987ff895d00e30429cefc54067dd8ccc4e59f4617cd38d1b7ccd4eb79a6afaab",
    "pos": "f46a3bd640327ceac5b2f8c04bdcaf5fed28b80e95f1b387dcbfbd84dcc46b4f",
}
CONTENT_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
EYE_MEASURES = ("RTfirstfix", "RTfirstpass", "RTrightbound", "RTgopast")


def require(condition, message):
    # Explicit exceptions remain active even under python -O.
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_inputs(paths):
    result = {}
    for name, expected in INPUT_HASHES.items():
        actual = digest(paths[name])
        require(actual == expected, f"Wrong {name} input: SHA-256 differs")
        result[name] = actual
    return result


def read_aggregate(path):
    require(digest(path) == INPUT_HASHES["aggregate"], "Unrecognized aggregate")
    frame = pd.read_csv(path)
    # Preserve literal words/contexts such as 'None'; numeric NA handling is separate.
    text = pd.read_csv(path, keep_default_na=False, usecols=["word", "item", "sentence"])
    for column in ("word", "item", "sentence"):
        frame[column] = text[column]
    require(len(frame) == 1726, "Aggregate row count differs")
    require(not frame.duplicated(["sent_id", "context_length"]).any(), "Duplicate word key")
    return frame


def keyed_predecessor(frame, values):
    require(not frame.duplicated(["sent_id", "context_length"]).any(), "Duplicate lag key")
    keys = pd.MultiIndex.from_arrays([frame.sent_id, frame.context_length])
    previous = pd.MultiIndex.from_arrays([frame.sent_id, frame.context_length - 1])
    return pd.Series(np.asarray(values), index=keys).reindex(previous).to_numpy()


def order_summary(raw):
    prior = raw.groupby("sent_id").context_length.shift()
    distance = raw.context_length - prior
    available = prior.notna()
    out = {"available_row_predecessors": int(available.sum()),
           "immediate_predecessors": int(distance.eq(1).sum()),
           "earlier_nonadjacent": int(distance.gt(1).sum()),
           "later_words": int(distance.lt(0).sum())}
    require(out == {"available_row_predecessors": 1521, "immediate_predecessors": 165,
                    "earlier_nonadjacent": 575, "later_words": 781}, "Order audit differs")
    return out


def word_annotations(rt_path, pos_path):
    require(Path(pos_path).read_text().splitlines()[0].split() == ["sent_nr", "pos"], "POS header differs")
    tags = pd.read_csv(pos_path, sep="\t", skiprows=1, names=["sent_nr", "pos"], keep_default_na=False).set_index("sent_nr")
    rt = pd.read_csv(rt_path, sep="\t", keep_default_na=False, usecols=["sent_nr", "word_pos", "word"])
    words = rt.drop_duplicates().sort_values(["sent_nr", "word_pos"])
    require(not words.duplicated(["sent_nr", "word_pos"]).any(), "Duplicate annotation key")
    tokenizer, records = TreebankWordTokenizer(), []
    for sentence, group in words.groupby("sent_nr"):
        sentence_text = " ".join(group.word)
        word_spans = list(re.finditer(r"\S+", sentence_text))
        spans = list(tokenizer.span_tokenize(sentence_text))
        sequence = str(tags.loc[sentence, "pos"]).split()
        require(len(spans) == len(sequence), "Unresolved Penn tag alignment")
        per_word = [[] for _ in word_spans]
        for (start, stop), tag in zip(spans, sequence):
            indexes = [i for i, span in enumerate(word_spans) if start >= span.start() and stop <= span.end()]
            require(len(indexes) == 1, "Token crosses displayed-word boundary")
            if tag in {",", ".", ":", "-LRB-", "-RRB-", "``", "''"}:
                require(not any(ch.isalpha() for ch in sentence_text[start:stop]), "Punctuation alignment differs")
            per_word[indexes[0]].append(tag)
        for row, local in zip(group.itertuples(), per_word):
            require(bool(local), "Word has no aligned tag")
            records.append((row.sent_nr, row.word_pos, row.word, sentence_text, bool(set(local) & CONTENT_TAGS)))
    result = pd.DataFrame(records, columns=["sent_nr", "word_pos", "word", "sentence", "content_penn"])
    require(len(result) == len(words) == 1931, "Annotated word count differs")
    return result


def reconstruct_observations(aggregate, rt_path, fix_path):
    raw = pd.read_csv(rt_path, sep="\t")
    raw["word"] = pd.read_csv(rt_path, sep="\t", keep_default_na=False, usecols=["word"]).word
    fix = pd.read_csv(fix_path, sep="\t")
    keys = ["subj_nr", "sent_nr", "word_pos"]
    require(not raw.duplicated(keys).any(), "Duplicate reader-word record")
    words = raw[["sent_nr", "word_pos", "word"]].drop_duplicates().sort_values(["sent_nr", "word_pos"])
    sentences = words.groupby("sent_nr").word.apply(lambda x: " ".join(x)).rename("sentence").reset_index()
    correspondence = aggregate[["sent_id", "sentence"]].drop_duplicates().merge(sentences, on="sentence", validate="one_to_one")
    require(len(correspondence) == 205, "Sentence correspondence incomplete")
    target = aggregate.merge(correspondence[["sent_id", "sent_nr"]], on="sent_id", validate="many_to_one")
    target["word_pos"] = target.context_length + 1
    target = target.merge(words, on=["sent_nr", "word_pos"], suffixes=("", "_original"), validate="one_to_one")
    require(len(target) == 1726 and target.word.eq(target.word_original).all(), "Target correspondence differs")
    target_keys = pd.MultiIndex.from_frame(target[["sent_nr", "word_pos"]])
    grouped = raw.groupby(["sent_nr", "word_pos"])
    means, counts = grouped[list(EYE_MEASURES)].mean(), grouped.size().reindex(target_keys)
    comparisons = {}
    for measure in EYE_MEASURES:
        error = means[measure].reindex(target_keys).to_numpy() - target[measure].to_numpy()
        require(np.isfinite(error).all() and np.max(abs(error)) < 1e-9, f"{measure} means do not reconstruct")
        comparisons[measure] = {"words": len(error), "max_absolute_error_ms": float(abs(error).max()),
                                "minimum_readers": int(counts.min()), "maximum_readers": int(counts.max())}
    raw_trials = set(map(tuple, raw[["subj_nr", "sent_nr"]].drop_duplicates().to_numpy()))
    fix_trials = set(map(tuple, fix[["subj_nr", "sent_nr"]].drop_duplicates().to_numpy()))
    require(raw_trials == fix_trials, "Word and fixation trial sets differ")
    entries = []
    # Input documentation guarantees chronological order within each reader/sentence.
    for (reader, sentence), group in fix.groupby(["subj_nr", "sent_nr"], sort=False):
        high, seen = 0, set()
        for record in group.itertuples():
            word = record.word_pos
            if not np.isfinite(word):
                continue
            require(word == int(word) and word >= 1, "Invalid fixation word position")
            word = int(word)
            if word not in seen:
                entries.append((reader, sentence, word, "L" if high > word else "E", float(record.fix_duration)))
                seen.add(word)
            high = max(high, word)
    events = pd.DataFrame(entries, columns=[*keys, "entry_state", "first_fixation_ms"])
    observations = raw.merge(events, on=keys, how="left", validate="one_to_one")
    observations["entry_state"] = observations.entry_state.fillna("N")
    early = observations.entry_state.eq("E")
    require(np.max(abs(observations.loc[early, "RTfirstfix"] - observations.loc[early, "first_fixation_ms"])) == 0,
            "First-fixation event does not reconstruct")
    for measure in EYE_MEASURES:
        require(np.isfinite(observations[measure]).all() and observations[measure].gt(0).equals(early),
                "Released zero and entry definition differ")
    return {"raw_word_rows": len(raw), "raw_fixations": len(fix), "readers": int(raw.subj_nr.nunique()),
            "represented_trials": len(raw_trials), "aggregate_comparisons": comparisons,
            "entry_counts": observations.entry_state.value_counts().to_dict(),
            "missing_trials_are_not_zero": True}


def analysis_frame(raw, scores, model, annotations):
    column = {"gpt2": "GPT2", "gpt_neo_125m": "GPTNeo_125M"}[model]
    d = raw.rename(columns={"Subtlex_log10": "frequency"}).copy()
    d["position"] = (d.context_length + 1) / (d.groupby("sent_id").context_length.transform("max") + 1)
    d["wrong_position"] = (d.groupby("sent_id").cumcount() + 1) / d.groupby("sent_id").context_length.transform("size")
    d["stored"], d["multitoken"] = d["s_" + column], d["is_multitoken_" + column]
    d["canonical"] = scores["canonical_nats"]
    require(len(scores["canonical_nats"]) == len(d), "Score row count differs")
    for name in ("stored", "canonical", "length", "frequency"):
        d["lag_" + name] = keyed_predecessor(d, d[name])
        d["wrong_lag_" + name] = d.groupby("sent_id")[name].shift()
    d["correct_lag_affected"] = keyed_predecessor(d, d.multitoken)
    d["wrong_lag_affected"] = d.groupby("sent_id").multitoken.shift()
    punctuation = ~d.word.str.contains(r"[.,']", regex=True)
    finite = np.isfinite(d[["length", "frequency", "position", "stored", "canonical", "RTfirstfix", "RTfirstpass"]]).all(axis=1)
    # The exploratory implementation also screened these pilot outcome fields.
    # Verify that omitting them from the public core does not change its sample.
    pilot_finite = np.isfinite(d[["self_paced_reading_time", "N400", "P600"]]).all(axis=1)
    require(np.array_equal(punctuation & finite, punctuation & finite & pilot_finite), "Pilot-field screening changes core sample")
    require(int(punctuation.sum()) == 1487 and int((punctuation & finite).sum()) == 1466, "Core sample differs")
    keys = pd.MultiIndex.from_arrays([d.sentence, d.context_length + 1])
    aligned = annotations.set_index(["sentence", "word_pos"]).reindex(keys)
    require(aligned.content_penn.notna().all() and np.array_equal(aligned.word.to_numpy(), d.word.to_numpy()), "Content labels do not align")
    d["content"] = aligned.content_penn.to_numpy(dtype=bool)
    selected = d.loc[punctuation & finite].copy()
    require(int(selected.content.sum()) == 761, "Content sample differs")
    return selected
