"""Offline canonical-token and continuation-target reconstruction."""
from pathlib import Path
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from .data import digest, require, INPUT_HASHES
from .output import guarded, write_arrays, write_json

TOLERANCE_NATS = 0.001


def substitution_ids(tokenizer, target):
    """Reconstruct the audited lookup, retaining the actual conditioning path."""
    result = [target[0]]
    for token in target[1:]:
        decoded = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        alternatives = tokenizer.encode(" " + decoded, add_special_tokens=False)
        require(bool(alternatives), "Empty retokenized continuation")
        result.append(alternatives[0])
    return result


def verify_model(model_dir, name):
    manifest = json.loads((Path(__file__).parents[1] / "model_manifest.json").read_text())[name]
    allowed = {record["file"] for record in manifest["files"]} | {"README.md", "LICENSE", ".gitattributes", ".cache", ".DS_Store"}
    unexpected = {path.name for path in Path(model_dir).iterdir()} - allowed
    require(not unexpected, f"Unmanifested model/tokenizer files: {sorted(unexpected)}")
    for record in manifest["files"]:
        require(digest(Path(model_dir) / record["file"]) == record["sha256"], f"Wrong {name} model file: {record['file']}")
    return manifest


def score(raw, model_dir, name, output):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    output = guarded(output)
    guarded(output.with_suffix(".json"))
    manifest = verify_model(model_dir, name)
    torch.set_num_threads(4)
    torch.manual_seed(20260926)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, attn_implementation="eager")
    model.eval()
    contexts = [tokenizer.encode(text, add_special_tokens=False) for text in raw.item]
    targets = [tokenizer.encode(" " + word, add_special_tokens=False) for word in raw.word]
    for c, t, text, word in zip(contexts, targets, raw.item, raw.word):
        require(bool(c) and bool(t), "Empty context or target")
        require(c + t == tokenizer.encode(text + " " + word, add_special_tokens=False), "Noncompositional tokenization")
    sequences = [c + t for c, t in zip(contexts, targets)]
    checked = {i for i, target in enumerate(targets) if len(target) > 1} | set(range(10))
    canonical, substituted = np.empty(len(raw)), np.empty(len(raw))
    max_prefix_error = 0.0
    with torch.inference_mode():
        for offset in range(0, len(sequences), 12):
            part = sequences[offset:offset + 12]
            ids = torch.full((len(part), max(map(len, part))), tokenizer.eos_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, seq in enumerate(part):
                ids[row, :len(seq)] = torch.tensor(seq)
                mask[row, :len(seq)] = 1
            logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1)
            for row, seq in enumerate(part):
                i = offset + row
                n = len(contexts[i])
                target = targets[i]
                alternative = substitution_ids(tokenizer, target)
                canonical[i] = -sum(float(logp[row, n + j - 1, token]) for j, token in enumerate(target))
                substituted[i] = -sum(float(logp[row, n + j - 1, token]) for j, token in enumerate(alternative))
                if i in checked:
                    direct = 0.0
                    for j, token in enumerate(target):
                        prefix = torch.tensor([seq[:n + j]], dtype=torch.long)
                        single = torch.log_softmax(model(input_ids=prefix).logits[0, -1], dim=-1)
                        direct -= float(single[token])
                    max_prefix_error = max(max_prefix_error, abs(direct - canonical[i]))
            if offset % 120 == 0:
                print(f"{name}: scored {min(offset + 12, len(raw))}/{len(raw)}", flush=True)
    require(np.isfinite(canonical).all() and np.isfinite(substituted).all(), "Nonfinite scores")
    require(max_prefix_error < TOLERANCE_NATS, "Independent prefix discrepancy exceeds declared tolerance")
    column = {"gpt2": "GPT2", "gpt_neo_125m": "GPTNeo_125M"}[name]
    multitoken = np.asarray([len(target) > 1 for target in targets])
    require(np.array_equal(multitoken, raw["is_multitoken_" + column].astype(bool)), "Released token flags differ")
    error = substituted - raw["s_" + column].to_numpy()
    require(float(abs(error).max()) < TOLERANCE_NATS, "Substitution does not reconstruct released scores")
    record = {"model": manifest["repo"], "revision": manifest["revision"], "input_sha256": INPUT_HASHES["aggregate"],
              "dtype": "float32", "device": "cpu", "batch_size": 12, "prefix_check_rows": len(checked),
              "prefix_check_max_error_nats": max_prefix_error, "tolerance_nats": TOLERANCE_NATS,
              "substituted_stored_max_error_nats": float(abs(error).max()), "matched_rows": int((abs(error) < TOLERANCE_NATS).sum()),
              "multitoken_rows": int(multitoken.sum()),
              "canonical_minus_stored_multitoken_mean_nats": float((canonical - raw["s_" + column].to_numpy())[multitoken].mean())}
    output = Path(output)
    write_arrays(output, canonical_nats=canonical, substituted_nats=substituted,
                 input_sha256=np.array(INPUT_HASHES["aggregate"]))
    record["cache_sha256"] = digest(output)
    write_json(output.with_suffix(".json"), record)
    return {"canonical_nats": canonical, "substituted_nats": substituted}, record


def load_cache(path, raw, name):
    path = guarded(path)
    record = json.loads(guarded(path.with_suffix(".json")).read_text())
    require(record["cache_sha256"] == digest(path), "Score cache digest differs")
    expected_model = json.loads((Path(__file__).parents[1] / "model_manifest.json").read_text())[name]
    require(record["revision"] == expected_model["revision"] and record["model"] == expected_model["repo"], "Cache model differs")
    require(record["input_sha256"] == INPUT_HASHES["aggregate"], "Cache source differs")
    arrays = np.load(path, allow_pickle=False)
    require(arrays["input_sha256"].item() == INPUT_HASHES["aggregate"], "Cache source differs")
    result = {key: arrays[key] for key in ("canonical_nats", "substituted_nats")}
    require(all(len(value) == len(raw) and np.isfinite(value).all() for value in result.values()), "Invalid cache arrays")
    column = {"gpt2": "GPT2", "gpt_neo_125m": "GPTNeo_125M"}[name]
    require(np.max(abs(result["substituted_nats"] - raw["s_" + column])) < TOLERANCE_NATS, "Cached scores do not reconstruct")
    return result, {**record, "execution": "VERIFIED_CACHE_REUSE_NOT_FRESH_INFERENCE"}
