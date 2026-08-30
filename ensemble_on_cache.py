import json
import numpy as np
import glob
import os
from collections import Counter, defaultdict


DB_PATH = f"../input_multiple_choice_videoQA/mc_vqa_unified_valid.json"

def subset_of(vid):
    suffix = vid.rsplit("_", 1)[-1]
    return suffix if suffix in ("oai", "pa", "ao", "pt") else "orig"


def report(records):
    """Breakdown by subsets, areas and tags. The overall figure hides
    where exactly the failure is: 58 new questions give only 12.8% of the weight."""
    scored = [r for r in records if r["real"] is not None]
    if not scored:
        print("Ground truth is missing.")
        return None

    acc = sum(r["pred"] == r["real"] for r in scored) / len(scored)
    baseline = sum(1 / r["n_options"] for r in scored) / len(scored)
    print(f"\n{'=' * 70}")
    print(f"Overall accuracy: {acc:.4f}  ({len(scored)} questions)")
    print(f"Random baseline: {baseline:.4f}")

    buckets = defaultdict(lambda: [0, 0])
    for r in scored:
        keys = [f"subset:{subset_of(r['vid'])}", f"options:{r['n_options']}",
                f"area:{r['area']}", f"reasoning:{r['reasoning']}"]
        keys += [f"tag:{t}" for t in r["tag"]]
        for k in keys:
            buckets[k][1] += 1
            buckets[k][0] += int(r["pred"] == r["real"])

    print(f"\n{'slice':42s} {'acc':>7s} {'n':>5s} {'errors':>7s}")
    print("-" * 70)
    for k, (c, t) in sorted(buckets.items(), key=lambda x: x[1][0] / x[1][1]):
        if t >= 6:  # samples smaller than 6 are pure noise, do not optimize based on them
            print(f"{k:42s} {100 * c / t:6.1f}% {t:5d} {t - c:7d}")

    print(f"\nSelected 'None of the other options': "
          f"{sum(r['is_none'] for r in scored)} times")
    print("Prediction distribution:",
          dict(sorted(Counter(r["pred"] for r in scored).items())))
    n_failed = sum(r.get("failed", False) for r in scored)
    if n_failed:
        print(f"\nWARNING: {n_failed} questions out of {len(scored)} are stubs "
              f"after exceptions, not model predictions")
    return acc


def ensemble(cache_dirs, weights=None):
    """Averaging distributions of several models.

    cache_dirs: [(cache_path, cfg_hash), ...]
    Only works if all models saved real probabilities —
    one-hot from the migrated cache will only give a hard vote.
    """
    with open(DB_PATH) as f:
        db = json.load(f)
    w = weights or [1.0] * len(cache_dirs)
    submission, records = {}, []

    for vid, data in db.items():
        submission[vid] = []
        for q in data.get("mc_question", []):
            k = len(q["options"])
            agg, used = np.zeros(k), 0.0
            for (folder, h), wi in zip(cache_dirs, w):
                p = os.path.join(folder, f"{vid}_q{q['id']}_{h}.json")
                if os.path.isfile(p):
                    with open(p) as f:
                        agg += wi * np.array(json.load(f)["probs"])
                    used += wi
            if used == 0:
                continue
            pred = int((agg / used).argmax())
            submission[vid].append({"id": q["id"], "answer_id": pred,
                                    "answer": q["options"][pred]})
            records.append({
                "vid": vid, "qid": q["id"], "pred": pred,
                "real": q.get("answer_id"), "conf": float((agg / used).max()),
                "n_options": k, "tag": q.get("tag", []),
                "area": q.get("area", "?"), "reasoning": q.get("reasoning", "?"),
                "is_none": "None of the other" in q["options"][pred],
            })
    report(records)
    return submission


def check_cache_dirs(cache_dirs, weights=None, expect_split=None, min_files=1):
    """Checks that every cache folder exists and contains files with the required hash.

    expect_split: for example 'valid' — then it warns if the path does not contain '_valid'.
    """
    if weights is not None and len(weights) != len(cache_dirs):
        raise ValueError(
            f'len(weights)={len(weights)} != len(cache_dirs)={len(cache_dirs)}'
        )

    problems = []
    for i, (path, h) in enumerate(cache_dirs):
        if not os.path.isdir(path):
            problems.append(f'[{i}] folder not found: {path}')
            continue

        files = glob.glob(os.path.join(path, f'*{h}*'))
        if len(files) < min_files:
            available = sorted({
                # pull out the hashes that are actually present in the folder
                os.path.basename(f) for f in glob.glob(os.path.join(path, '*'))
            })[:5]
            problems.append(
                f'[{i}] no files with hash {h} found in {path}. '
                f'Example files: {available or "folder is empty"}'
            )
            continue

        if expect_split and expect_split not in path:
            problems.append(
                f'[{i}] path {path} does not look like the "{expect_split}" split — '
                f'train/valid/test may have been mixed up'
            )
        else:
            print(f'[{i}] OK: {path} hash={h} files={len(files)}')

    if problems:
        raise FileNotFoundError('Problems with cache_dirs:\n  ' + '\n  '.join(problems))


if __name__ == "__main__":
    cache_dirs = [
        ('./cache/qwen3-32bfp8_data/', '5e0cf10376'),
        ('./cache/qwen38-27bfp8_data/', 'd47dc5d75d'),
    ]
    weights = [8, 7]
    check_cache_dirs(cache_dirs)
    ensemble(cache_dirs, weights=weights)