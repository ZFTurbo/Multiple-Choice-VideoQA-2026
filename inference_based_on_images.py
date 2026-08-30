# https://eval.ai/web/challenges/challenge-page/2702/overview
#
# Version with client-side frame sampling.
# Frames are sampled and resized here, a ready-made list of JPEGs is sent to the server.
# Neither do_sample_frames, fps, nor min/max_pixels are passed to the server:
# it takes exactly what was sent.

import argparse
import base64
import json
import math
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from io import BytesIO
from threading import Lock

import numpy as np
from PIL import Image
from openai import OpenAI

warnings.filterwarnings("ignore")

# ==========================================================================
# Configuration (defaults, every value can be overridden from the command line)
# ==========================================================================

SUPPORTED_MODELS = [
    'qwen3-32bfp8',
    'qwen3-8b',
    'qwen3-30ba3bfp8',
    'qwen38-27bfp8',
    'qwen3-235b',
    'internvl3_5-1b',
    'internvl3_5-8b',
    'internvl3_5-38b',
    'gemma-4-31B-it',
]

MODEL_NAME = "qwen38-27bfp8"
PORT = 8000
N_FRAMES = 100           # how many frames are actually sent to the model
MAX_FRAME_PIXELS = 512 * 32 * 32   # ceiling per frame (token = 32x32 px)
JPEG_QUALITY = 90
THINKING = True

# Total pixel budget per video for Qwen3-VL: size.longest_edge.
# If N_FRAMES * MAX_FRAME_PIXELS exceeds it, the server will compress
# the frames itself anyway — it's better to compress in advance and not send extra bytes over the network.
TOTAL_PIXEL_BUDGET = 25165824
PATCH = 32               # patch_size(16) * merge_size(2): multiple for sides
DEBIAS = True            # option permutations: +1-3 p.p., but N times more expensive
MAX_WORKERS = 4
PROMPT_VERSION = "v5"    # if you change the prompt text — change the version
REQUEST_TIMEOUT = 1800.0  # without a timeout, a hung request keeps the worker busy forever
MAX_RETRIES = 2
SAVE_RAW = True          # dump prompt + full model answer next to the .json cache

DB_PATH = "../input_multiple_choice_videoQA/mc_vqa_unified_valid.json"
VIDEO_DIR = "../input_multiple_choice_videoQA/unified_vqa_valid_split/valid_split"
CACHE_DIR = None         # None -> ./cache/{MODEL_NAME}_data
SUBM_DIR = "./subm"


INSTRUCTION_V1 = (
    "Instruction: Output ONLY the number of the correct option "
    "(e.g., 0, 1, 2 etc). Do not include any other text, punctuation, "
    "or explanations. Note that 'None of the other options' is sometimes "
    "the correct answer."
)

INSTRUCTION = "Instruction: Output ONLY the number of the correct option (e.g., 0, 1, 2 etc). Do not include any other text, punctuation, or explanations."

# Created in apply_args(): the base_url depends on --port, so the client
# cannot be built at import time any more.
client = None
CFG_HASH = None
print_lock = Lock()


# ==========================================================================
# Command line
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multiple choice VideoQA inference with client-side frame sampling')

    # --- server / model ---
    parser.add_argument('--port', type=int, default=PORT,
                        help='Port of the local OpenAI-compatible server (vLLM etc.)')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, choices=SUPPORTED_MODELS,
                        help='Model to run inference with; must be one of SUPPORTED_MODELS')
    parser.add_argument('--timeout', type=float, default=REQUEST_TIMEOUT,
                        help='Per-request timeout in seconds')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES,
                        help='How many times the OpenAI client retries a failed request')

    # --- paths ---
    parser.add_argument('--json-path', type=str, default=DB_PATH,
                        help='JSON file with questions and ground truth answers')
    parser.add_argument('--videos-folder', type=str, default=VIDEO_DIR,
                        help='Folder with videos (<video_id>.mp4 files)')
    parser.add_argument('--cache-path', type=str, default=None,
                        help='Folder where per-question probability distributions are cached '
                             '(default: ./cache/<model-name>_data)')
    parser.add_argument('--out-subm-folder', type=str, default=SUBM_DIR,
                        help='Output folder for the resulting submission JSON')

    # --- sampling ---
    parser.add_argument('--n-frames', type=int, default=N_FRAMES,
                        help='How many frames are sampled and sent to the model')
    parser.add_argument('--max-frame-pixels', type=int, default=MAX_FRAME_PIXELS,
                        help='Pixel ceiling per single frame')
    parser.add_argument('--total-pixel-budget', type=int, default=TOTAL_PIXEL_BUDGET,
                        help='Total pixel budget per video; frames are shrunk to fit it')
    parser.add_argument('--jpeg-quality', type=int, default=JPEG_QUALITY,
                        help='JPEG quality of the encoded frames (1-100)')
    parser.add_argument('--patch', type=int, default=PATCH,
                        help='Frame sides are rounded to a multiple of this value')

    # --- inference ---
    parser.add_argument('--thinking', dest='thinking', action='store_true', default=THINKING,
                        help='Enable reasoning mode: model may emit a <think>...</think> block before the answer')
    parser.add_argument('--no-thinking', dest='thinking', action='store_false',
                        help='Disable reasoning mode: single-token answer constrained by guided_choice')
    parser.add_argument('--debias', dest='debias', action='store_true', default=DEBIAS,
                        help='Average over cyclic permutations of the options (N times more requests)')
    parser.add_argument('--no-debias', dest='debias', action='store_false',
                        help='One request per question, no permutations')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS,
                        help='Number of videos processed in parallel')
    parser.add_argument('--prompt-version', type=str, default=PROMPT_VERSION,
                        help='Prompt version tag; it goes into the cache hash, '
                             'so bump it whenever the prompt text changes')

    # --- raw log ---
    parser.add_argument('--save-raw', dest='save_raw', action='store_true', default=SAVE_RAW,
                        help='Save the prompt and the full model answer as a .txt next to the .json cache')
    parser.add_argument('--no-save-raw', dest='save_raw', action='store_false',
                        help='Do not write the .txt dumps (thinking traces can be large)')

    # --- alternative entry points ---
    parser.add_argument('--probe', nargs='?', const='', default=None, metavar='VIDEO_ID',
                        help='Send one diagnostic request (optionally for a given video id) and exit')

    args = parser.parse_args()

    if args.model_name not in SUPPORTED_MODELS:
        parser.error("Model '{}' is not supported. Choose from: {}".format(
            args.model_name, ', '.join(SUPPORTED_MODELS)))
    if args.n_frames < 1:
        parser.error("--n-frames must be >= 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in the 1..100 range")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    return args


def apply_args(args):
    """Pushes the parsed arguments into the module globals and builds the client.

    The functions below read the configuration from globals, so everything has to
    be assigned before the first call — including CFG_HASH, which depends on
    the model, the frame count and the prompt version.
    """
    global PORT, MODEL_NAME, REQUEST_TIMEOUT, MAX_RETRIES
    global DB_PATH, VIDEO_DIR, CACHE_DIR, SUBM_DIR
    global N_FRAMES, MAX_FRAME_PIXELS, TOTAL_PIXEL_BUDGET, JPEG_QUALITY, PATCH
    global THINKING, DEBIAS, MAX_WORKERS, PROMPT_VERSION, SAVE_RAW
    global client, CFG_HASH

    PORT = args.port
    MODEL_NAME = args.model_name
    REQUEST_TIMEOUT = args.timeout
    MAX_RETRIES = args.max_retries

    DB_PATH = args.json_path
    VIDEO_DIR = args.videos_folder
    CACHE_DIR = args.cache_path or f"./cache/{MODEL_NAME}_data"
    SUBM_DIR = args.out_subm_folder

    N_FRAMES = args.n_frames
    MAX_FRAME_PIXELS = args.max_frame_pixels
    TOTAL_PIXEL_BUDGET = args.total_pixel_budget
    JPEG_QUALITY = args.jpeg_quality
    PATCH = args.patch

    THINKING = args.thinking
    DEBIAS = args.debias
    MAX_WORKERS = args.max_workers
    PROMPT_VERSION = args.prompt_version
    SAVE_RAW = args.save_raw

    client = OpenAI(
        base_url=f"http://127.0.0.1:{PORT}/v1",
        api_key="EMPTY",
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )
    CFG_HASH = config_hash()
    return args


# ==========================================================================
# Client-side frame sampling
# ==========================================================================

def frame_geometry(w_src, h_src, n_frames):
    """Target frame size taking into account both budgets, sides are multiples of PATCH."""
    budget = min(MAX_FRAME_PIXELS, TOTAL_PIXEL_BUDGET // max(n_frames, 1))
    scale = min(1.0, math.sqrt(budget / (w_src * h_src)))
    w = max(PATCH, int(round(w_src * scale / PATCH)) * PATCH)
    h = max(PATCH, int(round(h_src * scale / PATCH)) * PATCH)
    return w, h


def sample_frames(video_path):
    """Uniformly samples N_FRAMES frames and encodes them into base64 JPEG.

    We decode directly in the target resolution (decord can resize on the fly):
    300 frames of 1080p in memory is 1.8 GB per video, and with four workers
    it's a full 7 GB.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n_src = len(vr)
    src_fps = vr.get_avg_fps()
    duration = n_src / src_fps if src_fps else 0.0
    h_src, w_src = vr[0].shape[:2]
    del vr

    n = min(N_FRAMES, n_src)
    idx = np.linspace(0, n_src - 1, n).astype(int)
    w, h = frame_geometry(w_src, h_src, n)

    vr = VideoReader(video_path, ctx=cpu(0), width=w, height=h, num_threads=1)
    batch = vr.get_batch(idx).asnumpy()
    del vr

    frames = []
    for arr in batch:
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=JPEG_QUALITY)
        frames.append(base64.b64encode(buf.getvalue()).decode())

    info = {"n_src": n_src, "duration": duration, "n_sent": len(frames),
            "size": (w, h), "eff_fps": len(frames) / duration if duration else 0}
    return "data:video/jpeg;base64," + ",".join(frames), info


# ==========================================================================
# Distribution cache
# ==========================================================================

def config_hash():
    cfg = (f"{MODEL_NAME}|{N_FRAMES}|{MAX_FRAME_PIXELS}|{JPEG_QUALITY}|"
           f"{DEBIAS}|{PROMPT_VERSION}")
    return md5(cfg.encode()).hexdigest()[:10]


def cache_path(vid, qid, cfg_hash=None):
    return os.path.join(CACHE_DIR, f"{vid}_q{qid}_{cfg_hash or CFG_HASH}.json")


def raw_cache_path(vid, qid, cfg_hash=None):
    """Same name as the prediction cache, but with a .txt extension."""
    return os.path.splitext(cache_path(vid, qid, cfg_hash))[0] + ".txt"


def cache_load(vid, qid):
    p = cache_path(vid, qid)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return np.array(json.load(f)["probs"], dtype=float)
    except Exception:
        return None  # ignore corrupted file, do not inherit


def cache_save(vid, qid, probs):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(vid, qid), "w") as f:
        json.dump({"probs": probs.tolist(), "cfg": CFG_HASH}, f)


def raw_cache_save(vid, qid, entries):
    """Human-readable dump of every request/response pair for one question.

    With DEBIAS there are N calls per question (one per cyclic permutation of
    the options), so all of them go into a single file, separated by headers.
    The file is written even when a call raised: whatever was collected before
    the failure is still useful for debugging.
    """
    if not SAVE_RAW or not entries:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(raw_cache_path(vid, qid), "w", encoding="utf-8") as f:
        f.write(f"# video={vid} question={qid}\n")
        f.write(f"# model={MODEL_NAME} cfg={CFG_HASH} thinking={THINKING} "
                f"debias={DEBIAS} n_frames={N_FRAMES}\n")
        f.write(f"# calls={len(entries)}\n")
        for e in entries:
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"CALL {e['call']}/{e['n_calls']}  perm={e['perm']}  "
                    f"finish_reason={e.get('finish_reason')}  "
                    f"prompt_tokens={e.get('prompt_tokens')}  "
                    f"completion_tokens={e.get('completion_tokens')}\n")
            if e.get("note"):
                f.write(f"NOTE: {e['note']}\n")
            f.write("-" * 78 + "\nPROMPT\n" + "-" * 78 + "\n")
            f.write(e["prompt"].rstrip() + "\n")
            if e.get("reasoning"):
                f.write("-" * 78 + "\nREASONING\n" + "-" * 78 + "\n")
                f.write(e["reasoning"].rstrip() + "\n")
            f.write("-" * 78 + "\nRESPONSE\n" + "-" * 78 + "\n")
            f.write((e.get("response") or "").rstrip() + "\n")
            if e.get("probs") is not None:
                f.write("-" * 78 + "\nPROBS (order as shown to the model)\n")
                f.write("[" + ", ".join(f"{p:.4f}" for p in e["probs"]) + "]\n")
            if e.get("error"):
                f.write("-" * 78 + f"\nERROR: {e['error']}\n")


# ==========================================================================
# Model request
# ==========================================================================

def _to_index(tok, num_options):
    """isdigit() allows '₀','¹','²' — they do not convert to int.
    isdecimal() is strictly for decimal digits."""
    tok = tok.strip()
    if len(tok) == 1 and tok.isdecimal():
        i = int(tok)
        if i < num_options:
            return i
    return None


def format_prompt(question, options):
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    return ("Watch the video and answer the multiple-choice question below.\n"
            f"Question: {question}\n\nOptions:\n{opts}\n\n{INSTRUCTION}")


def build_messages(video_url, prompt_text):
    """Video first, text last. The order cannot be changed: it relies on
    prefix caching — 3 questions for one video and all option permutations
    reuse a single visual prefix."""
    return [{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": prompt_text},
        ],
    }]


def _raw_info(r):
    """Everything worth keeping from a response, except the probabilities."""
    ch = r.choices[0]
    return {
        "response": ch.message.content or "",
        # vLLM can put the thinking block into a separate field instead of content
        "reasoning": getattr(ch.message, "reasoning_content", None) or "",
        "finish_reason": ch.finish_reason,
        "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
        "completion_tokens": getattr(r.usage, "completion_tokens", None),
    }


def _strip_think(text):
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def _parse_answer(text, num_options):
    tail = _strip_think(text).strip()
    for s in reversed(re.findall(r"\d+", tail)):   # последнее число в хвосте
        if int(s) < num_options:
            return int(s)
    return None


def _soft_from_logprobs(lp, num_options):
    if not lp or not lp.content:
        return None
    for item in reversed(lp.content):
        if _to_index(item.token, num_options) is None:
            continue
        probs = np.zeros(num_options)
        for t in item.top_logprobs:
            i = _to_index(t.token, num_options)
            if i is not None:
                probs[i] += math.exp(t.logprob)
        return probs / probs.sum() if probs.sum() > 0 else None
    return None


def ask_probs_thinking(video_url, prompt_text, num_options):
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=build_messages(video_url, prompt_text),
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.0,
        max_tokens=8 * 4096,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
        reasoning_effort="xhigh",
    )
    ch = r.choices[0]
    text = ch.message.content or ""
    raw = _raw_info(r)

    if ch.finish_reason == "length" and "</think>" not in text:
        raw["note"] = "truncated inside <think>, uniform distribution used"
        return np.ones(num_options) / num_options, r.usage.prompt_tokens, raw

    probs = _soft_from_logprobs(ch.logprobs, num_options)
    if probs is not None:
        raw["note"] = "probabilities from logprobs"
        return probs, r.usage.prompt_tokens, raw

    idx = _parse_answer(text, num_options)          # one-hot fallback
    raw["note"] = f"one-hot fallback, parsed answer={idx}"
    probs = np.full(num_options, 1e-9)
    if idx is not None:
        probs[idx] = 1.0
    return probs / probs.sum(), r.usage.prompt_tokens, raw


def ask_probs(video_url, prompt_text, num_options):
    if THINKING:
        return ask_probs_thinking(video_url, prompt_text, num_options)

    if MODEL_NAME in ['qwen38-27bfp8']:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=build_messages(video_url, prompt_text),
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            max_tokens=16,
            logprobs=True,
            top_logprobs=20,
            extra_body={
                "guided_choice": [str(i) for i in range(num_options)],
                "mm_processor_kwargs": {"do_sample_frames": False},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    else:
        """Normalized distribution over options."""
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=build_messages(video_url, prompt_text),
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            extra_body={
                "guided_choice": [str(i) for i in range(num_options)],
                # Frames are already sampled — disable resampling on the server.
                # If the vLLM version does not accept this key, the line can be removed.
                "mm_processor_kwargs": {"do_sample_frames": False},
            },
        )

    raw = _raw_info(r)

    probs = np.zeros(num_options)
    lp = r.choices[0].logprobs
    if lp and lp.content:
        for t in lp.content[0].top_logprobs:
            i = _to_index(t.token, num_options)
            if i is not None:
                probs[i] += math.exp(t.logprob)  # += in case of duplicates

    if probs.sum() > 0:
        raw["note"] = "probabilities from logprobs"
        return probs / probs.sum(), r.usage.prompt_tokens, raw

    # Fallback: some vLLM builds do not return logprobs with guided_choice.
    # The answer is still limited to valid digits.
    txt = (r.choices[0].message.content or "").strip()
    i = _to_index(txt[:1], num_options)
    raw["note"] = f"one-hot fallback, parsed answer={i}"
    probs = np.full(num_options, 1e-9)
    if i is not None:
        probs[i] = 1.0
    return probs / probs.sum(), r.usage.prompt_tokens, raw


def predict(video_url, question, options, log=None):
    """Without DEBIAS — one request. With DEBIAS — cyclic permutations with
    averaging, which removes the positional bias of the model.

    log: an optional list the caller owns. Every request/response pair is
    appended to it as it happens, so a failure in the middle of the
    permutations still leaves the earlier calls on disk.
    """
    n = len(options)
    shifts = range(n) if DEBIAS else [0]

    agg, tok = np.zeros(n), 0
    for call, shift in enumerate(shifts, start=1):
        perm = [(i + shift) % n for i in range(n)]
        shown = [options[j] for j in perm]
        prompt_text = format_prompt(question, shown)

        entry = {"call": call, "n_calls": len(shifts), "perm": perm,
                 "prompt": prompt_text}
        if log is not None:
            log.append(entry)

        try:
            probs, t, raw = ask_probs(video_url, prompt_text, n)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            raise

        entry.update(raw)
        entry["probs"] = probs.tolist()

        tok += t
        for pos, orig in enumerate(perm):
            agg[orig] += probs[pos]

    return agg / len(shifts), tok


# ==========================================================================
# Processing a single video (all its questions)
# ==========================================================================

def process_video(vid, data):
    questions = data.get("mc_question", [])
    if not questions:
        return vid, [], []

    cached = [cache_load(vid, q["id"]) for q in questions]

    # Encode frames once per video: the same payload is reused by all
    # questions and all permutations of options.
    video_url = None
    if any(c is None for c in cached):
        video_path = os.path.abspath(f"{VIDEO_DIR}/{vid}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)
        video_url, info = sample_frames(video_path)
        with print_lock:
            print(f"[{vid}] {info['n_src']} frames / {info['duration']:.1f}s -> "
                  f"{info['n_sent']} frames @ {info['size'][0]}x{info['size'][1]}, "
                  f"eff_fps={info['eff_fps']:.2f}, "
                  f"payload={len(video_url) / 1e6:.1f} MB")

    results, records = [], []
    for q, probs in zip(questions, cached):
        if probs is None:
            raw_log = []
            try:
                probs, ptok = predict(video_url, q["question"], q["options"], raw_log)
                cache_save(vid, q["id"], probs)
                failed = False
            except Exception as e:
                with print_lock:
                    msg = str(e)
                    if len(msg) > 400:  # base64 in traceback clutters the console
                        msg = msg[:400] + " ...[truncated]"
                    print(f"  [{vid} q{q['id']}] FAILURE {type(e).__name__}: {msg}")
                probs = np.ones(len(q["options"])) / len(q["options"])
                ptok, failed = None, True
            finally:
                # written on success and on failure alike
                raw_cache_save(vid, q["id"], raw_log)
        else:
            ptok, failed = None, False

        pred = int(probs.argmax())
        real = q.get("answer_id")
        with print_lock:
            mark = "  " if real is None else ("OK " if pred == real else "ERR")
            tk = f" tok={ptok}" if ptok else " (cache)"
            print(f"  [{vid} q{q['id']}] {mark} pred={pred} real={real} "
                  f"conf={probs.max():.2f}{tk}")

        results.append({"id": q["id"], "answer_id": pred,
                        "answer": q["options"][pred]})
        records.append({
            "vid": vid, "qid": q["id"], "pred": pred, "real": real,
            "conf": float(probs.max()), "n_options": len(q["options"]),
            "tag": q.get("tag", []), "area": q.get("area", "?"),
            "reasoning": q.get("reasoning", "?"),
            "is_none": "None of the other" in q["options"][pred],
        })

    del video_url  # payload can be tens of megabytes
    return vid, results, records


# ==========================================================================
# Report
# ==========================================================================

def subset_of(vid):
    suffix = vid.rsplit("_", 1)[-1]
    return suffix if suffix in ("oai", "pa", "ao", "pt") else "orig"


def report(records):
    """Breakdown by subsets, areas, and tags. The overall figure hides
    where exactly the failure is: 58 new questions provide only 12.8% of the weight."""
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
    """Averaging distributions of multiple models.

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


def compare_submissions(subm_path):
    """Checking the submission against ground truth. The order of questions within a video
    must match the original."""
    with open(DB_PATH) as f:
        db1 = json.load(f)
    with open(subm_path) as f:
        db2 = json.load(f)

    real, pred = [], []
    for vid, data in db1.items():
        for i, q in enumerate(data.get("mc_question", [])):
            d2 = db2[vid][i]
            if q["id"] != d2["id"]:
                raise ValueError(f"ID mismatch: {vid} {q['id']} != {d2['id']}")
            real.append(q["answer_id"])
            pred.append(d2["answer_id"])

    acc = (np.array(pred) == np.array(real)).mean()
    print(f"compare_submissions: {acc:.4f} ({len(pred)} questions)")
    return acc


def probe(vid=None):
    """A single request on one video: check that the frames arrive entirely.

    prompt_tokens should be approximately n_sent * (w/32) * (h/32) / 2 plus text.
    If it is significantly less — the server still resampled the frames.
    """
    with open(DB_PATH) as f:
        db = json.load(f)
    vid = vid or next(iter(db))
    video_path = os.path.abspath(f"{VIDEO_DIR}/{vid}.mp4")
    video_url, info = sample_frames(video_path)
    w, h = info["size"]
    expect = info["n_sent"] * (w // PATCH) * (h // PATCH) // 2
    print(f"[{vid}] sent {info['n_sent']} frames @ {w}x{h}, "
          f"expecting ~{expect} video tokens")

    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=build_messages(video_url, "Describe the video in one sentence."),
        max_tokens=64,
        temperature=0.0,
        extra_body={"mm_processor_kwargs": {"do_sample_frames": False}},
    )
    print(f"actual prompt_tokens={r.usage.prompt_tokens}")
    print("response:", r.choices[0].message.content)


# ==========================================================================
# Main
# ==========================================================================

def main():
    with open(DB_PATH) as f:
        db = json.load(f)
    n_q = sum(len(d.get("mc_question", [])) for d in db.values())
    print(f"Videos: {len(db)}, questions: {n_q}")
    print(f"Config: {MODEL_NAME}, {N_FRAMES} frames, "
          f"max_frame_px={MAX_FRAME_PIXELS}, debias={DEBIAS}, "
          f"thinking={THINKING}, workers={MAX_WORKERS}, hash={CFG_HASH}")
    print(f"Cache: {CACHE_DIR} (raw dumps: {'on' if SAVE_RAW else 'off'})")

    start = time.time()
    submission, records = {}, []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_video, vid, data): vid
                   for vid, data in db.items()}
        done = 0
        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                vid, results, recs = fut.result()
                submission[vid] = results
                records.extend(recs)
            except Exception as e:
                with print_lock:
                    print(f"[{vid}] ERROR: {type(e).__name__}: {e}")
                submission[vid] = []
                for q in db[vid].get("mc_question", []):
                    submission[vid].append({"id": q["id"], "answer_id": 0,
                                            "answer": q["options"][0]})
                    records.append({
                        "vid": vid, "qid": q["id"], "pred": 0,
                        "real": q.get("answer_id"), "conf": 0.0,
                        "n_options": len(q["options"]), "tag": q.get("tag", []),
                        "area": q.get("area", "?"),
                        "reasoning": q.get("reasoning", "?"),
                        "is_none": False, "failed": True,
                    })
            done += 1
            if done % 5 == 0:
                scored = [r for r in records if r["real"] is not None]
                cur = (sum(r["pred"] == r["real"] for r in scored) / len(scored)
                       if scored else 0)
                with print_lock:
                    print(f"--- {done}/{len(db)} videos, {len(scored)} questions, "
                          f"acc={cur:.4f}, {time.time() - start:.0f}s")

    submission = {vid: submission.get(vid, []) for vid in db}  # db order

    os.makedirs(SUBM_DIR, exist_ok=True)
    subm_path = (f"{SUBM_DIR}/submission_valid_{MODEL_NAME}_"
                 f"{N_FRAMES}f_{MAX_FRAME_PIXELS}px_{CFG_HASH}.json")
    with open(subm_path, "w") as f:
        json.dump(submission, f)
    print(f"\nSaved to: {subm_path}")
    print(f"Time: {time.time() - start:.1f} sec")

    report(records)
    return subm_path


if __name__ == "__main__":
    args = apply_args(parse_args())

    if args.probe is not None:
        probe(args.probe or None)
        raise SystemExit(0)

    subm_path = main()
    try:
        compare_submissions(subm_path)
    except Exception as e:
        print("Can't calculate score", str(e))