# A Training-Free Ensemble of Open-Source Video-Language Models for Unified Multiple-Choice VideoQA

**The Fourth Perception Test Challenge (ECCV 2026) — Task 1: Unified Multiple-Choice VideoQA**
 
**Author**: Roman Solovyev ([MVSep.com](https://mvsep.com)).
**Project repository**: https://github.com/ZFTurbo/Multiple-Choice-VideoQA-2026

---

## 1. Summary

This report describes a training-free, single-GPU solution to the unified mc-vQA task. No model was trained or fine-tuned for this challenge, and no Perception Test data was used for any form of learning. The system is a thin inference layer around off-the-shelf, openly released video-language models (VLMs) served locally with vLLM, plus a set of decoding-side techniques that turn a generative model into a well-calibrated multiple-choice classifier:

1. controlled frame sampling under a fixed visual-token budget;
2. constrained decoding (`guided_choice`) with logprob extraction, giving a probability distribution over the answer options instead of a single token;
3. positional debiasing by averaging over cyclic permutations of the options;
4. optional extended reasoning ("thinking") mode;
5. a probability-level ensemble of independently cached model runs.

The best single model reaches **0.8194** top-1 accuracy on the validation split; the two-model ensemble reaches **0.8546**. The one configuration evaluated on the official leaderboard, Qwen3-VL-32B-FP8, scored **0.7759**, closely tracking its validation score of 0.7709–0.7841.

## 2. Models and evaluation mode

* All models are used exactly as published by their authors. The only "learned" component of the pipeline is the ensemble weighting, which consists of two integers chosen on the 454-question validation split.

* The strongest results come from the Qwen3-VL family. Qwen3-VL uses the now-standard three-part design — a SigLIP-2-based vision encoder trained at dynamic resolution, an MLP vision–language merger, and an LLM decoder — with three modifications that matter for this task: *interleaved MRoPE*, which spreads temporal, height and width positional components evenly across the embedding spectrum for better long-video modelling; *DeepStack*, which injects multi-level ViT features into several LLM layers rather than one; and explicit textual timestamp alignment for video. The family ships dense (8B/32B) and mixture-of-experts (30B-A3B/235B-A22B) variants with a native 256K-token interleaved context, which is what makes it feasible to place 100–300 frames and the question in a single prompt. The second primary model is a reasoning-capable 27B checkpoint (`qwen38-27bfp8`) whose post-training exposes a thinking / non-thinking switch.

* InternVL3.5 at 1B/8B/38B and Gemma-4-31B-it, included to check whether the conclusions are family-specific.

* All large checkpoints are run in FP8. Every experiment except the 235B-A22B model (3–4 GPUs) fits on a **single NVIDIA A6000 96GB**, served through `vllm.entrypoints.openai.api_server`; the inference scripts talk to it over the OpenAI-compatible API.

* Nothing was trained here. The underlying checkpoints were pretrained and post-trained by their respective vendors on large-scale interleaved image/video–text corpora; we make no claim about Perception Test contamination, and the validation-to-leaderboard correlation is the only evidence we have that validation numbers transfer.

## 3. Method

### 3.1 Two inference paths

Two entry points are implemented. The server-side variant (`inference_based_on_video.py`) hands the model a `file://` URL and lets vLLM decode the video, controlling sampling through `mm_processor_kwargs` (`fps`, `min_pixels`, `max_pixels`); the target FPS is computed per video as `N_FRAMES / duration`, clipped to the source FPS. The client-side variant (`inference_based_on_images.py`) does the sampling itself: frames are decoded by decord directly at the target resolution, JPEG-encoded, and sent as a base64 frame list with server-side resampling disabled (`do_sample_frames: False`). The second variant is the one used for all final results, because it makes the visual input exactly reproducible and removes any ambiguity about what the server actually saw. A `--probe` mode compares the observed `prompt_tokens` against the analytically expected token count to confirm frames are not being silently downsampled.

### 3.2 Frame budget

Frames are sampled uniformly over the whole clip. Frame sides are rounded to a multiple of 32 (patch 16 × merge 2), each frame is capped at `max_frame_pixels = 524288`, and the whole video is capped at a total budget of 25,165,824 pixels — roughly 12.3k visual tokens after spatial and temporal merging. This budget is the key constraint of the whole system: below ~48 frames the per-frame cap binds, above it the total budget binds, so frame count and per-frame resolution trade off directly against each other. At 100 frames each frame gets ≈250k pixels (~580×430); at 300 frames only ≈84k (~330×250).

### 3.3 From generation to a distribution

For non-thinking runs the model is asked for a single token, decoding is constrained to the valid option digits via `guided_choice`, and the option distribution is recovered from `top_logprobs` (top-20) rather than from the sampled token. This gives soft scores that are usable for both confidence analysis and ensembling. For thinking runs the model emits a `<think>` block with `reasoning_effort="xhigh"` and up to 32k tokens; the answer is parsed from the tail after `</think>`, with a one-hot fallback, and a truncation inside the thinking block falls back to a uniform distribution rather than to a silent wrong answer. Failures are recorded explicitly (`failed: True`) so that stub predictions are never mistaken for real ones in the metrics.

### 3.4 Positional debiasing

VLMs have a measurable preference for particular option positions. Each question is therefore asked once per cyclic permutation of its options (3 or 5 calls), and the resulting distributions are mapped back to the original option order and averaged. This costs *k×* more requests and buys 1–2 points.

### 3.5 Engineering

The prompt is built video-first, text-last, so that all questions of a video and all option permutations share one visual prefix and hit vLLM's prefix cache. Videos are processed by a thread pool (4 workers); frames are encoded once per video and reused. Every question's distribution is cached to `{vid}_q{qid}_{cfg_hash}.json`, where the hash covers model, frame count, pixel cap, JPEG quality, debias flag and prompt version — so runs resume for free and configurations can never contaminate one another. Raw prompts, reasoning traces and responses are dumped alongside for error analysis.

### 3.6 Ensemble

Because every run leaves calibrated per-option distributions on disk, `ensemble_on_cache.py` combines any set of previous runs post-hoc by weighted averaging of the distributions, with no re-inference. The best ensemble uses Qwen3-VL-32B-FP8 and the 27B thinking model with weights 8:7.

### 3.7 Analysis tooling

The reporting function breaks accuracy down by *subset* (original vs. the new `pt`/`ao`/`pa`/`oai` suffixes), *number of options*, *skill area*, *reasoning type* and *tag*, alongside the per-slice random baseline (≈0.32 overall for the validation mix). This matters because the new 5-option questions are only ~13% of validation weight, and the aggregate number hides them entirely; slices with fewer than 6 samples are printed but treated as noise.

## 4. Results (validation split: 189 videos, 454 questions)

**Server-side decoding, 300 frames / 2 FPS**

| Model | Settings | Top-1 |
| :--- | :--- | :--- |
| qwen3-32bfp8 | default | 0.7709 |
| qwen3-32bfp8 | max_pixels 256·28², debias | 0.7841 |
| qwen38-27bfp8 | thinking off | 0.6872 |
| qwen38-27bfp8 | **thinking on** | **0.8128** |
| qwen3-30ba3bfp8 | max_pixels 256·28², debias | 0.7291 |
| qwen3-8b | max_pixels 256·28², debias | 0.6608 |
| qwen3-235b | max_pixels 512·28², debias | 0.7753 |
| internvl3_5-1b / 8b / 38b | default | 0.3855 / 0.5705 / 0.6410 |
| gemma-4-31B-it | default | 0.6101 |

**Client-side sampling** (`max_frame_px = 524288`)

| Model | Settings | Top-1 | Time (s) |
| :--- | :--- | :--- | :--- |
| qwen3-32bfp8 | 32 f, debias | 0.7467 | 591 |
| qwen3-32bfp8 | **100 f, debias** | **0.7819** | 880 |
| qwen3-32bfp8 | 150 f, debias | 0.7731 | 895 |
| qwen3-32bfp8 | 200 f, debias | 0.7577 | 992 |
| qwen3-32bfp8 | 300 f, debias | 0.7357 | — |
| qwen3-235b | 150 f, no debias | 0.7709 | 746 |
| qwen38-27bfp8 | 100 f, debias, **thinking** | **0.8194** | 29,711 |
| qwen38-27bfp8 | 100 f, debias, no thinking | 0.7203 | 3,226 |
| qwen38-27bfp8 | 100 f, no debias, no thinking | 0.7026 | 1,014 |
| internvl3_5-8b / 38b | 32 f, ≤768px | 0.5617 / 0.6322 | — |
| gemma-4-31B-it | 48 f, no debias | 0.6520 | 991 |

**Ensemble (32B + 27B-thinking, 8:7): 0.8546.**
**Test leaderboard, qwen3-32bfp8: 0.7759.**

## 5. Findings

**More frames are not better.** Under a fixed token budget, accuracy peaks at 100 frames and falls monotonically to 0.7357 at 300. Temporal coverage is bought with spatial detail, and for Perception Test questions — many of which hinge on small objects and short interactions — spatial detail wins. This is the single largest configuration effect we measured and it is the opposite of the usual "sample more frames" instinct.

**Reasoning is worth ~10 points and costs ~10×.** Thinking mode moves the 27B model from 0.7203 to 0.8194 (and 0.6872 → 0.8128 in the server-side setting), at 29,711 s vs. 3,226 s on 454 questions. Extrapolated to the 13,370-question test set, this is the reason only the 32B model was submitted to the leaderboard.

**Debiasing is cheap and consistent.** Cyclic permutation averaging gives +1.8 points (0.7026 → 0.7203) for 3× the requests, and helped in every configuration where it was tried.

**Scale saturates.** 8B → 32B is worth ~12 points, but 235B-A22B does not beat 32B (0.7753 / 0.7709 vs. 0.7841 / 0.7819) while requiring 3–4 GPUs. Reasoning ability and input configuration matter more than parameter count here. The same ordering holds within InternVL3.5 (0.39 → 0.57 → 0.64), so this is not a Qwen-specific artefact.

**Diversity beats the best member.** The 32B non-thinking and 27B thinking models fail on different questions: averaging their distributions adds 3.5 points over the better of the two. Soft probabilities are what make this work — hard voting over two models would be undefined.

**Validation transfers.** 0.7841 validation vs. 0.7759 leaderboard for the one submitted configuration suggests the validation split, despite its size, is a usable proxy. On that basis the 27B thinking model would be expected around 0.81 on the leaderboard.

## 6. Limitations and future work

Frame sampling is uniform - no question-conditioned or keyframe-based retrieval was attempted, which is likely where the point-tracking and action-localisation questions in the new 5-option subsets are lost. The ensemble weights (8:7) were picked on 454 questions and should be treated as a rough prior rather than a tuned parameter. Finally, the thinking model was never evaluated on the leaderboard: the 0.81 estimate rests on a single validation-to-test correlation point. Also ensemble should be tested on LB too.

## References

1. V. Pătrăucean et al. *Perception Test: A Diagnostic Benchmark for Multimodal Video Models.* NeurIPS 2023 (Datasets and Benchmarks).
2. Perception Test Challenge 2026, Task 1 — Unified Multiple-Choice VideoQA. https://eval.ai/web/challenges/challenge-page/2702/overview
3. S. Bai et al. *Qwen3-VL Technical Report.* arXiv:2511.21631. https://arxiv.org/abs/2511.21631
4. W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).* SOSP 2023.
5. Qwen Team. *Qwen3-VL.* GitHub repository. https://github.com/QwenLM/Qwen3-VL
6. Zhang et al. *Decord: an efficient video loader for deep learning with smart shuffling.* https://github.com/dmlc/decord

### Model checkpoints

All checkpoints are used as released, without fine-tuning.

| Ref | Model | Role in this study | Model card |
| :--- | :--- | :--- | :--- |
| M1 | Qwen3-VL-32B-Instruct-FP8 | primary non-reasoning model; only leaderboard submission | https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8 |
| M2 | Qwen3.8-27B-FP8 | primary reasoning model; best single-model result (0.8194) | https://huggingface.co/Qwen/Qwen3.8-27B-FP8 |
| M3 | Qwen3-VL-30B-A3B-Instruct | MoE scale point | https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct |
| M4 | Qwen3-VL-8B-Instruct | small dense scale point | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct |
| M5 | Qwen3-VL-235B-A22B-Instruct-FP8 | large MoE scale point (3–4 GPUs) | https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 |
| M6 | InternVL3.5-1B | cross-family control | https://huggingface.co/OpenGVLab/InternVL3_5-1B |
| M7 | InternVL3.5-8B | cross-family control | https://huggingface.co/OpenGVLab/InternVL3_5-8B |
| M8 | InternVL3.5-38B | cross-family control | https://huggingface.co/OpenGVLab/InternVL3_5-38B |
| M9 | Gemma-4-31B-it | cross-family control | https://huggingface.co/google/gemma-4-31B-it |