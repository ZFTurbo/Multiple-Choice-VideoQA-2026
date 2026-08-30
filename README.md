# Solution for Perception Test Challenge 2026

This repository contains a solution for the [Perception Test Challenge 2026 - Task 1 - Multiple-choice videoQA](https://eval.ai/web/challenges/challenge-page/2702/overview) by Google DeepMind. The solution is based entirely on open-source models and can be run on a single GPU.

## Tested Models

**Best performing models:** 
* [Qwen3-VL 32B FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8)
* [Qwen3.8 27B FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

**Additional models evaluated:** 
* [Qwen3-VL 30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
* [Qwen3-VL 8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
* [Qwen3-VL 235B-A22B FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8)
* [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
* [InternVL3.5-38B](https://huggingface.co/OpenGVLab/InternVL3_5-38B)
* [Gemma4 31B](https://huggingface.co/google/gemma-4-31B-it)

## System Requirements

All experiments were performed on a single NVIDIA A6000 96GB GPU. The only exception is `Qwen3-VL 235B-A22B`, which required 3-4 GPUs.

## Data Preparation

You can download the dataset from the `Dataset` section on the [challenge page](https://eval.ai/web/challenges/challenge-page/2702/overview), or use the direct links below:

| Split | Videos | Questions | Download Links |
| :--- | :--- | :--- | :--- |
| **Validation** | 189 | 454 | [Videos (2.4 GB)](https://storage.googleapis.com/dm-perception-test/zip_data/unified_vqa_valid_split.zip), [Annotations](https://storage.googleapis.com/dm-perception-test/zip_data/mc_vqa_unified_valid.zip) |
| **Test** | 3525 + 1842 | 11528 + 1842 | Videos: [Original set (41.8 GB)](https://storage.googleapis.com/dm-perception-test/zip_data/test_videos.zip) + [New set (32 GB)](https://storage.googleapis.com/dm-perception-test/zip_data/unified_vqa_test_videos.zip)<br>Annotations: [Original set](https://storage.googleapis.com/dm-perception-test/zip_data/test_annotations.zip) + [New set](https://storage.googleapis.com/dm-perception-test/zip_data/mc_vqa_unified_test.json) |

## Run Inference

There are two inference variants: video-based - `inference_based_on_images.py` and image-based (extracting frames) - `inference_based_on_video.py`. Before running the inference script, you need to start the vLLM server.

Example for `Qwen3-VL-8B-Instruct`:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3-VL-8B-Instruct \
--served-model-name qwen3-8b \
--tensor-parallel-size 1 \
--max-num-seqs 32 \
--gpu-memory-utilization 0.95 \
--host localhost \
--port 8000 \
--dtype auto \
--max-model-len 131072 \
--allowed-local-media-path <path_to_your_dataset> \
--mm-processor-kwargs '{"fps": 2}' \
--media-io-kwargs '{"video": {"num_frames": -1}}'
```

**Notes:**
* The model is served under the name `qwen3-8b`, which is controlled by the `--served-model-name` parameter (see the supported models list below).
* You **must** provide the correct dataset path via the `--allowed-local-media-path` argument, otherwise the server won't be able to access the media files.

The codebase currently supports the following models:

| Parameter | Model |
| :--- | :--- |
| `qwen3-32bfp8` | [Qwen3-VL 32B FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) |
| `qwen3-8b` | [Qwen3-VL 8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| `qwen3-30ba3bfp8` | [Qwen3-VL 30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |
| `qwen38-27bfp8` | [Qwen3.8 27B FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| `qwen3-235b` | [Qwen3-VL 235B-A22B FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8) |
| `internvl3_5-1b` | [InternVL3.5-1B](https://huggingface.co/OpenGVLab/InternVL3_5-1B) |
| `internvl3_5-8b` | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |
| `internvl3_5-38b` | [InternVL3.5-38B](https://huggingface.co/OpenGVLab/InternVL3_5-38B) |
| `gemma-4-31B-it` | [Gemma4 31B](https://huggingface.co/google/gemma-4-31B-it) |

### Run on Validation Data

```bash
python3 inference_based_on_video.py \
--model-name qwen3-8b \
--port 8000 \
--videos-folder ../input_multiple_choice_videoQA/unified_vqa_valid_split/valid_split/ \
--json-path ../input_multiple_choice_videoQA/mc_vqa_unified_valid.json
```
*Note: Metrics are calculated automatically during the validation run.*

### Run on Test Data

```bash
python3 inference_based_on_video.py \
--model-name qwen3-8b \
--port 8000 \
--videos-folder ../input_multiple_choice_videoQA/unified_vqa_test_videos/unified_vqa_test_videos/ \
--json-path ../input_multiple_choice_videoQA/mc_vqa_unified_test.json
```

## Validation Results

### Video-based Inference
*Frames setup: 300 frames or 2 FPS*

| Model | Settings | Metric |
| :--- | :--- | :--- |
| qwen3-32bfp8 | Default | 0.7709 |
| qwen3-32bfp8 | max_pixels: 256*28*28, debias: True | 0.7841 |
| qwen38-27bfp8 | Thinking: off | 0.6872 |
| qwen38-27bfp8 | Thinking: on | 0.8128 |
| qwen3-30ba3bfp8 | max_pixels: 256*28*28, debias: True | 0.7291 |
| qwen3-8b | max_pixels: 256*28*28, debias: True | 0.6608 |
| qwen3-235b | max_pixels: 512*28*28, debias: True | 0.7753 |
| internvl3_5-1b | Default | 0.3855 |
| internvl3_5-8b | Default | 0.5705 |
| internvl3_5-38b | Default | 0.6410 |
| gemma-4-31B-it | Default | 0.6101 |

### Image-based Inference

| Model                         | Settings                                      | Metric | Time (sec) |
|:------------------------------|:----------------------------------------------|:-------|:-----------|
| qwen3-32bfp8                  | 100 frames, max_frame_px=524288, debias=True  | 0.7819 | 880        |
| qwen3-32bfp8                  | 150 frames, max_frame_px=524288, debias=True  | 0.7731 | 895        |
| qwen3-32bfp8                  | 200 frames, max_frame_px=524288, debias=True  | 0.7577 | 992        |
| qwen3-32bfp8                  | 300 frames, max_frame_px=524288, debias=True  | 0.7357 | ---        |
| qwen3-32bfp8                  | 32 frames, max_frame_px=524288, debias=True   | 0.7467 | 591        |
| qwen3-235b                    | 150 frames, max_frame_px=524288, debias=False | 0.7709 | 746        |
| qwen3-2bfp8                   | 100 frames, max_frame_px=524288, debias=True  | 0.5617 | ---        |
| internvl3_5-1b                | N_FRAMES 12, orig image size                  | 0.4097 | ---        |
| internvl3_5-8b                | N_FRAMES 32 (Max size: 768px)                 | 0.5617 | ---        |
| internvl3_5-38b               | N_FRAMES 32 (Max size: 768px)                 | 0.6322 | ---        |
| gemma-4-31B-it                | 48 frames, budget=140, debias=False           | 0.6520 | 991        |
| qwen38-27bfp8 (Thinking: on)  | 100 frames, max_frame_px=524288, debias=True  | 0.8194 | 29711      |
| qwen38-27bfp8 (Thinking: off) | 100 frames, max_frame_px=524288, debias=True  | 0.7203 | 3226       |
| qwen38-27bfp8 (Thinking: off) | 100 frames, max_frame_px=524288, debias=False | 0.7026 | 1014       |

## Test Results (Leaderboard)

Due to the massive size of the test dataset, I was only able to evaluate the **Qwen3-VL 32B FP8** model on the official Leaderboard (LB). It achieved a score of **0.7759**, which aligns closely with its validation score. 

Based on this correlation, I expect the **Qwen3.8 27B FP8** (with thinking enabled) to achieve an LB score of around **0.81**. However, it is worth noting that running this model in "thinking" mode is significantly slower than the Qwen3-VL 32B FP8.

## Ensemble

If you use `inference_based_on_images.py` you will have cache with probabilities for every model you tried. In thi case you can use `ensemble_on_cache.py` script. 
For example ensemble of qwen3-32bfp8 and qwen38-27bfp8 (Thinking: on) on validation increases score up to **0.8546**

## Docs

[Report](docs/perception_test_2026_report.md)