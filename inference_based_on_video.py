import os
import re
import time
import json
import warnings
import numpy as np
import argparse
from decord import VideoReader, cpu
from openai import OpenAI

warnings.filterwarnings("ignore")

PORT = 8000
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

VIDEOS_FOLDER = '../input_multiple_choice_videoQA/unified_vqa_valid_split/valid_split/'
JSON_PATH = '../input_multiple_choice_videoQA/mc_vqa_unified_valid.json'
OUT_SUBM_FOLDER = './subm/'
CACHE_PATH = './cache/'
N_FRAMES = 300
MODEL_NAME = "qwen3-32bfp8"
THINKING = True


def parse_args():
    parser = argparse.ArgumentParser(description='Multiple choice VideoQA inference')
    parser.add_argument('--port', type=int, default=PORT,
                        help='Port of the local OpenAI-compatible server (vLLM etc.)')
    parser.add_argument('--videos-folder', type=str, default=VIDEOS_FOLDER,
                        help='Folder with videos (<video_id>.mp4 files)')
    parser.add_argument('--json-path', type=str, default=JSON_PATH,
                        help='JSON file with questions and ground truth answers')
    parser.add_argument('--out-subm-folder', type=str, default=OUT_SUBM_FOLDER,
                        help='Output folder for the resulting submission JSON')
    parser.add_argument('--cache-path', type=str, default=CACHE_PATH,
                        help='Folder where raw model predictions are cached (to resume runs)')
    parser.add_argument('--n-frames', type=int, default=N_FRAMES,
                        help='Target number of frames sampled per video (defines the FPS passed to the model)')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, choices=SUPPORTED_MODELS,
                        help='Model to run inference with; must be one of SUPPORTED_MODELS')
    parser.add_argument('--thinking', dest='thinking', action='store_true', default=THINKING,
                        help='Enable reasoning mode: model may emit a <think>...</think> block before the answer')
    parser.add_argument('--no-thinking', dest='thinking', action='store_false',
                        help='Disable reasoning mode: expect a short direct answer')
    args = parser.parse_args()

    if args.model_name not in SUPPORTED_MODELS:
        parser.error("Model '{}' is not supported. Choose from: {}".format(
            args.model_name, ', '.join(SUPPORTED_MODELS)))

    return args


def compare_submissions(subm_path):
    with open(JSON_PATH, 'r') as f:
        db1 = json.load(f)
    print(len(db1))

    with open(subm_path, 'r') as f:
        db2 = json.load(f)
    print(len(db2))

    answers_real = []
    answers_pred = []
    for vid, data in db1.items():
        print(vid)
        print(data)
        for i, q in enumerate(data.get('mc_question', [])):
            print(i, q['id'], q['answer_id'])
            data2 = db2[vid][i]
            print(data2['id'], data2['answer_id'])
            if q['id'] != data2['id']:
                print('Error!', q['id'], data2['id'])
                exit()
            answers_real.append(q['answer_id'])
            answers_pred.append(data2['answer_id'])

    answers_real = np.array(answers_real)
    answers_pred = np.array(answers_pred)
    accuracy = (answers_pred == answers_real).sum() / len(answers_pred)
    print("{:.4f}".format(accuracy))
    return accuracy


def proc_with_qwen():
    model_name = MODEL_NAME
    port = PORT

    client = OpenAI(
        base_url="http://127.0.0.1:{}/v1".format(port),
        api_key="EMPTY"
    )

    with open(JSON_PATH, 'r') as f:
        db = json.load(f)
    print("Total videos to process:", len(db))

    os.makedirs(CACHE_PATH, exist_ok=True)
    out_cache_folder = CACHE_PATH + '{}_data/'.format(model_name)
    os.makedirs(out_cache_folder, exist_ok=True)

    possible_predictions_errors = 0
    thinking_error = 0
    total = 0
    correct = 0
    submission = {}
    start_time = time.time()
    for vid, data in db.items():
        submission[vid] = []
        print("Go for: {}".format(total + 1))
        for q in data.get('mc_question', []):
            num_options = len(q['options'])
            try:
                real_answer = q['answer_id']
            except Exception as e:
                real_answer = -1
            total += 1

            cache_file = out_cache_folder + vid + '_qid_{}_results_{}_{}_char.txt'.format(q['id'], N_FRAMES, model_name)

            process = True
            if os.path.isfile(cache_file):
                print("Restore from cache!")
                model_prediction = open(cache_file, 'r', encoding='utf8').read()
                if "</think>" in model_prediction:
                    print("Thinking complete!")
                    process = False

            if process:
                options_text = "\n".join([f"{i}. {opt}" for i, opt in enumerate(q['options'])])

                prompt_text = (
                    "Watch the video and answer the multiple-choice question below.\n"
                    f"Question: {q['question']}\n\n"
                    "Options:\n"
                    f"{options_text}\n\n"
                    "Instruction: Output ONLY the number of the correct option (e.g., 0, 1, 2 etc). Do not include any other text, punctuation, or explanations."
                )

                print("--- Prompt Text Start ----------------------------------")
                print(prompt_text)
                print("--- Prompt Text End ----------------------------------")

                video_path = os.path.abspath(VIDEOS_FOLDER + '{}.mp4'.format(vid))
                print(os.path.isfile(video_path))

                prompt_multimodal = []

                vid_path = video_path
                if not vid_path.startswith("file://"):
                    vid_path = f"file://{vid_path}"

                prompt_multimodal.append({
                    "type": "video_url",
                    "video_url": {
                        "url": vid_path
                    }
                })

                prompt_multimodal.append({
                    "type": "text",
                    "text": prompt_text
                })

                messages = [
                    {
                        "role": "user",
                        "content": prompt_multimodal,
                    }
                ]

                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                original_fps = vr.get_avg_fps()
                video_duration = total_frames / original_fps

                target_fps = N_FRAMES / video_duration
                target_fps = min(target_fps, original_fps)

                print(f"Total frames: {total_frames}")
                print(f"Length of video: {video_duration:.2f} sec, FPS: {original_fps:.2f}")
                print(f"Target FPS: {target_fps:.2f}")

                try:
                    if MODEL_NAME in ['qwen38-27bfp8'] and THINKING is True:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=1.0,
                            top_p=0.95,
                            presence_penalty=0.0,
                            max_tokens=8 * 4096,
                            extra_body={
                                "mm_processor_kwargs": {
                                    "fps": target_fps,
                                    "min_pixels": 4 * 32 * 32,
                                    "max_pixels": 360 * 420
                                },
                                "chat_template_kwargs": {
                                    "enable_thinking": True,
                                    "preserve_thinking": True,
                                },
                            },
                            reasoning_effort="xhigh",
                        )
                    elif MODEL_NAME in ['qwen38-27bfp8'] and THINKING is False:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=0.7,
                            top_p=0.8,
                            presence_penalty=1.5,
                            max_tokens=16,
                            extra_body={
                                "mm_processor_kwargs": {
                                    "fps": target_fps,
                                    "min_pixels": 4 * 32 * 32,
                                    "max_pixels": 360 * 420
                                },
                                "top_k": 20,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                        )
                    elif MODEL_NAME in ['internvl3_5-1b', 'internvl3_5-8b', 'internvl3_5-38b']:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=16,
                        )
                    elif MODEL_NAME in ['gemma-4-31B-it']:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=16,
                            extra_body={
                                "mm_processor_kwargs": {
                                    "do_sample_frames": True,
                                    "num_frames": 100,
                                    "size": {"height": 432, "width": 768},
                                    "do_center_crop": False,
                                }
                            }
                        )
                    else:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=16,
                            extra_body={
                                "mm_processor_kwargs": {
                                    "fps": target_fps,
                                    "min_pixels": 4 * 32 * 32,
                                    "max_pixels": 360 * 420
                                }
                            }
                        )

                    model_prediction = response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"Error for answering question {q['id']} for video {vid}: {e}")
                    model_prediction = '0'

                model_prediction = model_prediction.strip()
                out = open(cache_file, 'w', encoding='utf-8')
                out.write(model_prediction)
                out.close()

            print("--- Model Prediction Start ----------------------------------")
            print(f"Question ID: {q['id']} | Model prediction: {model_prediction}")
            print("--- Model Prediction End ----------------------------------")

            if THINKING:
                if "</think>" in model_prediction:
                    model_prediction = model_prediction.split("</think>")[-1].strip()
                else:
                    model_prediction = model_prediction
                    thinking_error += 1

            match = re.search(r'\d+', model_prediction)
            if match:
                model_prediction = match.group(0)
            mp = int(model_prediction)

            if mp >= num_options:
                possible_predictions_errors += 1
                mp = num_options - 1

            if mp == real_answer:
                correct += 1

            if real_answer >= 0:
                print(f"Question ID: {q['id']} | Final model prediction: {mp} Real answer: {real_answer} Current quality: {100 * correct / total:.2f}%")
            else:
                print(f"Question ID: {q['id']} | Final model prediction: {mp}")

            submission[vid].append({
                'id': q['id'],
                'answer_id': mp,
                'answer': q['options'][mp]
            })
            print("\n")

    os.makedirs(OUT_SUBM_FOLDER, exist_ok=True)
    subm_path = OUT_SUBM_FOLDER + 'submission_v1_{}_{}.json'.format(model_name, N_FRAMES)
    with open(subm_path, 'w') as f:
        json.dump(submission, f)

    print(f"Saved submission to: {subm_path}")
    print('Possible predictions errors: {} from {}'.format(possible_predictions_errors, total))
    if THINKING:
        print('Thinking errors: {} from {}'.format(thinking_error, total))
    print("Time: {:.2f} sec".format(time.time() - start_time))

    return subm_path


if __name__ == "__main__":
    args = parse_args()

    PORT = args.port
    VIDEOS_FOLDER = args.videos_folder
    JSON_PATH = args.json_path
    OUT_SUBM_FOLDER = args.out_subm_folder
    CACHE_PATH = args.cache_path
    N_FRAMES = args.n_frames
    MODEL_NAME = args.model_name
    THINKING = args.thinking

    subm_path = proc_with_qwen()
    try:
        compare_submissions(subm_path)
    except Exception as e:
        print("Can't calculate score", str(e))
