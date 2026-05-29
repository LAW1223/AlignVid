import os
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Environment variables
os.environ["DECORD_EOF_RETRY_MAX"] = "2048"

def extract_yes_no(response):
    """Extract a yes/no answer from the model response."""
    response = response.lower().strip()

    # Check the first few words directly
    first_words = response.split(",")[0].strip()

    if first_words.startswith("yes"):
        return "yes"
    elif first_words.startswith("no"):
        return "no"

    # If no explicit yes/no at the start, infer from the content
    if "yes" in response and "no" not in response:
        return "yes"
    elif "no" in response and "yes" not in response:
        return "no"

    # Last resort: look at the very first few words
    first_few_words = ' '.join(response.split()[:5])
    if "yes" in first_few_words:
        return "yes"
    elif "no" in first_few_words:
        return "no"

    # Default: undetermined
    return "unknown"

def evaluate_single_video(model, processor, tokenizer, item_data, video_dir, fps=8.0):
    """Evaluate a single video."""
    item_id = item_data.get("id", "unknown")
    video_path = os.path.join(video_dir, f"{item_id}.mp4")

    print(f"\nProcessing video: {video_path}")

    # Check that the video file exists
    if not os.path.exists(video_path):
        print(f"Warning: video file not found: {video_path}")
        return {
            "id": item_id,
            "video_path": video_path,
            "error": "Video file not found",
            "questions": {},
            "correct": 0,
            "total": 0,
            "accuracy": 0
        }

    # Get the list of questions
    questions = item_data.get("questions", [])
    if not questions:
        print(f"Warning: no questions found for item {item_id}")
        return {
            "id": item_id,
            "video_path": video_path,
            "error": "No questions found",
            "questions": {},
            "correct": 0,
            "total": 0,
            "accuracy": 0
        }

    # Initialize the result
    results = {
        "id": item_id,
        "video_path": video_path,
        "prompt": item_data.get("prompt", ""),
        "expected_change": item_data.get("expected-change", ""),
        "main_category": item_data.get("main-category", ""),
        "sub_category": item_data.get("sub-category", ""),
        "domain": item_data.get("domain", ""),
        "questions": {},
        "correct": 0,
        "total": 0,
        "accuracy": 0
    }

    # Evaluate each question
    for q in questions:
        q_id = q.get("id", f"q{len(results['questions'])+1}")
        question_text = q.get("question", "")
        expected_answer = q.get("expected_answer", "unknown")
        category = q.get("category", "unknown")

        if not question_text:
            print(f"Skipping empty question: {q_id}")
            continue

        print(f"Evaluating question {q_id}: {question_text}")

        # Build the prompt - important: provide no prompt hint, so the model
        # answers purely from the video.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "max_pixels": 360 * 420
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Question: {question_text}\n\n"
                            "Please watch the video carefully and answer the question based solely on what you observe. "
                            "Reply with 'yes' or 'no' followed by a brief explanation of your reasoning."
                        )
                    }
                ]
            }
        ]

        try:
            # Prepare inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")

            # Generate the answer
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print(f"Response: {response}")

            # Extract yes/no from the response
            answer = extract_yes_no(response)
            is_correct = (answer == expected_answer)
            print(f"Extracted answer: {answer}, expected: {expected_answer}, correct: {is_correct}")

        except Exception as e:
            print(f"Error generating response: {e}")
            response = f"Error generating response: {str(e)}"
            answer = "unknown"
            is_correct = False

        # Save the question result
        results["questions"][q_id] = {
            "question": question_text,
            "expected_answer": expected_answer,
            "model_answer": answer,
            "correct": is_correct,
            "response": response,
            "category": category
        }

        if is_correct:
            results["correct"] += 1
        results["total"] += 1

        # Clear the GPU cache
        torch.cuda.empty_cache()

    # Compute accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate OmitI2V generated videos with Qwen2.5-VL")
    parser.add_argument("--json_file", type=str, default="/path/to/OmitI2V/meta.json",
                        help="JSON file containing the questions")
    parser.add_argument("--video_dir", type=str, default="/path/to/generated_videos",
                        help="directory of generated videos")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="output file for the evaluation results")
    parser.add_argument("--model_path", type=str,
                        default="/path/to/Qwen2.5-VL-32B-Instruct",
                        help="path to the model")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="video sampling frame rate")
    parser.add_argument("--max_items", type=int, default=None,
                        help="max number of items to evaluate (for testing)")

    args = parser.parse_args()

    try:
        # Load the model and processor - use the local path directly
        print(f"Loading model: {args.model_path}")

        # Check that the model path exists
        if not os.path.exists(args.model_path):
            print(f"Error: model path not found: {args.model_path}")
            return

        # Use local_files_only to avoid network requests
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,   # use local files only
                trust_remote_code=True   # trust remote code
            )

            processor = AutoProcessor.from_pretrained(
                args.model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            print(f"Model loaded: {args.model_path}")

        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Please check that the model path is correct and the model files are complete")
            return

        # Load the JSON file
        try:
            with open(args.json_file, "r", encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return

        if not isinstance(data, list):
            print("Invalid JSON format: expected a list")
            return

        # Keep only items that have questions
        valid_items = []
        for item in data:
            if "questions" in item and len(item["questions"]) > 0:
                valid_items.append(item)
            else:
                print(f"Skipping item without questions: {item.get('id', 'unknown')}")

        print(f"Found {len(valid_items)} valid items")

        # Limit the number of items (for testing)
        if args.max_items:
            valid_items = valid_items[:args.max_items]
            print(f"Limiting evaluation to: {args.max_items}")

        # Evaluate each item
        all_results = []
        total_correct = 0
        total_questions = 0

        for item in tqdm(valid_items, desc="Evaluating"):
            result = evaluate_single_video(model, processor, tokenizer, item, args.video_dir, args.fps)
            all_results.append(result)

            if "error" not in result:
                total_correct += result["correct"]
                total_questions += result["total"]

        # ---- Aggregate metrics: per-sample macro-average ----
        # Each sample is weighted equally regardless of how many questions it
        # has, so samples with more questions do not dominate the score.
        valid_results = [r for r in all_results if "error" not in r]

        def macro_accuracy(samples):
            accs = [r["accuracy"] for r in samples]
            return sum(accs) / len(accs) if accs else 0.0

        def group_by(samples, key):
            groups = {}
            for r in samples:
                groups.setdefault(r.get(key, "unknown"), []).append(r)
            return groups

        overall_accuracy = macro_accuracy(valid_results)
        category_stats = {
            cat: {"num_samples": len(g), "accuracy": macro_accuracy(g)}
            for cat, g in group_by(valid_results, "main_category").items()
        }
        domain_stats = {
            dom: {"num_samples": len(g), "accuracy": macro_accuracy(g)}
            for dom, g in group_by(valid_results, "domain").items()
        }

        # Per-question dimension accuracy (a dimension is a property of a
        # question, so this breakdown stays at the question level).
        dimension_stats = {}
        for r in valid_results:
            for q_info in r["questions"].values():
                if q_info.get("correct") is None:
                    continue
                d = dimension_stats.setdefault(
                    q_info.get("category", "unknown"),
                    {"correct": 0, "total": 0, "accuracy": 0})
                d["correct"] += int(bool(q_info["correct"]))
                d["total"] += 1
        for d in dimension_stats.values():
            if d["total"] > 0:
                d["accuracy"] = d["correct"] / d["total"]

        # ---- Build & save results ----
        final_results = {
            "model_path": args.model_path,
            "overall": {
                "total_items": len(all_results),
                "valid_items": len(valid_results),
                "total_questions": total_questions,
                "total_correct": total_correct,
                "overall_accuracy": overall_accuracy,
            },
            "statistics": {
                "by_category": category_stats,
                "by_dimension": dimension_stats,
                "by_domain": domain_stats,
            },
            "detailed_results": all_results,
        }

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"\n=== Evaluation done ===")
        print(f"Saved to: {args.output_file}")
        print(f"Overall accuracy (per-sample macro): {overall_accuracy:.4f} "
              f"over {len(valid_results)} samples")
        print(f"\n=== By main-category ===")
        for cat, s in category_stats.items():
            print(f"  {cat}: {s['accuracy']:.4f} (n={s['num_samples']})")
        print(f"\n=== By dimension (per-question) ===")
        for dim, s in dimension_stats.items():
            print(f"  {dim}: {s['accuracy']:.4f} ({s['correct']}/{s['total']})")
        print(f"\n=== By domain ===")
        for dom, s in domain_stats.items():
            print(f"  {dom}: {s['accuracy']:.4f} (n={s['num_samples']})")

    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

'''
python qwen.py \
    --json_file /path/to/OmitI2V/meta.json \
    --video_dir /path/to/generated_videos \
    --model_path /path/to/Qwen2.5-VL-32B-Instruct \
    --output_file evaluation_results.json
'''
