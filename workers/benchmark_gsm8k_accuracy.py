#!/usr/bin/env python3
"""
Full GSM8K benchmark for speculative decoding with accuracy tracking.

Similar to benchmark_gsm8k.py, but focused on GSM8K and includes:
- per-question correctness (exact numeric match)
- final benchmark accuracy
- per-question accepted token rates logged to CSV/JSON
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

# Add proto directory to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "proto",
    ),
)

import common_pb2
import speculative_decoding_pb2
from draft_node.client import DraftNodeClient


# Match \boxed{number} - common LaTeX format for final answers (check first)
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def flexible_extract_math(text: str) -> str:
    """
    Pure Python equivalent of lm-eval's flexible-extract for numeric tasks.
    Handles \\boxed{18}, $18, 1,234, etc. Requires at least one digit to avoid
    matching spurious sequences like "$." at end of "\\boxed{18}$."
    """
    # Prefer \boxed{number} - common format for math answers
    boxed_match = BOXED_PATTERN.search(text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # The regex pattern used in the lm-eval YAML
    pattern = r"(-?[ $0-9.,]{2,})|(-?[ 0-9]+)"

    # 1. Regex Filter: Find all matches
    matches = re.findall(pattern, text)
    if not matches:
        return text

    # 2. Filter: Only keep matches that contain at least one digit (avoids "$." etc.)
    digit_pattern = re.compile(r"\d")
    valid_matches = [
        m for m in matches
        if digit_pattern.search(next((g for g in m if g), ""))
    ]
    if not valid_matches:
        return text

    # 3. Group Select: Grab the last valid match
    last_match = valid_matches[-1]
    extracted = next((g for g in last_match if g), "")

    # 4. Take First Filter: Clean and return
    return extracted.strip()


def parse_decimal(value: str) -> Optional[Decimal]:
    """Parse numeric string to Decimal after stripping commas, $, and trailing period."""
    if not value:
        return None
    normalized = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def extract_numeric_answer(text: str) -> Tuple[Optional[str], Optional[Decimal]]:
    """
    Extract a final numeric answer from generated text using lm-eval's flexible-extract.
    Returns (raw_extracted_string, parsed_decimal_or_none).
    """
    if not text:
        return None, None

    extracted = flexible_extract_math(text)
    parsed = parse_decimal(extracted)
    return (extracted, parsed) if parsed is not None else (None, None)


def load_gsm8k_testset(num_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """Load GSM8K test split from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    print("Loading GSM8K test split from HuggingFace...")
    dataset = load_dataset("gsm8k", "main", split="test")
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    samples: List[Dict[str, str]] = []
    for item in dataset:
        answer_text = item["answer"].split("####")[-1].strip()
        samples.append(
            {
                "question": item["question"],
                "answer": answer_text,
            }
        )

    print(f"Loaded {len(samples)} GSM8K questions")
    return samples


def get_completion_text(full_text: str, prompt_text: str) -> str:
    """Best-effort extraction of completion-only text from generated output."""
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    marker = "Answer: Let's solve this step by step."
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    return full_text.strip()


def run_benchmark(
    draft_model: str,
    target_model: str,
    verification_server: str,
    num_samples: Optional[int],
    max_tokens: int,
    temperature: float
) -> None:
    print("\n" + "=" * 100)
    print("GSM8K FULL BENCHMARK - Acceptance Rates + Accuracy")
    print("=" * 100)
    print("Configuration:")
    print(f"  Draft model:  {draft_model}")
    print(f"  Target model: {target_model}")
    print(f"  Server:       {verification_server}")
    print(f"  Max tokens:   {max_tokens}")
    print(f"  Temperature:  {temperature}")
    print(f"  Samples:      {num_samples if num_samples is not None else 'ALL GSM8K test questions'}")
    print("=" * 100 + "\n")

    questions = load_gsm8k_testset(num_samples)

    print("Initializing draft node client...")
    client = DraftNodeClient(
        draft_model=draft_model,
        verification_server=verification_server,
        num_draft_tokens=5,
    )

    results: List[Dict[str, object]] = []
    total_start_time = time.time()
    running_correct = 0
    running_parsed = 0
    running_total_acceptance = 0.0

    for i, item in enumerate(questions):
        question_num = i + 1
        question = item["question"]
        expected_answer_text = item["answer"]

        model_prompt = (
            f"Question: {question}\n\n"
            "Answer: Let's solve this step by step.\n"
        )

        request = speculative_decoding_pb2.InferenceJobRequest(
            request_id=f"gsm8k-{i}",
            prompt=model_prompt,
            params=common_pb2.InferenceParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=-1 if temperature == 0 else 50,
                draft_tokens=5,
            ),
            model_id=target_model,
            timestamp=int(time.time() * 1000),
        )

        start_time = time.time()
        response = client.execute_inference(request)
        elapsed = time.time() - start_time

        completion_text = get_completion_text(response.generated_text, model_prompt)
        predicted_raw, predicted_value = extract_numeric_answer(completion_text)
        expected_raw, expected_value = extract_numeric_answer(expected_answer_text)

        is_correct = (
            predicted_value is not None
            and expected_value is not None
            and predicted_value == expected_value
        )

        if predicted_value is not None:
            running_parsed += 1
        if is_correct:
            running_correct += 1
        running_total_acceptance += response.acceptance_rate

        result = {
            "question_num": question_num,
            "question": question,
            "expected_answer": expected_answer_text,
            "predicted_answer_raw": predicted_raw,
            "predicted_answer_value": str(predicted_value) if predicted_value is not None else None,
            "expected_answer_raw": expected_raw,
            "expected_answer_value": str(expected_value) if expected_value is not None else None,
            "is_correct": is_correct,
            "generated_text": response.generated_text,
            "completion_text": completion_text,
            "total_tokens": response.total_tokens,
            "draft_generated": response.draft_tokens_generated,
            "draft_accepted": response.draft_tokens_accepted,
            "acceptance_rate": response.acceptance_rate,
            "speculation_rounds": response.speculation_rounds,
            "generation_time_ms": response.generation_time_ms,
            "wall_time_seconds": elapsed,
            "tokens_per_sec": (
                response.total_tokens / (response.generation_time_ms / 1000)
                if response.generation_time_ms > 0
                else 0.0
            ),
        }
        results.append(result)

        running_accuracy = running_correct / question_num
        running_parse_rate = running_parsed / question_num
        running_avg_acceptance = running_total_acceptance / question_num
        print(
            f"[{question_num}/{len(questions)}] "
            f"correct={running_correct} "
            f"accuracy={running_accuracy:.2%} "
            f"parse_rate={running_parse_rate:.2%} "
            f"avg_acceptance={running_avg_acceptance:.2%}"
        )

    total_elapsed = time.time() - total_start_time

    avg_acceptance = sum(r["acceptance_rate"] for r in results) / len(results)
    avg_speed = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_time_per_question = sum(r["wall_time_seconds"] for r in results) / len(results)
    total_tokens = sum(r["total_tokens"] for r in results)
    total_draft_generated = sum(r["draft_generated"] for r in results)
    total_draft_accepted = sum(r["draft_accepted"] for r in results)
    total_rounds = sum(r["speculation_rounds"] for r in results)

    total_correct = sum(1 for r in results if r["is_correct"])
    total_parsed = sum(1 for r in results if r["predicted_answer_value"] is not None)
    accuracy = total_correct / len(results)
    parse_rate = total_parsed / len(results)
    overall_acceptance = (
        total_draft_accepted / total_draft_generated if total_draft_generated > 0 else 0.0
    )

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print("Quality:")
    print(f"  Accuracy (exact numeric): {accuracy:.2%} ({total_correct}/{len(results)})")
    print(f"  Parse rate:               {parse_rate:.2%} ({total_parsed}/{len(results)})")
    print("Speculative Decoding:")
    print(f"  Avg acceptance rate:      {avg_acceptance:.2%}")
    print(f"  Overall acceptance:       {overall_acceptance:.2%} ({total_draft_accepted}/{total_draft_generated})")
    print(f"  Avg speed:                {avg_speed:.1f} tokens/sec")
    print(f"  Total speculation rounds: {total_rounds}")
    print("Timing:")
    print(f"  Total benchmark time:     {total_elapsed:.1f}s")
    print(f"  Avg per question:         {avg_time_per_question:.1f}s")
    print("=" * 100)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = int(time.time())

    json_path = os.path.join(logs_dir, f"gsm8k_full_accuracy_benchmark_{timestamp}.json")
    csv_path = os.path.join(logs_dir, f"gsm8k_full_accuracy_benchmark_{timestamp}.csv")

    payload = {
        "config": {
            "dataset": "gsm8k/main:test",
            "draft_model": draft_model,
            "target_model": target_model,
            "verification_server": verification_server,
            "num_samples": len(results),
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        "summary": {
            "accuracy": accuracy,
            "num_correct": total_correct,
            "num_questions": len(results),
            "parse_rate": parse_rate,
            "avg_acceptance_rate": avg_acceptance,
            "overall_acceptance_rate": overall_acceptance,
            "avg_speed_tokens_per_sec": avg_speed,
            "avg_time_per_question_sec": avg_time_per_question,
            "total_tokens": total_tokens,
            "total_draft_generated": total_draft_generated,
            "total_draft_accepted": total_draft_accepted,
            "total_speculation_rounds": total_rounds,
            "total_benchmark_time_sec": total_elapsed,
        },
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_num",
                "is_correct",
                "predicted_answer_value",
                "expected_answer_value",
                "acceptance_rate",
                "draft_accepted",
                "draft_generated",
                "tokens_per_sec",
                "generation_time_ms",
                "speculation_rounds",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["question_num"],
                    row["is_correct"],
                    row["predicted_answer_value"],
                    row["expected_answer_value"],
                    row["acceptance_rate"],
                    row["draft_accepted"],
                    row["draft_generated"],
                    row["tokens_per_sec"],
                    row["generation_time_ms"],
                    row["speculation_rounds"],
                ]
            )

    print(f"Saved JSON results to: {json_path}")
    print(f"Saved CSV logs to:     {csv_path}")

    del client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GSM8K benchmark with accepted-token rates and final accuracy."
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Draft model (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Target model served by verification server (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="Verification server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of GSM8K samples to run (0 means full test split)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per question (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )

    args = parser.parse_args()
    sample_limit = args.num_samples if args.num_samples > 0 else None

    run_benchmark(
        draft_model=args.draft_model,
        target_model=args.target_model,
        verification_server=args.server,
        num_samples=sample_limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
