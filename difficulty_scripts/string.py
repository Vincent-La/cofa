import csv
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional


BASE_DIR = Path(__file__).resolve().parent


def normalize_answer(raw: Optional[str]) -> Optional[str]:
    """Normalize an answer string for comparison."""
    if raw is None:
        return None
    cleaned = raw.strip().strip("$").replace(",", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.rstrip(".")
    if cleaned == "":
        return None
    try:
        return str(Fraction(cleaned))
    except (ValueError, ZeroDivisionError):
        try:
            value = float(cleaned)
        except ValueError:
            return cleaned
        return str(int(value)) if value.is_integer() else str(value)


def normalize_text_for_search(text: str) -> str:
    """Light normalization for substring matching."""
    cleaned = text.lower()
    cleaned = cleaned.replace(",", "").replace("$", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def extract_correct_answer(correct_field: str) -> Optional[str]:
    """Pull the ground truth answer that follows '#### ' in the field."""
    if "####" not in correct_field:
        return None
    after_marker = correct_field.split("####", maxsplit=1)[1]
    return normalize_answer(after_marker)


def extract_boxed_answers(responses: Iterable[str]) -> list[Optional[str]]:
    """Find the value inside the last \\boxed{} block for each response."""
    results: list[Optional[str]] = []
    for response in responses:
        matches = re.findall(r"\\boxed\{([^}]*)\}", response)
        answer = normalize_answer(matches[-1]) if matches else None
        results.append(answer)
    return results


def response_contains_correct(response: str, correct: str) -> bool:
    """Check if the normalized correct answer appears in the response string."""
    normalized_response = normalize_text_for_search(response)
    escaped = re.escape(correct)
    # Avoid matching as part of a larger digit sequence when possible.
    pattern = rf"(?<!\d){escaped}(?!\d)"
    return re.search(pattern, normalized_response) is not None


def evaluate_file(path: Path) -> None:
    prefix = path.stem
    accuracies_path = BASE_DIR / f"{prefix}_accuracies.csv"
    summary_path = BASE_DIR / f"{prefix}_accuracy_summary.csv"

    accuracy_rows = []
    summary_counter: Counter[int] = Counter()

    with path.open() as infile:
        for line in infile:
            record = json.loads(line)
            idx = record.get("idx")
            question = record.get("question", "")
            correct = extract_correct_answer(record.get("correct_answer", ""))
            responses = record.get("responses", [])
            boxed_answers = extract_boxed_answers(responses)

            correct_count = 0
            for resp, boxed in zip(responses, boxed_answers):
                if correct is None:
                    continue
                if boxed is not None and boxed == correct:
                    correct_count += 1
                    continue
                if response_contains_correct(resp, correct):
                    correct_count += 1

            total = len(responses) if responses else 1
            accuracy = correct_count / total

            accuracy_rows.append(
                {"idx": idx, "accuracy": accuracy, "question": question}
            )
            summary_counter[correct_count] += 1

    with accuracies_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["idx", "accuracy", "question"])
        writer.writeheader()
        writer.writerows(accuracy_rows)

    with summary_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["correct_answers", "count"])
        writer.writeheader()
        for correct_answers in sorted(summary_counter):
            writer.writerow(
                {"correct_answers": correct_answers, "count": summary_counter[correct_answers]}
            )


def main() -> None:
    for jsonl_path in BASE_DIR.glob("*.jsonl"):
        evaluate_file(jsonl_path)


if __name__ == "__main__":
    main()
