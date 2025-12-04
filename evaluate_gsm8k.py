import torch
import re
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Must match the prompt used in training exactly for fair comparison
SYSTEM_PROMPT = """You are a helpful math assistant. Solve the following problem step by step. You must put your final answer inside \\boxed{}.

Here are examples of the required format:

Question: In Professor Plum's biology class there are 40 students. Of those students, 80 percent have puppies. Of those who have puppies, 25% also have parrots. How many students have both puppies and parrots?
Step-by-step reasoning:
We start with the initial numbers of students, 40 and multiply that by .8 for 40 * 0.8 = <<40*0.8=32>>32 who own puppies.
That the number of students with puppies, 32, and multiply that by .25 to find out how many own both puppies and parrots, 32 * 0.25 = <<32*0.25=8>>8 who own puppies and parrots.
The answer is <<8=8>>8.
\\boxed{8}

Question: A baker has 10 cakes. He sells 4 of them. Then he bakes 12 more. How many does he have now?
Step-by-step reasoning:
The baker starts with 10 cakes and sells 4, so 10 - 4 = <<10-4=6>>6 cakes left.
Then he bakes 12 more, so 6 + 12 = <<6+12=18>>18 cakes.
The answer is <<18=18>>18.
\\boxed{18}"""

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Note: adapter is hard-coded for now.
ADAPTER_PATH = "finetuning/qwen_25_05b_bmc_fixed_final_v2"


def extract_numeric_value(text: str) -> str:
    """Extracts the last number found in text."""
    # Matches integers, decimals, and simple fractions, ignoring commas
    matches = re.findall(r"[-+]?\d[\d,]*\.?\d?", text)
    if matches:
        return matches[-1].replace(",", "")
    return ""


def extract_boxed_content(text: str) -> str:
    """Extracts content inside \\boxed{...}."""
    if "\\boxed{" in text:
        match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def check_correctness(completion, ground_truth):
    """
    Checks if the completion matches the ground truth.
    Prioritizes \\boxed{} content, falls back to loose matching.
    """
    if "####" in ground_truth:
        gold_val = extract_numeric_value(ground_truth.split("####")[-1])
    else:
        gold_val = extract_numeric_value(ground_truth)

    # 1. Try Boxed Extraction
    boxed_pred = extract_boxed_content(completion)
    if boxed_pred and extract_numeric_value(boxed_pred) == gold_val:
        return True, "boxed_match"

    # 2. Try Loose Extraction (Last number in text)
    loose_pred = extract_numeric_value(completion)
    if loose_pred == gold_val:
        return True, "loose_match"

    return False, "mismatch"


def run_evaluation(model, tokenizer, dataset, desc="Evaluating"):
    model.eval()
    correct_count = 0
    total = len(dataset)

    results = []

    # Iterate with progress bar
    for i, item in tqdm(enumerate(dataset), total=total, desc=desc):
        question = item["question"]
        ground_truth = item["answer"]

        # Prepare Prompt (Same as training)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding for consistent eval
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        completion = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Check Answer
        is_correct, reason = check_correctness(completion, ground_truth)
        if is_correct:
            correct_count += 1

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "generated": completion,
                "correct": is_correct,
                "match_type": reason,
            }
        )

    accuracy = correct_count / total
    print(f"\n{desc} Accuracy: {accuracy:.2%}")
    return accuracy, results


def main():
    # Load Dataset (Test Split)
    # Using a small subset (e.g. 100) for speed, or remove .select() for full run
    print("Loading GSM8K Test Set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(50))  # Comment this out for full evaluation!

    # 1. Setup Base Model
    print(f"Loading Base Model: {BASE_MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )

    # 2. Evaluate Base Model
    print("\n--- Running Baseline Evaluation ---")
    base_acc, base_results = run_evaluation(
        model, tokenizer, dataset, desc="Base Model"
    )

    # 3. Load Adapter (Fine-Tuned)
    print(f"\nLoading Adapter from: {ADAPTER_PATH}")
    # This merges the LoRA weights onto the base model efficiently
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # 4. Evaluate Fine-Tuned Model
    print("\n--- Running Fine-Tuned Evaluation ---")
    ft_acc, ft_results = run_evaluation(
        model, tokenizer, dataset, desc="Fine-Tuned Model"
    )

    # 5. Summary
    print("\n" + "=" * 30)
    print("FINAL RESULTS")
    print("=" * 30)
    print(f"Base Model Accuracy:      {base_acc:.2%}")
    print(f"Fine-Tuned Model Accuracy: {ft_acc:.2%}")
    print(f"Improvement:              {ft_acc - base_acc:+.2%}")
    print("=" * 30)

    # Save detailed logs
    with open("eval_results_finetuned.json", "w") as f:
        json.dump(ft_results, f, indent=2)


if __name__ == "__main__":
    main()
