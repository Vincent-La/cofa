import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging
import os
import multiprocessing as mp
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(input_file):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_max_var(data):
    vars_list = []
    for item in data:
        if "responses" in item:
            responses = item["responses"]
            lengths = [len(re.findall(r"\w+", r)) for r in responses]
            var = np.var(lengths)
            vars_list.append(var)
    max_var = max(vars_list) if vars_list else 1.0
    if max_var == 0:
        max_var = 1.0  # Avoid division by zero
    return max_var


def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding


def compute_score_for_item(item, max_var, tokenizer, model, device):
    if "correct_answer" not in item or "responses" not in item:
        logger.warning("Skipping item missing 'correct_answer' or 'responses'")
        return None

    correct_answer = item["correct_answer"]
    # Extract correct final answer
    if "#### " in correct_answer:
        correct_final = correct_answer.split("#### ")[-1].strip()
    else:
        logger.warning("Skipping item without '#### ' in correct_answer")
        return None

    responses = item["responses"]
    num_responses = len(responses)

    correct_emb = get_embedding(correct_answer, tokenizer, model, device)

    num_correct = 0
    dists = []
    for r in responses:
        # Check correctness
        boxed_match = re.search(r"\\boxed\{(.*?)\}", r, re.DOTALL)
        if boxed_match:
            pred = boxed_match.group(1).strip()
            if pred == correct_final:
                num_correct += 1
        else:
            if correct_final in r:
                num_correct += 1

        r_emb = get_embedding(r, tokenizer, model, device)

        if correct_emb.norm() > 0 and r_emb.norm() > 0:
            sim = torch.nn.functional.cosine_similarity(
                correct_emb, r_emb, dim=1
            ).item()
            dist = 1 - sim
        else:
            dist = 1.0
        dists.append(dist)

    avg_dist = np.mean(dists)

    # Variance of lengths (word count)
    lengths = [len(re.findall(r"\w+", r)) for r in responses]
    var = np.var(lengths)
    norm_var = var / max_var

    # Fraction incorrect
    frac_inc = (
        (num_responses - num_correct) / float(num_responses)
        if num_responses > 0
        else 0.0
    )

    # Final score
    score = (frac_inc + norm_var + avg_dist) / 3.0
    item["score"] = score
    return item


def worker_init():
    global tokenizer, model, device
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to(device)
    model.eval()


def process_wrapper(item_max_var):
    item, max_var = item_max_var
    return compute_score_for_item(item, max_var, tokenizer, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--visualize-distribution",
        action="store_true",
        help="Visualize the distribution of scores after processing.",
    )
    args = parser.parse_args()

    filename = "qwen_25_05b_responses.jsonl"
    input_file = f"responses/gsm8k_responses/{filename}"
    output_file = f"scores/gsm8k_scores/{filename.replace('responses', 'scores')}"

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading data from {input_file}...")
    data = load_data(input_file)
    logger.info(f"Loaded {len(data)} items.")

    logger.info("Computing max_var...")
    max_var = compute_max_var(data)
    logger.info(f"Max variance: {max_var}")

    # Load existing processed idxs for resuming
    existing_idxs = set()
    if os.path.exists(output_file):
        logger.info(
            f"Found existing output file: {output_file}. Loading processed idxs..."
        )
        with open(output_file, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "idx" in item:
                        existing_idxs.add(item["idx"])
                except json.JSONDecodeError:
                    pass
        logger.info(f"Found {len(existing_idxs)} previously processed items.")

    data_to_process = [
        item for item in data if "idx" not in item or item["idx"] not in existing_idxs
    ]
    logger.info(f"Items to process: {len(data_to_process)}")

    processed_items = []
    if data_to_process:
        logger.info(f"Processing items and saving incrementally to {output_file}...")
        if device == "cuda":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to(device)
            model.eval()

            with open(output_file, "a" if existing_idxs else "w") as f:
                for item in tqdm(data_to_process, desc="Processing items"):
                    processed_item = compute_score_for_item(
                        item, max_var, tokenizer, model, device
                    )
                    if processed_item:
                        json.dump(processed_item, f)
                        f.write("\n")
                        f.flush()
        else:
            num_processes = mp.cpu_count()
            logger.info(f"Using multiprocessing with {num_processes} processes on CPU.")
            with mp.Pool(processes=num_processes, initializer=worker_init) as pool:
                item_max_var_pairs = [(item, max_var) for item in data_to_process]
                processed_items = list(
                    tqdm(
                        pool.imap(process_wrapper, item_max_var_pairs),
                        total=len(data_to_process),
                        desc="Processing items",
                    )
                )

            with open(output_file, "a" if existing_idxs else "w") as f:
                for processed_item in processed_items:
                    if processed_item:
                        json.dump(processed_item, f)
                        f.write("\n")
                        f.flush()

    logger.info("Processing complete.")

    if args.visualize_distribution:
        logger.info("Visualizing score distribution...")
        scores = []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "score" in item:
                            scores.append(item["score"])
                    except json.JSONDecodeError:
                        pass

        if scores:
            figures_dir = "scores/gsm8k_scores/figures"
            os.makedirs(figures_dir, exist_ok=True)
            model_filename_safe = model_name.replace("/", "_")

            # Histogram with KDE
            plt.figure(figsize=(10, 6))
            sns.histplot(scores, kde=True)
            plt.title(f"Score Distribution (Histogram + KDE) for {model_name}")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.savefig(f"{figures_dir}/{model_filename_safe}_hist_kde.png")
            plt.close()

            # Boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(scores)
            plt.title(f"Score Boxplot for {model_name}")
            plt.xlabel("Score")
            plt.savefig(f"{figures_dir}/{model_filename_safe}_box.png")
            plt.close()

            # Violin plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(scores)
            plt.title(f"Score Violin Plot for {model_name}")
            plt.xlabel("Score")
            plt.savefig(f"{figures_dir}/{model_filename_safe}_violin.png")
            plt.close()

        else:
            logger.warning("No scores found to visualize.")
