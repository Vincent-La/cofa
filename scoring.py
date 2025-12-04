import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import evaluate
from tqdm import tqdm
import logging
import os
import multiprocessing as mp
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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


def compute_item_perplexity(metric, item, model_name, device):
    
    
    result = metric.compute(model_id=model_name, predictions = item['responses'], device = device)
    
    # perplexity averaged across rollouts
    item['score'] = float(result['mean_perplexity'])
    # print(item["score"])
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--visualize-distribution",
        action="store_true",
        help="Visualize the distribution of scores after processing.",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="Scoring method"
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help = "Normalize computed score to [0,1] by dividing by max(score)"
    )

    args = parser.parse_args()

    if args.method == "perplexity":
        score_func = compute_item_perplexity
    else:
        raise NotImplementedError(f"{args.method} not implemented")
    

    # filename = "qwen_25_05b_responses.jsonl"
    filename = "qwen25-05b-instruct_arc1k_evaluations.json"
    input_file = f"./difficulty_scripts/{filename}"
    raw_output_file = f"./scores/arc_scores/{args.method}/qwen_scores.jsonl"
    output_file = f"./scores/arc_scores/{args.method}/qwen_scores_normalized.jsonl"

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading data from {input_file}...")
    data = load_data(input_file)
    logger.info(f"Loaded {len(data)} items.")


    if device == "cuda":
        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        # model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to(device)
        # model.eval()

        # with open(output_file, "a" if existing_idxs else "w") as f:
        gather = []
        perplexity = evaluate.load("perplexity", module_type="metric")
        for item in tqdm(data, desc="Processing items"):
            processed_item = score_func(
                perplexity, item, model_name, device
            )

            gather.append(processed_item)

    df = pd.DataFrame(gather)
    df.to_json(raw_output_file, orient = 'records', lines=True)

    if args.normalize:
        df["score"] = df["score"] / df.score.max()

    df.to_json(output_file, orient = 'records', lines=True)

    logger.info("Processing complete.")

    if args.visualize_distribution:
        logger.info("Visualizing score distribution...")
        scores = []\
        
        print(os.path.exists(output_file))
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
            figures_dir = f"scores/arc_scores/{args.method}/figures"
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
