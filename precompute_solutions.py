# generate_responses.py
import os
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
import argparse
import re
import pandas as pd
import numpy as np


def make_prompt(question: str) -> str:
    few_shot_cot_prompt = """Answer the question **step by step** and provide the final answer at the end. Put your final answer within $\\boxed{}$. Below is an example:
Question: BoatsRUs built 7 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatsRUs by the end of May of this year?
### Step1: To find the result of the total number of canoes built by BoatsRUs by the end of May, I need to find the number of canoes built in each month from January to May and then add them up.
### Step2: To find the number of canoes built in each month, I need to use the formula for the number of canoes built in a given month, which is the number of canoes built in the previous month times 2.
### Step3: So, the number of canoes built in January is 7, the number of canoes built in February is 7 times 2, which is 14, the number of canoes built in March is 14 times 2, which is 28, the number of canoes built in April is 28 times 2, which is 56, and the number of canoes built in May is 56 times 2, which is 112.
### Step4: Now, I can add up these numbers to get the total number of canoes built by BoatsRUs by the end of May: 7 plus 14 plus 28 plus 56 plus 112, which is 217.
### Final Answer: The answer is: $\\boxed{217}$.
Remember to answer the question **step by step**! 
Question:
"""
    return few_shot_cot_prompt + question


class BoxedStoppingCriteria(StoppingCriteria):
    def __init__(self, prompt_length, tokenizer):
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_length :]

        # If very little has been generated, do not stop
        if gen_ids.numel() <= 5:
            return False

        # Decode ONLY the generated portion
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        # Strip trailing spaces/newlines to avoid false positives
        gen_text = gen_text.rstrip()

        # Match ONLY if a full boxed expression occurs AT THE END
        # i.e., the model has finished its answer.
        if re.search(r"\$\\boxed\{[^}]+\}\$$", gen_text):
            return True

        return False


def main(args):
    if args.data == "gsm":
        parquet_path = "gsm_1k.parquet"
        output_dir = "./responses/gsm_responses"
        dataset_name = "gsm1k"
        correct_answer_key = "answer"
    elif args.data == "arc":
        parquet_path = "./data/arc/arc_1k_train.parquet"
        output_dir = "./responses/arc_responses"
        dataset_name = "arc1k"
        correct_answer_key = "reward_model"
    else:
        raise ValueError("Invalid data choice")

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset from local Parquet file
    df = pd.read_parquet(parquet_path)
    dataset = df.to_dict(orient="records")
    print(f"Loaded {len(dataset)} examples from {parquet_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model_name = args.model.split("/")[-1].lower().replace(".", "")
    filename = f"{model_name}_{dataset_name}_evaluations.json"
    output_file = os.path.join(output_dir, filename)

    # Check for existing file and find starting index for resumability
    start_idx = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    last_result = json.loads(last_line)
                    start_idx = last_result["idx"] + 1

    for idx in tqdm(range(start_idx, len(dataset)), desc="Generating responses"):
        item = dataset[idx]
        if args.data == "gsm":
            question = item["question"]
            prompt = make_prompt(question)
        else:
            # Handle prompt as numpy array
            prompt_array = item["prompt"]
            if isinstance(prompt_array, np.ndarray) and len(prompt_array) > 0:
                prompt_dict = prompt_array[0]
                if isinstance(prompt_dict, dict) and "content" in prompt_dict:
                    prompt = prompt_dict["content"]
                else:
                    raise ValueError(f"Unexpected prompt structure at index {idx}")
            else:
                raise ValueError(f"Unexpected prompt structure at index {idx}")
            question = None  # Not used for ARC

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        prompt_length = input_ids.shape[-1]
        stopping_criteria = StoppingCriteriaList(
            [BoxedStoppingCriteria(prompt_length, tokenizer)]
        )

        responses = []

        attention_mask = torch.ones_like(input_ids).to(args.device)

        for _ in range(args.num_samples):
            outputs = model.generate(
                input_ids.clone(),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                attention_mask=attention_mask,
                top_p=args.top_p,
                do_sample=True,
                eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_text[len(prompt) :].strip()
            responses.append(answer)

        result = {
            "idx": idx,
            "prompt": prompt,
            "correct_answer": item[correct_answer_key],
            "responses": responses,
        }
        if args.data == "gsm":
            result["question"] = question
        if args.data == "arc":
            result["extra_info"] = item.get("extra_info", None)

        # Append to JSON file continuously (as JSONL)
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"Finished! Saved responses to {output_file}")


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model name, e.g. Qwen/Qwen2.5-Math-7B-Instruct",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["gsm", "arc"],
        help="Dataset to use: 'gsm' or 'arc'",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Samples per question"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parsing()
    main(args)
