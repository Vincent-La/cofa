import re
import os
import json
import logging
import sys
import torch
from datasets import Dataset as HFDataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

# Setup standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. System Prompt & Helpers
# ------------------------------------------------------------------

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


def extract_numeric_value(text: str) -> str:
    matches = re.findall(r"[-+]?\d[\d,]*\.?\d?", text)
    if matches:
        return matches[-1].replace(",", "")
    return ""


def extract_boxed_content(text: str) -> str:
    if "\\boxed{" in text:
        match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


# ------------------------------------------------------------------
# 2. Reward Function
# ------------------------------------------------------------------


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Standard stateless reward function for Vanilla GRPO.
    Checks if the completion matches the gold answer.
    """
    rewards = []

    for completion, gold_answer in zip(completions, answer):
        if "####" in gold_answer:
            gold_str = gold_answer.split("####")[-1].strip()
        else:
            gold_str = gold_answer.strip()

        gold_val = extract_numeric_value(gold_str)
        boxed_pred = extract_boxed_content(completion)

        if boxed_pred and extract_numeric_value(boxed_pred) == gold_val:
            rewards.append(1.0)
        elif gold_val in completion.split():
            # Partial credit for finding number but missing box format
            rewards.append(0.8)
        elif gold_val in completion:
            rewards.append(0.8)
        else:
            rewards.append(0.0)

    return rewards


# ------------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------------


def main():
    # UPDATED: Distinct output directory for the baseline
    output_dir = "finetuning/qwen_25_05b_vanilla_grpo_baseline"

    # --- Load Data ---
    # We load the exact same file to ensure the dataset is identical,
    # but we ignore the 'score' field since this is vanilla training.
    questions = []
    answers = []

    data_path = "scores/gsm8k_scores/qwen_25_05b_scores.jsonl"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return

    logger.info("Loading data...")
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
            ans = item.get("correct_answer", item.get("answer", ""))
            answers.append(ans)
            # Scores are ignored here

    # --- Config ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Format Prompts
    formatted_prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}"},
        ]
        full_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(full_txt)

    train_dataset = HFDataset.from_dict(
        {
            "prompt": formatted_prompts,
            "answer": answers,
        }
    )

    # Eval Data (Same as reference script)
    eval_data_raw = load_dataset("gsm8k", "main", split="test")
    eval_prompts = []
    for q in eval_data_raw["question"]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}"},
        ]
        full_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        eval_prompts.append(full_txt)

    eval_dataset = HFDataset.from_dict(
        {"prompt": eval_prompts, "answer": eval_data_raw["answer"]}
    )
    eval_dataset = eval_dataset.select(range(50))

    # Model Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPO Config (Kept identical to reference)
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=1000,
        logging_steps=5,
        save_steps=100,
        fp16=True,
        num_generations=4,
        max_completion_length=300,
        report_to="none",
        use_vllm=False,
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=4,
    )

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # Standard GRPO Trainer
    # We do NOT pass beta_mixture, bin_lists, or custom callbacks
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("Starting VANILLA GRPO training...")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Vanilla training finished. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
