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
        # if "####" in gold_answer:
        #     gold_str = gold_answer.split("####")[-1].strip()
        # else:
        #     gold_str = gold_answer.strip()

        # gold_val = extract_numeric_value(gold_str)
        boxed_pred = extract_boxed_content(completion)

        # wrong formatting
        if boxed_pred is None:
            rewards.append(0.0)
        # correct answer
        elif boxed_pred == gold_answer:
            rewards.append(1.0)
        # parital formatting score
        else:
            rewards.append(0.1)

        # if boxed_pred and extract_numeric_value(boxed_pred) == gold_val:
        #     rewards.append(1.0)
        # elif gold_val in completion.split():
        #     # Partial credit for finding number but missing box format
        #     rewards.append(0.8)
        # elif gold_val in completion:
        #     rewards.append(0.8)
        # else:
        #     rewards.append(0.0)


    return rewards


# ------------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------------


def main():
    # UPDATED: Distinct output directory for the baseline
    output_dir = "finetuning/qwen_25_05b_arc_vanilla_grpo_baseline"

    # --- Load Data ---
    # We load the exact same file to ensure the dataset is identical,
    # but we ignore the 'score' field since this is vanilla training.
    questions = []
    answers = []

    # data_path = "scores/gsm8k_scores/basic/qwen_scores.jsonl"
    data_path = "scores/arc_scores/basic/qwen_scores.jsonl"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return

    logger.info("Loading data...")
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["prompt"])
            ans = item["correct_answer"]["ground_truth"]["target"]
            answers.append(ans)
            # Scores are ignored here

    # --- Config ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # NOTE: arc prompts already stored as formatted
    # # Format Prompts
    # formatted_prompts = []
    # for q in questions:
    #     messages = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": f"Question: {q}"},
    #     ]
    #     full_txt = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     formatted_prompts.append(full_txt)

    train_dataset = HFDataset.from_dict(
        {
            "prompt": questions,
            "answer": answers,
        }
    )


    test_questions = []
    test_answers = []

    data_path = "data/arc/test.jsonl"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return

    logger.info("Loading data...")
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            test_questions.append(item["prompt"][0]["content"])
            ans = item["reward_model"]["ground_truth"]["target"]
            test_answers.append(ans)


    eval_dataset = HFDataset.from_dict(
        {"prompt": test_questions, "answer":test_answers}
    )
    # eval_dataset = eval_dataset.select(range(50))


    # print(f"SAMPLE PROMPT:\n{eval_dataset['prompt'][0]}")
    # print('\n---' * 5)
    # print(f"SAMPLE ANSWER:\n{eval_dataset['answer'][0]}")
    # import sys
    # sys.exit(0)

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
        max_completion_length=600,
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
