import re
import os
import json
import torch
import logging
import csv
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch.nn as nn
from torch.distributions import Beta, Bernoulli
from datasets import Dataset as HFDataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import DataLoader, BatchSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. Logging & Tracking Infrastructure
# ------------------------------------------------------------------


class TrainingLogger:
    """
    Centralized logger to track curriculum stats, params, and sampled difficulties.
    Writes to a JSONL file for easy parsing/plotting later.
    """

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_stats.jsonl")
        # clear previous log
        with open(self.log_file, 'w') as f:
            pass

    def log_step(self, step_data: Dict[str, Any]):
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(step_data) + "\n")


# Global logger instance (initialized in main)
experiment_tracker = None

# ------------------------------------------------------------------
# 2. Curriculum & Sampling Modules
# ------------------------------------------------------------------


class BetaMixtureCurriculum(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize logits
        # Component 1: Ideally "Easy" (High Alpha, Low Beta -> Skewed right towards 1.0)
        # Component 2: Ideally "Hard" (Low Alpha, High Beta -> Skewed left towards 0.0) or Uniform
        self.alpha1_logit = nn.Parameter(torch.tensor(0.0))
        self.beta1_logit = nn.Parameter(torch.tensor(1.0))
        self.alpha2_logit = nn.Parameter(torch.tensor(1.0))
        self.beta2_logit = nn.Parameter(torch.tensor(0.0))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        # Buffers for current state (for loss calc)
        self.register_buffer('current_d', None)
        self.register_buffer('current_components', None)

        # Buffer for visualization tracking
        # We store the last sampled batch's difficulties here so the Trainer can log them
        self.last_sampled_difficulties_cpu = []

    def sample_difficulty(self, batch_size: int):
        pi = torch.sigmoid(self.mix_logit)
        components = Bernoulli(pi).sample((batch_size,)).long()

        alpha = torch.where(components == 1, torch.exp(
            self.alpha1_logit), torch.exp(self.alpha2_logit))
        beta = torch.where(components == 1, torch.exp(
            self.beta1_logit), torch.exp(self.beta2_logit))

        dist = Beta(alpha, beta)
        samples = dist.rsample()

        # Save state for Loss
        self.current_d = samples.detach()
        self.current_components = components.detach()

        # Save state for Logging (moved to CPU list)
        self.last_sampled_difficulties_cpu = samples.detach().cpu().tolist()

        return samples, components

    def log_prob(self, difficulties: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        pi = torch.sigmoid(self.mix_logit)
        log_p_c = Bernoulli(pi).log_prob(components.float())

        alpha = torch.where(components == 1, torch.exp(
            self.alpha1_logit), torch.exp(self.alpha2_logit))
        beta = torch.where(components == 1, torch.exp(
            self.beta1_logit), torch.exp(self.beta2_logit))

        log_p_d_given_c = Beta(alpha, beta).log_prob(difficulties)
        return log_p_c + log_p_d_given_c

    def get_params_dict(self):
        """Returns current parameters for logging."""
        return {
            "alpha1": torch.exp(self.alpha1_logit).item(),
            "beta1": torch.exp(self.beta1_logit).item(),
            "alpha2": torch.exp(self.alpha2_logit).item(),
            "beta2": torch.exp(self.beta2_logit).item(),
            "mix_prob": torch.sigmoid(self.mix_logit).item(),
        }


class BetaMixtureBatchSampler(BatchSampler):
    def __init__(self, beta_mixture, bin_lists, dataset_size, batch_size):
        self.beta_mixture = beta_mixture
        self.bin_lists = bin_lists
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.device = next(beta_mixture.parameters()).device

    def __iter__(self):
        num_batches = self.dataset_size // self.batch_size
        for _ in range(num_batches):
            d, _ = self.beta_mixture.sample_difficulty(self.batch_size)
            d = d.to(self.device)

            # Map sampled 'target difficulty' to actual data bins
            bin_edges = torch.linspace(
                0, 1, len(self.bin_lists) + 1, device=self.device)
            bin_idx = torch.searchsorted(bin_edges, d.contiguous()) - 1
            bin_idx = bin_idx.clamp(0, len(self.bin_lists) - 1)

            indices = []
            bin_idx_cpu = bin_idx.cpu()

            for i in range(self.batch_size):
                b_i = bin_idx_cpu[i].item()
                available_indices = self.bin_lists[b_i]

                if len(available_indices) == 0:
                    idx = torch.randint(0, self.dataset_size, (1,)).item()
                else:
                    selection = torch.randint(
                        len(available_indices), (1,)).item()
                    idx = available_indices[selection].item()
                indices.append(idx)

            yield indices

    def __len__(self):
        return self.dataset_size // self.batch_size


def precompute_bins(difficulties, num_bins=100):
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    bin_lists = [[] for _ in range(num_bins)]
    for idx, score in enumerate(difficulties):
        bin_idx = int(torch.searchsorted(bin_edges, score) - 1)
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        bin_lists[bin_idx].append(idx)
    return [torch.tensor(b) for b in bin_lists]

# ------------------------------------------------------------------
# 3. Helper Functions & Reward
# ------------------------------------------------------------------


SYSTEM_PROMPT = """You are a helpful math assistant. Solve the following problem step by step and put your final answer inside \\boxed{}."""


def extract_final_answer(text: str) -> str:
    if "\\boxed{" in text:
        match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    parts = text.split()
    return parts[-1] if parts else ""


def reward_func(prompts, completions, **kwargs):
    answers = kwargs.get("answer")
    rewards = []
    for completion, answer in zip(completions, answers):
        pred_ans = extract_final_answer(completion)
        gold_ans = extract_final_answer(answer)
        # Exact match 1.0 or 0.0
        rewards.append(1.0 if pred_ans == gold_ans else 0.0)
    return rewards

# ------------------------------------------------------------------
# 4. Custom GRPOTrainer with Deep Logging
# ------------------------------------------------------------------


class BMCGRPOTrainer(GRPOTrainer):
    def __init__(self, beta_mixture, bin_lists, **kwargs):
        super().__init__(**kwargs)
        self.beta_mixture = beta_mixture
        self.bin_lists = bin_lists
        self.last_batch_rewards = None

        if self.accelerator:
            self.beta_mixture = self.accelerator.prepare(self.beta_mixture)

    def create_optimizer(self):
        decay_parameters = []
        no_decay_parameters = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if "bias" in n or "LayerNorm" in n:
                    no_decay_parameters.append(p)
                else:
                    decay_parameters.append(p)

        curriculum_params = [
            p for p in self.beta_mixture.parameters() if p.requires_grad]

        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
            {"params": curriculum_params, "weight_decay": 0.0, "lr": 1e-2},
        ]

        optimizer_cls, optimizer_kwargs = GRPOTrainer.get_optimizer_cls_and_kwargs(
            self.args)
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sampler = BetaMixtureBatchSampler(
            self.beta_mixture,
            self.bin_lists,
            len(self.train_dataset),
            self.args.per_device_train_batch_size,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
        )

    def _compute_rewards(self, prompts, completions, **kwargs):
        rewards = super()._compute_rewards(prompts, completions, **kwargs)
        self.last_batch_rewards = rewards.detach()
        return rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs)
            outputs = None

        curriculum_loss_val = 0.0

        # --- Curriculum Update Logic ---
        if self.last_batch_rewards is not None:
            mean_rewards = self.last_batch_rewards.mean(dim=1)
            curr_d = self.beta_mixture.current_d
            curr_c = self.beta_mixture.current_components

            if curr_d.shape[0] == mean_rewards.shape[0]:
                log_probs = self.beta_mixture.log_prob(curr_d, curr_c)
                curriculum_loss = -(log_probs * mean_rewards).mean()
                total_loss = loss + 0.1 * curriculum_loss
                curriculum_loss_val = curriculum_loss.item()
            else:
                total_loss = loss
        else:
            total_loss = loss

        # --- Deep Logging ---
        # We log every step to the JSONL file
        if experiment_tracker is not None and self.state.global_step > 0:
            stats = {
                "step": self.state.global_step,
                "total_loss": loss.item(),
                "curriculum_loss": curriculum_loss_val,
                "avg_reward": mean_rewards.mean().item() if self.last_batch_rewards is not None else 0.0,
                # Sampled difficulties this batch (list of floats)
                "sampled_difficulties": self.beta_mixture.last_sampled_difficulties_cpu,
                # Current Beta Params
                **self.beta_mixture.get_params_dict()
            }
            experiment_tracker.log_step(stats)

        if return_outputs:
            return total_loss, outputs
        return total_loss

# ------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------


def main():
    global experiment_tracker

    output_dir = "finetuning/qwen_25_05b_bmc_v2"
    log_dir = os.path.join(output_dir, "logs")
    experiment_tracker = TrainingLogger(log_dir)

    # --- Data Loading ---
    # 1. Load Scored Training Data
    questions = []
    answers = []
    scores = []

    data_path = "scores/gsm8k_scores/qwen_25_05b_scores.jsonl"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return

    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item['question'])
            # Normalize answer key
            ans = item.get('correct_answer', item.get('answer', ''))
            answers.append(ans)
            scores.append(item.get('score', 0.5))

    difficulties = torch.tensor(scores, dtype=torch.float32)
    bin_lists = precompute_bins(difficulties)

    train_dataset = HFDataset.from_dict({
        "prompt": [
            f"{SYSTEM_PROMPT}\n\nQuestion: {q}\n\nStep-by-step reasoning:"
            for q in questions
        ],
        "answer": answers,
    })

    # 2. Load Evaluation Data (Held-out GSM8K Test)
    # Using 'main' config, 'test' split
    eval_data_raw = load_dataset("gsm8k", "main", split="test")

    # Format eval data to match training structure
    eval_dataset = HFDataset.from_dict({
        "prompt": [
            f"{SYSTEM_PROMPT}\n\nQuestion: {q}\n\nStep-by-step reasoning:"
            for q in eval_data_raw['question']
        ],
        "answer": eval_data_raw['answer']
    })
    # Optional: subset for speed during debugging
    # eval_dataset = eval_dataset.select(range(100))

    # --- Model & Config ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=1000,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        num_generations=4,
        max_completion_length=256,
        report_to="none",
        use_vllm=False,
        # Evaluation Settings
        eval_strategy="steps",  # Evaluate periodically
        eval_steps=100,        # Every 100 steps
        per_device_eval_batch_size=4,
        eval_on_start=True,    # Baseline check
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    beta_mixture = BetaMixtureCurriculum()

    trainer = BMCGRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Pass held-out set here
        peft_config=lora_config,
        beta_mixture=beta_mixture,
        bin_lists=bin_lists,
        processing_class=tokenizer,
    )

    logger.info("Starting training with full tracking...")
    trainer.train()

    trainer.save_model(output_dir)
    logger.info(
        f"Training complete. Stats saved to {log_dir}/training_stats.jsonl")


if __name__ == "__main__":
    main()
