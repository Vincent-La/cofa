import re
import os
import json
import torch
import logging
import sys
import numpy as np
from typing import Dict, Any, List
import torch.nn as nn
from torch.distributions import Beta, Bernoulli
from datasets import Dataset as HFDataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import DataLoader, BatchSampler
from tqdm.auto import tqdm

# Setup standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. System Prompt & Reward Logic
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


class RewardCapturer:
    def __init__(self):
        self.last_rewards = None
        # FIX: TRL expects the callable to have a __name__ property
        self.__name__ = "curriculum_reward_func"

    def __call__(self, prompts, completions, **kwargs):
        answers = kwargs.get("answer")
        rewards = []

        for completion, answer in zip(completions, answers):
            if "####" in answer:
                gold_str = answer.split("####")[-1].strip()
            else:
                gold_str = answer.strip()

            gold_val = extract_numeric_value(gold_str)
            boxed_pred = extract_boxed_content(completion)

            if boxed_pred and extract_numeric_value(boxed_pred) == gold_val:
                rewards.append(1.0)
            elif gold_val in completion.split():
                rewards.append(0.8)
            elif gold_val in completion:
                rewards.append(0.8)
            else:
                rewards.append(0.0)

        # Capture for Curriculum
        self.last_rewards = torch.tensor(rewards, dtype=torch.float32)
        return rewards


# ------------------------------------------------------------------
# 2. Callback
# ------------------------------------------------------------------


class BetaTrackingCallback(TrainerCallback):
    def __init__(self, log_file_path, beta_mixture):
        self.log_file_path = log_file_path
        self.beta_mixture = beta_mixture
        self.trainer = None

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(self.log_file_path, "w") as f:
            f.write("")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 1. Check Link
        if not hasattr(self.beta_mixture, "get_params_dict"):
            return

        params = self.beta_mixture.get_params_dict()
        stats = {}
        if self.trainer and hasattr(self.trainer, "latest_step_stats"):
            stats = self.trainer.latest_step_stats

        log_entry = {"step": state.global_step, **params, **stats}
        with open(self.log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Force print to console
        if state.global_step % args.logging_steps == 0:
            print(
                f"   >> Beta Stats | Mix: {params['mix_prob']:.2f} | "
                f"Alpha1: {params['alpha1']:.2f} | Beta1: {params['beta1']:.2f} | "
                f"AvgRew: {stats.get('avg_reward', 0.0):.3f}"
            )


# ------------------------------------------------------------------
# 3. Curriculum Modules
# ------------------------------------------------------------------


class BetaMixtureCurriculum(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha1_logit = nn.Parameter(torch.tensor(0.0))
        self.beta1_logit = nn.Parameter(torch.tensor(1.0))
        self.alpha2_logit = nn.Parameter(torch.tensor(1.0))
        self.beta2_logit = nn.Parameter(torch.tensor(0.0))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("current_d", None)
        self.register_buffer("current_components", None)
        self.last_sampled_difficulties_cpu = []

    def sample_difficulty(self, batch_size: int):
        pi = torch.sigmoid(self.mix_logit)
        components = Bernoulli(pi).sample((batch_size,)).long()
        alpha = torch.where(
            components == 1, torch.exp(self.alpha1_logit), torch.exp(self.alpha2_logit)
        )
        beta = torch.where(
            components == 1, torch.exp(self.beta1_logit), torch.exp(self.beta2_logit)
        )

        dist = Beta(alpha, beta)
        samples = dist.rsample()

        self.current_d = samples.detach()
        self.current_components = components.detach()
        self.last_sampled_difficulties_cpu = samples.detach().cpu().tolist()
        return samples, components

    def log_prob(
        self, difficulties: torch.Tensor, components: torch.Tensor
    ) -> torch.Tensor:
        pi = torch.sigmoid(self.mix_logit)
        log_p_c = Bernoulli(pi).log_prob(components.float())
        alpha = torch.where(
            components == 1, torch.exp(self.alpha1_logit), torch.exp(self.alpha2_logit)
        )
        beta = torch.where(
            components == 1, torch.exp(self.beta1_logit), torch.exp(self.beta2_logit)
        )
        log_p_d_given_c = Beta(alpha, beta).log_prob(difficulties)
        return log_p_c + log_p_d_given_c

    def get_params_dict(self):
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
            bin_edges = torch.linspace(
                0, 1, len(self.bin_lists) + 1, device=self.device
            )
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
                    idx = available_indices[
                        torch.randint(len(available_indices), (1,)).item()
                    ].item()
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


class BMCGRPOTrainer(GRPOTrainer):
    def __init__(self, beta_mixture, bin_lists, reward_capturer, **kwargs):
        super().__init__(**kwargs)
        self.beta_mixture = beta_mixture
        self.bin_lists = bin_lists
        self.reward_capturer = reward_capturer
        self.latest_step_stats = {}
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
            p for p in self.beta_mixture.parameters() if p.requires_grad
        ]

        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
            {"params": curriculum_params, "weight_decay": 0.0, "lr": 1e-2},
        ]

        optimizer_cls, optimizer_kwargs = GRPOTrainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs)
            outputs = None

        curriculum_loss_val = 0.0
        avg_reward = 0.0

        captured_rewards = self.reward_capturer.last_rewards

        if captured_rewards is not None:
            captured_rewards = captured_rewards.to(loss.device)
            avg_reward = captured_rewards.mean().item()

            curr_d = self.beta_mixture.current_d.to(loss.device)
            curr_c = self.beta_mixture.current_components.to(loss.device)

            log_probs = self.beta_mixture.log_prob(curr_d, curr_c)
            curriculum_loss = -(log_probs.mean() * avg_reward)

            total_loss = loss + 0.1 * curriculum_loss
            curriculum_loss_val = curriculum_loss.item()
        else:
            total_loss = loss

        self.latest_step_stats = {
            "total_loss": loss.item(),
            "curriculum_loss": curriculum_loss_val,
            "avg_reward": avg_reward,
            "sampled_difficulties": self.beta_mixture.last_sampled_difficulties_cpu,
        }

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------


def main():
    output_dir = "finetuning/qwen_25_05b_bmc_fixed_final_v2"
    log_file = os.path.join(output_dir, "logs", "active_training_log.jsonl")

    # --- Load Data ---
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
            questions.append(item["question"])
            ans = item.get("correct_answer", item.get("answer", ""))
            answers.append(ans)
            scores.append(item.get("score", 0.5))

    difficulties = torch.tensor(scores, dtype=torch.float32)
    bin_lists = precompute_bins(difficulties)

    # --- Config ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

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

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=4,  # Your setting
        gradient_accumulation_steps=4,  # Your setting
        max_steps=1000,
        logging_steps=5,
        save_steps=100,
        fp16=True,  # Your setting
        num_generations=4,
        max_completion_length=300,
        report_to="none",
        use_vllm=False,
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=4,
    )

    # SAFETY CHECK: Prevent TRL Crash
    if training_args.per_device_train_batch_size % training_args.num_generations != 0:
        raise ValueError(
            f"CRITICAL CONFIG ERROR: Batch Size ({training_args.per_device_train_batch_size}) must be divisible by Num Generations ({training_args.num_generations})"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    beta_mixture = BetaMixtureCurriculum()
    reward_spy = RewardCapturer()
    beta_callback = BetaTrackingCallback(log_file, beta_mixture)

    trainer = BMCGRPOTrainer(
        model=model,
        reward_funcs=[reward_spy],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        beta_mixture=beta_mixture,
        bin_lists=bin_lists,
        reward_capturer=reward_spy,
        processing_class=tokenizer,
        callbacks=[beta_callback],
    )

    beta_callback.trainer = trainer

    logger.info("Starting training with Goldilocks Config (Batch=2, Gens=2)...")
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
