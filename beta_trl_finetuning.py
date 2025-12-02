import re
import os
import json
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
from torch.distributions import Beta, Bernoulli
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import Dataset, DataLoader, BatchSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom Dataset for GSM8K
class GSM8KDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"You are a helpful math assistant. Solve the following problem step by step and put your final answer inside \\boxed{{}}.\n\nQuestion: {item['question']}\n\nStep-by-step reasoning:"
        correct_answer = item.get("correct_answer", item.get("answer", ""))
        return {
            "prompt": prompt,
            "labels": correct_answer,
        }


# BetaMixtureCurriculum class
class BetaMixtureCurriculum(nn.Module):
    def __init__(self, num_bins=100):
        super().__init__()
        self.alpha1_logit = nn.Parameter(torch.tensor(1.0))  # easy
        self.beta1_logit = nn.Parameter(torch.tensor(2.0))
        self.alpha2_logit = nn.Parameter(torch.tensor(2.0))  # hard
        self.beta2_logit = nn.Parameter(torch.tensor(1.0))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.num_bins = num_bins
        self.current_d = None
        self.current_components = None

    def sample_difficulty(self, batch_size: int):
        pi = torch.sigmoid(self.mix_logit)
        components = Bernoulli(pi).sample(
            (batch_size,)).long()  # [B], 1=easy, 0=hard
        alpha = torch.where(
            components == 1, self.alpha1_logit.exp(), self.alpha2_logit.exp()
        )
        beta = torch.where(
            components == 1, self.beta1_logit.exp(), self.beta2_logit.exp()
        )
        # Return components for log_prob
        return Beta(alpha, beta).rsample(), components

    def log_prob(
        self, difficulties: torch.Tensor, components: torch.Tensor
    ) -> torch.Tensor:
        pi = torch.sigmoid(self.mix_logit)
        log_p_c = Bernoulli(pi).log_prob(components.float())
        alpha = torch.where(
            components == 1, self.alpha1_logit.exp(), self.alpha2_logit.exp()
        )
        beta = torch.where(
            components == 1, self.beta1_logit.exp(), self.beta2_logit.exp()
        )
        log_p_d_given_c = Beta(alpha, beta).log_prob(difficulties)
        return log_p_c + log_p_d_given_c

    def get_params(self):
        return {
            "alpha1": self.alpha1_logit.exp().item(),
            "beta1": self.beta1_logit.exp().item(),
            "alpha2": self.alpha2_logit.exp().item(),
            "beta2": self.beta2_logit.exp().item(),
            "mix_prob": torch.sigmoid(self.mix_logit).item(),
        }


# Custom Batch Sampler
class BetaMixtureBatchSampler(BatchSampler):
    def __init__(self, beta_mixture, bin_lists, dataset_size, batch_size):
        self.beta_mixture = beta_mixture
        self.bin_lists = bin_lists
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            d, components = self.beta_mixture.sample_difficulty(
                self.batch_size)
            bin_edges = torch.linspace(
                0, 1, len(self.bin_lists) + 1, device=d.device)
            bin_idx = (
                torch.searchsorted(bin_edges, d.contiguous()) - 1
            )  # Make contiguous
            bin_idx = bin_idx.clamp(0, len(self.bin_lists) - 1)
            indices = []
            for i in tqdm(range(self.batch_size), desc="Sampling indices", leave=False):
                bin_list = self.bin_lists[bin_idx[i].item()]
                if len(bin_list) == 0:
                    idx = torch.randint(0, self.dataset_size, (1,)).item()
                else:
                    idx = bin_list[torch.randint(
                        len(bin_list), (1,)).item()].item()
                indices.append(idx)
            self.beta_mixture.current_d = d
            self.beta_mixture.current_components = components
            yield indices


# Precompute bins function
def precompute_bins(difficulties, num_bins=100):
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    bin_lists = [[] for _ in range(num_bins)]
    for idx, d in enumerate(difficulties):
        bin_idx = int(torch.searchsorted(bin_edges, d) - 1)
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        bin_lists[bin_idx].append(idx)
    return [torch.tensor(b) for b in bin_lists]


# Extract final answer function


def extract_final_answer(text: str) -> str:
    match = re.search(r"\\boxed\{(.*)\}", text, re.DOTALL)
    return match.group(1).strip() if match else text.split()[-1] if text.split() else ""


# Reward function
def reward_func(prompts, completions, completion_ids, **kwargs):
    labels = kwargs.get("labels", [])
    rewards = []
    for completion, label in zip(completions, labels):
        pred_ans = extract_final_answer(completion)
        gold_ans = extract_final_answer(label)
        rewards.append(1.0 if pred_ans == gold_ans else 0.0)
    return rewards


# Custom GRPO Trainer
class BMCGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, beta_mixture, bin_lists, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_mixture = beta_mixture
        self.bin_lists = bin_lists
        self.param_history = []
        self.eval_history = []
        self.loss_history = []
        self.current_scores = None

    def _generate_and_score_completions(self, generation_batch):
        generation_batch = super()._generate_and_score_completions(generation_batch)
        # FIX 1: Use "rewards"
        self.current_scores = torch.tensor(generation_batch["rewards"])
        return generation_batch

    def get_train_dataloader(self):
        batch_sampler = BetaMixtureBatchSampler(
            self.beta_mixture,
            self.bin_lists,
            len(self.train_dataset),
            self.args.per_device_train_batch_size,
        )

        # FIX 2: Simple passthrough collator
        def collate_fn(features):
            return features

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0
        )

    def create_optimizer(self):
        opt = super().create_optimizer()
        opt.add_param_group(
            {"params": self.beta_mixture.parameters(), "lr": 1e-3})
        return opt

    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None

        # Average rewards per prompt
        num_gen = self.args.num_generations
        mean_scores = self.current_scores.view(-1, num_gen).mean(dim=1)

        # Curriculum REINFORCE
        log_probs = self.beta_mixture.log_prob(
            self.beta_mixture.current_d, self.beta_mixture.current_components
        )
        loss_curr = -(log_probs * mean_scores.detach()).mean()
        total_loss = loss + 0.1 * loss_curr

        self.loss_history.append(
            {"grpo_loss": loss.item(), "curr_loss": loss_curr.item()}
        )

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def log(self, logs):
        super().log(logs)
        if self.state.global_step % 100 == 0:
            params = self.beta_mixture.get_params()
            self.param_history.append(
                {"step": self.state.global_step, **params})
            with open(
                os.path.join(self.args.output_dir, "param_history.json"), "w"
            ) as f:
                json.dump(self.param_history, f)
            with open(
                os.path.join(self.args.output_dir, "loss_history.json"), "w"
            ) as f:
                json.dump(self.loss_history, f)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.eval_history.append({"step": self.state.global_step, **metrics})
        with open(os.path.join(self.args.output_dir, "eval_history.json"), "w") as f:
            json.dump(self.eval_history, f)
        return metrics


# Main function
def main():
    output_dir = "finetuning/qwen_25_05b"
    os.makedirs(output_dir, exist_ok=True)

    # Load data from JSONL
    data = []
    with open("scores/gsm8k_scores/qwen_25_05b_scores.jsonl", "r") as f:
        for line in tqdm(f, desc="Loading data"):
            data.append(json.loads(line))
    difficulties = torch.tensor([item["score"] for item in data])
    bin_lists = precompute_bins(difficulties)

    # Load tokenizer and model with 4-bit quant for memory
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config
    )

    # LoRA config (low rank for 3080)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Datasets
    train_dataset = GSM8KDataset(data)
    eval_data = load_dataset("gsm8k", "main")["test"]
    eval_dataset = GSM8KDataset(eval_data)

    # Beta mixture
    beta_mixture = BetaMixtureCurriculum()

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Small for 3080
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        max_steps=5000,  # Adjust for budget
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        report_to="none",  # Or "tensorboard" if installed
        num_generations=4,
        fp16=True,
    )

    # Trainer
    trainer = BMCGRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        beta_mixture=beta_mixture,
        bin_lists=bin_lists,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
