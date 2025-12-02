import re
import os
import json
import torch
import logging
import torch.nn as nn
from torch.distributions import Beta, Bernoulli
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
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
# 1. Curriculum & Sampling Modules
# ------------------------------------------------------------------


class BetaMixtureCurriculum(nn.Module):
    def __init__(self, num_bins=100):
        super().__init__()
        # Initialize logits to reasonable starting distributions
        self.alpha1_logit = nn.Parameter(
            torch.tensor(0.0))  # Starts around 1.0
        self.beta1_logit = nn.Parameter(torch.tensor(1.0))  # Starts around 2.7
        self.alpha2_logit = nn.Parameter(torch.tensor(1.0))
        self.beta2_logit = nn.Parameter(torch.tensor(0.0))
        self.mix_logit = nn.Parameter(
            torch.tensor(0.0))    # 50/50 mix initially

        self.num_bins = num_bins
        # Buffers to store current step's state for loss calculation
        self.register_buffer('current_d', None)
        self.register_buffer('current_components', None)

    def sample_difficulty(self, batch_size: int):
        pi = torch.sigmoid(self.mix_logit)
        # 1 = easy, 0 = hard
        components = Bernoulli(pi).sample((batch_size,)).long()

        # Softplus or Exp to ensure positive alpha/beta
        alpha = torch.where(
            components == 1,
            torch.exp(self.alpha1_logit),
            torch.exp(self.alpha2_logit)
        )
        beta = torch.where(
            components == 1,
            torch.exp(self.beta1_logit),
            torch.exp(self.beta2_logit)
        )

        dist = Beta(alpha, beta)
        samples = dist.rsample()

        # Store for loss computation later
        self.current_d = samples.detach()
        self.current_components = components.detach()

        return samples, components

    def log_prob(self, difficulties: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        """Calculates log probability for REINFORCE / Policy Gradient on the curriculum."""
        pi = torch.sigmoid(self.mix_logit)

        # Log prob of choosing the component (easy vs hard)
        log_p_c = Bernoulli(pi).log_prob(components.float())

        alpha = torch.where(
            components == 1, torch.exp(
                self.alpha1_logit), torch.exp(self.alpha2_logit)
        )
        beta = torch.where(
            components == 1, torch.exp(
                self.beta1_logit), torch.exp(self.beta2_logit)
        )

        # Log prob of the difficulty score given component
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
        self.bin_lists = bin_lists  # List of tensors containing indices
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.device = next(beta_mixture.parameters()).device

    def __iter__(self):
        # We assume infinite training or one massive epoch for the curriculum
        # Usually easier to yield loop forever and let Trainer control max_steps
        num_batches = self.dataset_size // self.batch_size

        for _ in range(num_batches):
            # Sample target difficulties
            d, _ = self.beta_mixture.sample_difficulty(self.batch_size)
            d = d.to(self.device)

            # Map difficulties to bins
            bin_edges = torch.linspace(
                0, 1, len(self.bin_lists) + 1, device=self.device)
            bin_idx = torch.searchsorted(bin_edges, d.contiguous()) - 1
            bin_idx = bin_idx.clamp(0, len(self.bin_lists) - 1)

            indices = []
            # We must move bin_idx to CPU to access the list structure efficiently
            bin_idx_cpu = bin_idx.cpu()

            for i in range(self.batch_size):
                b_i = bin_idx_cpu[i].item()
                available_indices = self.bin_lists[b_i]

                if len(available_indices) == 0:
                    # Fallback if bin is empty: random sample
                    idx = torch.randint(0, self.dataset_size, (1,)).item()
                else:
                    # Sample random index from the specific bin
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

    # Simple CPU based binning
    for idx, score in enumerate(difficulties):
        # score is expected to be 0.0 to 1.0
        bin_idx = int(torch.searchsorted(bin_edges, score) - 1)
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        bin_lists[bin_idx].append(idx)

    return [torch.tensor(b) for b in bin_lists]

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------


SYSTEM_PROMPT = """You are a helpful math assistant. Solve the following problem step by step and put your final answer inside \\boxed{}."""


def extract_final_answer(text: str) -> str:
    if "\\boxed{" in text:
        # Extract content inside \boxed{}
        match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: look for the last number
    parts = text.split()
    return parts[-1] if parts else ""


def reward_func(prompts, completions, **kwargs):
    """
    TRL standard reward function signature:
    prompts: list[str]
    completions: list[str] (generated responses)
    kwargs: contains 'answer' or 'label' from dataset
    """
    answers = kwargs.get("answer")  # Access the ground truth column

    rewards = []
    for completion, answer in zip(completions, answers):
        pred_ans = extract_final_answer(completion)
        gold_ans = extract_final_answer(answer)

        # Exact match logic (could be softened)
        if pred_ans == gold_ans:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

# ------------------------------------------------------------------
# 3. Custom GRPOTrainer
# ------------------------------------------------------------------


class BMCGRPOTrainer(GRPOTrainer):
    def __init__(self, beta_mixture, bin_lists, **kwargs):
        super().__init__(**kwargs)
        self.beta_mixture = beta_mixture
        self.bin_lists = bin_lists

        # Metric storage
        self.param_history = []
        self.curriculum_loss_history = []

        # Buffer for rewards from the generation phase
        self.last_batch_rewards = None

        # Ensure mixture is on the right device
        if self.accelerator:
            self.beta_mixture = self.accelerator.prepare(self.beta_mixture)

    def create_optimizer(self):
        """
        We override this to ensure BetaMixture params are included in the optimizer.
        """
        # Get model parameters (handled by default logic usually, but we reconstruct here)
        decay_parameters = []
        no_decay_parameters = []

        # Model params
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if "bias" in n or "LayerNorm" in n:
                    no_decay_parameters.append(p)
                else:
                    decay_parameters.append(p)

        # Curriculum params (usually no weight decay)
        curriculum_params = [
            p for p in self.beta_mixture.parameters() if p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": decay_parameters,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
            {
                "params": curriculum_params,
                "weight_decay": 0.0,
                "lr": 1e-2,  # Usually needs higher LR than LLM
            },
        ]

        optimizer_cls, optimizer_kwargs = GRPOTrainer.get_optimizer_cls_and_kwargs(
            self.args)

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def get_train_dataloader(self) -> DataLoader:
        """
        Inject the BetaMixtureBatchSampler.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # We pass the raw dataset size to the sampler
        sampler = BetaMixtureBatchSampler(
            self.beta_mixture,
            self.bin_lists,
            len(self.train_dataset),
            self.args.per_device_train_batch_size,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,  # Use TRL's default collator
            num_workers=0,  # Keep 0 to avoid multiprocessing complexity with dynamic samplers
        )

    def _compute_rewards(self, prompts, completions, **kwargs):
        """
        Hook into reward computation to save scores for curriculum update.
        """
        # Call parent to get rewards
        rewards = super()._compute_rewards(prompts, completions, **kwargs)

        # Store rewards for the loss step (tensor shape: [batch_size, num_generations])
        # We detach to ensure no gradient flows back through the reward calc itself
        self.last_batch_rewards = rewards.detach()
        return rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Compute the standard GRPO Loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs)
            outputs = None

        # 2. Compute Curriculum Loss (REINFORCE)
        # We maximize the reward obtained by the curriculum's choices
        # Loss = - (log_prob(difficulty) * mean_reward)

        if self.last_batch_rewards is not None:
            # Calculate mean reward per prompt across generations
            # shape: [batch_size]
            mean_rewards = self.last_batch_rewards.mean(dim=1)

            # Recalculate log_prob for the *current* difficulties sampled in this step
            # Note: We rely on beta_mixture.current_d being set during sampling
            curr_d = self.beta_mixture.current_d
            curr_c = self.beta_mixture.current_components

            # Ensure dimensions match (DataLoader might drop last batch partials)
            if curr_d.shape[0] == mean_rewards.shape[0]:
                log_probs = self.beta_mixture.log_prob(curr_d, curr_c)

                # We want to increase prob of difficulties that yielded high rewards
                # aux_loss = - (log_probs * reward)
                curriculum_loss = -(log_probs * mean_rewards).mean()

                # Combine losses.
                # Note: 'loss' gradients update Model. 'curriculum_loss' updates BetaMixture.
                # They are disjoint graphs, so simple addition works.
                total_loss = loss + 0.1 * curriculum_loss

                # Logging
                self.curriculum_loss_history.append(curriculum_loss.item())
            else:
                total_loss = loss
        else:
            total_loss = loss

        # Periodically dump stats
        if self.state.global_step % 50 == 0:
            params = self.beta_mixture.get_params_dict()
            self.log(params)

        if return_outputs:
            return total_loss, outputs
        return total_loss


def main():
    output_dir = "finetuning/qwen_25_05b_bmc"

    # Assuming 'scores/gsm8k_scores/qwen_25_05b_scores.jsonl' exists
    # and has keys: "question", "answer", "score" (float 0-1)

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

    # Precompute Bins for sampling
    difficulties = torch.tensor(scores, dtype=torch.float32)
    bin_lists = precompute_bins(difficulties)

    # Create HF Dataset
    # TRL expects 'prompt' or 'messages'. For GRPO, simple keys are often best.
    train_dataset = HFDataset.from_dict({
        "prompt": [
            f"{SYSTEM_PROMPT}\n\nQuestion: {q}\n\nStep-by-step reasoning:"
            for q in questions
        ],
        "answer": answers,
        # We don't pass score here, the sampler handles it via indices
    })

    # --- Model & Config ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # FIX: Align compute dtype with Trainer's bf16=True
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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

    # GRPO Config
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=500,  # Short run for demo
        logging_steps=10,
        save_steps=100,
        fp16=True,
        num_generations=4,  # How many samples per prompt for GRPO
        max_completion_length=256,
        report_to="none",  # or "wandb"
        use_vllm=False,
    )

    # --- Initialization ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # FIX: Prepare model for k-bit training (casts layers to fp32)
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize Mixture
    beta_mixture = BetaMixtureCurriculum()

    # Initialize Trainer
    trainer = BMCGRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        beta_mixture=beta_mixture,
        bin_lists=bin_lists,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final adapters
    trainer.save_model(output_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
