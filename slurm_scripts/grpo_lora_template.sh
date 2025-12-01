#!/bin/bash

project_name=$1
experiment_name=$2
train_path=$3
test_path=$4
model_dir=$5

# install verl into container
pip install --no-deps -e verl/

echo "Start verl script"
echo "project_name: $project_name"
echo "experiment_name: $experiment_name"
echo "train_path: $train_path"
echo "test_path: $test_path"
echo "model_dir: $model_dir"


python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=64 \
    data.max_response_length=1024 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_dir \
    actor_rollout_ref.actor.optim.lr=3e-6\
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15

  # data.max_prompt_length=512 \
    # data.filter_overlong_prompts=True \