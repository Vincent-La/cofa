#!/bin/bash

# cp -r /home/vince/hf_models/qwen2.5_0.5b /tmp/qwen2.5_0.5b
cp -r /home/vince/hf_models/qwen3_0.6b /tmp/qwen3_0.6b

# model_dir='/tmp/qwen2.5_0.5b'
model_dir='/tmp/qwen3_0.6b'
# model_dir='Qwen/Qwen2.5-0.5B'

train_files='/home/vince/cofa/data/gsm/train_subset.parquet'
test_files='/home/vince/cofa/data/gsm/test.parquet'

# install verl into container
pip install --no-deps -e verl/

# run verl script
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
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
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2.5_0.5b_gsm8k_lora' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

#cleanup
# rm -rf /tmp/qwen2.5_0.5b
rm -rf /tmp/qwen3_0.6b
