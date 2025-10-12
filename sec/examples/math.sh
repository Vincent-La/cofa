exp_name=$1
shift 1
other_args=$@




# Get the number of GPUs available
export N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUs available: $N_GPUS"


MODEL_PATH="Qwen/Qwen2.5-3B"
log_dir=logs/math
mkdir -p $log_dir


export VLLM_ATTENTION_BACKEND=XFORMERS

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/math/math_train.parquet \
    data.val_files=data/math/math_test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.00\ \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='math_sec' \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    $other_args 2>&1 | tee ${log_dir}/${exp_name}.log