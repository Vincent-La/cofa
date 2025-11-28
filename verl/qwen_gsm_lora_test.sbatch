#!/bin/bash
#SBATCH -A cmsc848n-class
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=80000
#SBATCH --tmp=60000
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -t 00:24:30

# ldd --version
# source ~/.bashrc
module load cuda
# micromamba activate VERL
cd /scratch/zt1/project/cmsc848n/user/vla/cofa/verl 
source .venv/bin/activate


# export RAY_LOG_TO_STDERR=1

# verl_workdir=/scratch/zt1/project/cmsc848n/user/vla/cofa/verl 


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=("$nodes")

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"


# printenv

echo "Starting HEAD at $head_node"

srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus 1 --num-gpus 1 --include-dashboard=false --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# NOTE: only doing single-node training for now
# # number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))

# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     srun --nodes=1 --ntasks=1 -w "$node_i" \
#         ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
#     sleep 5
# done

export VLLM_USE_V1=1

cp -r /scratch/zt1/project/cmsc848n/user/vla/hf_models/qwen2.5_0.5b /tmp/qwen2.5_0.5b/
# attmempting to resolve some ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found according to https://gcc.gnu.org/onlinedocs/libstdc++/faq.html#faq.how_to_install
# export LD_LIBRARY_PATH=/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-x86_64/gcc-rh8-8.5.0/gcc-11.3.0-oedkmii7vhd6rbnqm6xufmg7d3jx4w6l/lib64:$LD_LIBRARY_PATH

# bash run_qwen2.5-0.5b_gsm8k_lora.sh 

echo "Start verl script"

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/scratch/zt1/project/cmsc848n/user/vla/cofa/verl/data/gsm8k/train.parquet \
    data.val_files=/scratch/zt1/project/cmsc848n/user/vla/cofa/verl/data/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_response_length=1024 \
    data.truncation='error' \
    actor_rollout_ref.model.path=/tmp/qwen2.5_0.5b \
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
    trainer.total_epochs=1 $@

    # data.max_prompt_length=512 \
    # data.filter_overlong_prompts=True \


rm -rf /tmp/qwen2.5_0.5b/

