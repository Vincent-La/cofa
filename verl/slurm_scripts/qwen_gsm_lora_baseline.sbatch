#!/bin/bash
#SBATCH -A cmsc848n-class
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=80000
#SBATCH --tmp=60000
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -t 15:59:00

module load cuda
module load apptainer

# export RAY_LOG_TO_STDERR=1

verl_workdir=/scratch/zt1/project/cmsc848n/user/vla/cofa/verl 
apptainer_image_path=/scratch/zt1/project/cmsc848n/user/vla/cofa/verl_vllm.sif


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
    apptainer exec --nv --bind $verl_workdir $apptainer_image_path \
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

# export VLLM_USE_V1=1


# NOTE: set verl params
project_name='cofa_experiments'
experiment_name='qwen3-0.6b_gsm8k_grpo_lora'

# note these need to be in binded dir for container to have access
train_path='/home/vla/scratch/cofa/verl/data/gsm/train_subset.parquet'
test_path='/home/vla/scratch/cofa/verl/data/gsm/test.parquet'

# /tmp dir included in container by default
model_dir='/scratch/zt1/project/cmsc848n/user/vla/hf_models/qwen3_0.6b'
tmp_model_dir='/tmp/qwen3_0.6b'

# output_dir='/scratch/zt1/project/cmsc848n/user/vla/cofa/verl/outputs/'

echo "copying model dir to /tmp"
cp -r $model_dir $tmp_model_dir

# cd /scratch/zt1/project/cmsc848n/user/vla/cofa

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    apptainer exec --nv --bind $verl_workdir $apptainer_image_path \
    bash verl/slurm_scripts/grpo_lora_template.sh $project_name $experiment_name $train_path $test_path $tmp_model_dir
  
echo "cleanup /tmp model dir"
rm -rf $tmp_model_dir

