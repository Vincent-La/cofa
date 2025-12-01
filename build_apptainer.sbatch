#!/bin/bash
#SBATCH -A cmsc848n-class
#SBATCH -n 1
#SBATCH --tasks-per-node=1
#SBATCH --mem=64g
#SBATCH -t 16:00:00

module load apptainer

cd /scratch/zt1/project/cmsc848n/user/vla/cofa
apptainer --verbose pull verl_vllm.sif docker://verlai/verl:vllm011.latest

