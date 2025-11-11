# cofa
Curriculum Optimization for Agents (COfA)

## Installation

### uv

uv installation docs can be found [here](https://docs.astral.sh/uv/getting-started/installation/)

### Download pre-built flash-attn wheel

+ Building flash-attn takes forever... using a pre-built one instead
+ uv lock will look for this wheel in the top-level directory of this git repository

```
# torch and python versions match uv lock requirments
# this wheel is compatible with CUDA 12.xx, other releases can be found at 
# https://github.com/Dao-AILab/flash-attention/releases/

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

### Create venv
```
uv sync
```

### Activate venv
```
source .venv/bin/activate
```

### Cleanup
```
rm flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
```

## Experiments



+ TODO: run scripts follow the original SEC repo setup, maybe better to just put all the params into the file itself 

### ARC-1D Random (uniform sampling) Baseline
```
sh experiments/arc_baselines.sh arc_random_baseline \
    trainer.sec.enable=True \
    trainer.sec.strategy=random \
    trainer.total_training_steps=240 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B
``` 

### ARC-1D Sec Baseline
```
sh experiments/arc_baselines.sh arc_sec_baseline \
    trainer.sec.enable=True \
    trainer.sec.strategy=bandit \
    trainer.sec.bandit.lr=0.5 \
    trainer.sec.bandit.objective=adv \
    trainer.total_training_steps=240 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B
``` 
