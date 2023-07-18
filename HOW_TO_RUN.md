## Train

```bash
WANDB_MODE=offline WANDB_CACHE_DIR=wandb_cache torchrun --nproc_per_node=1 run_pretraining_multimae.py --config cfgs/pretrain/v1.0.19-pr.yaml
```

## Train v2
```bash
PYTHONWARNINGS=ignore python run_pretraining_multimae_v2.py --config cfgs/pretrain/v2.0.3-pr.yaml 
```

## Qualitative Evaluation

```bash
WANDB_MODE=offline WANDB_CACHE_DIR=wandb_cache python eval_v2.py --config cfgs/pretrain/v2.0.2-pr.yaml 
```