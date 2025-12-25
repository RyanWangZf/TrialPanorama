export WANDB_API_KEY="c00a0bfe1fa6781c8a744ce43706d32e7d4bd78d"
export WANDB_PROJECT="TrialPanorama"  
export WANDB_RUN_NAME="llama3-stage2-full-sft"  
export ALLOW_EXTRA_ARGS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

llamafactory-cli train recipe/llama3_full_sft_stage_2.yaml