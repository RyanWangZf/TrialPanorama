export N_GPUS=4
# export BASE_MODEL=/shared/rsaas/jl254/code/TrialPanorama/checkpoints/llama3-8b/full/sft_stage_2/checkpoint-14190
# export BASE_MODEL=/workspace/verl/checkpoints/base/checkpoint-14190
export BASE_MODEL=/workspace/verl/checkpoints/base/sft_stage_3
export DATA_DIR=data/sample_size_estimation
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=sample-size-estimation-llama3-8b-stage-1-${N_GPUS}gpus-new
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="c00a0bfe1fa6781c8a744ce43706d32e7d4bd78d"
export HF_HOME="/srv/local/data/linjc/hub"
export PROJECT_NAME="TrailPanorama-R1"

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash script_my/train_sample_size_estimation.sh