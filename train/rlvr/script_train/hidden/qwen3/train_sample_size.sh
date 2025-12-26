export N_GPUS=4
export BASE_MODEL="/workspace/verl/checkpoints/base/qwen-sft-stage-3"
export DATA_DIR=data/qwen3/sample_size_estimation
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=sample-size-estimation-qwen3-8b-stage-1-${N_GPUS}gpus
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="c00a0bfe1fa6781c8a744ce43706d32e7d4bd78d"
export HF_HOME="/srv/local/data/linjc/hub"
export PROJECT_NAME="TrailPanorama-R1"

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash script_my/train_sample_size_estimation_qwen.sh