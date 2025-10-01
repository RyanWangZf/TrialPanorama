export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model DeepRetrieval/DeepRetrieval-PubMed-3B-Llama \
    --port 8000 \
    --max-model-len 2048 \
    --tensor-parallel-size 4 > vllm_host.log 2>&1 &