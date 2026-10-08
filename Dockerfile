FROM vllm/vllm-openai:v0.29.0-cu129

WORKDIR /workspace

EXPOSE 8000

CMD ["bash", "-c", "vllm serve ${MODEL_NAME} --host 0.0.0.0 --port 8000 --max-model-len ${MAX_MODEL_LEN:-8192} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.85} --max-num-seqs ${MAX_NUM_SEQS:-1}"]
