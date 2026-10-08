FROM vllm/vllm-openai:v0.29.0-cu129

# The assigned RunPod driver is older than CUDA 12.9. Skip the NVIDIA
# prestart check, then load the CUDA compatibility libraries in the image.
ENV NVIDIA_DISABLE_REQUIRE=true
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

WORKDIR /workspace

EXPOSE 8000

CMD ["bash", "-c", "vllm serve ${MODEL_NAME} --host 0.0.0.0 --port 8000 --max-model-len ${MAX_MODEL_LEN:-8192} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.85} --max-num-seqs ${MAX_NUM_SEQS:-1}"]
