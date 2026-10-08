FROM vllm/vllm-openai:v0.29.0-cu129

# The assigned RunPod driver is older than CUDA 12.9. Skip the NVIDIA
# prestart check, then load the CUDA compatibility libraries in the image.
ENV NVIDIA_DISABLE_REQUIRE=true
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

WORKDIR /workspace

RUN pip install --no-cache-dir runpod requests

COPY handler.py /workspace/handler.py

EXPOSE 8000

# The base image entrypoint is the vllm CLI. Clear it so this command
# is Python, not `vllm -u /workspace/handler.py`.
ENTRYPOINT []
CMD ["python3", "-u", "/workspace/handler.py"]
