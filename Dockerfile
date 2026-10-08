FROM vllm/vllm-openai:v0.29.0-cu129
#
# Skip the NVIDIA prestart CUDA version check. Do not enable
# VLLM_ENABLE_CUDA_COMPATIBILITY: this GPU rejects those libraries
# with CUDA error 804.
ENV NVIDIA_DISABLE_REQUIRE=true

WORKDIR /workspace

RUN pip install --no-cache-dir runpod requests

COPY handler.py /workspace/handler.py

EXPOSE 8000

# The base image entrypoint is the vllm CLI. Clear it so this command
# is Python, not `vllm -u /workspace/handler.py`.
ENTRYPOINT []
CMD ["python3", "-u", "/workspace/handler.py"]
