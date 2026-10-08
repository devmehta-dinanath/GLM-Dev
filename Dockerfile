FROM vllm/vllm-openai:v0.29.0-cu129
#
# Skip the NVIDIA prestart CUDA version check. Do not enable
# VLLM_ENABLE_CUDA_COMPATIBILITY: this GPU rejects those libraries
# with CUDA error 804.
ENV NVIDIA_DISABLE_REQUIRE=true
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0

# cuda-compat's libcuda shadows the host driver and raises CUDA error 804
# on this GPU. Drop it and search the host driver path first.
RUN rm -f /etc/ld.so.conf.d/*compat* \
    && rm -rf /usr/local/cuda/compat /usr/local/cuda-12.9/compat /usr/local/cuda-12/compat \
    && ldconfig
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /workspace

RUN pip install --no-cache-dir runpod requests

COPY handler.py /workspace/handler.py

EXPOSE 8000

# The base image entrypoint is the vllm CLI. Clear it so this command
# is Python, not `vllm -u /workspace/handler.py`.
ENTRYPOINT []
CMD ["python3", "-u", "/workspace/handler.py"]
