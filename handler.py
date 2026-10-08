#!/usr/bin/env python3
import json
import os
import sys
import time
import subprocess
import requests
import runpod


MODEL_PATH = os.getenv(
    "MODEL_NAME",
    "/runpod-volume/models/glm-4.6v-flash",
)

HOST = "127.0.0.1"
PORT = 8000

MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "8192")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.85")
MAX_NUM_SEQS = os.getenv("MAX_NUM_SEQS", "1")
VIDEO_NUM_FRAMES = int(os.getenv("VIDEO_NUM_FRAMES", "16"))
READY_TIMEOUT_SECONDS = int(os.getenv("VLLM_READY_TIMEOUT", "900"))

# GLM-4.6V-Flash profiles one sample of each modality at startup.
# Cap that sample so it fits inside MAX_MODEL_LEN.
LIMIT_MM_PER_PROMPT = os.getenv(
    "LIMIT_MM_PER_PROMPT",
    json.dumps(
        {
            "image": 4,
            "audio": 1,
            "video": {
                "count": 1,
                "num_frames": VIDEO_NUM_FRAMES,
                "width": 224,
                "height": 224,
            },
        },
        separators=(",", ":"),
    ),
)
MEDIA_IO_KWARGS = os.getenv(
    "MEDIA_IO_KWARGS",
    json.dumps(
        {
            "video": {
                "num_frames": VIDEO_NUM_FRAMES,
                "backend": "opencv",
            }
        },
        separators=(",", ":"),
    ),
)


vllm_process = None


def prepare_compile_cache():
    """Keep torch.compile output on the network volume.

    A new serverless worker has an empty container disk, so the compile
    under /root/.cache otherwise runs on every cold start.
    """
    cache_root = os.getenv("VLLM_CACHE_ROOT", "/runpod-volume/vllm-cache")
    inductor_cache = os.path.join(cache_root, "inductor")
    try:
        os.makedirs(inductor_cache, exist_ok=True)
    except OSError as exc:
        print(f"Compile cache not available ({exc}).")
        return

    os.environ["VLLM_CACHE_ROOT"] = cache_root
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
    print(f"vLLM compile cache: {cache_root}")


def start_vllm():
    global vllm_process

    if vllm_process is not None:
        return

    prepare_compile_cache()

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL_PATH,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--limit-mm-per-prompt",
        LIMIT_MM_PER_PROMPT,
        "--media-io-kwargs",
        MEDIA_IO_KWARGS,
        "--safetensors-load-strategy",
        "prefetch",
    ]

    print("Starting vLLM:")
    print(" ".join(command))

    vllm_process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
        env=os.environ.copy(),
    )

    print("vLLM process started.")


def wait_for_vllm():
    print("Waiting for vLLM server...")

    url = f"http://{HOST}:{PORT}/health"
    deadline = time.time() + READY_TIMEOUT_SECONDS
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("vLLM server is ready.")
                return
        except Exception:
            pass

        if vllm_process is not None:
            return_code = vllm_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {return_code}"
                )

        if attempt % 10 == 1:
            print(f"Still waiting for vLLM... attempt {attempt}")

        time.sleep(2)

    raise TimeoutError(
        f"vLLM server did not become ready within {READY_TIMEOUT_SECONDS} seconds."
    )


def handler(job):
    try:
        job_input = job.get("input", job)

        if not isinstance(job_input, dict):
            return {"error": "Input must be a JSON object."}

        messages = job_input.get("messages")
        if not messages:
            return {
                "error": (
                    "Missing 'messages'. "
                    "Expected OpenAI-compatible chat messages."
                )
            }

        if job_input.get("stream"):
            return {
                "error": (
                    "Streaming is not supported on this queue worker. "
                    "Omit stream."
                )
            }

        payload = {
            "model": MODEL_PATH,
            "messages": messages,
        }

        optional_parameters = [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
            "seed",
            "mm_processor_kwargs",
        ]

        for parameter in optional_parameters:
            if parameter in job_input:
                payload[parameter] = job_input[parameter]

        print("Sending request to vLLM...")

        response = requests.post(
            f"http://{HOST}:{PORT}/v1/chat/completions",
            json=payload,
            timeout=600,
        )

        if response.status_code != 200:
            return {
                "error": "vLLM request failed",
                "status_code": response.status_code,
                "details": response.text,
            }

        print("vLLM request completed.")
        return response.json()

    except Exception as e:
        print(f"Handler error: {type(e).__name__}: {e}")
        return {
            "error": str(e),
            "type": type(e).__name__,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("GLM-4.6V-Flash Runpod Serverless Worker")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Max model length: {MAX_MODEL_LEN}")
    print(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"Max sequences: {MAX_NUM_SEQS}")
    print(f"Video frames: {VIDEO_NUM_FRAMES}")
    print(f"Multimodal limits: {LIMIT_MM_PER_PROMPT}")
    print("=" * 60)

    start_vllm()
    wait_for_vllm()
    print("Starting Runpod Serverless handler...")


# Must stay at the start of the line. Runpod's repo scan looks for this call.
runpod.serverless.start({"handler": handler})
