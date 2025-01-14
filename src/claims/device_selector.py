import sys
import platform
import torch


def select_best_device() -> str:
    """
    Automatically choose the best available device:
    - 'cuda' if an NVIDIA GPU is available.
    - 'mps' if on Apple Silicon with Metal Performance Shaders available.
    - 'cpu' otherwise.
    """
    # 1. Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # 2. Check for MPS (Mac with Apple Silicon)
    #    mps is available in PyTorch 1.12+ for MacOS 12.3+
    #    and only if the underlying hardware/OS supports it
    if (platform.system() == "Darwin"  # macOS
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return "mps"

    # 3. Fallback to CPU
    return "cpu"
