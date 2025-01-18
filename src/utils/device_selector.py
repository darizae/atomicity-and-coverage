import platform
import torch


def select_best_device() -> str:
    """
    Automatically choose the best available device:
    - 'cuda' if an NVIDIA GPU is available.
    - 'mps' if on Apple Silicon with Metal Performance Shaders.
    - 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"

    # Check for MPS on macOS (PyTorch 1.12+)
    if (platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return "mps"

    return "cpu"


def check_or_select_device(requested_device: str | None = None) -> str:
    """
    Check if the user-specified device is actually available.
    If `requested_device` is None, auto-detect via `select_best_device`.
    Otherwise, validate that the requested device is available.

    :param requested_device: e.g., "cuda", "cpu", "mps", or None.
    :return: The validated device string.
    :raises ValueError: if a user explicitly requests an unavailable device.
    """
    # Auto-detect if user hasn't specified
    if requested_device is None:
        return select_best_device()

    # If user requests cuda, ensure it's available
    if requested_device.lower() == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available on this machine.")
        return "cuda"

    # If user requests mps, ensure it's available
    if requested_device.lower() == "mps":
        if not (platform.system() == "Darwin"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            raise ValueError("MPS not available on this machine.")
        return "mps"

    # If user explicitly requests "cpu", always fine
    if requested_device.lower() == "cpu":
        return "cpu"

    # For any other device string (e.g., "cuda:0", "cuda:1", etc.)
    # you might do deeper checks if needed. Here we illustrate a simple approach:
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"{requested_device} is unavailable because CUDA is not available.")
        return requested_device

    # Otherwise, assume user knows what they're doing (e.g., "xla", custom device, etc.)
    # or handle it as an error if you want to restrict devices strictly.
    raise ValueError(f"Unknown or unsupported device requested: '{requested_device}'.")
