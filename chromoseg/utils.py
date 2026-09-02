import torch

def get_device() -> torch.device:
    """
    Returns the appropriate device (CPU, MPS, or GPU) for PyTorch operations.
        - CUDA for Windows/Linux with NVIDIA GPUs
        - MPS for macOS with Apple Silicon (M1, M2, etc.)
        - CPU as a fallback for other platforms or when no GPU is available.
    Returns:
        torch.device: The device to be used for computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")