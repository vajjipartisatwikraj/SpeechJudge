"""GPU readiness helpers (NVIDIA RTX 50 series / Blackwell, compute capability 12.0 included).

A Blackwell card (RTX 5070 = sm_120) needs CUDA 12.8+ builds: PyTorch >= 2.7 from the ``cu128`` wheel
index and CTranslate2 >= 4.6 (faster-whisper). Older builds either fail with "no kernel image is
available for execution on the device" or silently fall back to the CPU. These helpers turn that into
a clear start-up error instead of a mystery.
"""
from __future__ import annotations

from typing import Any


def prepare_cuda_runtime() -> None:
    """Import PyTorch first when it is installed.

    PyTorch's CUDA wheels bundle cuBLAS / cuDNN and load them globally on import. CTranslate2 looks
    for ``libcublas.so.12`` / ``libcudnn*.so.9`` (``cublas64_12.dll`` on Windows) by name when the first
    GPU model is created; after ``import torch`` the already-loaded copies are reused, so no
    LD_LIBRARY_PATH juggling is needed. A missing torch is not an error here (the caller reports it).
    """
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


def torch_cuda_problem(torch: Any, index: int = 0) -> str | None:
    """Why this PyTorch build cannot run on GPU ``index``; None when it can."""
    if not torch.cuda.is_available():
        return ("torch.cuda.is_available() is False: this is a CPU-only PyTorch build, the NVIDIA driver is "
                "missing or too old, or no GPU is visible. For an RTX 50 series card install the CUDA 12.8 "
                "build:  pip install torch --index-url https://download.pytorch.org/whl/cu128")
    major, minor = torch.cuda.get_device_capability(index)
    capability = major * 10 + minor
    arches = list(torch.cuda.get_arch_list())
    native = {int(a[3:]) for a in arches if a.startswith("sm_") and a[3:].isdigit()}
    ptx = {int(a[8:]) for a in arches if a.startswith("compute_") and a[8:].isdigit()}
    if arches and capability not in native and not any(p <= capability for p in ptx):
        name = torch.cuda.get_device_name(index)
        return (f"This PyTorch build ({torch.__version__}, kernels for {', '.join(arches)}) has no kernels "
                f"for {name} (sm_{capability}). Install the CUDA 12.8 build:  "
                "pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128")
    return None


def ctranslate2_cuda_problem() -> str | None:
    """Why CTranslate2 (faster-whisper) cannot use the GPU; None when it can."""
    try:
        import ctranslate2  # type: ignore
    except ImportError:
        return "ctranslate2 is not installed (pip install -r requirements-ml.txt)"
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        return (f"CTranslate2 cannot initialise CUDA ({exc}). Needs the NVIDIA driver plus the CUDA 12 "
                "cuBLAS and cuDNN 9 libraries (they come with the PyTorch cu128 wheel: import torch first).")
    if count < 1:
        return ("CTranslate2 sees no CUDA device. Check nvidia-smi, and that ctranslate2 >= 4.6 is installed "
                "(older builds have no Blackwell / RTX 50 support).")
    return None
