from types import SimpleNamespace

from app.core.gpu import torch_cuda_problem


def fake_torch(available=True, capability=(12, 0), arches=("sm_80", "sm_90", "sm_100", "sm_120", "compute_120")):
    cuda = SimpleNamespace(is_available=lambda: available, get_device_capability=lambda i=0: capability,
                           get_arch_list=lambda: list(arches), get_device_name=lambda i=0: "NVIDIA GeForce RTX 5070")
    return SimpleNamespace(cuda=cuda, __version__="2.7.0+cu128")


def test_blackwell_with_a_cuda_128_build_is_fine():
    assert torch_cuda_problem(fake_torch()) is None


def test_cpu_only_build_is_explained():
    problem = torch_cuda_problem(fake_torch(available=False))
    assert "cu128" in problem and "is_available() is False" in problem


def test_old_build_without_blackwell_kernels_is_rejected():
    old = fake_torch(arches=("sm_50", "sm_60", "sm_70", "sm_80", "sm_86", "sm_90"))
    problem = torch_cuda_problem(old)
    assert "sm_120" in problem and "cu128" in problem and "RTX 5070" in problem


def test_ptx_for_an_older_architecture_is_accepted():
    # no native sm_120 kernels, but PTX that the driver can compile for the newer card
    assert torch_cuda_problem(fake_torch(arches=("sm_90", "compute_90"))) is None


def test_older_gpu_with_matching_kernels_is_fine():
    assert torch_cuda_problem(fake_torch(capability=(8, 6), arches=("sm_80", "sm_86"))) is None
