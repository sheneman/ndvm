"""Ahead-of-time build of the NDVM native extension (the Phase-2 PyTorch boundary).

Uses torch's setuptools CppExtension with use_ninja=False, so it builds with the plain system C++
compiler and does NOT require ninja (unlike the JIT torch.utils.cpp_extension.load path). Run on an
HPC compute node where torch + a C++17 compiler are available:

    cd ndvm/python && /path/to/.venv/bin/python setup.py build_ext --inplace

This drops ndvm_native.*.so next to ndvm_autograd.py, which imports it directly (no ninja needed).
"""
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

HERE = Path(__file__).resolve().parent
NDVM = HERE.parent
sources = [str(HERE / "ndvm_ext.cpp")] + sorted(str(p) for p in (NDVM / "src").glob("*.cpp"))

setup(
    name="ndvm_native",
    ext_modules=[
        CppExtension(
            "ndvm_native",
            sources,
            include_dirs=[str(NDVM / "include"), str(NDVM / "src")],
            extra_compile_args=["-O2", "-std=c++17", "-pthread"],   # Phase 5: parallel.cpp uses std::thread
            extra_link_args=["-pthread"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
