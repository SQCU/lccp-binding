# trickery/setup.py
import os
import sys
import subprocess
from pathlib import Path
import shlex

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self) -> None:
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        # The source directory, e.g., 'llama.cpp/'
        source_dir = Path(ext.sourcedir)
        # The build directory, e.g., 'llama.cpp/build/'
        build_dir = source_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- Configuring CMake for {ext.name} ---")

        # *** THE FIX IS HERE: We now run CMake from INSIDE the build directory ***
        cmake_args = [
            "cmake",
            "..",  # The source directory is now '..' relative to the build directory
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        
        extra_cmake_args = os.environ.get("CMAKE_ARGS", "")
        if extra_cmake_args:
            cmake_args.extend(shlex.split(extra_cmake_args))

        # Run the configuration step from within the build directory
        subprocess.check_call(cmake_args, cwd=build_dir)

        print(f"--- Building {ext.name} with CMake ---")
        
        # *** THE FIX IS HERE: We now run the build command from INSIDE the build directory ***
        build_args = ["cmake", "--build", ".", "--config", "Release"]
        
        # Run the build step from within the build directory
        subprocess.check_call(build_args, cwd=build_dir)

        print(f"--- CMake build successful. Executables are in {build_dir / 'bin'} ---")

setup(
    name="llama_cpp_builder",
    version="0.1.0",
    ext_modules=[CMakeExtension("llama_cpp_server", sourcedir='.')],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)