# build_engine.py
import os
import sys
import shutil
import subprocess
import importlib.util

# --- ANSI Color Codes for Better Output ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def run_command(command, step_name, cwd=None, env=None):
    """Runs a command, streams output, and handles errors."""
    print(f"{Colors.HEADER}--- {step_name} ---{Colors.ENDC}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            cwd=cwd,
            env=env,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"\n{Colors.FAIL}Error: {step_name} failed with return code {process.returncode}.{Colors.ENDC}")
            sys.exit(1)
        print(f"{Colors.OKGREEN}--- {step_name} completed successfully. ---\n{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}An unexpected error occurred during '{step_name}': {e}{Colors.ENDC}")
        sys.exit(1)

def main():
    """
    Finds the llama-cpp-python installation and builds the 'server' executable.
    """
    print(f"{Colors.BOLD}Starting the native inference engine build process...{Colors.ENDC}")

    # 1. Find the installed llama-cpp-python package location
    try:
        spec = importlib.util.find_spec("llama_cpp")
        if spec is None or spec.origin is None:
            raise ImportError
        package_dir = os.path.dirname(spec.origin)
    except ImportError:
        print(f"{Colors.FAIL}Could not find the 'llama-cpp-python' package. "
              f"Please ensure it's installed via 'uv sync' first.{Colors.ENDC}")
        sys.exit(1)
        
    print(f"Found llama-cpp-python in: {package_dir}")

    # 2. Define source, build, and destination directories
    source_dir = os.path.join(package_dir, "vendor", "llama.cpp")
    build_dir = os.path.join(source_dir, "build")
    engine_dir = os.path.abspath("engine") # Our project's engine directory
    
    executable_name = "server.exe" if sys.platform == "win32" else "server"
    destination_path = os.path.join(engine_dir, executable_name)

    # 3. Clean up old builds and create directories
    if os.path.exists(build_dir):
        print(f"Removing old build directory: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(engine_dir, exist_ok=True)

    # 4. Configure the build with CMake
    # We will enable CUBLAS for GPU acceleration. This is the crucial step.
    cmake_args = [
        "cmake",
        "-S", source_dir,
        "-B", build_dir,
        "-DLLAMA_CUBLAS=ON",  # Enable CUDA
        "-DLLAMA_SERVER=ON"    # Ensure the server is a build target
    ]
    # Add generator arguments for Windows if using MSVC
    if sys.platform == "win32":
        cmake_args.extend(["-G", "Ninja"])

    run_command(cmake_args, "Step 1: Configuring CMake for the engine")

    # 5. Build the 'server' target with CMake
    run_command(
        ["cmake", "--build", build_dir, "--target", "server", "-j"],
        "Step 2: Compiling the 'server' executable"
    )

    # 6. Copy the compiled executable to our project's engine directory
    built_executable_path = os.path.join(build_dir, "bin", executable_name)
    print(f"Copying '{built_executable_path}' to '{destination_path}'...")
    shutil.copy(built_executable_path, destination_path)

    print(f"{Colors.OKGREEN}{Colors.BOLD}✅ Native inference engine built successfully!{Colors.ENDC}")
    print(f"The executable is now at: {destination_path}")

if __name__ == "__main__":
    main()