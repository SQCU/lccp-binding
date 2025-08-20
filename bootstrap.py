# bootstrap.py
import subprocess
import sys
import os
import shutil
import importlib.util
from pathlib import Path

# --- ANSI Color Codes for Better Output ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# bootstrap.py
# ... (All other code remains the same) ...

def handle_windows_dependencies(base_dir: Path):
    """
    On Windows, this function ensures vcpkg is set up and provides the necessary
    CMake arguments to find the curl dependency. On other platforms, it does nothing.
    """
    if sys.platform != "win32":
        return []

    print(f"{Colors.OKBLUE}--- Windows platform detected. Checking for dependencies... ---{Colors.ENDC}")
    
    vcpkg_dir = base_dir / "vcpkg"
    curl_pkg_dir = vcpkg_dir / "packages" / "curl_x64-windows"
    toolchain_file = (vcpkg_dir / "scripts" / "buildsystems" / "vcpkg.cmake").resolve()
    
    # --- ⭐ START OF FIX ⭐ ---
    # Find PowerShell's path and directory reliably.
    system_root = os.environ.get("SystemRoot", "C:\\Windows")
    powershell_path = (
        Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    
    if not powershell_path.exists():
        print(f"{Colors.FAIL}FATAL: Could not find PowerShell at '{powershell_path}'.")
        sys.exit(1)
        
    # Get the directory containing powershell.exe
    powershell_dir = powershell_path.parent
    # --- ⭐ END OF FIX ⭐ ---

    if not curl_pkg_dir.exists():
        print(f"{Colors.WARNING}vcpkg or curl dependency not found. Setting it up...{Colors.ENDC}")
        if not vcpkg_dir.exists():
            run_command(
                ["git", "clone", "https://github.com/microsoft/vcpkg.git", str(vcpkg_dir)],
                "Cloning vcpkg"
            )
        
        print(f"Found PowerShell at: {powershell_path}")
        
        powershell_script_relative_path = Path(".") / "scripts" / "bootstrap.ps1"
        bootstrap_command = [
            str(powershell_path),
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-Command", f"& '{powershell_script_relative_path}' -disableMetrics"
        ]
        
        run_command(bootstrap_command, "Bootstrapping vcpkg", cwd=vcpkg_dir)
        
        vcpkg_exe = vcpkg_dir / "vcpkg.exe"
        run_command([str(vcpkg_exe), "install", "curl:x64-windows"], "Installing curl via vcpkg", cwd=vcpkg_dir)

    else:
        print(f"{Colors.OKGREEN}Found existing vcpkg curl installation.{Colors.ENDC}")

    # ... (Pre-flight checks remain the same) ...
    print(f"{Colors.HEADER}--- Running Pre-flight Build Checks ---{Colors.ENDC}")
    if not toolchain_file.exists():
        print(f"{Colors.FAIL}Build Sanity Check FAILED: vcpkg toolchain file not found at '{toolchain_file}'{Colors.ENDC}")
        sys.exit(1)
    else:
        print(f"{Colors.OKGREEN}OK: Found vcpkg toolchain file.{Colors.ENDC}")
    print(f"{Colors.OKGREEN}--- Pre-flight Checks Passed ---\n{Colors.ENDC}")
    
    # Prepend the PowerShell directory to the PATH environment variable for the build subprocess.
    # This is the correct way to ensure child processes (like cmake/msbuild) can find it.
    print(f"Adding '{powershell_dir}' to PATH for the build process.")
    os.environ['PATH'] = f"{powershell_dir}{os.pathsep}{os.environ['PATH']}"
    # --- ⭐ END OF FIX ⭐ ---
    
    # We return ONLY the toolchain file argument.
    return [
        "-DLLAMA_CURL=ON",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file.as_posix()}"
    ]

    """
    print(f"Providing absolute curl library path: {curl_lib}")
    print(f"Providing absolute curl include path: {curl_include}")
    
    return [
        "-DLLAMA_CURL=ON",
        f"-DCURL_LIBRARY={curl_lib}",
        f"-DCURL_INCLUDE_DIR={curl_include}"
    ]
    """

def copy_windows_dlls(base_dir: Path):
    """
    After a successful build on Windows, copy the required DLLs next to the executable.
    """
    if sys.platform != "win32":
        return

    print(f"{Colors.HEADER}--- Copying required DLLs on Windows ---{Colors.ENDC}")
    vcpkg_bin_dir = base_dir / "vcpkg" / "installed" / "x64-windows" / "bin"
    build_bin_dir = base_dir / "build" / "bin"
    
    dlls_to_copy = ["libcurl.dll", "zlib1.dll"]
    for dll in dlls_to_copy:
        source_dll = vcpkg_bin_dir / dll
        if source_dll.exists():
            print(f"Copying '{dll}' to '{build_bin_dir}'")
            shutil.copy(source_dll, build_bin_dir)
        else:
            print(f"{Colors.WARNING}Warning: Could not find '{dll}' to copy.{Colors.ENDC}")


def run_command(command, step_name, cwd=None, log_file=None):
    log_writer = open(log_file, 'w', encoding='utf-8') if log_file else None
    print(f"{Colors.HEADER}--- {step_name} ---{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Executing command: {' '.join(command)}{Colors.ENDC}")
    if cwd:
        print(f"{Colors.OKBLUE}In working directory: {cwd}{Colors.ENDC}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            cwd=cwd
        )
        for line in process.stdout:
            print(line, end='')
            if log_writer:
                log_writer.write(line)
        process.wait()
        if process.returncode != 0:
            print(f"\n{Colors.FAIL}Error: {step_name} failed with return code {process.returncode}.{Colors.ENDC}")
            # If logging was enabled, tell the user where to find the detailed log
            if log_file:
                print(f"{Colors.OKCYAN}A detailed build log has been saved to:{Colors.ENDC}")
                print(f"{Colors.BOLD}{log_file}{Colors.ENDC}")
            sys.exit(1)
        print(f"{Colors.OKGREEN}--- {step_name} completed successfully. ---\n{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}An unexpected error occurred during '{step_name}': {e}{Colors.ENDC}")
        if log_file:
            print(f"{Colors.OKCYAN}A partial build log may be available at: {log_file}{Colors.ENDC}")
        sys.exit(1)
    finally:
        # Ensure the log file is always closed
        if log_writer:
            log_writer.close()


def ensure_uv_is_available():
    """
    Checks if 'uv' is installed. If not, installs it using pip.
    This is the core of the self-bootstrapping logic.
    """
    # Use importlib to check if the 'uv' module can be found.
    # This is more reliable than checking the PATH.
    spec = importlib.util.find_spec("uv")
    if spec is None:
        print(f"{Colors.WARNING}'uv' is not installed. Attempting to install it now...{Colors.ENDC}")
        # Use the current Python interpreter to run pip, ensuring uv is installed
        # for this Python version. This is the most robust method.
        run_command(
            [sys.executable, "-m", "pip", "install", "uv"],
            "Bootstrapping: Installing 'uv'"
        )
    else:
        print(f"{Colors.OKGREEN}'uv' is already installed. Proceeding.{Colors.ENDC}\n")

# We modify get_git_repo to accept and use a specific commit/tag
def get_git_repo(url, target_dir, commit="main"):
    """Clones a git repository and checks out a specific commit/tag."""
    target_path = Path(target_dir)
    if target_path.exists():
        print(f"{Colors.OKBLUE}Directory '{target_dir}' already exists. Checking commit...{Colors.ENDC}\n")
        # You could add logic here to check if the commit is correct, but for now we assume it is.
        return

    try:
        import git
    except ImportError:
        run_command([sys.executable, "-m", "uv", "pip", "install", "GitPython"], "Installing GitPython")
        import git

    print(f"{Colors.HEADER}--- Cloning {url} at commit/tag '{commit}' ---{Colors.ENDC}")
    repo = git.Repo.clone_from(url, target_path)
    
    # Check out the specific commit after cloning
    if commit != "main":
        repo.git.checkout(commit)
    
    print(f"{Colors.OKGREEN}--- Clone complete ---\n{Colors.ENDC}")


def main():
    """
    Main function to orchestrate the entire setup process.
    """
    print(f"{Colors.BOLD}Starting project setup...\n{Colors.ENDC}")

    LLAMACPP_DIR = Path("llama.cpp")
    SETUP_PY_SOURCE = Path("trickery/setup.py")
    build_log_path = LLAMACPP_DIR / "build_log.txt"

    LLAMACPP_VERSION = "b6208"

    # Step 1: Self-bootstrap uv if it's not present.
    ensure_uv_is_available()

    # Step 2: Run 'uv sync' to create the environment and compile dependencies.
    # We run 'uv' as a module to avoid any PATH issues.
    run_command(
        [sys.executable, "-m", "uv", "sync"],
        "Step 1: Creating environment and compiling dependencies"
    )

    # Step 3: Clone llama.cpp (if not present)
    get_git_repo("https://github.com/ggerganov/llama.cpp.git", LLAMACPP_DIR, commit=LLAMACPP_VERSION)
    
    # Step 4: The 'Trickery' - Copy the custom setup.py into llama.cpp
    if not SETUP_PY_SOURCE.exists():
        print(f"{Colors.FAIL}Error: The build script at '{SETUP_PY_SOURCE}' was not found.{Colors.ENDC}")
        sys.exit(1)
    
    print(f"{Colors.HEADER}--- Step 2: Preparing llama.cpp for building ---{Colors.ENDC}")
    shutil.copy(SETUP_PY_SOURCE, LLAMACPP_DIR / "setup.py")
    print(f"Copied '{SETUP_PY_SOURCE}' to '{LLAMACPP_DIR / 'setup.py'}'")
    #msvc libcurl error?
    cmake_extra_args = handle_windows_dependencies(LLAMACPP_DIR)
    if cmake_extra_args:
        # Pass the arguments to the setup.py script via an environment variable
        os.environ["CMAKE_ARGS"] = " ".join(f'"{arg}"' for arg in cmake_extra_args)
    print(f"{Colors.OKGREEN}--- Preparation complete ---\n{Colors.ENDC}")


    # Step 2: Build llama.cpp using the Python packaging infrastructure
    # This is the key change. We let pip and setuptools handle the build.
    run_command(
        [sys.executable, "setup.py", "build_ext"],
        "Step 3: Building llama.cpp C++ server via custom setup.py",
        cwd=LLAMACPP_DIR,
        log_file=build_log_path # Pass the path for the log file
    )

    # Step 3: Run the model download script using the new environment.
    run_command(
        [sys.executable, "-m", "uv", "run", "python", "download_model.py"],
        "Step 2: Downloading the GGUF model"
    )

    print(f"{Colors.OKGREEN}{Colors.BOLD}🎉 Setup complete! 🎉{Colors.ENDC}")
    print("The llama.cpp server executable should be in the 'llama.cpp/build' directory.")
    print("You can now start the server by running:")
    # The exact output path might need adjustment based on the setup.py
    print(f"{Colors.OKCYAN}./llama.cpp/build/bin/server -m models/your_model.gguf{Colors.ENDC}")

if __name__ == "__main__": 
    main()