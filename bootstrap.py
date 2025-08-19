# bootstrap.py
import subprocess
import sys
import shutil
import importlib.util

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

def run_command(command, step_name):
    """
    Runs a command in a subprocess, streams its output, and checks for errors.
    """
    print(f"{Colors.HEADER}--- {step_name} ---{Colors.ENDC}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
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

def main():
    """
    Main function to orchestrate the entire setup process.
    """
    print(f"{Colors.BOLD}Starting project setup...\n{Colors.ENDC}")

    # Step 1: Self-bootstrap uv if it's not present.
    ensure_uv_is_available()

    # Step 2: Run 'uv sync' to create the environment and compile dependencies.
    # We run 'uv' as a module to avoid any PATH issues.
    run_command(
        [sys.executable, "-m", "uv", "sync"],
        "Step 1: Creating environment and compiling dependencies"
    )

    # Step 3: Run the model download script using the new environment.
    run_command(
        [sys.executable, "-m", "uv", "run", "python", "download_model.py"],
        "Step 2: Downloading the GGUF model"
    )

    print(f"{Colors.OKGREEN}{Colors.BOLD}🎉 Setup complete! 🎉{Colors.ENDC}")
    print("You can now start the server by running:")
    print(f"{Colors.OKCYAN}uv run python server.py{Colors.ENDC}")
    print(f"{Colors.OKCYAN}# or: {sys.executable} -m uv run python server.py{Colors.ENDC}")

if __name__ == "__main__":
    main()