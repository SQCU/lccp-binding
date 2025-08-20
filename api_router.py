# api_router.py
import subprocess
import time
import sys
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

# --- Configuration ---
LLAMACPP_DIR = Path("./llama.cpp")
BUILD_DIR = LLAMACPP_DIR / "build"
SERVER_EXECUTABLE_NAME = "server.exe" if sys.platform == "win32" else "server"
SERVER_EXECUTABLE = BUILD_DIR / "bin" / SERVER_EXECUTABLE_NAME

# Model and server settings
MODEL_PATH = Path("./models/gemma-3-1b-pt-q4_0.gguf") # Make sure this matches your downloaded model
HOST = "127.0.0.1"
LLAMA_PORT = 8080
ROUTER_PORT = 8000
LLAMA_CPP_URL = f"http://{HOST}:{LLAMA_PORT}"

# --- FastAPI Application ---
app = FastAPI()
llama_server_process = None

@app.on_event("startup")
async def startup_event():
    """On startup, launch the llama.cpp server as a subprocess."""
    global llama_server_process
    if not SERVER_EXECUTABLE.exists():
        print(f"FATAL: llama.cpp server executable not found at '{SERVER_EXECUTABLE}'")
        print("Please ensure you have run the bootstrap.py script successfully.")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"FATAL: Model file not found at '{MODEL_PATH}'")
        sys.exit(1)

    print(f"Starting llama.cpp server...")
    command = [
        str(SERVER_EXECUTABLE),
        "-m", str(MODEL_PATH),
        "--host", HOST,
        "--port", str(LLAMA_PORT),
        "-c", "4096", # Context size
        # Add other llama.cpp server arguments here
        # "--n-gpu-layers", "35",
    ]
    
    # Start the subprocess
    llama_server_process = subprocess.Popen(command)
    
    # Wait for the server to be ready
    # This is a simple health check loop; more robust solutions could be used
    print(f"Waiting for llama.cpp server to be ready at {LLAMA_CPP_URL}...")
    is_ready = False
    for _ in range(30): # Wait for max 30 seconds
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{LLAMA_CPP_URL}/health")
                if response.status_code == 200 and response.json().get("status") == "ok":
                    is_ready = True
                    break
        except httpx.ConnectError:
            time.sleep(1) # Wait a second before retrying
            
    if is_ready:
        print("llama.cpp server is ready.")
    else:
        print("FATAL: llama.cpp server did not become ready in time.")
        llama_server_process.terminate()
        sys.exit(1)


@app.on_event("shutdown")
def shutdown_event():
    """On shutdown, terminate the llama.cpp server process."""
    global llama_server_process
    if llama_server_process:
        print("Shutting down llama.cpp server...")
        llama_server_process.terminate()
        try:
            # Wait for a few seconds for the process to terminate gracefully
            llama_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # If it doesn't terminate, force kill it
            print("llama.cpp server did not terminate gracefully, forcing kill.")
            llama_server_process.kill()
        print("llama.cpp server shut down.")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def relay_to_llama_cpp(request: Request):
    """
    Catches all incoming requests and forwards them to the llama.cpp server.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        # Construct the target URL
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
        
        # Prepare the request for forwarding
        rp_req = client.build_request(
            method=request.method,
            url=url,
            headers=request.headers.raw,
            content=await request.body(),
        )
        
        # Make the actual request to the llama.cpp server
        rp_resp = await client.send(rp_req, base_url=LLAMA_CPP_URL)
        
        # Return the response from llama.cpp back to the original client
        return Response(
            content=rp_resp.content,
            status_code=rp_resp.status_code,
            headers=rp_resp.headers,
        )

if __name__ == "__main__":
    print(f"Starting API Router on http://{HOST}:{ROUTER_PORT}")
    print(f"It will forward requests to the llama.cpp server at {LLAMA_CPP_URL}")
    uvicorn.run(app, host=HOST, port=ROUTER_PORT)