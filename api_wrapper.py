import asyncio
import subprocess
import json
import time
import sys # <-- Import sys for platform checking
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager 

# --- 1. Configuration ---
LLAMACPP_DIR = Path("./llama.cpp")
DEFAULT_SERVER_EXECUTABLE_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"
DEFAULT_SERVER_PATH = Path("./llama.cpp/build/bin/Release") if sys.platform == "win32" else Path("./llama.cpp/build/bin")
DEFAULT_SERVER_PATH = DEFAULT_SERVER_PATH / DEFAULT_SERVER_EXECUTABLE_NAME

MODEL_PATH = Path("./models/gemma-3-1b-pt-q4_0.gguf")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
API_PORT = 8080
LLAMA_CPP_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
llama_server_process = None

MAX_BATCH_SIZE = 8
BATCH_TIMEOUT_SECONDS = 0.1 # This is now a more reliable timeout
CTX_SIZE_PER_REQUEST = 4096 # Assumed context size for a single request in a batch

# --- 2. Lifecycle Management (Unchanged) ---
class ServerLifecycleManager:
    # ... (This class remains exactly the same) ...
    """Manages the startup and shutdown of the llama.cpp server process."""
    def __init__(self, executable_path: Path, model_path: Path, host: str, port: int):
        if not executable_path.exists():
            raise FileNotFoundError(f"Server executable not found at: {executable_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.command = [
            str(executable_path),
            "-m", str(model_path),
            "--host", host,
            "--port", str(port),
            "--n-gpu-layers", "-1", # Use -1 for max possible, or adjust
            "--threads", "4",
            # --- ⭐ FIX 4: Add server arguments for batching support ---
            # Set number of parallel sequences to our max batch size
            "--parallel", str(MAX_BATCH_SIZE),
            "--batch-size", "32768",
            "--ubatch-size", "4096",
            # Set total context size to accommodate all parallel requests
            "--ctx-size", str(CTX_SIZE_PER_REQUEST * MAX_BATCH_SIZE),
            "--flash-attn",
            "--top-k", "0",
            "--top-p", "1.0",
            "--min-p", "0.02",
        ]
        self.process: Optional[subprocess.Popen] = None
        self.server_url = f"http://{host}:{port}"

    async def start(self):
        """Starts the server process and waits for it to be ready."""
        print("🚀 Starting llama.cpp server...")
        print(f"   Command: {' '.join(self.command)}")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            # --- ⭐ THE FIX IS HERE: Enforce line-buffering ---
            bufsize=1
        )
        
        loop = asyncio.get_running_loop()
        # The _wait_for_startup function is now guaranteed to not deadlock
        await loop.run_in_executor(None, self._wait_for_startup)

    def _wait_for_startup(self):
        """
        Monitors the server's stdout to confirm it has started AND loaded the model.
        """
        if not self.process or not self.process.stdout:
            return

        for line in iter(self.process.stdout.readline, ''):
            print(f"[llama.cpp server]: {line.strip()}")
            
            # --- ⭐ THE FIX IS HERE ⭐ ---
            # Wait for the "model loaded" message, which appears AFTER the server
            # is truly ready to accept and process inference requests.
            if "main: model loaded" in line:
                print("✅ llama.cpp server has loaded the model and is ready.")
                return # Now we can safely proceed
            
            # Also, check if the process died unexpectedly during startup
            if self.process.poll() is not None:
                print("❌ llama.cpp server failed to start or exited prematurely.")
                return

    async def stop(self):
        """Stops the server process gracefully."""
        if self.process:
            print("🛑 Stopping llama.cpp server...")
            self.process.terminate()
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self.process.wait(timeout=5)
                )
            except subprocess.TimeoutExpired:
                print("   Server did not terminate gracefully, forcing kill.")
                self.process.kill()
            print("   Server stopped.")


# --- 3. External Logit Processing (Unchanged) ---
class LogitProcessor:
    # ... (This class remains exactly the same) ...
    def __init__(self):
        self._queue = asyncio.Queue()
        self._worker_task = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._process_logits())
        print("📊 Logit processor started.")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            print("   Logit processor stopped.")

    async def submit_logits(self, logits_data: list):
        await self._queue.put(logits_data)

    async def _process_logits(self):
        while True:
            try:
                logits = await self._queue.get()
                print(f"[Logit Processor]: Received {len(logits)} logits for analysis.")
                top_token = logits[0]['token']
                top_prob = logits[0]['probability']
                print(f"    -> Top predicted token: {repr(top_token)} ({top_prob*100:.2f}%)")
                self._queue.task_done()
            except asyncio.CancelledError:
                break


# --- 4. Batched Inference Manager ---
# Pydantic model fix for arbitrary types
class RequestContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request_data: Dict[str, Any]
    future: asyncio.Future

class BatchingManager:
    # ... (This class remains mostly the same, just needs the client_session) ...
    def __init__(self, server_url: str, client_session: aiohttp.ClientSession):
        self.server_url = f"{server_url}/v1/completions"
        self.client_session = client_session
        self._queue = asyncio.Queue()
        self._worker_task = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._batch_processor())
        print("📦 Batching manager started.")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            print("   Batching manager stopped.")

    async def submit_request(self, request_data: Dict[str, Any]) -> Any:
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(RequestContext(request_data=request_data, future=future))
        return await future

    async def _batch_processor(self):
        """The core background task that creates and processes batches."""
        while True:
            try:
                # --- ⭐ FIX 2: Correct, efficient, and robust batch accumulation ---
                batch: List[RequestContext] = []
                
                # Wait for the first request indefinitely
                first_ctx = await self._queue.get()
                batch.append(first_ctx)

                # Then, wait for a short period to gather more requests
                while len(batch) < MAX_BATCH_SIZE:
                    try:
                        ctx = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=BATCH_TIMEOUT_SECONDS
                        )
                        batch.append(ctx)
                    except asyncio.TimeoutError:
                        # Timeout reached, stop collecting for this batch
                        break
                
                print(f"[Batch Processor]: Assembled batch of {len(batch)} request(s).")
                
                # Prepare payload based on batch size
                batch_payload = batch[0].request_data.copy()
                if len(batch) > 1:
                    batch_payload["prompt"] = [ctx.request_data["prompt"] for ctx in batch]
                else:
                    # For a single request, keep the prompt as a simple string
                    batch_payload["prompt"] = batch[0].request_data["prompt"]
                
                try:
                    async with self.client_session.post(self.server_url, json=batch_payload) as response:
                        response.raise_for_status()
                        results = await response.json()

                        # --- ⭐ FIX 3: Handle Polymorphic Response from llama.cpp ---
                        if len(batch) == 1:
                            # Single request -> server returns a single JSON object (dict)
                            if isinstance(results, dict):
                                batch[0].future.set_result(results)
                            else:
                                raise TypeError(f"Expected a dict for single request, but got {type(results)}")
                        else:
                            # Batch request -> server returns a JSON array (list)
                            if isinstance(results, list) and len(results) == len(batch):
                                for i, ctx in enumerate(batch):
                                    ctx.future.set_result(results[i])
                            else:
                                raise TypeError(f"Expected a list of {len(batch)} items for batch request, got {type(results)} of length {len(results) if isinstance(results, list) else 'N/A'}")

                except Exception as e:
                    # --- ⭐ FIX 1: Log the actual error here for visibility ---
                    print(f"❌ Error processing batch: {type(e).__name__}: {e}")
                    # Propagate the error to all waiting clients
                    for ctx in batch:
                        ctx.future.set_exception(e)

            except asyncio.CancelledError:
                break


# --- 5. FastAPI Application ---
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    stream: bool = False
    # --- ADD THIS LINE ---
    n_probs: int = Field(alias="n_probs", default=0) # Number of logit probabilities to return
    # The rest remains the same...
    include_logits: bool = Field(alias="include_logits", default=False)
    include_bigram_logits: bool = Field(alias="include_bigram_logits", default=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    global client_session, batching_manager, llama_server_process
    print("--- Application Lifespan: Startup ---")
    # Create the session now that the event loop is running
    client_session = aiohttp.ClientSession()
    # Now we can initialize the batching manager which depends on the session
    server_executable_path = DEFAULT_SERVER_PATH
    print(f"Starting llama.cpp server from: {server_executable_path}")
    server_manager = ServerLifecycleManager(server_executable_path, MODEL_PATH, SERVER_HOST, SERVER_PORT)
    batching_manager = BatchingManager(server_manager.server_url, client_session)

    await server_manager.start()
    await logit_processor.start()
    await batching_manager.start()

    # --- This is where the application runs ---
    yield
    # --- This code runs after the application is shut down ---
    
    print("\n--- Application Lifespan: Shutdown ---")
    if batching_manager: await batching_manager.stop()
    await logit_processor.stop()
    if client_session: await client_session.close()
    await server_manager.stop()


# Initialize components
# --- ⭐ FIX 3: Register the lifespan manager with the app ---
app = FastAPI(title="Llama.cpp API Wrapper", lifespan=lifespan)
# Mount a 'static' directory to serve our HTML, CSS, and JS
app.mount("/static", StaticFiles(directory="static"), name="static")
logit_processor = LogitProcessor()

# Defer these initializations until the lifespan startup event
client_session: Optional[aiohttp.ClientSession] = None
batching_manager: Optional[BatchingManager] = None

# Add a root endpoint to serve the HTML file
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/v1/completions")
async def handle_completion(request: CompletionRequest, raw_request: Request):
    request_data = await raw_request.json()

    if request.stream:
        print("⚡ Handling streaming request directly.")
        async def stream_generator():
            try:
                async with client_session.post(
                    f"{server_manager.server_url}/v1/completions",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=3600)
                ) as response:
                    async for line in response.content:
                        yield line
            except Exception as e:
                print(f"Error during streaming proxy: {e}")
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        print(f"📥 Queuing request for batching: '{request.prompt[:30]}...'")
        try:
            result = await batching_manager.submit_request(request_data)
            if "full_context_logits" in result:
                await logit_processor.submit_logits(result["full_context_logits"])
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

from fastapi import WebSocket, WebSocketDisconnect

def parse_sse_event(line: bytes) -> Optional[Dict[str, Any]]:
    """Parses a single line from an SSE stream."""
    line_str = line.decode('utf-8').strip()
    if line_str.startswith('data: '):
        json_str = line_str[len('data: '):]
        if "[DONE]" in json_str:
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    Handles a WebSocket connection for real-time generation with logits.
    """
    await websocket.accept()
    print("🤝 WebSocket connection established.")
    
    try:
        config_data = await websocket.receive_json()
        
        request_data = {
            "prompt": config_data.get("prompt", ""),
            "max_tokens": config_data.get("max_tokens", 200),
            "stream": True,
            "n_probs": 5,
        }

        print(f"⚡ Starting stream for prompt: '{request_data['prompt'][:50]}...'")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LLAMA_CPP_URL}/v1/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=3600)
            ) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    parsed_data = parse_sse_event(line)
                    if parsed_data:
                        #print(f"[RAW SSE DATA]: {parsed_data}")
                        
                        choice = parsed_data.get("choices", [{}])[0]
                        content = choice.get("text", "")
                        
                        # --- ⭐ THIS IS THE FIX ⭐ ---
                        logprobs = None
                        logprobs_obj = choice.get("logprobs")
                        if logprobs_obj and logprobs_obj.get("content"):
                            # The top logprobs are nested inside the first item of the 'content' list.
                            top_logprobs_list = logprobs_obj["content"][0].get("top_logprobs", [])
                            
                            # Reformat the data to match what the client expects:
                            # A simple list of {'token': str, 'probability': float}
                            logprobs = []
                            for item in top_logprobs_list:
                                # The raw data is in log-probability, convert to linear probability.
                                # Use math.exp() for this.
                                import math
                                logprobs.append({
                                    "token": item.get("token"),
                                    "probability": math.exp(item.get("logprob", -float('inf')))
                                })
                        
                        await websocket.send_json({
                            "content": content,
                            "logprobs": logprobs
                        })
        
        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("👋 WebSocket connection closed by client.")
    except Exception as e:
        print(f"❌ Error in WebSocket handler: {type(e).__name__}: {e}")
        await websocket.send_json({"error": f"{type(e).__name__}: {e}"})
    finally:
        print("🛑 WebSocket session ended.")

class TokenizeRequest(BaseModel):
    content: str

class DetokenizeRequest(BaseModel):
    tokens: List[int]

@app.post("/v1/tokenize")
async def handle_tokenize(request: TokenizeRequest):
    """Relays a tokenize request to the llama.cpp server."""
    try:
        async with client_session.post(
            f"{LLAMA_CPP_URL}/tokenize",
            json={"content": request.content}
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/v1/detokenize")
async def handle_detokenize(request: DetokenizeRequest):
    """Relays a detokenize request to the llama.cpp server."""
    try:
        async with client_session.post(
            f"{LLAMA_CPP_URL}/detokenize",
            json={"tokens": request.tokens}
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- NEW: Context Slicing Endpoint ---
class ContextProbeRequest(BaseModel):
    prompt: str
    n_probs: int = 1000
    slices: List[float] = Field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])


def format_logprobs(choice: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Extracts and formats logprobs from a choice object."""
    logprobs_obj = choice.get("logprobs")
    if logprobs_obj and logprobs_obj.get("content"):
        top_logprobs_list = logprobs_obj["content"][0].get("top_logprobs", [])
        formatted = []
        for item in top_logprobs_list:
            formatted.append({
                "token": item.get("token"),
                "probability": math.exp(item.get("logprob", -float('inf')))
            })
        return formatted
    return None

@app.post("/v1/probe_context_slices")
async def probe_context_slices(request: ContextProbeRequest):
    """
    Takes a prompt, slices it into trailing subsequences, and gets next-token
    logits for each slice in parallel.
    """
    print(f"🔎 Probing context slices for prompt: '{request.prompt[:50]}...'")
    try:
        # 1. Tokenize the full prompt
        async with client_session.post(f"{LLAMA_CPP_URL}/tokenize", json={"content": request.prompt}) as response:
            response.raise_for_status()
            token_data = await response.json()
            tokens = token_data.get("tokens", [])
            if not tokens:
                return JSONResponse(status_code=400, content={"error": "Prompt could not be tokenized."})

        # 2. Create token slices
        token_slices = []
        for factor in request.slices:
            if factor <= 0 or factor > 1: continue
            start_index = len(tokens) - int(len(tokens) * factor)
            token_slices.append(tokens[start_index:])

        # 3. Detokenize all slices concurrently
        detokenize_tasks = []
        for ts in token_slices:
            task = client_session.post(f"{LLAMA_CPP_URL}/detokenize", json={"tokens": ts})
            detokenize_tasks.append(task)
        detokenize_responses = await asyncio.gather(*detokenize_tasks)

        prompt_slices = []
        for resp in detokenize_responses:
            resp.raise_for_status()
            data = await resp.json()
            prompt_slices.append(data.get("content", ""))

        # 4. Submit all sliced prompts for inference concurrently via the batching manager
        inference_tasks = []
        for prompt_text in prompt_slices:
            payload = {
                "prompt": prompt_text,
                "max_tokens": 1,
                "n_probs": request.n_probs
            }
            task = batching_manager.submit_request(payload)
            inference_tasks.append(task)
        inference_results = await asyncio.gather(*inference_tasks)

        # 5. Format the final response
        final_results = []
        for i, raw_result in enumerate(inference_results):
            choice = raw_result.get("choices", [{}])[0]
            final_results.append({
                "slice_factor": request.slices[i],
                "prompt_slice": prompt_slices[i],
                "logprobs": format_logprobs(choice) or []
            })

        return JSONResponse(content=final_results)

    except Exception as e:
        print(f"❌ Error during context slice probe: {type(e).__name__}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- 6. Run the API Server ---
if __name__ == "__main__":
    print(f"🔥 Starting FastAPI wrapper server on http://127.0.0.1:{API_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=API_PORT)