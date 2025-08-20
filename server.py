# server.py
import os
import json
import uvicorn
import asyncio
import uuid
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Coroutine, List, Dict
from dataclasses import dataclass
import aiohttp

# --- Configuration ---
MODEL_FILE = "gemma-3-1b-pt-q4_0.gguf"
MODEL_PATH = os.path.abspath(os.path.join("models", MODEL_FILE))

MAX_BATCH_SIZE = 8
BATCH_TIMEOUT = 0.05
ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 8080
ENGINE_URL = f"http://{ENGINE_HOST}:{ENGINE_PORT}"

# --- Pydantic Models for API ---
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8

@dataclass
class BatchedRequest:
    uid: str
    request: CompletionRequest
    future: asyncio.Future

# Global variable to hold the engine subprocess
engine_process = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the lifecycle of the native llama.cpp server."""
    global engine_process
    
    executable_name = "server.exe" if sys.platform == "win32" else "server"
    engine_path = os.path.abspath(os.path.join("engine", executable_name))
    
    if not os.path.exists(engine_path):
        print(f"\nFATAL: Engine executable not found at '{engine_path}'")
        print("Please run the bootstrap script first: python bootstrap.py\n")
        sys.exit(1)

    print("Starting native llama.cpp server engine...")
    engine_command = [
        engine_path,
        "-m", MODEL_PATH,
        "-c", "2048", # Context size
        "--port", str(ENGINE_PORT),
        "--host", ENGINE_HOST,
        "-b", "512", # Batch size
        "-ngl", "99" # Number of GPU layers
    ]
    
    engine_process = await asyncio.create_subprocess_exec(
        *engine_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # --- Wait for the server to be ready ---
    # We can do this by trying to connect to it.
    is_ready = False
    for _ in range(20): # Try for 10 seconds
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ENGINE_URL}/health") as resp:
                    if resp.status == 200:
                        print("Native engine is healthy and ready.")
                        is_ready = True
                        break
        except aiohttp.ClientConnectorError:
            await asyncio.sleep(0.5) # Wait and retry
            
    if not is_ready:
        print("\nFATAL: Native engine failed to start in time.")
        engine_process.terminate()
        await engine_process.wait()
        sys.exit(1)
        
    yield # The application is now running
    
    print("Shutting down native llama.cpp server engine...")
    engine_process.terminate()
    await engine_process.wait()
    print("Engine shut down gracefully.")

app = FastAPI(title="Llama.cpp Relay Server", lifespan=lifespan)

# --- Batch Manager ---
class BatchManager:
    def __init__(self):
        self.requests_queue = asyncio.Queue()

    async def add_request(self, request: CompletionRequest) -> Coroutine:
        future = asyncio.Future()
        await self.requests_queue.put(
            BatchedRequest(uid=str(uuid.uuid4()), request=request, future=future)
        )
        return await future

    async def process_requests_loop(self):
        print("Batch processor loop started.")
        while True:
            first_request = await self.requests_queue.get()
            batch = [first_request]
            start_time = asyncio.get_event_loop().time()
            while (len(batch) < MAX_BATCH_SIZE and (asyncio.get_event_loop().time() - start_time) < BATCH_TIMEOUT and not self.requests_queue.empty()):
                try:
                    batch.append(self.requests_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            prompts = [item.request.prompt for item in batch]
            print(f"Forwarding batch of size {len(prompts)} to native engine...")

            # --- Forward the entire batch to the C++ server ---
            payload = {
                "prompt": prompts,
                "n_predict": batch[0].request.max_tokens,
                "temperature": batch[0].request.temperature,
                # The C++ server uses slots to manage concurrent requests
                "id_slot": -1 # Let the server manage slots
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{ENGINE_URL}/completion", json=payload) as resp:
                        if resp.status == 200:
                            results = await resp.json()
                            # The C++ server response format is different
                            for i, item in enumerate(batch):
                                # This is a simplified mapping. The actual result
                                # structure might differ slightly.
                                content = results.get('content', '') if len(prompts) == 1 else results.get(f'content_{i}', '')
                                choice = {"text": content, "finish_reason": "stop"}
                                item.future.set_result(choice)
                        else:
                            error_text = await resp.text()
                            raise Exception(f"Engine returned error {resp.status}: {error_text}")
            except Exception as e:
                print(f"[ERROR] Failed to process batch: {e}")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)


# --- Instantiate and Register Batch Manager ---
batch_manager = BatchManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_manager.process_requests_loop())

@app.post("/v1/completions")
async def get_completion(request: CompletionRequest):
    try:
        result_choice = await batch_manager.add_request(request)
        # We can't easily get token counts from the relay, so we return dummy values
        return JSONResponse(content={
            "id": "cmpl-" + str(uuid.uuid4()), "object": "text_completion",
            "model": MODEL_FILE, "choices": [result_choice],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Uvicorn relay server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)