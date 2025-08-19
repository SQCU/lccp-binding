# server.py
import os
import json
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama, LogitsProcessor
from typing import Iterator, List, Dict, Any

# --- Pydantic Models for API ---
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    stream: bool = False
    include_logits: bool = False

# --- FastAPI App Initialization ---
app = FastAPI(title="Llama.cpp Server", description="...")

# --- Load the Model ---
MODEL_DIR = "models"
MODEL_FILE = "gemma-3-1b-pt-q4_0.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

print("Loading model...")
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=True)
print("Model loaded successfully.")

# --- LogitsProcessor Definitions ---

class CaptureLogitsProcessor(LogitsProcessor):
    """Captures the logits of the very last token generated."""
    def __init__(self):
        self.last_logits = None

    def __call__(self, input_ids: List[int], logits: List[float]) -> List[float]:
        # --- NOTE THE UPDATED SIGNATURE ---
        # We accept input_ids but ignore it, as we only need the logits.
        self.last_logits = np.array(logits)
        return logits

class PenalizeWordProcessor(LogitsProcessor):
    """A custom sampler that penalizes a specific token to make it less likely."""
    def __init__(self, penalty: float = 2.0):
        self.word_to_penalize = " the" # Note the leading space
        self.token_id_to_penalize = llm.tokenize(self.word_to_penalize.encode('utf-8'))[0]
        self.penalty = penalty
        print(f"[Custom Sampler] Initialized. Penalizing token ID {self.token_id_to_penalize} ('{self.word_to_penalize}')")

    def __call__(self, input_ids: List[int], logits: List[float]) -> List[float]:
        # --- NOTE THE UPDATED SIGNATURE ---
        logits[self.token_id_to_penalize] -= self.penalty
        return logits

# --- Helper Functions ---

def generate_json_chunks(iterator: Iterator[dict]) -> Iterator[str]:
    for chunk in iterator:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

def process_logits(final_logits: np.ndarray) -> List[Dict[str, Any]]:
    """Calculates and decodes top-k logits from the captured numpy array."""
    if final_logits is None: return []
    probs = np.exp(final_logits - np.max(final_logits)) / np.sum(np.exp(final_logits - np.max(final_logits)))
    top_k_indices = np.argsort(probs)[-24:][::-1]
    return [{
        "token_id": int(token_id),
        "token": llm.detokenize([int(token_id)]).decode('utf-8', 'ignore'),
        "probability": float(probs[token_id])
    } for token_id in top_k_indices]

# --- API Endpoint ---

@app.post("/v1/completions")
async def get_completion(request: CompletionRequest):
    """Handles requests for model completions."""
    if request.stream:
        # Streaming logic
        result_iterator = llm(
            request.prompt, max_tokens=request.max_tokens, temperature=request.temperature, stream=True
        )
        return StreamingResponse(generate_json_chunks(result_iterator), media_type="application/x-ndjson")
    else:
        # Non-streaming logic
        logit_capturer = CaptureLogitsProcessor() if request.include_logits else None
        
        # ACTIVATE YOUR CUSTOM SAMPLERS HERE
        # For now, let's just use the capturer when requested.
        processors = []
        if logit_capturer:
            processors.append(logit_capturer)
        # To test the custom sampler, uncomment the line below:
        # processors.append(PenalizeWordProcessor())

        result = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
            logits_processor=processors if processors else None,
        )
        if logit_capturer:
            result["top_logits"] = process_logits(logit_capturer.last_logits)
        return result

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)