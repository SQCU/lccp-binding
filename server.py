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
    include_bigram_logits: bool = False # <-- ADDED: Flag for the new feature

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
        # Streaming logic (unchanged)
        result_iterator = llm(
            request.prompt, max_tokens=request.max_tokens, temperature=request.temperature, stream=True
        )
        return StreamingResponse(generate_json_chunks(result_iterator), media_type="application/x-ndjson")
    else:
        # --- MODIFIED Non-streaming logic ---
        full_context_capturer = CaptureLogitsProcessor() if request.include_logits or request.include_bigram_logits else None
        
        processors = []
        if full_context_capturer:
            processors.append(full_context_capturer)
        # processors.append(PenalizeWordProcessor())

        # --- 1. Full Context Evaluation ---
        result = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
            logits_processor=processors if processors else None,
        )

        if request.include_logits and full_context_capturer:
            result["full_context_logits"] = process_logits(full_context_capturer.last_logits)
        
        # --- 2. Bigram Context Evaluation (Corrected Logic) ---
        if request.include_bigram_logits:
            print("\n[Bigram] Performing second evaluation for the last token.")
            
            prompt_tokens = llm.tokenize(request.prompt.encode('utf-8'))
            
            if prompt_tokens:
                last_token = prompt_tokens[-1]
                last_token_text = llm.detokenize([last_token]).decode('utf-8', 'ignore')
                print(f"[Bigram] Last token ID: {last_token}, Text: '{repr(last_token_text)}'")

                # Step A: CRITICAL - Reset model state to forget the full context
                llm.reset()
                
                # Step B: Create a NEW capturer for this isolated evaluation
                bigram_capturer = CaptureLogitsProcessor()
                
                # Step C: Perform a separate, 1-token generation to capture the bigram logits
                # We detokenize the last token back to a string to pass it as a prompt.
                # We don't care about the text output, only the side-effect on the capturer.
                _ = llm(
                    prompt=last_token_text,
                    max_tokens=1, # We must generate at least one token to trigger the processor
                    temperature=0.1, # Temperature is irrelevant as we only need the first set of logits
                    logits_processor=[bigram_capturer]
                )
                
                # Step D: Process the logits captured by the second processor
                result["bigram_context_logits"] = process_logits(bigram_capturer.last_logits)
                result["bigram_source_token"] = {
                    "token_id": last_token,
                    "token": last_token_text
                }

            else:
                print("[Bigram] Prompt was empty, skipping bigram evaluation.")
                result["bigram_context_logits"] = []
        
        # Reset the model state again after the bigram evaluation to ensure the next API call is clean.
        llm.reset()

        return result

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)