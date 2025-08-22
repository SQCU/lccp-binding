#concurrent_client.py
import asyncio
import aiohttp
import json
import time

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8080/v1/completions"
PROMPTS = [
    "The first prompt is about the history of Rome.",
    "Write a python function that calculates a fibonacci sequence.",
    "What is the capital of Mongolia? Explain its significance.",
    "The best thing about AI is its ability to learn from vast amounts of data.",
    "Summarize the plot of the movie 'The Matrix' in three sentences.",
    "Explain the concept of quantum entanglement to a five-year-old.",
]

async def make_request(session: aiohttp.ClientSession, prompt: str):
    """Sends a single request to the server and prints the response."""
    print(f"[-->] Sending prompt: '{prompt[:40]}...'")
    
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": 150
    }

    try:
        async with session.post(BASE_URL, headers=headers, data=json.dumps(data)) as response:
            response_json = await response.json()
            if response.status == 200:
                text = response_json.get("choices", [{}])[0].get("text", "[No text found]")
                print(f"[<-- OK] Response for '{prompt[:40]}...':\n{text.strip()}\n")
            else:
                print(f"[<-- ERR] Error for '{prompt[:40]}...': {response_json}")
    except aiohttp.ClientConnectorError as e:
        print(f"\n[FATAL] Connection Error: Could not connect to the server at {BASE_URL}. Is it running?")
        print(f"Details: {e}")
        # Terminate other tasks
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

async def main():
    """Sets up the client session and runs all requests concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, prompt) for prompt in PROMPTS]
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    print(f"--- Starting Concurrent Client: Sending {len(PROMPTS)} requests ---")
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"--- All requests completed in {end_time - start_time:.2f} seconds ---")