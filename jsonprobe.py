# jsonprobe.py
# uv run jsonprobe.py --logits
import requests
import json
import argparse

BASE_URL = "http://127.0.0.1:8000/v1/completions"

def print_top_logits(top_logits):
    """Formats and prints the top_logits data."""
    if not top_logits:
        print("[No logit data returned]")
        return
    
    print("\n--- Top 24 Next Token Predictions ---")
    print(f"{'Token':<20} | {'Probability':<15}")
    print("-" * 38)
    for item in top_logits:
        # Represent special characters like newline for better display
        token_str = repr(item['token'])[1:-1]
        prob_percent = f"{item['probability'] * 100:.2f}%"
        print(f"{token_str:<20} | {prob_percent:<15}")
    print("-" * 38)

def print_logit_table(title: str, logits_data: list, source_token_info: dict = None):
    """Formats and prints a table of top logits."""
    if not logits_data:
        print(f"\n--- {title} ---")
        print("[No logit data returned]")
        return
    
    header = f"--- {title} ---"
    if source_token_info:
        token_repr = repr(source_token_info['token'])[1:-1]
        header += f"\n(Prediction from token: '{token_repr}')"

    print(f"\n{header}")
    print(f"{'Token':<20} | {'Probability':<15}")
    print("-" * 38)
    for item in logits_data:
        token_str = repr(item['token'])[1:-1]
        prob_percent = f"{item['probability'] * 100:.2f}%"
        print(f"{token_str:<20} | {prob_percent:<15}")
    print("-" * 38)

def main():
    parser = argparse.ArgumentParser(description="Client for the Llama.cpp server")
    parser.add_argument("prompt", type=str, nargs="?", default="The best thing about AI is", help="The prompt to send.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming response.")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate.")
    parser.add_argument("--logits", action="store_true", help="Request and display top-24 logits for the full context.")
    parser.add_argument("--bigram", action="store_true", help="Also request and display top-24 logits for the last token (bigram model).")
    args = parser.parse_args()

    # If --bigram is used, we imply --logits to get both sets.
    request_logits = args.logits or args.bigram
    is_streaming = args.stream and not request_logits # Logits mode forces non-streaming

    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "stream": is_streaming,
        "include_logits": request_logits,
        "include_bigram_logits": args.bigram
    }

    print(f"Sending prompt: '{args.prompt}' (Stream: {is_streaming}, Logits: {request_logits}, Bigram: {args.bigram})")
    print("-" * 30)

    try:
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(data), stream=is_streaming)
        response.raise_for_status()

        if is_streaming:
            # (Streaming code is unchanged)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[len('data: '):]
                        if "[DONE]" in json_str: break
                        try:
                            chunk = json.loads(json_str)
                            print(chunk.get("choices", [{}])[0].get("text", ""), end='', flush=True)
                        except json.JSONDecodeError: pass
            print("\n" + "-" * 30 + "\nStream complete.")
        else:
            # Handle standard JSON response
            result = response.json()
            text = result.get("choices", [{}])[0].get("text", "")
            print("--- Generated Text ---")
            print(text)
            
            if "full_context_logits" in result:
                print_logit_table("Top 24 (Full Context)", result["full_context_logits"])

            if "bigram_context_logits" in result:
                print_logit_table("Top 24 (Bigram Context)", result["bigram_context_logits"], result.get("bigram_source_token"))

            print("\nRequest complete.")

    except requests.exceptions.RequestException as e:
        print(f"\n[Error] Could not connect to the server: {e}")

if __name__ == "__main__":
    main()