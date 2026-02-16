# Copyright © 2025 Apple Inc.

import time
from mlx_lm import batch_generate, load

# Specify the checkpoint
checkpoint = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)

# A batch of prompts
prompts = [
    "Write a story about Einstein.",
    "Why is the sky blue?",
    "What time is it?",
    "How tall is Mt Everest?",
]

# Apply the chat template and encode to tokens
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        add_generation_prompt=True,
    )
    for p in prompts
]

print(f"\n{'='*60}")
print("BATCH 1: Processing {len(prompts)} prompts")
print(f"{'='*60}\n")

# Set `verbose=True` to see generation statistics and max_kv_size for true batching
start_time = time.time()
result = batch_generate(
    model, 
    tokenizer, 
    prompts, 
    verbose=True,  # Enable to see stats
    return_prompt_caches=True,
    max_kv_size=8192,  # Allow larger KV cache for batching
)
batch_time = time.time() - start_time

print(f"\n{'='*60}")
print("RESULTS FROM BATCH 1:")
print(f"{'='*60}")
for i, text in enumerate(result.texts):
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Response: {text}\n")
print(f"Total batch processing time: {batch_time:.2f}s")
print(f"{'='*60}\n")

prompts = [
    "Could you summarize that?",
    "And what about the sea?",
    "Try again?",
    "And Mt Olympus?",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        add_generation_prompt=True,
    )
    for p in prompts
]

result = batch_generate(
    model, tokenizer, prompts, verbose=False, prompt_caches=result.caches
)
print(result.texts[-1])

