Home
Search
New Chat
Bodega Hub


















































Unload All Models
http://localhost:44468/v1/chat/completions
View Docs



Bodega Inference Engine
A high-performance local inference server running on localhost:44468, providing OpenAI-compatible endpoints for language models, embeddings, audio transcription, and image generation.

Overview
Bodega Inference Engine delivers enterprise-grade AI inference capabilities directly on your machine. Built specifically for Apple Silicon, it provides a seamless OpenAI-compatible API while keeping your data private and your costs predictable.

Server Address: http://localhost:44468

Key Capabilities:

Language model inference with streaming support
Text embeddings generation
Audio transcription via Whisper models
Image generation and editing (coming soon)
Multimodal language models (coming soon)
Dynamic model loading and management
Built-in tool system for web search, calculations, and more
Table of Contents
Getting Started
Core Endpoints
Model Management
Model Discovery
Built-in Tools System
Advanced Features
Monitoring and Health
Best Practices
Getting Started
Quick Start
Start the server and load your first model:

# Load a language model
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "context_length": 32768
  }'

# Make your first inference request
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "current",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

Python Quick Start
import requests

BASE_URL = "http://localhost:44468"

# Load a model
response = requests.post(
    f"{BASE_URL}/v1/admin/load-model",
    json={
        "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
        "model_type": "lm",
        "context_length": 32768
    }
)
print(response.json())

# Chat completion
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
)
print(response.json()["choices"][0]["message"]["content"])

Core Endpoints
Chat Completions
Generate text responses using loaded language models. Fully compatible with OpenAI's chat completions API.

Endpoint: POST /v1/chat/completions

Basic Request
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "current",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }'

Python Example
import requests

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])

Streaming Response
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "current",
    "messages": [
      {"role": "user", "content": "Write a short story about AI."}
    ],
    "stream": true
  }'

Python Streaming Example
import requests
import json

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {"role": "user", "content": "Write a short story about AI."}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)

Request Parameters
Parameter	Type	Default	Description
model	string	required	Model identifier (use "current" for loaded model)
messages	array	required	Array of message objects with role and content
max_tokens	integer	null	Maximum tokens to generate
temperature	float	0.7	Sampling temperature (0.0 to 2.0)
top_p	float	1.0	Nucleus sampling parameter
stream	boolean	false	Enable streaming responses
tools	array	null	Available tools for function calling
tool_choice	string/object	"auto"	Control tool selection behavior
response_format	object	null	Specify output format (e.g., JSON)
presence_penalty	float	0.0	Penalize new tokens based on presence
frequency_penalty	float	0.0	Penalize new tokens based on frequency
stop	string/array	null	Stop sequences
seed	integer	null	Random seed for reproducibility
Response Format
{
  "id": "chatcmpl_1234567890",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "current",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 120,
    "total_tokens": 135
  }
}

Image Generation
Generate images from text prompts. Coming soon.

Endpoint: POST /v1/images/generations

Status: This endpoint will be available in upcoming releases.

Image Editing
Edit existing images with text instructions. Coming soon.

Endpoint: POST /v1/images/edits

Status: This endpoint will be available in upcoming releases.

Model Management
The Bodega Inference Engine supports dynamic model loading, allowing you to switch between models without restarting the server.

Load Model
Load a new model into memory. If a model is already loaded, it will be automatically unloaded first.

Endpoint: POST /v1/admin/load-model

Load Language Model
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "context_length": 32768,
    "max_concurrency": 1
  }'

Python Example
import requests

response = requests.post(
    "http://localhost:44468/v1/admin/load-model",
    json={
        "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
        "model_type": "lm",
        "context_length": 32768,
        "max_concurrency": 1,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3"
    }
)

print(response.json())

Request Parameters
Parameter	Type	Default	Description
model_path	string	required	HuggingFace repo ID or local path
model_type	string	"lm"	Model type: "lm", "multimodal", "embeddings", "whisper", "image-generation", "image-edit"
context_length	integer	32768	Maximum context length for language models
max_concurrency	integer	1	Maximum concurrent requests
queue_timeout	integer	300	Request timeout in seconds
queue_size	integer	100	Maximum queue size
quantize	integer	8	Quantization level for Flux models (4, 8, or 16)
config_name	string	null	Config for image models (flux-schnell, flux-dev, etc.)
lora_paths	array	null	Paths to LoRA adapters
lora_scales	array	null	Scale factors for LoRA adapters
disable_auto_resize	boolean	false	Disable auto-resize for vision models
enable_auto_tool_choice	boolean	false	Enable automatic tool selection
tool_call_parser	string	null	Parser for tool calls (qwen3, harmony, etc.)
reasoning_parser	string	null	Parser for reasoning content (qwen3, harmony, etc.)
trust_remote_code	boolean	false	Allow custom model code execution
chat_template_file	string	null	Path to custom chat template
Available Parsers
Both tool_call_parser and reasoning_parser support:

qwen3 - Qwen3 model formats
glm4_moe - GLM4 MoE model formats
qwen3_coder - Qwen3 Coder formats (tool calls only)
qwen3_moe - Qwen3 MoE formats
qwen3_next - Qwen3 Next formats
qwen3_vl - Qwen3 Vision-Language formats
harmony - Unified parser for Harmony/GPT-OSS models
minimax_m2 - MiniMax model formats
Unload Model
Unload the currently loaded model to free memory.

Endpoint: POST /v1/admin/unload-model

curl -X POST http://localhost:44468/v1/admin/unload-model

Python Example
response = requests.post("http://localhost:44468/v1/admin/unload-model")
print(response.json())

Get Current Model
Retrieve information about the currently loaded model.

Endpoint: GET /v1/admin/current-model

curl http://localhost:44468/v1/admin/current-model

Python Example
response = requests.get("http://localhost:44468/v1/admin/current-model")
model_info = response.json()

if model_info["loaded"]:
    print(f"Model: {model_info['model_info']['model_path']}")
    print(f"Type: {model_info['model_info']['model_type']}")
else:
    print("No model currently loaded")

Response Format
{
  "loaded": true,
  "model_info": {
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "context_length": 32768,
    "max_concurrency": 1
  }
}

Model Discovery
Discover, download, and manage models from HuggingFace.

List Available Models
List all models available in your local HuggingFace cache.

Endpoint: GET /v1/models

curl http://localhost:44468/v1/models

With Hub Verification
Verify download completeness against HuggingFace API (slower but more accurate):

curl "http://localhost:44468/v1/models?verify_with_hub=true"

Python Example
response = requests.get("http://localhost:44468/v1/models")
models = response.json()["data"]

for model in models:
    print(f"ID: {model['id']}")
    print(f"  Created: {model['created']}")
    print(f"  Size: {model['size_gb']:.2f} GB")
    print(f"  Download: {model['download_percentage']:.1f}%")
    print()

Response Format
{
  "object": "list",
  "data": [
    {
      "id": "SRSWTI/bodega-raptor-8b-mxfp4",
      "object": "model",
      "created": 1704067200,
      "owned_by": "SRSWTI",
      "size_gb": 4.8,
      "download_percentage": 100.0,
      "is_complete": true
    }
  ]
}

Download Model
Proactively download a model to your local cache for faster loading later.

Endpoint: POST /v1/admin/download-model

Download Language Model
curl -X POST http://localhost:44468/v1/admin/download-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4"
  }'

Python Example
# Download by model path
response = requests.post(
    "http://localhost:44468/v1/admin/download-model",
    json={"model_path": "SRSWTI/bodega-raptor-8b-mxfp4"}
)
print(response.json())


### Download Model with Progress

Download a model with real-time progress updates via Server-Sent Events.

**Endpoint:** `POST /v1/admin/download-model-stream`

```bash
curl -X POST http://localhost:44468/v1/admin/download-model-stream \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4"
  }'

Python Example with Progress
import requests
import json

response = requests.post(
    "http://localhost:44468/v1/admin/download-model-stream",
    json={"model_path": "SRSWTI/bodega-raptor-8b-mxfp4"},
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                progress = json.loads(data)
                print(f"Status: {progress['status']}")
                print(f"Progress: {progress.get('progress', 0)}%")
                if 'current_file' in progress:
                    print(f"File: {progress['current_file']}")
                print()

Progress Event Format
{
  "status": "downloading",
  "progress": 45.5,
  "current_file": "model.safetensors",
  "file_progress": 67.3,
  "downloaded_files": 3,
  "total_files": 8,
  "message": "Downloading model files..."
}

Delete Model
Remove a model from your local cache to free disk space.

Endpoint: DELETE /v1/models/{model_id}

curl -X DELETE "http://localhost:44468/v1/models/SRSWTI/bodega-raptor-8b-mxfp4"

Python Example
model_id = "SRSWTI/bodega-raptor-8b-mxfp4"
response = requests.delete(f"http://localhost:44468/v1/models/{model_id}")
print(response.json())

Response Format
{
  "id": "SRSWTI/bodega-raptor-8b-mxfp4",
  "object": "model",
  "deleted": true
}

Built-in Tools System
The Bodega Inference Engine includes a powerful built-in tools system that enables language models to perform web searches, calculations, and more.

Understanding Tools
Tools extend your language model's capabilities by allowing it to:

Search the web for current information
Perform mathematical calculations
Look up news articles
Search academic papers
And more
Important: You can use the built-in parsers to automatically extract tool calls, or parse the model's output yourself if you prefer custom handling.

How Tool Calling Works
The Bodega Inference Engine provides two approaches to tool calling:

Option 1: Using Built-in Parsers (Recommended)
Load your model with a tool call parser to automatically extract and format tool calls:

# Load model with parser
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/axe-turbo-31b",
    "model_type": "lm",
    "tool_call_parser": "qwen3",
    "reasoning_parser": "qwen3"
  }'

Then make requests with tools:

import requests
import json

# Get available tools
tools_response = requests.get("http://localhost:44468/v1/tools")
tools = tools_response.json()["data"]

# Chat with tools
response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {"role": "user", "content": "What's the latest news about AI?"}
        ],
        "tools": tools,
        "tool_choice": "auto"
    }
)

result = response.json()
choice = result["choices"][0]

# Check if model wants to call a tool
if choice.get("finish_reason") == "tool_calls":
    for tool_call in choice["message"]["tool_calls"]:
        tool_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        # Execute the tool
        tool_result = requests.post(
            "http://localhost:44468/v1/tools/execute",
            json={
                "tool_name": tool_name,
                "arguments": arguments
            }
        ).json()
        
        print(f"Tool {tool_name} result: {tool_result['result']}")
        
        # Send result back to model
        response = requests.post(
            "http://localhost:44468/v1/chat/completions",
            json={
                "model": "current",
                "messages": [
                    {"role": "user", "content": "What's the latest news about AI?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result["result"])
                    }
                ]
            }
        )
        
        final_response = response.json()
        print(final_response["choices"][0]["message"]["content"])

Option 2: Custom Parsing
If you prefer to parse tool calls yourself, load the model without a parser:

curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm"
  }'

The model's raw output will be returned, and you can parse it according to your needs:

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {"role": "user", "content": "Search for AI news"}
        ]
    }
)

# Parse the raw response yourself
raw_content = response.json()["choices"][0]["message"]["content"]
# Your custom parsing logic here

Advanced Features
Reasoning Models
Some models support explicit reasoning/thinking processes. Configure parsers to extract this content.

Load Model with Reasoning Parser
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-90m",
    "model_type": "lm",
    "reasoning_parser": "qwen3"
  }'

Request with Thinking Enabled
response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {"role": "user", "content": "Solve this logic puzzle: ..."}
        ],
        "chat_template_kwargs": {
            "enable_thinking": True
        }
    }
)

result = response.json()
message = result["choices"][0]["message"]

# Reasoning content (thinking process)
if "reasoning_content" in message:
    print("Thinking:", message["reasoning_content"])

# Final answer
print("Answer:", message["content"])

JSON Mode
Force the model to output valid JSON.

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "List three colors with their hex codes."
            }
        ],
        "response_format": {"type": "json_object"}
    }
)

import json
result = json.loads(response.json()["choices"][0]["message"]["content"])
print(result)

Context Length Configuration
Optimize memory usage by configuring context length:

curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-90m",
    "model_type": "lm",
    "context_length": 8192
  }'

Smaller context lengths reduce memory usage, while larger ones support longer conversations.

Custom Chat Templates
Override the model's default chat template:

curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-90m",
    "model_type": "lm",
    "chat_template_file": "/path/to/custom_template.jinja"
  }'

Monitoring and Health
Health Check
Check if the server is running and if a model is loaded.

Endpoint: GET /health

curl http://localhost:44468/health

Response (Healthy)
{
  "status": "ok",
  "model_id": "SRSWTI/bodega-raptor-90m",
  "model_status": "initialized"
}

Response (No Model)
{
  "status": "unhealthy",
  "model_id": null,
  "model_status": "uninitialized"
}

Queue Statistics
Monitor request queue performance.

Endpoint: GET /v1/queue/stats

curl http://localhost:44468/v1/queue/stats

Python Example
response = requests.get("http://localhost:44468/v1/queue/stats")
stats = response.json()["queue_stats"]

print(f"Queue size: {stats.get('queue_size', 0)}")
print(f"Active requests: {stats.get('active_requests', 0)}")

Best Practices
Model Selection
Choose models based on your requirements:

**Fastest variants **

SRSWTI/bodega-raptor-90m - Extreme edge variant. Sub-100M params for amazing tool calling support and reasoning onedge
SRSWTI/bodega-raptor-0.9b - 900M params. Runs with 400+ tok/s. Handles document classification, query reformulation, and lightweight reasoning at the edge
SRSWTI/axe-turbo-1b - 1B params, 150 tok/s, sub-50ms first token. Edge-first architecture for agentic coding
Balanced Performance:

SRSWTI/bodega-raptor-1b-reasoning-opus4.5-distill - Distilled from Claude Opus 4.5 reasoning patterns. Enhanced logical deduction chains
SRSWTI/bodega-vertex-4b - 4B params. Optimized for structured data processing
SRSWTI/bodega-raptor-8b-mxfp4 - Balanced power/performance for laptops. Excellent for general-purpose tasks
Multimodal and Agentic:

SRSWTI/bodega-solomon-9b - Multimodal and best for agentic coding workflows
High-Capacity Models:

SRSWTI/bodega-raptor-15b-6bit - Better raptor variant with enhanced capabilities
SRSWTI/bodega-centenario-21b-mxfp4 - Production workhorse. 21B params optimized for sustained inference workloads. Behemoth in all categories
SRSWTI/blackbird-she-doesnt-refuse-21b - Uncensored 21B variant for unrestricted generation
SRSWTI/axe-turbo-31b - High-capacity variant for desktop/server deployments with agentic coding focus
Flagship Intelligence:

SRSWTI/deepseek-v3.2-speciale-distilled-raptor-32b-4bit - DeepSeek V3.2 distilled to 32B with Raptor reasoning. Exceptional math/code generation in 5-7GB footprint. 120 tok/s on M1 Max
Memory Management
Use appropriate context lengths for your use case
Unload models when switching to free memory
Monitor queue stats to prevent overload
Use quantized models (4-bit) for better memory efficiency
Performance Optimization
Set max_concurrency: 1 for single-user scenarios
Use streaming for long responses to improve perceived latency
Enable appropriate parsers only when needed
Cache embeddings for frequently used texts
Error Handling
Always check response status codes:

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={...}
)

if response.status_code == 503:
    print("No model loaded. Load a model first.")
elif response.status_code == 400:
    print("Invalid request parameters.")
elif response.status_code == 200:
    result = response.json()
    # Process result
else:
    print(f"Error: {response.status_code}")

Security Considerations
The server runs on localhost:44468 only - not accessible from external networks
No authentication is required for local access
Do not expose this port to the internet without proper security measures
Trust remote code only from verified sources
Coming Soon
The following features are in active development:

Image Generation Image Editing Multimodal Language Models

Stay tuned for updates as these features become available.

Support
For issues, questions, or feature requests, please refer to the project documentation or contact the development team.

Server Address: http://localhost:44468

Default Port: 44468 (fixed)

Documentation last updated: January 2026 _