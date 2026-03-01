# Bodega Inference Engine

A high-performance local inference server running on localhost:44468, providing OpenAI-compatible endpoints for language models, embeddings, audio transcription, and image generation.

---

## Overview

Bodega Inference Engine delivers enterprise-grade AI inference capabilities directly on your machine. Built specifically for Apple Silicon, it provides a seamless OpenAI-compatible API while keeping your data private and your costs predictable.

**Server Address:** `http://localhost:44468`
**Architecure:** Multi-process isolated handler architecture prevents Metal memory leaks.

**Key Capabilities:**
- Language model inference with streaming support
- Text embeddings generation
- Audio transcription via Whisper models
- Image generation and editing (coming soon)
- Multimodal language models (coming soon)
- Dynamic model loading and management
- Built-in tool system for web search, calculations, and more

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Endpoints](#core-endpoints)
3. [Model Management](#model-management)
4. [Model Discovery](#model-discovery)
5. [Built-in Tools System](#built-in-tools-system)
6. [Advanced Features](#advanced-features)
7. [Monitoring and Health](#monitoring-and-health)
8. [Best Practices](#best-practices)

---

## Getting Started

### Quick Start (Single Model)

Start the server using CLI and load your first model:

```bash
# Start the server (in a separate terminal)
python -m bodega_mlx_engine.main launch --model srswti/bodega-raptor-8b-mxfp4

# Or dynamically load a language model via API later
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_id": "bodega-raptor-8b",
    "model_type": "lm",
    "context_length": 32768,
    "prompt_cache_size": 10
  }'

# Make your first inference request
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bodega-raptor-8b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Quick Start (Multi-Model Registry)

For advanced workflows, start the engine with a `config.yaml` to launch multiple models concurrently. Each model spins up in its own hardware-isolated child process.

**1. Create `config.yaml`:**
```yaml
server:
  host: "0.0.0.0"
  port: 44468

models:
  - model_id: "bodega-solomon-9b"
    model_type: "multimodal"
    model_path: "srswti/bodega-solomon-9b"
    max_concurrency: 1

  - model_id: "bodega-raptor-8b"
    model_type: "lm"
    model_path: "srswti/bodega-raptor-8b-mxfp4"
    prompt_cache_size: 10
```

<!-- **2. Launch the Engine:**
```bash
# Using CLI flag
python -m bodega_mlx_engine.main launch --config config.yaml

# Or using the environment variable (ideal for systemd/Docker setups)
export BODEGA_INFERENCE_CONFIG="/path/to/config.yaml"
python -m bodega_mlx_engine.main launch
``` -->

### Python Quick Start

```python
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
```

---

## Core Endpoints

### Chat Completions

Generate text responses using loaded language models. Fully compatible with OpenAI's chat completions API.

**Endpoint:** `POST /v1/chat/completions`

#### Basic Request

```bash
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
```

#### Python Example

```python
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
```

#### Streaming Response

```bash
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "current",
    "messages": [
      {"role": "user", "content": "Write a short story about AI."}
    ],
    "stream": true
  }'
```

#### Python Streaming Example

```python
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
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model identifier (use "current" for loaded model) |
| `messages` | array | required | Array of message objects with role and content |
| `max_tokens` | integer | null | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0 to 2.0) |
| `top_p` | float | 1.0 | Nucleus sampling parameter |
| `stream` | boolean | false | Enable streaming responses |
| `tools` | array | null | Available tools for function calling |
| `tool_choice` | string/object | "auto" | Control tool selection behavior |
| `response_format` | object | null | Specify output format (e.g., JSON) |
| `presence_penalty` | float | 0.0 | Penalize new tokens based on presence |
| `frequency_penalty` | float | 0.0 | Penalize new tokens based on frequency |
| `stop` | string/array | null | Stop sequences |
| `seed` | integer | null | Random seed for reproducibility |

#### Response Format

```json
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
```

---

### Structured Outputs (JSON Schema)

Force the language model to output data that perfectly strictly adheres to a predefined JSON schema structure you provide. This happens natively within the inference engine using outlines constraints.

**Endpoint:** `POST /v1/chat/completions`

#### Python JSON Schema Example

```python
import requests

schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "AddressExtractor",
        "schema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {
                            "type": "string",
                            "description": "2 letter abbreviation of the state"
                        },
                        "zip": {
                            "type": "string",
                            "description": "5 digit zip code"
                        }
                    },
                    "required": ["street", "city", "state", "zip"]
                }
            },
            "required": ["address"]
        }
    }
}

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": [
            {"role": "system", "content": "Extract the address from the user input into the specified JSON format."},
            {"role": "user", "content": "Please format this address: 1 Hacker Wy Menlo Park CA 94025"}
        ],
        "response_format": schema,
        "stream": False # Set to True for streaming JSON tokens
    }
)

# Returns: '{"address": {"street": "1 Hacker Wy", "city": "Menlo Park", "state": "CA", "zip": "94025"}}'
print(response.json()["choices"][0]["message"]["content"])
```

#### Streaming Output

Structured output also works dynamically if you pass `"stream": true`. The model will stream partial JSON tokens.

```python
# Assuming same schema/messages as above
response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "current",
        "messages": messages,
        "response_format": schema,
        "stream": True
    },
    stream=True
)

import json
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: ') and line[6:] != '[DONE]':
            chunk = json.loads(line[6:])
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                print(content, end="", flush=True)
```

---

### Multimodal Completions (Vision)

Pass images alongside your text prompts for models with vision capabilities (e.g. `bodega-solomon-9b`).

**Endpoint:** `POST /v1/chat/completions`

#### Basic Image Request
```bash
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bodega-solomon-9b",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image? Provide a detailed description."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
```

#### Python Local Base64 Example
```python
import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("document_scan.png")

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "bodega-solomon-9b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this scanned document."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

---

### Image Generation

Generate images from text prompts using MLX Flux (`mflux`).

**Endpoint:** `POST /v1/images/generations`

#### Image Generation Request
```bash
curl -X POST http://localhost:44468/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-image-generation-model",
    "prompt": "A highly detailed portrait of a tiny red dragon wearing a chef hat, pulling a fresh loaf of sourdough bread out of a medieval stone oven.",
    "size": "1024x1024",
    "guidance_scale": 3.5,
    "steps": 14,
    "seed": 42
  }'
```

**Response Format**
By default, the engine returns a standard OpenAI object containing the `b64_json` image dictionary rather than a URL:
```json
{
  "created": 1709428581,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

---

### Image Editing

Edit existing images with text instructions. Coming soon.

**Endpoint:** `POST /v1/images/edits`

**Status:** This endpoint will be available in upcoming releases.

---

### Document Indexing (RAG)

Bodega Inference Engine includes a fully self-contained Retrieval-Augmented Generation (RAG) pipeline for PDF documents. It uses a lightweight local embedding model (`all-MiniLM-L6-v2`) and FAISS for on-disk vector storage.

#### 1. Upload & Index PDF
Upload a PDF file to be chunked, embedded, and stored locally.

**Endpoint:** `POST /v1/rag/upload`

```bash
curl -X POST http://localhost:44468/v1/rag/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "file_id": "rag-c6cd8f10",
  "filename": "document.pdf",
  "num_chunks": 71,
  "status": "indexed"
}
```

#### 2. Query Indexed PDF
Query a previously-indexed document. The engine embeds your question, retrieves the most relevant passage chunks (via FAISS cosine-similarity), and automatically passes the context alongside your query to the active chat model to generate an answer.

**Endpoint:** `POST /v1/rag/query`

```bash
curl -X POST http://localhost:44468/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "rag-c6cd8f10",
    "query": "What is the main conclusion of this document?",
    "model": "current",
    "top_k": 5
  }'
```

**Note:** You can also add `"stream": true` to receive the answer in a Server-Sent Events stream, exactly like the standard `/v1/chat/completions` endpoint.

#### 3. List Indexed Documents

**Endpoint:** `GET /v1/rag/documents`

```bash
curl http://localhost:44468/v1/rag/documents
```

#### 4. Delete Indexed Document

**Endpoint:** `DELETE /v1/rag/documents/{file_id}`

```bash
curl -X DELETE http://localhost:44468/v1/rag/documents/rag-c6cd8f10
```

---

The Bodega Inference Engine supports a Multi-Model Registry. This allows you to dynamically spawn, route to, and unload process-isolated model handlers without disrupting the server.

### Load Model

Dynamically spawn a new handler process for a model. The model becomes immediately available in the registry for inference requests.

**Endpoint:** `POST /v1/admin/load-model`

#### Load Language Model

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_id": "bodega-raptor-8b",
    "model_type": "lm",
    "context_length": 32768,
    "max_concurrency": 1,
    "prompt_cache_size": 10
  }'
```

#### Load Image Generation Model

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "srswti/solomon",
    "model_id": "flux-local",
    "model_type": "image-generation",
    "config_name": "solomon",
    "quantize": 8
  }'
```

#### Python Example

```python
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
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | required | HuggingFace repo ID or local path |
| `model_id` | string | null | Alias used in API requests (defaults to path) |
| `model_type` | string | "lm" | Model type: "lm", "multimodal", "embeddings", "whisper", "image-generation", "image-edit" |
| `context_length` | integer | 32768 | Maximum context length for language models |
| `max_concurrency` | integer | 1 | Maximum concurrent requests |
| `queue_timeout` | integer | 300 | Request timeout in seconds |
| `queue_size` | integer | 100 | Maximum queue size |
| `quantize` | integer | 8 | Quantization level for Flux models (4, 8, or 16) |
| `config_name` | string | null | Config for image models (solomon, solomon-max, rehoboam, omri-4b, etc.) |
| `lora_paths` | array | null | Paths to LoRA adapters |
| `lora_scales` | array | null | Scale factors for LoRA adapters |
| `disable_auto_resize` | boolean | false | Disable auto-resize for vision models |
| `enable_auto_tool_choice` | boolean | false | Enable automatic tool selection |
| `tool_call_parser` | string | null | Parser for tool calls (qwen3, harmony, etc.) |
| `reasoning_parser` | string | null | Parser for reasoning content (qwen3, harmony, etc.) |
| `trust_remote_code` | boolean | false | Allow custom model code execution |
| `chat_template_file` | string | null | Path to custom chat template |

#### Available Parsers

Both `tool_call_parser` and `reasoning_parser` support:
- `qwen3` - Qwen3 model formats
- `glm4_moe` - GLM4 MoE model formats
- `qwen3_coder` - Qwen3 Coder formats (tool calls only)
- `qwen3_moe` - Qwen3 MoE formats
- `qwen3_next` - Qwen3 Next formats
- `qwen3_vl` - Qwen3 Vision-Language formats
- `harmony` - Unified parser for Harmony/GPT-OSS models
- `minimax_m2` - MiniMax model formats

---

### Unload Model

Gracefully shutdown a model's running subprocess and unregister it from the engine, instantly freeing up unified GPU/CPU memory.

**Endpoint:** `DELETE /v1/admin/unload-model/{model_id}`

```bash
curl -X DELETE http://localhost:44468/v1/admin/unload-model/bodega-raptor-8b
```

#### Python Example

```python
response = requests.delete("http://localhost:44468/v1/admin/unload-model/bodega-raptor-8b")
print(response.json())
```

---

### List Loaded Models & Memory Usage

Retrieve real-time operating system metrics (Metal Unified Memory + CPU RSS) for all dynamically loaded models running inside the engine.

**Endpoint:** `GET /v1/admin/loaded-models`

```bash
curl http://localhost:44468/v1/admin/loaded-models
```

#### Python Example

```python
response = requests.get("http://localhost:44468/v1/admin/loaded-models")
models = response.json().get("data", [])

if not models:
    print("No models currently loaded")
else:
    for model in models:
        print(f"[{model['status'].upper()}] {model['id']} - Pid: {model['pid']}")
        mem = model.get('memory', {})
        print(f"  └ Metal Active (GPU): {mem.get('metal_active_mb', 0):.1f} MB")
        print(f"  └ Python Overhead (CPU): {mem.get('rss_mb', 0):.1f} MB")
        print(f"  └ Total System Pool: {mem.get('total_mb', 0):.1f} MB\n")
```

#### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "id": "bodega-raptor-8b",
      "type": "lm",
      "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
      "context_length": 32768,
      "created_at": 1704067200,
      "status": "running",
      "pid": 83932,
      "memory": {
        "metal_active_mb": 4150.2,
        "metal_cache_mb": 0.0,
        "metal_peak_mb": 4150.2,
        "rss_mb": 408.2,
        "total_mb": 4558.4
      }
    }
  ],
  "total": 1
}
```

---

## Model Discovery

Discover, download, and manage models from HuggingFace.

### List Available Models

List all models available in your local HuggingFace cache.

**Endpoint:** `GET /v1/models`

```bash
curl http://localhost:44468/v1/models
```

#### With Hub Verification

Verify download completeness against HuggingFace API (slower but more accurate):

```bash
curl "http://localhost:44468/v1/models?verify_with_hub=true"
```

#### Python Example

```python
response = requests.get("http://localhost:44468/v1/models")
models = response.json()["data"]

for model in models:
    print(f"ID: {model['id']}")
    print(f"  Created: {model['created']}")
    print(f"  Size: {model['size_gb']:.2f} GB")
    print(f"  Download: {model['download_percentage']:.1f}%")
    print()
```

#### Response Format

```json
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
```

---

### Download Model

Proactively download a model to your local cache for faster loading later.

**Endpoint:** `POST /v1/admin/download-model`

#### Download Language Model

```bash
curl -X POST http://localhost:44468/v1/admin/download-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4"
  }'
```


#### Python Example

```python
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
```

#### Python Example with Progress

```python
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
```

#### Progress Event Format

```json
{
  "status": "downloading",
  "progress": 45.5,
  "current_file": "model.safetensors",
  "file_progress": 67.3,
  "downloaded_files": 3,
  "total_files": 8,
  "message": "Downloading model files..."
}
```

---

### Delete Model

Remove a model from your local cache to free disk space.

**Endpoint:** `DELETE /v1/models/{model_id}`

```bash
curl -X DELETE "http://localhost:44468/v1/models/SRSWTI/bodega-raptor-8b-mxfp4"
```

#### Python Example

```python
model_id = "SRSWTI/bodega-raptor-8b-mxfp4"
response = requests.delete(f"http://localhost:44468/v1/models/{model_id}")
print(response.json())
```

#### Response Format

```json
{
  "id": "SRSWTI/bodega-raptor-8b-mxfp4",
  "object": "model",
  "deleted": true
}
```

---

## Built-in Tools System

The Bodega Inference Engine includes a powerful built-in tools system that enables language models to perform web searches, calculations, and more.

### Understanding Tools

Tools extend your language model's capabilities by allowing it to:
- Search the web for current information
- Perform mathematical calculations
- Look up news articles
- Search academic papers
- And more

**Important:** You can use the built-in parsers to automatically extract tool calls, or parse the model's output yourself if you prefer custom handling.

### How Tool Calling Works

The Bodega Inference Engine provides two approaches to tool calling:

#### Option 1: Using Built-in Parsers (Recommended)

Load your model with a tool call parser to automatically extract and format tool calls:

```bash
# Load model with parser
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/axe-turbo-31b",
    "model_type": "lm",
    "tool_call_parser": "qwen3",
    "reasoning_parser": "qwen3"
  }'
```

Then make requests with tools:

```python
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
```

#### Option 2: Custom Parsing

If you prefer to parse tool calls yourself, load the model without a parser:

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm"
  }'
```

The model's raw output will be returned, and you can parse it according to your needs:

```python
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
```

---


## Advanced Features

### Reasoning Models

Some models support explicit reasoning/thinking processes. Configure parsers to extract this content.

#### Load Model with Reasoning Parser

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-90m",
    "model_type": "lm",
    "reasoning_parser": "qwen3"
  }'
```

#### Request with Thinking Enabled

```python
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
```

---

### JSON Mode

Force the model to output valid JSON.

```python
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
```

---

### Context Length Configuration

Optimize memory usage by configuring context length:

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-90m",
    "model_type": "lm",
    "context_length": 8192
  }'
```

Smaller context lengths reduce memory usage, while larger ones support longer conversations.

---

### Prompt Caching Configuration

Bodega uses dynamic prompt caching for extremely fast time-to-first-token on recurring sequences. The cache operates natively on MLX token indices.

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b",
    "model_type": "lm",
    "prompt_cache_size": 25
  }'
```

By setting `prompt_cache_size` > 0 (e.g. `25` slots), overlapping prefixes across subsequent calls bypass the matrix multiplication completely.

---

### Speculative Decoding

Speculative decoding is a technique that can significantly accelerate the generation speed of large models without sacrificing output quality.

**How it works under the hood:**
Instead of generating one token at a time with a massive "target" model (which is bottlenecked by loading the large model's weights into GPU memory over and over), the engine simultaneously runs a much smaller, faster "draft" model. 

1. **The Guess:** The small `draft_model` quickly guesses the next $N$ tokens (controlled by `num_draft_tokens`).
2. **The Verification:** The large target model evaluates all $N$ guessed tokens in a single parallel forward pass.
3. **The Result:** If the target model agrees with the draft model's guesses, it accepts them all instantly. This essentially yields multiple tokens in the time it takes the large model to generate just one token! 

**Try it out:**
For speculative decoding to work well, the draft model must share the same tokenizer as the target model (e.g., using a 1B param variant to accelerate an 8B param variant).

```bash
# Example: Accelerating Bodega Raptor 8B using the tiny Qwen 0.6B variant as a draft
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b",
    "model_type": "lm",
    "draft_model_path": "Qwen/Qwen3-0.6B-MLX-8bit",
    "num_draft_tokens": 4
  }'
```

Or in your `config.yaml`:
```yaml
models:
  - model_id: "raptor-fast"
    model_type: "lm"
    model_path: "srswti/bodega-raptor-8b-mxfp4"
    draft_model_path: "Qwen/Qwen3-0.6B-MLX-8bit"
    num_draft_tokens: 3
```

**What happens in the Response Payload?**
Because speculative decoding is a mathematically transparent optimization, the resulting OpenAI response format is exactly the same as normal. There are no proprietary metrics leaked into the response. The absolute number of `completion_tokens` reported will be identical—the only difference is that you receive the payload *much faster*.

Here is an example response demonstrating speculative decoding alongside `prompt_cache_size`:
```json
{
  "id": "chatcmpl_2fa419e...",
  "object": "chat.completion",
  "model": "raptor-fast",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "<think>\nOkay, the user wants a 3-sentence story... ",
        "role": "assistant"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 21,
    "total_tokens": 121,
    "completion_tokens": 100,
    "prompt_tokens_details": {
      "cached_tokens": 3
    }
  }
}
```

---

### Custom Chat Templates

Override the model's default chat template:

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-90m",
    "model_type": "lm",
    "chat_template_file": "/path/to/custom_template.jinja"
  }'
```

---

## Monitoring and Health

### Health Check

Check if the server is running and if a model is loaded.

**Endpoint:** `GET /health`

```bash
curl http://localhost:44468/health
```

#### Response (Healthy Multi-Model Registry)

```json
{
  "status": "ok",
  "model_id": "bodega-solomon-9b, bodega-raptor-8b",
  "model_status": "initialized (2 model(s))",
  "models_detail": [
    {
      "id": "bodega-solomon-9b",
      "type": "multimodal",
      "status": "running",
      "ram_usage_mb": 11645.8
    },
    {
      "id": "bodega-raptor-8b",
      "type": "lm",
      "status": "running",
      "ram_usage_mb": 4558.4
    }
  ]
}
```

#### Response (No Models Loaded)

```json
{
  "status": "unhealthy",
  "model_id": null,
  "model_status": "no_models"
}
```

---

### Queue Statistics

Monitor request queue performance.

**Endpoint:** `GET /v1/queue/stats`

```bash
curl http://localhost:44468/v1/queue/stats
```

#### Python Example

```python
response = requests.get("http://localhost:44468/v1/queue/stats")
stats = response.json()["queue_stats"]

print(f"Queue size: {stats.get('queue_size', 0)}")
print(f"Active requests: {stats.get('active_requests', 0)}")
```

---

## Best Practices

### Model Selection

Choose models based on your requirements:

**Fastest variants **
- `SRSWTI/bodega-raptor-90m` - Extreme edge variant. Sub-100M params for amazing tool calling support and reasoning onedge 
- `SRSWTI/bodega-raptor-0.9b` - 900M params. Runs with 400+ tok/s. Handles document classification, query reformulation, and lightweight reasoning at the edge
- `SRSWTI/axe-turbo-1b` - 1B params, 150 tok/s, sub-50ms first token. Edge-first architecture for agentic coding

**Balanced Performance:**
- `SRSWTI/bodega-raptor-1b-reasoning-opus4.5-distill` - Distilled from Claude Opus 4.5 reasoning patterns. Enhanced logical deduction chains
- `SRSWTI/bodega-vertex-4b` - 4B params. Optimized for structured data processing
- `SRSWTI/bodega-raptor-8b-mxfp4` - Balanced power/performance for laptops. Excellent for general-purpose tasks

**Multimodal and Agentic:**
- `SRSWTI/bodega-solomon-9b` - Multimodal and best for agentic coding workflows

**High-Capacity Models:**
- `SRSWTI/bodega-raptor-15b-6bit` - Better raptor variant with enhanced capabilities
- `SRSWTI/bodega-centenario-21b-mxfp4` - Production workhorse. 21B params optimized for sustained inference workloads. Behemoth in all categories
- `SRSWTI/blackbird-she-doesnt-refuse-21b` - Uncensored 21B variant for unrestricted generation
- `SRSWTI/axe-turbo-31b` - High-capacity variant for desktop/server deployments with agentic coding focus

**Flagship Intelligence:**
- `SRSWTI/deepseek-v3.2-speciale-distilled-raptor-32b-4bit` - DeepSeek V3.2 distilled to 32B with Raptor reasoning. Exceptional math/code generation in 5-7GB footprint. 120 tok/s on M1 Max
### Memory Management

1. Use appropriate context lengths for your use case
2. Unload models when switching to free memory
3. Monitor queue stats to prevent overload
4. Use quantized models (4-bit) for better memory efficiency

### Performance Optimization

1. Set `max_concurrency: 1` for single-user scenarios
2. Use streaming for long responses to improve perceived latency
3. Enable appropriate parsers only when needed
4. Cache embeddings for frequently used texts

### Error Handling

Always check response status codes:

```python
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
```

### Security Considerations

1. The server runs on localhost:44468 only - not accessible from external networks
2. No authentication is required for local access
3. Do not expose this port to the internet without proper security measures
4. Trust remote code only from verified sources

---

## Coming Soon

The following features are in active development:

**Image Generation**
**Image Editing**
**Multimodal Language Models**

Stay tuned for updates as these features become available.

---

## Support

For issues, questions, or feature requests, please refer to the project documentation or contact the development team.

**Server Address:** `http://localhost:44468`

**Default Port:** 44468 (fixed)

---

*Documentation last updated: January 2026*
