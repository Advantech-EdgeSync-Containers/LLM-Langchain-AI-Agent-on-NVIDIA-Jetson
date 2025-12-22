# LLM Langchain AI Agent on NVIDIA Jetson™

**Version:** 2.0
**Release Date:** November 2025
**Copyright:** © 2025 Advantech Corporation. All rights reserved.
>  Check our [Troubleshooting Wiki](https://github.com/Advantech-EdgeSync-Containers/GPU-Passthrough-on-NVIDIA-Jetson/wiki/Advantech-Containers'-Troubleshooting-Guide) for common issues and solutions.

## Overview
The LLM Langchain AI Agent on NVIDIA Jetson™ Image is a high-performance, modular AI chat solution for Jetson™ edge devices. It integrates Ollama with the Meta Llama 3.2 3B model for LLM inference, FastAPI-based Langchain middleware, and OpenWebUI. Supporting RAG, tool-augmented reasoning, conversational memory, and custom workflows, it features an AI agent with EdgeSync Device Library integration for natural language-driven control of peripherals and edge hardware. Optimized for Jetson™ acceleration, it enables real-time, context-aware edge AI applications.

## Host System Requirements

| Component | Version/Requirement |
|-----------|---------|
| **JetPack** | 6.x |
| **CUDA** | 12.6.68 |
| **cuDNN** | 9.3.0.75 |
| **TensorRT** | 10.3.0.30 |
| **OpenCV** | 4.8.0 |

* CUDA , CuDNN , TensorRT , OpenCV versions Depends on JetPack version 6.x
* Please refer to the [NVIDIA JetPack Documentation](https://developer.nvidia.com/embedded/jetpack) for more details on compatible versions.

## Key Features

| Feature | Description |
|--------|-------------|
| AI Agent with EdgeSync Device Library | Tool-enabled agent for interacting with edge peripherals (e.g., sensors, actuators) through natural language commands |
| Integrated OpenWebUI | Clean, user-friendly frontend for LLM chat interface |
| Meta Llama 3.2 3B Inference | Efficient on-device LLM via Ollama; minimal memory, high performance |
| Model Customization | Create or fine-tune models using `ollama create` |
| REST API Access | Simple local HTTP API for model interaction |
| Flexible Parameters | Adjust inference with `temperature`, `top_k`, `repeat_penalty`, etc. |
| Modelfile Customization | Configure model behavior with Docker-like `Modelfile` syntax |
| Prompt Templates | Supports formats like `chatml`, `llama`, and more |
| LangChain Integration | Multi-turn memory with `ConversationChain` support |
| FastAPI Middleware | Lightweight interface between OpenWebUI and LangChain |
| Offline Capability | Fully offline after container image setup; no internet required |

## Architecture
![ai-agent-llama.png](..%2Fdata%2Farchitectures%2Fai-agent-llama.png)

## Repository Structure
```
LLM-Langchain-AI-Agent-on-NVIDIA-Jetson/
├── .env                                      # Environment configuration
├── build.sh                                  # Build helper script
├── wise-bench.sh                             # Wise Bench script
├── docker-compose.yml                        # Docker Compose setup
├── README.md                                 # Overview
├── quantization-readme.md                    # Model quantization steps
├── other-AI-capabilities-readme.md           # Other AI capabilities supported by container image
├── llm-models-performance-notes-readme.md    # Performance notes of LLM Models
├── efficient-prompting-for-compact-models.md # Craft better prompts for small and quantized language models
├── customization-readme.md                   # Customization, optimization & configuration guide
├── .gitignore                                # Git ignore specific files
├── data/                                     # Data assets (diagrams, gifs, images)
│   ├── architectures/
│   │   └── ai-agent-llama.png                # LLaMA Agent architecture diagram
│   ├── gifs/
│   │   └── llama-ai-agent.gif                # LLaMA AI Agent demo GIF
│   └── images/
│       ├── ai-agent-llama-curl.png           # Example cURL request to LLaMA Agent
│       ├── fast-api-curl.png                 # FastAPI cURL example
│       ├── fast-api.png                      # FastAPI reference diagram
│       ├── ggml-repo.png                     # GGML repository reference
│       ├── gguf-convert.png                  # GGUF conversion guide
│       ├── hugging-face-token.png            # Hugging Face token setup
│       ├── kvcache-after.png                 # KV cache after optimization
│       ├── kvcache-before.png                # KV cache before optimization
│       ├── kvcache.png                       # KV cache illustration
│       ├── langchain-wise-bench.png          # LangChain Wise Bench diagram
│       ├── quantization.png                  # Quantization workflow
│       ├── quantize-help.png                 # Quantization helper guide
│       └── select-model-llm.png              # Model selection interface
└── langchain-agent-service/                  # Core LangChain Agent API service
    ├── app.py                                # Main LangChain-FastAPI app
    ├── llm_loader.py                         # LLM loader (Ollama, Llama, etc.)
    ├── requirements.txt                      # Python dependencies
    ├── agent_setup.py                        # Agent setup code
    ├── start_services.sh                     # Start script
    ├── edgesync_utils.py                     # Wrapper module for EdgeSync interactions
    └── tools.py                              # Defines tools
```

## Container Description

### Quick Information

`build.sh` will start following two containers:

| Container Name | Description |
|-----------|---------|
| LLM-Langchain-AI-Agent-on-NVIDIA-Jetson | Provides a hardware-accelerated development environment using various AI software components along with Meta Llama 3.2 3B, Ollama & Langchain-based AI agent that integrates the EdgeSync Device Library to interact with low-level edge hardware components |
| openweb-ui-service | Optional, provides UI which is accessible via browser for inferencing |

### LLM Langchain AI Agent on NVIDIA Jetson™ Container Highlights

This container leverages [**LangChain**](https://www.langchain.com/) as the core orchestration framework for building powerful, modular LLM applications directly on NVIDIA Jetson™ devices. It integrates with the local inference engine Ollama, enabling offline, edge-optimized AI workflows without relying on cloud services.

| Feature                   | Description |
|------------------------------|-----------------|
| Middleware Logic Engine | FastAPI-based LangChain server handles agent logic, tools, memory, and RAG pipelines. |
| LLM Integration          | Connects to On-device model (Meta Llama 3.2 3B) via Ollama. |
| EdgeSync Integration | Integration of EdgeSync Device Library to interact with low-level edge hardware components |
| RAG-Enabled              | Supports Retrieval-Augmented Generation using vector stores and document loaders. |
| Agent & Tool Support     | Easily define and run LangChain agents with tool integration (e.g., search, calculator). |
| Conversational Memory    | Includes support for memory modules like buffer, summary, or vector-based recall. |
| Streaming & Async Support | Real-time response streaming for chat UIs via FastAPI endpoints. |
| Offline-First            | All components run locally after model download—ensures low latency and data privacy. |
| Modular Architecture     | Plug-and-play design with support for custom chains, tools, and prompts. |
| Developer Friendly       | Exposes RESTful APIs; works with OpenWebUI, custom frontends, or CLI tools. |
| Hardware Accelerated     | Optimized for Jetson™ devices using quantized models and accelerated inference. |

### OpenWebUI Container Highlights

OpenWebUI serves as a clean and responsive frontend interface for interacting with LLMs via APIs like Ollama or OpenAI-compatible endpoints. When containerized, it provides a modular, portable, and easily deployable chat interface suitable for local or edge deployments.

| Feature                          | Description |
|----------------------------------|-------------|
| User-Friendly Interface      | Sleek, chat-style UI for real-time interaction. |
| OpenAI-Compatible Backend    | Works with Ollama, OpenAI, and similar APIs with minimal setup. |
| Container-Ready Design       | Lightweight and optimized for edge or cloud deployments. |
| Streaming Support            | Enables real-time response streaming for interactive UX. |
| Authentication & Access Control | Basic user management for secure access. |
| Offline Operation            | Runs fully offline with local backends like Ollama. |

## List of READMEs

| Module   | Link                | Description                     |
|----------|----------------------------|---------------------------------|
| Quick Start | [README](./README.md) | Overview of the container image   |
| Customization & optimization | [README](./customization-readme.md) | Steps to customize a model, configure environment, and optimize |
| Model Performances | [README](./llm-models-performance-notes-readme.md) | Performance stats of various LLM Models  |
| Other AI Capabilities  | [README](./other-AI-capabilities-readme.md) | Other AI capabilities supported by the container |
| Quantization  | [README](./quantization-readme.md) | Steps to quantize a model |


## Model Information  

This image uses Meta Llama 3.2 3B instead of 1B (to avoid accuracy issues in the LangChain agent) for inferencing. Here are the details about the model used:

| Item  | Description                |
|---------|----------------------------|
| Model source  | Ollama Model (llama3.2:3b)  |
| Model architecture | Llama |
| Model quantization | Q4_K_M   |
| Ollama command | ollama pull llama3.2:3b |
| Number of Parameters | ~3.21 B  |
| Model size | ~2 GB  |
| Default context size (unless changed using parameters) | 2048 |

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| Target Hardware | NVIDIA Jetson™ |
| GPU | NVIDIA® Ampere architecture with 1024 CUDA® cores |
| DLA Cores | 1 (Deep Learning Accelerator) |
| Memory | 4/8/16 GB shared GPU/CPU memory |
| JetPack Version | 6.0 |

## Software Components

The following software components are available in the base image:

| Component    | Version        | Description                        |
|--------------|----------------|------------------------------------|
| CUDA®        | 12.6.68        | GPU computing platform             |
| cuDNN        | 9.3.0.75       | Deep Neural Network library        |
| TensorRT™    | 10.3.0.30      | Inference optimizer and runtime    |
| PyTorch      | 2.0.0+nv23.02  | Deep learning framework            |
| TensorFlow   | 2.12.0         | Machine learning framework         |
| ONNX Runtime | 1.16.3         | Cross-platform inference engine    |
| VPI          | 3.2.4          | Vision Programming Interface       |
| Vulkan       | 1.3.204        | Graphics and compute API           |
| OpenCV       | 4.8.0          | Computer vision library with CUDA® |
| GStreamer    | 1.16.2         | Multimedia framework               |


The following software components/packages are provided further inside the container image:

| Component               | Version     | Description                                                                                                              |
|-------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------|
| Ollama                  | 0.5.7       | LLM inference engine                                                                                                     |
| LangChain               | 0.2.17      | Installed via PIP, framework to build LLM applications                                                                   |
| FastAPI                 | 0.115.12    | Installed via PIP, develop OpenAI-compatible APIs for serving LangChain                                                  |
| OpenWebUI               | 0.6.5       | Provided via separate OpenWebUI container for UI                                                                         |
| Meta Llama 3.2 3B       | N/A         | Pulled inside the container and persisted via docker volume                                                              |
| FAISS                   | 1.8.0.post1 | Vector store backend for enabling RAG with efficient similarity search                                                   |
| EdgeSync Device Library | 1.0.0       | EdgeSync is provided as part of the container image for low-level edge hardware components interaction with the AI Agent |

## Before You Start
Make sure you have SUSI installed before using AI Agent tools. Refer to the below link for SUSI installation.
- Navigate to the [SUSI Release Packages](https://github.com/ADVANTECH-Corp/SUSI/tree/master/ReleasePackage) and select the appropriate package based on your hardware:
    Example:
   For EPC-R7300 (ARM64, Ubuntu 20.04): `RISC/Standard/Linux/EPC/EPC-R7300/Ubuntu 20.04/ARM64` e.g [SUSI package](https://github.com/ADVANTECH-Corp/SUSI/tree/master/ReleasePackage/RISC/Standard/Linux/EPC/EPC-R7300/Ubuntu%2020.04/ARM64)

-    After downloading the appropriate package for your device, follow the [SUSI installation guide](https://ess-wiki.advantech.com.tw/view/SUSI#Installation).

- Ensure the following components are installed on your host system:
  - **Docker** (v28.1.1 or compatible)
  - **Docker Compose** (v2.39.1 or compatible)
  - **NVIDIA Container Toolkit** (v1.11.0 or compatible)
  - **NVIDIA Runtime** configured in Docker

## Quick Start

### Installation
```
# Clone the repository
git clone https://github.com/Advantech-EdgeSync-Containers/LLM-Langchain-AI-Agent-on-NVIDIA-Jetson.git
cd LLM-Langchain-on-NVIDIA-Jetson-AI-Agent

# Make the build script executable
chmod +x build.sh

# Launch the container
sudo ./build.sh
```

### Run Services

After installation succeeds, by default control lands inside the container. Run the following command to start services within the container.

```
# Under /workspace/langchain-agent-service, run this command
# Provide executable rights
chmod +x start_services.sh

# Start services
./start_services.sh
```
Allow some time for the OpenWebUI and LLM Langchain AI Agent on NVIDIA Jetson™ to settle and become healthy.

### AI Accelerator and Software Stack Verification (Optional)
```
# Verify AI Accelerator and Software Stack Inside Docker Container
# Under /workspace, run this command
# Provide executable rights
chmod +x wise-bench.sh

# To run Wise-bench
./wise-bench.sh
```

![langchain-wise-bench.png](..%2Fdata%2Fimages%2Flangchain-wise-bench.png)

Wise-bench logs are saved in `wise-bench.log` file under `/workspace`

### Check Installation Status
Exit from the container and run the following command to check the status of the containers:
```
sudo docker ps
```
Allow some time for containers to become healthy.

### UI Access
Access OpenWebUI via any browser using the URL given below. Create an account and perform a login:
```
http://localhost_or_Jetson_IP:3000
```
### Select Model
In case Ollama has multiple models available, choose from the list of models on the top-left of OpenWebUI after signing up/logging in successfully. As shown below. Select Meta Llama 3.2 3B:

![Select Model](..%2Fdata%2Fimages%2Fselect-model-llm.png)

### Quick Demonstration:

![Demo](..%2Fdata%2Fgifs%2Fllama-ai-agent.gif)

## Prompt Guidelines

This [README](./efficient-prompting-for-compact-models.md) provides essential prompt guidelines to help you get accurate and reliable outputs from small and quantized language models.

## Sample Prompts for Calling AI Agent Tools

| Tool Name               | Tool Function Name        | Description                                                          | Sample Prompt |
|-------------------------|-------------------------|----------------------------------------------------------------------|--------|
| Device Information Tool | device_info_tool        | Retrieve detailed motherboard and BIOS information                   | Get me device information |
| Device Voltage Tool     | device_voltage_tool    | Get real-time voltage readings from all onboard voltage sources      | Get me voltage details |
| Device Temperature Tool | device_temperature_tool | Fetch current temperature data from all onboard temperature sensors | Get me temperature information |
| Device Fan Tool         | device_fans_tool | Check real-time fan speed (RPM) readings for each onboard fan sensor | What’s the fan speed? |
| GPIO Pins Overview      | gpio_pins_overview | Get an overview of GPIO pins directions and logic levels             | Show me GPIO overview |
| Set GPIO Pin Tool       | gpio_set_tool | Set the output level of a GPIO pin                                   | Set GPIO pin GPIO5 to low  |
| Read GPIO Pin Tool      | gpio_read_tool | Read the input level of a GPIO pin                                   | Read GPIO pin GPIO3  |


## Ollama Logs and Troubleshooting

### Log Files

Once services have been started inside the container, the following log files are generated:

| Log File | Description |
|-----------|---------|
| ollama.pid | Provides process-id for the currently running Ollama service   |
| ollama.log | Provides Ollama service logs |
| uvicorn.log | Provides FastAPI-Langchain service logs |
| uvicorn.pid | Provides FastAPI-Langchain service pid |

### Troubleshoot

Here are quick commands/instructions to troubleshoot issues with the LLM Langchain AI Agent on NVIDIA Jetson™:

- View service logs within the container
  ```
  tail -f ollama.log # or
  tail -f uvicorn.log
  ```

- Check if the model is loaded using CPU or GPU or partially both (ideally, it should be 100% GPU loaded).
  ```
  ollama ps
  ```

- Kill & restart services within the container (check pid manually via `ps -eaf` or use pid stored in `ollama.pid` or `uvicorn.pid`)
  ```
  kill $(cat ollama.pid)
  kill $(cat uvicorn.pid)
  ./start_services.sh
  ```

  Confirm there is no Ollama & FastAPI service running using:
  ```
  ps -eaf
  ```

- Enable debug mode for the Ollama service (kill the existing Ollama service first).
  ```
  export OLLAMA_DEBUG=true
  ./start_services.sh
  ```

- In some cases, it has been found that if Ollama is also present at the host, it may give permission issues during pulling models within the container. Uninstalling host Ollama may solve the issue quickly. Follow this link for uninstallation steps - [Uninstall Ollama.](https://github.com/ollama/ollama/blob/main/docs/linux.md#uninstall)


## Best Practices and Recommendations

### Memory Management & Speed
- Ensure models are fully loaded into GPU memory for best results.
- Batch inference for better throughput
- Use stream processing for continuous data
- Offload unwanted models from GPU (use the Keep-Alive parameter for customizing this behavior).
- Enable Jetson™ Clocks for better inference speed
- Used quantized models to balance speed and accuracy
- Increase swap size if models loaded are large

### Ollama Model Behavior Corrections 
- Restart Ollama services
- Remove the model once and pull it again
- Check if the model is correctly loaded into the GPU or not; it should show loaded as 100% GPU. 
- Create a new Modelfile and set parameters like temperature, repeat penalty, system, etc., as needed to get expected results.

### LangChain Middleware Tuning
- Use asynchronous chains and streaming response handlers to reduce latency in FastAPI endpoints.
- For RAG pipelines, use efficient vector stores (e.g., FAISS with cosine or inner product) and pre-filter data when possible.
- Avoid long chain dependencies; break workflows into smaller composable components.
- Cache prompt templates and tool results when applicable to reduce unnecessary recomputation
- For agent-based flows, limit tool calls per loop to avoid runaway execution or high memory usage.
- Log intermediate steps (using LangChain’s callbacks) for better debugging and observability
- Use models with ≥3B parameters (e.g., Llama 3.2 3B or larger) for agent development to ensure better reasoning depth and tool usage reliability.

## REST API Access

[**Official Documentation**](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Ollama APIs
Ollama APIs are accessible on the default endpoint (unless modified). If needed, APIs could be called using code or curl as below:

Inference Request:
```
curl http://localhost_or_Jetson_IP:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```
Here stream mode could be changed to true/false as per the needs.

Response:
```
{
  "model": "llama3.2:3b",
  "created_at": "2023-08-04T08:52:19.385406455-07:00",
  "response": "<HERE_WILL_THE_RESPONSE>",
  "done": false
}
```
Sample Screenshot:

![ollama-curl.png](..%2Fdata%2Fimages%2Fai-agent-llama-curl.png)

For further API details, please refer to the official documentation of Ollama as mentioned on top.

### FastAPI (Serving LangChain)
Swagger docs could be accessed on the following endpoint:
```
http://localhost_or_Jetson_IP:8000/docs
```
Sample Screenshot:

![fast-api.png](..%2Fdata%2Fimages%2Ffast-api.png)

Inference Request:
```
curl -X 'POST' \
  'http://localhost_or_Jetson_IP:8000/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "string",
  "messages": [
    {
      "role": "user",
      "content": "Hi"
    }
  ],
  "stream": true
}'
```
Response:
```
data: {"id": "1f8b8036-0933-4449-ada1-686ac3393f5b", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "I"}, "index": 0, "finish_reason": null}]}
data: {"id": "6a777430-fde1-4e0a-a986-7cf4c786b146", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "'m"}, "index": 0, "finish_reason": null}]}
data: {"id": "5c38efb4-93be-4483-b618-ee8231e55e06", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " doing"}, "index": 0, "finish_reason": null}]}
data: {"id": "b3daadba-2677-429a-87ec-7bd309278e1c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " well"}, "index": 0, "finish_reason": null}]}
data: {"id": "0e7095a0-08bc-4048-a1cd-ef1c706917ee", "object": "chat.completion.chunk", "choices": [{"delta": {"content": ","}, "index": 0, "finish_reason": null}]}
data: {"id": "9fe20ab3-98ae-48ce-9b10-ee05e3677056", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " thanks"}, "index": 0, "finish_reason": null}]}
data: {"id": "76dc47b9-ecac-4ad9-8c52-87a1ea6e2f09", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " for"}, "index": 0, "finish_reason": null}]}
data: {"id": "4f0dc4c2-4c3a-4a29-8af8-446037a93271", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " asking"}, "index": 0, "finish_reason": null}]}
data: {"id": "ad0b47a7-1855-4a89-9357-9ffa795eaf4c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "!"}, "index": 0, "finish_reason": null}]}
data: {"id": "518e206a-d4c8-4570-8f54-e4b7f9ad05fa", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " I"}, "index": 0, "finish_reason": null}]}
data: {"id": "dc3a73cb-653e-4afd-9b64-b3c6cb8bb91e", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "'ve"}, "index": 0, "finish_reason": null}]}
data: {"id": "7b0fae1e-bf21-4763-8a6c-644738731797", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " been"}, "index": 0, "finish_reason": null}]}
data: {"id": "aa9b5f1c-791f-4209-8e9c-0bed8e8ef805", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " running"}, "index": 0, "finish_reason": null}]}
data: {"id": "14a3d3bd-74d9-4f39-8efd-d8208b631203", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " some"}, "index": 0, "finish_reason": null}]}
data: {"id": "b7dd3250-f45a-47f5-bc39-6947e703c5f2", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " simulations"}, "index": 0, "finish_reason": null}]}
data: {"id": "b90f071f-cb9d-412b-8718-a3deb11c7e14", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " in"}, "index": 0, "finish_reason": null}]}
data: {"id": "8e17b519-7249-41b0-bb32-1b44ad11ea34", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " my"}, "index": 0, "finish_reason": null}]}
data: {"id": "73002c3a-6aa3-4930-86bf-9c33987d7dbc", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " training"}, "index": 0, "finish_reason": null}]}
data: {"id": "82baa0c5-2f6b-4f82-bc43-3951237b74e6", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " data"}, "index": 0, "finish_reason": null}]}
data: {"id": "d26ccf4f-2a83-4b20-a420-64a6c5bab7cb", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " and"}, "index": 0, "finish_reason": null}]}
data: {"id": "19c24661-1f41-495c-86c1-9b9200425ba9", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " I"}, "index": 0, "finish_reason": null}]}
data: {"id": "52c49cae-9fa5-47f6-9613-742e265fed17", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " think"}, "index": 0, "finish_reason": null}]}
data: {"id": "c7a6c07c-8dfa-46b9-92a5-e66a6eebb061", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " I"}, "index": 0, "finish_reason": null}]}
data: {"id": "c779ed9d-a7d7-4d0f-9ebd-d95f114b9b84", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " can"}, "index": 0, "finish_reason": null}]}
data: {"id": "ce2d2b38-8d56-4862-8dfe-16edc23cde65", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " help"}, "index": 0, "finish_reason": null}]}
data: {"id": "083afd6c-be5b-445d-b0e0-034e71c7a0e4", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " you"}, "index": 0, "finish_reason": null}]}
data: {"id": "6843da75-3e2b-449c-b480-1d4e71ced105", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " with"}, "index": 0, "finish_reason": null}]}
data: {"id": "657f3820-8846-48c9-9d57-c7aea41abd41", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " something"}, "index": 0, "finish_reason": null}]}
data: {"id": "c8f51cdc-6599-4e60-87c4-474faced062f", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "."}, "index": 0, "finish_reason": null}]}
data: {"id": "38dc2ca7-544d-4591-b3c2-9e23e6848586", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " What"}, "index": 0, "finish_reason": null}]}
data: {"id": "1eb01bdd-7e65-4333-91bc-f7769b66cf82", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "'s"}, "index": 0, "finish_reason": null}]}
data: {"id": "88698b05-ee29-4953-ac6b-bad6cd87e9e8", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " on"}, "index": 0, "finish_reason": null}]}
data: {"id": "9284ad84-bf10-41d6-bd5a-61ac44c7f32c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " your"}, "index": 0, "finish_reason": null}]}
data: {"id": "3cbcfa9b-45e9-45c2-b1e2-e2acce900782", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " mind"}, "index": 0, "finish_reason": null}]}
data: {"id": "9b5a6076-df17-4958-9a76-6be6cbc1b91c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "?"}, "index": 0, "finish_reason": null}]}
data: [DONE]
```
Please note that the inference response will be in streaming mode only in the case of FastAPI.

Sample Screenshot:

![fast-api-curl.png](..%2Fdata%2Fimages%2Ffast-api-curl.png)

The same requests can also be made from Fast-API swagger docs.

## Known Limitations

1. Execution Time: The model, when inferred for the first time via OpenWebUI, takes longer time (within 10 seconds) as the model gets loaded into the GPU. 
2. RAM Utilization: RAM utilization for running this container image occupies approximately >5 GB RAM when running on NVIDIA® Orin™ NX – 8 GB. Running this image on Jetson™ Nano may require some additional steps, like increasing swap size or using lower quantization as suited. 
3. OpenWebUI Dependencies: When OpenWebUI is started for the first time, it installs a few dependencies that are then persisted in the associated Docker volume. Allow it some time to set up these dependencies. This is a one-time activity.


Copyright © 2025 Advantech Corporation. All rights reserved.
