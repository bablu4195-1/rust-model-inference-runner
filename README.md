//! # Rust Inference Model Runner
//!
//! A lightweight CLI for running LLM inference locally using [Candle](https://github.com/huggingface/candle) — HuggingFace's ML framework for Rust, **or via cloud APIs**.
//!
//! ## Features
//!
//! - **GGUF model support** — Quantized models (Q4_K_M, Q5_K_M, Q8_0, etc.)
//! - **Hugging Face Hub** — One-shot download and cache
//! - **Local model files** — Point to any `.gguf` on disk
//! - **Cloud API support** — OpenAI, Anthropic, Google, Azure, Ollama
//! - **Single-turn** — `--prompt "your prompt"`
//! - **Interactive chat** -- `--chat` mode
//! - **Configurable** — temperature, top-p, seed, context window, tokens
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build
//! cargo build --release
//! ```
//!
//! ## Local Model Inference
//!
//! ### Single-turn inference with a small model
//! ```bash
//! cargo run --release -- \
//!  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
//!  --hf-file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
//!  --prompt "Explain quantum computing in one sentence."
//! ```
//!
//! ### Interactive chat
//! ```bash
//! cargo run --release -- \
//!  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
//!  --hf-file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
//!  --chat
//! ```
//!
//! ### Use a local model
//! ```bash
//! cargo run --release -- \
//!  --model ~/models/my-model.Q4_K_M.gguf \
//!  --prompt "Hello!"
//! ```
//!
//! ## Cloud API Inference
//!
//! ### OpenAI
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --release -- --cloud openai --prompt "Explain quantum computing"
//! cargo run --release -- --cloud openai --cloud-model gpt-4 --chat
//! ```
//!
//! ### Anthropic (Claude)
//! ```bash
//! export ANTHROPIC_API_KEY="your-api-key"
//! cargo run --release -- --cloud anthropic --prompt "Hello!"
//! cargo run --release -- --cloud anthropic --cloud-model claude-3-sonnet-20240229 --chat
//! ```
//!
//! ### Google Gemini
//! ```bash
//! export GOOGLE_API_KEY="your-api-key"
//! cargo run --release -- --cloud google --prompt "What's the weather?"
//! ```
//!
//! ### Ollama (local or remote)
//! ```bash
//! # With local Ollama server
//! cargo run --release -- --cloud ollama --prompt "Hi there!"
//! cargo run --release -- --cloud ollama --cloud-model llama3.2 --chat
//!
//! # With custom Ollama endpoint
//! cargo run --release -- --cloud ollama --base-url http://192.168.1.100:11434 --prompt "Hello"
//! ```
//!
//! ### Azure OpenAI
//! ```bash
//! export AZURE_OPENAI_API_KEY="your-api-key"
//! cargo run --release -- \
//!   --cloud azure \
//!   --base-url "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT" \
//!   --prompt "Test message"
//! ```
//!
//! ## CLI Arguments
//!
//! | Flag | Description | Default |
//! |------|-------------|---------|
//! | `--hf-repo` | HuggingFace repo ID | — |
//! | `--hf-file` | GGUF filename in repo | — |
//! | `--model, -m` | Local GGUF file path | — |
//! | `--cloud` | Cloud provider (openai, anthropic, google, azure, ollama) | — |
//! | `--api-key` | API key for cloud provider | From env var |
//! | `--cloud-model` | Cloud model name | Provider default |
//! | `--base-url` | Custom API base URL | Provider default |
//! | `--prompt, -p` | Single-turn prompt | "Hello, how are you?" |
//! | `--chat, -c` | Interactive chat mode | false |
//! | `-n` | Tokens to generate | 256 |
//! | `--temperature` | Sampling temperature (0 = greedy) | 0.8 |
//! | `--top-p` | Nucleus sampling threshold | 0.95 |
//! | `--seed` | Random seed | 42 |
//! | `--context-size` | Max context (tokens) | 2048 |
//! | `--cache-dir` | Model download cache dir | ~/.cache |
//! | `--verbose, -v` | Debug logging | false |
//!
//! ## Recommended Models
//!
//! ### Local Models
//! | Model | Size | Quality | Command |
//! |---|---|---|---|
//! | [TinyLlama 1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | ~600MB | Fast, basic | Q4_K_M |
//! | [Phi-2 2.7B](https://huggingface.co/TheBloke/phi-2-GGUF) | ~1.6GB | Excellent for size | Q4_K_M |
//! | [Mistral 7B](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) | ~4.1GB | Great quality | Q4_K_M |
//! | [Llama 3 8B](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) | ~4.7GB | State-of-the-art small | Q4_K_M |
//!
//! ### Cloud Models
//! | Provider | Model | Best For |
//! |---|---|---|
//! | OpenAI | gpt-4-turbo | General purpose, reasoning |
//! | OpenAI | gpt-3.5-turbo | Fast, cost-effective |
//! | Anthropic | claude-3-opus | Complex tasks, analysis |
//! | Anthropic | claude-3-haiku | Speed, efficiency |
//! | Google | gemini-pro | Multi-modal tasks |
//! | Ollama | llama3.2 | Local privacy-focused |
//!
//! ## Architecture
//!
//! ```
//! src/
//! ├── main.rs    # Entry point, argument parsing, orchestration, cloud APIs
//! ├── model.rs   # GGUF loading, forward pass, generation
//! └── cli.rs     # Tokenization, text generation, interactive mode
//! ```
//!
//! ## Notes
//!
//! - **CPU only** for now — GGUF quantization makes this fast enough for small models
//! - **Apple Silicon** benefits from optimized Metal kernels in `candle-core`
//! - First run downloads the model (~minutes); subsequent runs use the cache
//! - Cloud providers require valid API keys (set via env vars or `--api-key`)
//!
//! ## Environment Variables
//!
//! - `OPENAI_API_KEY` - OpenAI API key
//! - `ANTHROPIC_API_KEY` - Anthropic API key
//! - `GOOGLE_API_KEY` - Google API key
//! - `AZURE_OPENAI_API_KEY` - Azure OpenAI API key
//!
//! ## License
//!
//! MIT