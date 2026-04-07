//! # Rust Inference Model Runner
//!
//! A lightweight CLI for running LLM inference locally using [Candle](https://github.com/huggingface/candle) — HuggingFace's ML framework for Rust.
//!
//! ## Features
//!
//! - **GGUF model support** — Quantized models (Q4_K_M, Q5_K_M, Q8_0, etc.)
//! - **Hugging Face Hub** — One-shot download and cache
//! - **Local model files** — Point to any `.gguf` on disk
//! - **Single-turn** — `--prompt "your prompt"`
//! - **Interactive chat** -- `--chat` mode
//! - **Configurable** — temperature, top-p, seed, context window, tokens
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build
//! cargo build --release
//!
//! # Single-turn inference with a small model
//! cargo run --release -- \
//!  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
//!  --hf-file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
//!  --prompt "Explain quantum computing in one sentence."
//!
//! # Interactive chat
//! cargo run --release -- \
//!  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
//!  --hf-file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
//!  --chat
//!
//! # Use a local model
//! cargo run --release -- \
//!  --model ~/models/my-model.Q4_K_M.gguf \
//!  --prompt "Hello!"
//! ```
//!
//! ## CLI Arguments
//!
//! | Flag | Description | Default |
//! |------|-------------|---------|
//! | `--hf-repo` | HuggingFace repo ID | — |
//! | `--hf-file` | GGUF filename in repo | — |
//! | `--model, -m` | Local GGUF file path | — |
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
//! | Model | Size | Quality | Command |
//! |---|---|---|---|
//! | [TinyLlama 1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | ~600MB | Fast, basic | Q4_K_M |
//! | [Phi-2 2.7B](https://huggingface.co/TheBloke/phi-2-GGUF) | ~1.6GB | Excellent for size | Q4_K_M |
//! | [Mistral 7B](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) | ~4.1GB | Great quality | Q4_K_M |
//! | [Llama 3 8B](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) | ~4.7GB | State-of-the-art small | Q4_K_M |
//!
//! ## Architecture
//!
//! ```
//! src/
//! ├── main.rs    # Entry point, argument parsing, orchestration
//! ├── model.rs   # GGUF loading, forward pass, generation
//! └── cli.rs     # Tokenization, text generation, interactive mode
//! ```
//!
//! ## Notes
//!
//! - **CPU only** for now — GGUF quantization makes this fast enough for small models
//! - **Apple Silicon** benefits from optimized Metal kernels in `candle-core`
//! - First run downloads the model (~minutes); subsequent runs use the cache
//!
//! ## License
//!
//! MIT