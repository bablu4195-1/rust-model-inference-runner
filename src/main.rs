//! # Rust Inference Model Runner

use anyhow::{bail, Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::{Parser, ValueEnum};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

// ─── Model type flag ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, ValueEnum)]
enum ModelType {
    /// LLaMA-family models (default): LLaMA 2/3, TinyLlama, Mistral, Phi, etc.
    Llama,
    /// Gemma-family models (Gemma 3 / Gemma 4 architecture): use with Gemma GGUF files
    Gemma,
}

// ─── Cloud provider selection ────────────────────────────────────────────────

#[derive(Clone, Debug, ValueEnum)]
enum CloudProvider {
    /// OpenAI API (gpt-3.5-turbo, gpt-4, etc.)
    Openai,
    /// Anthropic API (claude-3-haiku, claude-3-sonnet, etc.)
    Anthropic,
    /// Google Vertex AI / Gemini API
    Google,
    /// Azure OpenAI Service
    Azure,
    /// Ollama local/cloud inference
    Ollama,
}

// ─── CLI arguments ────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "rust-inference-model-runner")]
#[command(about = "Run LLM inference locally with GGUF models via Candle or use cloud APIs")]
struct Args {
    /// Hugging Face repo ID (e.g. bartowski/Llama-3.2-1B-Instruct-GGUF)
    #[arg(long)]
    hf_repo: Option<String>,

    /// GGUF filename within the repo
    #[arg(long)]
    hf_file: Option<String>,

    /// Local path to GGUF model file (overrides --hf-repo / --hf-file)
    #[arg(long, short)]
    model: Option<PathBuf>,

    /// Model architecture type
    #[arg(long, default_value = "llama")]
    model_type: ModelType,

    /// Use cloud provider instead of local model
    #[arg(long)]
    cloud: Option<CloudProvider>,

    /// API key for cloud provider (or set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    #[arg(long, env = "API_KEY")]
    api_key: Option<String>,

    /// Cloud model name (e.g., gpt-4, claude-3-sonnet-20240229, gemini-pro)
    #[arg(long)]
    cloud_model: Option<String>,

    /// Base URL for cloud API (for custom endpoints or Ollama)
    #[arg(long)]
    base_url: Option<String>,

    /// Prompt for single-turn inference
    #[arg(long, short)]
    prompt: Option<String>,

    /// Enable interactive chat mode
    #[arg(long, short)]
    chat: bool,

    /// Number of tokens to generate
    #[arg(long, short = 'n', default_value_t = 128)]
    sample_len: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Separate HF repo to download tokenizer.json from (useful when the GGUF repo
    /// doesn't bundle a tokenizer, e.g. use the base model repo)
    #[arg(long)]
    tokenizer_repo: Option<String>,

    /// Cache directory for downloaded models (default: ~/.cache)
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

// ─── Model enum ───────────────────────────────────────────────────────────────

/// Wraps either a LLaMA or Gemma quantized model behind a uniform interface.
enum Model {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Gemma(candle_transformers::models::quantized_gemma3::ModelWeights),
}

impl Model {
    /// Run one forward pass; `index_pos = 0` triggers full-context recomputation
    /// (no KV-cache reuse), which is correct for the sliding-window generation
    /// loop used below.
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Model::Llama(m) => m.forward(x, index_pos),
            Model::Gemma(m) => m.forward(x, index_pos),
        }
    }

    fn context_size(&self) -> usize {
        match self {
            // quantized_llama::MAX_SEQ_LEN = 4096
            Model::Llama(_) => candle_transformers::models::quantized_llama::MAX_SEQ_LEN,
            // quantized_gemma3::MAX_SEQ_LEN = 131072
            Model::Gemma(_) => candle_transformers::models::quantized_gemma3::MAX_SEQ_LEN,
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🧞‍♂️  Rust Inference Model Runner v{}", env!("CARGO_PKG_VERSION"));
    println!("══════════════════════════════════════");

    if let Err(e) = run(&args) {
        eprintln!("\n❌ Error: {e:?}");
        std::process::exit(1);
    }

    Ok(())
}

fn run(args: &Args) -> Result<()> {
    // Check if cloud mode is enabled
    if let Some(cloud_provider) = &args.cloud {
        return run_cloud(args, cloud_provider);
    }

    // Local model inference
    let model_path = resolve_model_path(args)?;
    println!("📦 Model: {}", model_path.display());

    println!("🧠 Loading model ({:?})...", args.model_type);
    let (mut model, tokenizer) = load_gguf_model(&model_path, &args.model_type, args)?;
    println!("✅ Model & tokenizer loaded.");

    if args.chat {
        chat_loop(&mut model, &tokenizer, args)?;
    } else {
        let prompt = args.prompt.as_deref().unwrap_or("Hello, how are you?");
        println!("\n📝 Prompt: {prompt}");
        println!("──────────────────────────────────────");
        let output = generate(&mut model, &tokenizer, prompt, args)?;
        println!("\n✅ Output:\n{output}");
        println!("──────────────────────────────────────");
    }

    Ok(())
}

// ─── Model resolution ────────────────────────────────────────────────────────

fn resolve_model_path(args: &Args) -> Result<PathBuf> {
    if let Some(path) = &args.model {
        if !path.exists() {
            bail!("Model file not found: {}", path.display());
        }
        return Ok(path.clone());
    }

    let repo = args.hf_repo.as_deref().context("--hf-repo required")?;
    let file = args.hf_file.as_deref().context("--hf-file required")?;

    let api = hf_hub::api::sync::Api::new()?;
    let repo_api = api.model(repo.to_string());
    let path = repo_api
        .get(file)
        .with_context(|| format!("Failed to download {repo}/{file}"))?;
    Ok(path)
}

// ─── Model loading ───────────────────────────────────────────────────────────

fn load_gguf_model(
    path: &PathBuf,
    model_type: &ModelType,
    args: &Args,
) -> Result<(Model, tokenizers::Tokenizer)> {
    let device = Device::Cpu;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open model file: {}", path.display()))?;

    // Parse the GGUF container (reads metadata + tensor index, not the weights yet)
    let ct = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {e}"))?;

    let model = match model_type {
        ModelType::Llama => {
            let weights =
                candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    ct, &mut file, &device,
                )
                .map_err(|e| anyhow::anyhow!("Failed to load LLaMA weights: {e}"))?;
            Model::Llama(weights)
        }
        ModelType::Gemma => {
            let weights =
                candle_transformers::models::quantized_gemma3::ModelWeights::from_gguf(
                    ct, &mut file, &device,
                )
                .map_err(|e| anyhow::anyhow!("Failed to load Gemma weights: {e}"))?;
            Model::Gemma(weights)
        }
    };

    let tokenizer = load_tokenizer(path, args)?;
    Ok((model, tokenizer))
}

fn load_tokenizer(model_path: &PathBuf, args: &Args) -> Result<tokenizers::Tokenizer> {
    // 1. Check alongside the .gguf (works for local files and already-cached HF downloads)
    if let Some(dir) = model_path.parent() {
        let tj = dir.join("tokenizer.json");
        if tj.exists() {
            return tokenizers::Tokenizer::from_file(&tj)
                .map_err(|e| anyhow::anyhow!("Bad tokenizer at {}: {e}", tj.display()));
        }
    }

    // 2. Try repos in order: --tokenizer-repo first, then --hf-repo as fallback
    let repos_to_try: Vec<&str> = [args.tokenizer_repo.as_deref(), args.hf_repo.as_deref()]
        .into_iter()
        .flatten()
        .collect();

    for repo in repos_to_try {
        println!("⬇️  Downloading tokenizer.json from {repo}...");
        let api = hf_hub::api::sync::Api::new()?;
        let repo_api = api.model(repo.to_string());
        match repo_api.get("tokenizer.json") {
            Ok(tj_path) => {
                return tokenizers::Tokenizer::from_file(&tj_path)
                    .map_err(|e| anyhow::anyhow!("Bad tokenizer at {}: {e}", tj_path.display()));
            }
            Err(e) => {
                eprintln!("⚠️  Could not fetch tokenizer.json from {repo}: {e}");
            }
        }
    }

    bail!(
        "No tokenizer.json found.\n\
         Options:\n\
         1. Place tokenizer.json next to your .gguf file\n\
         2. Use --hf-repo so it can be downloaded automatically\n\
         3. Download manually: huggingface-cli download <repo> tokenizer.json"
    );
}

// ─── Generation ──────────────────────────────────────────────────────────────

fn generate(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    args: &Args,
) -> Result<String> {
    let device = Device::Cpu;
    let mut logits_processor = LogitsProcessor::from_sampling(
        args.seed,
        Sampling::TopP {
            p: args.top_p,
            temperature: args.temperature,
        },
    );

    let mut tokens: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?
        .get_ids()
        .to_vec();

    let mut generated: Vec<u32> = Vec::new();
    let context_size = model.context_size();

    // Collect stop-token IDs: EOS variants for LLaMA and Gemma
    let stop_tokens: Vec<u32> = ["<eos>", "</s>", "<end_of_turn>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    'gen: for _ in 0..args.sample_len {
        // Clamp input to model's context window
        let context_len = tokens.len().min(context_size);
        let input = &tokens[tokens.len() - context_len..];
        let input_tensor = Tensor::new(input, &device)?.unsqueeze(0)?;

        // index_pos = 0: always do full-context recomputation (no KV-cache accumulation).
        // This is simpler and correct; KV-cache optimisation can be added later.
        let logits = model
            .forward(&input_tensor, 0)
            .map_err(|e| anyhow::anyhow!("Forward pass error: {e}"))?;
        let logits = logits.squeeze(0)?;

        let next_token = logits_processor
            .sample(&logits)
            .map_err(|e| anyhow::anyhow!("Sampling error: {e}"))?;

        // Stop on EOS / end-of-turn tokens
        if stop_tokens.contains(&next_token) {
            break 'gen;
        }

        tokens.push(next_token);
        generated.push(next_token);

        // Stream-print the decoded piece as it is generated
        if let Ok(piece) = tokenizer.decode(&[next_token], false) {
            if !piece.is_empty() {
                print!("{piece}");
                io::stdout().flush()?;
            }
        }
    }

    println!();
    tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
}

// ─── Interactive chat ────────────────────────────────────────────────────────

fn chat_loop(model: &mut Model, tokenizer: &tokenizers::Tokenizer, args: &Args) -> Result<()> {
    println!("\n💬 Interactive mode — type 'exit' or press Ctrl+C to quit");

    let stdin = io::stdin();
    let stdout = io::stdout();

    loop {
        print!("\n🧑 You: ");
        stdout.lock().flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("exit")
            || trimmed.eq_ignore_ascii_case("quit")
        {
            println!("\n👋 Goodbye!");
            break;
        }

        // Format using the correct chat template for the chosen model family
        let prompt = match args.model_type {
            ModelType::Llama => {
                // TinyLlama / LLaMA 3 instruct template
                format!("<|user|>\n{trimmed}\n<|assistant|>\n")
            }
            ModelType::Gemma => {
                // Gemma 3 / Gemma 4 instruct template
                format!("<start_of_turn>user\n{trimmed}<end_of_turn>\n<start_of_turn>model\n")
            }
        };

        print!("🤖 Assistant: ");
        stdout.lock().flush()?;
        generate(model, tokenizer, &prompt, args)?;
    }

    Ok(())
}

// ─── Cloud API helpers ───────────────────────────────────────────────────────

/// Get API key from args or environment variables
fn get_api_key(args: &Args, provider: &CloudProvider) -> Result<String> {
    if let Some(key) = &args.api_key {
        return Ok(key.clone());
    }

    let env_var = match provider {
        CloudProvider::Openai => "OPENAI_API_KEY",
        CloudProvider::Anthropic => "ANTHROPIC_API_KEY",
        CloudProvider::Google => "GOOGLE_API_KEY",
        CloudProvider::Azure => "AZURE_OPENAI_API_KEY",
        CloudProvider::Ollama => "", // Ollama doesn't require API key by default
    };

    if env_var.is_empty() {
        Ok(String::new())
    } else {
        std::env::var(env_var).with_context(|| format!("{} not set. Use --api-key or set {}", env_var, env_var))
    }
}

/// Get base URL for cloud provider
fn get_base_url(args: &Args, provider: &CloudProvider) -> String {
    if let Some(url) = &args.base_url {
        return url.clone();
    }

    match provider {
        CloudProvider::Openai => "https://api.openai.com/v1".to_string(),
        CloudProvider::Anthropic => "https://api.anthropic.com/v1".to_string(),
        CloudProvider::Google => "https://generativelanguage.googleapis.com/v1".to_string(),
        CloudProvider::Azure => {
            // Azure requires custom endpoint, use placeholder
            "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT".to_string()
        }
        CloudProvider::Ollama => "http://localhost:11434".to_string(),
    }
}

/// Get default model name for provider
fn get_default_model(provider: &CloudProvider) -> &'static str {
    match provider {
        CloudProvider::Openai => "gpt-3.5-turbo",
        CloudProvider::Anthropic => "claude-3-haiku-20240307",
        CloudProvider::Google => "gemini-pro",
        CloudProvider::Azure => "gpt-35-turbo",
        CloudProvider::Ollama => "llama3.2",
    }
}

/// Run inference using cloud API
fn run_cloud(args: &Args, provider: &CloudProvider) -> Result<()> {
    let api_key = get_api_key(args, provider)?;
    let base_url = get_base_url(args, provider);
    let model_name = args.cloud_model.as_deref().unwrap_or_else(|| get_default_model(provider));

    println!("☁️  Cloud Provider: {:?}", provider);
    println!("🔗 Base URL: {}", base_url);
    println!("🤖 Model: {}", model_name);

    if args.chat {
        cloud_chat_loop(args, provider, &api_key, &base_url, model_name)?;
    } else {
        let prompt = args.prompt.as_deref().unwrap_or("Hello, how are you?");
        println!("\n📝 Prompt: {prompt}");
        println!("──────────────────────────────────────");
        let output = cloud_generate(args, provider, &api_key, &base_url, model_name, prompt)?;
        println!("\n✅ Output:\n{output}");
        println!("──────────────────────────────────────");
    }

    Ok(())
}

/// Generate single response from cloud API
fn cloud_generate(
    args: &Args,
    provider: &CloudProvider,
    api_key: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
) -> Result<String> {
    match provider {
        CloudProvider::Openai => openai_generate(api_key, base_url, model, prompt, args.temperature),
        CloudProvider::Anthropic => anthropic_generate(api_key, base_url, model, prompt, args.temperature),
        CloudProvider::Google => google_generate(api_key, base_url, model, prompt, args.temperature),
        CloudProvider::Azure => azure_generate(api_key, base_url, model, prompt, args.temperature),
        CloudProvider::Ollama => ollama_generate(api_key, base_url, model, prompt, args.temperature),
    }
}

/// OpenAI API generation
fn openai_generate(
    api_key: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
    temperature: f64,
) -> Result<String> {
    use serde_json::json;

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/chat/completions", base_url);

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1024
        }))
        .send()?
        .error_for_status()?;

    let json: serde_json::Value = response.json()?;
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .context("Invalid response from OpenAI API")?;

    Ok(content.to_string())
}

/// Anthropic API generation
fn anthropic_generate(
    api_key: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
    temperature: f64,
) -> Result<String> {
    use serde_json::json;

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/messages", base_url);

    let response = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }))
        .send()?
        .error_for_status()?;

    let json: serde_json::Value = response.json()?;
    let content = json["content"][0]["text"]
        .as_str()
        .context("Invalid response from Anthropic API")?;

    Ok(content.to_string())
}

/// Google Gemini API generation
fn google_generate(
    api_key: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
    temperature: f64,
) -> Result<String> {
    use serde_json::json;

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/models/{}:generateContent?key={}", base_url, model, api_key);

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&json!({
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1024
            }
        }))
        .send()?
        .error_for_status()?;

    let json: serde_json::Value = response.json()?;
    let content = json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .context("Invalid response from Google API")?;

    Ok(content.to_string())
}

/// Azure OpenAI API generation
fn azure_generate(
    api_key: &str,
    base_url: &str,
    _model: &str,
    prompt: &str,
    temperature: f64,
) -> Result<String> {
    use serde_json::json;

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/chat/completions?api-version=2023-05-15", base_url);

    let response = client
        .post(&url)
        .header("api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&json!({
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1024
        }))
        .send()?
        .error_for_status()?;

    let json: serde_json::Value = response.json()?;
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .context("Invalid response from Azure API")?;

    Ok(content.to_string())
}

/// Ollama API generation (local or remote)
fn ollama_generate(
    _api_key: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
    temperature: f64,
) -> Result<String> {
    use serde_json::json;

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/api/generate", base_url);

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": temperature
            }
        }))
        .send()?
        .error_for_status()?;

    let json: serde_json::Value = response.json()?;
    let content = json["response"]
        .as_str()
        .context("Invalid response from Ollama API")?;

    Ok(content.to_string())
}

/// Interactive chat with cloud APIs
fn cloud_chat_loop(
    args: &Args,
    provider: &CloudProvider,
    api_key: &str,
    base_url: &str,
    model: &str,
) -> Result<()> {
    println!("\n💬 Interactive cloud mode — type 'exit' or press Ctrl+C to quit");

    let stdin = io::stdin();
    let stdout = io::stdout();

    loop {
        print!("\n🧑 You: ");
        stdout.lock().flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("exit")
            || trimmed.eq_ignore_ascii_case("quit")
        {
            println!("\n👋 Goodbye!");
            break;
        }

        print!("🤖 Assistant: ");
        stdout.lock().flush()?;
        let output = cloud_generate(args, provider, api_key, base_url, model, trimmed)?;
        println!("{}", output);
    }

    Ok(())
}
