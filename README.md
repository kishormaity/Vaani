# The Relay (Vaani)

**Production-Grade Citation-Grounded RAG System, Resilient LLM Gateway & Playwright MCP Integration**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-85%2F85%20passing-brightgreen.svg)](tests/)
[![Fast Inference](https://img.shields.io/badge/inference-Groq%20Cloud-orange.svg)](https://groq.com/)
[![MCP Protocol](https://img.shields.io/badge/MCP-Playwright%20stdio-purple.svg)](https://modelcontextprotocol.io/)

*The Relay* is a research-grade question-answering architecture that synthesizes verified academic evidence across foundational AI papers while enforcing strict factual grounding, application-layer rate backpressure, and multi-route query dispatching.

---

## Key Capabilities

1. **4-Step Cyclical RAG State Machine**:
   - `Analyze` $\to$ `Retrieve` $\to$ `Draft` $\to$ `Verify & Loop/Branch`.
   - Enforces exact inline provenance: `[arXiv:ID, p.X]`.
   - Single-attempt targeted feedback correction retry loop on ungrounded drafts.
   - Guaranteed safe refusal when evidence is missing or out-of-scope.

2. **Tri-Route Query Intent Classifier**:
   - **`CORPUS_RAG`**: FAISS dense vector search ($k=3$ or $k=5$) + Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) targeting the 3 indexed arXiv papers.
   - **`DIRECT_LLM`**: Conversational chit-chat, programming tasks, and common knowledge routed directly to `qwen/qwen3.8-27b`, completely bypassing vector retrieval.
   - **`EXTERNAL_WEB`**: Live web page retrieval via the **Playwright MCP Server** over `stdio` with automated HTML stripping and graceful HTTP fallback.

3. **Balanced Multi-Paper Partitioned Retrieval**:
   - When a query targets multiple papers (e.g. comparing *Moshi* and *Transformers*), the retriever dynamically partitions top-$k$ quotas across papers, eliminating context starvation where one paper floods the prompt.

4. **Resilient LLM Gateway & Backpressure**:
   - **Token Bucket Rate Limiter**: Strictly enforces per-model limits (30 RPM / 8,000 TPM rolling 60s windows).
   - **Concurrency Semaphore**: Caps in-flight streaming requests at 15 concurrent streams.
   - **Bounded Request Queue (50 slots)**: Absorbs bursts; sheds excess load cleanly with `503 QueueOverflowError` to prevent memory exhaustion and latency death spirals.
   - **Full Jitter Exponential Backoff**: Dynamic HTTP 429 `retry-after` header parsing (2.0s recovery window).

5. **Calibrated Stand-in LLM Service**:
   - Local, mathematically calibrated service ($\text{TTFT} \sim \text{Lognormal}(\mu=6.418, \sigma=0.55)$) for zero-quota, repeatable benchmarking.

---

## Indexed Research Corpus

The system indexes 120 pages across 254 chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) and FAISS:

| Paper ID | Title | Authors | Focus Area |
|:---:|:---|:---|:---|
| **`1706.03762v7`** | *Attention Is All You Need* | Vaswani et al. (Google Brain / Google Research) | Transformer architecture, multi-head attention |
| **`2410.00037v1`** | *Moshi: A Speech-Text Foundation Model for Real-Time Dialogue* | Defossez et al. (Kyutai) | Full-duplex audio, Mimi neural codec, Helium LM |
| **`2507.07955v2`** | *Dynamic Chunking for Retrieval-Augmented Generation* | DeepSeek / Gao et al. | Adaptive token boundaries, chunking-free RAG |

---

## Project Structure

```text
Vaani/
├── benchmarks/              # Citation and refusal evaluation metrics
│   └── evaluator.py         # Precision, recall, F1, refusal, accuracy scorers
├── chunking/                # Document chunking & passage boundary logic
├── data/
│   ├── benchmark_results/   # Benchmark artifacts (results.csv, table3_raw.csv)
│   ├── chunked/             # Segmented passages (chunks.json)
│   ├── evaluation_questions.json # 15 ground-truth evaluation questions
│   ├── extracted/           # Extracted PDF text (papers.json)
│   ├── papers/              # Source PDFs of the 3 indexed papers
│   └── vector_store/        # FAISS index (faiss.index) and chunk catalog
├── data_extraction/         # PyMuPDF PDF extraction engine
├── docs/
│   └── REPORT.md            # Comprehensive 8-section technical report (Tables 1, 2a, 2b, 3)
├── embeddings/              # SentenceTransformer embedding generators
├── gateway/                 # Production-grade LLM Gateway
│   ├── llm_gateway.py       # Concurrency semaphore & bounded queue manager
│   ├── retry_policy.py      # Exponential backoff with full jitter
│   └── token_bucket.py      # Rolling dual-budget rate limiter (30 RPM / 8k TPM)
├── llm/                     # LLM abstraction layer
│   ├── groq_client.py       # High-speed Groq cloud streaming client
│   └── standin_service.py   # Mathematically calibrated simulation service
├── pipeline/                # 4-step cyclical RAG pipeline
│   ├── citation_verifier.py # Provenance parser & page range validator
│   ├── classifier.py        # Tri-route intent classification engine
│   └── orchestrator.py      # End-to-end RAG & Gateway orchestrator
├── retrieval/               # Vector retrieval & reranking
│   ├── reranker.py          # Cross-Encoder (ms-marco-MiniLM-L-6-v2)
│   └── retriever.py         # Settings A, B, C & multi-paper partitioning
├── scripts/
│   ├── cli.py               # Interactive terminal REPL & inspection tool
│   ├── run_benchmarks.py    # Benchmark runner for Tables 2a, 2b, and 3
│   └── test_concurrency.py  # Concurrency & rate-limit probing script
├── tests/                   # 85 automated pytest unit tests
├── tools/                   # External tool integration
│   ├── playwright_mcp_client.py # Playwright MCP client over stdio
│   └── web_fetcher.py       # Playwright MCP with fallback to httpx
├── mcp_config.json          # Playwright MCP server definition
├── pytest.ini               # Test configuration
└── requirements.txt         # Project dependencies
```

---

## Installation & Setup

### 1. Prerequisites
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Node.js**: v18+ (Required for the Playwright MCP server via `npx`)
- **Git**

### 2. Clone and Setup Environment
```powershell
git clone https://github.com/kishormaity/Vaani.git
cd Vaani

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and configure your credentials:
```powershell
cp .env.example .env
```
Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_CHAT_MODEL=qwen/qwen3.8-27b
```
*(Note: If no API key is provided, the system seamlessly defaults to the offline calibrated `StandinLLMService`).*

---

## Usage Guide

### 1. Interactive Terminal CLI (`scripts/cli.py`)

Launch the real-time terminal REPL:
```powershell
# Interactive mode using Setting C (Cross-Encoder reranked) and live Groq API
python scripts/cli.py

# Interactive mode using offline calibrated stand-in service (zero API quota used)
python scripts/cli.py --standin

# Interactive mode with Setting B (top_k=5 dense)
python scripts/cli.py --setting B
```

### 2. One-Shot Query Execution

Execute single queries directly from the command line:

```powershell
# In-corpus factual question (CORPUS_RAG route)
python scripts/cli.py "Explain scaled dot-product attention"

# Multi-paper comparative question with balanced partitioning
python scripts/cli.py "Compare full-duplex speech in Moshi with standard Transformers"

# Conversational chat (DIRECT_LLM route, bypassing retriever)
python scripts/cli.py "Hello! Who are you and what can you do?"

# Live external web synthesis (EXTERNAL_WEB route via Playwright MCP)
python scripts/cli.py "Fetch https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"

# Out-of-scope query (triggers safe refusal)
python scripts/cli.py "What is the recipe for baking chocolate chip cookies?"
```

---

## Benchmark Suite & Empirical Telemetry

Execute the full empirical evaluation suite:
```powershell
# Run all benchmark suites (Retrieval ablations + Gateway load)
python scripts/run_benchmarks.py --suite all

# Run retrieval quality ablation suite only (Table 3)
python scripts/run_benchmarks.py --suite retrieval

# Run gateway load & queue telemetry suite only (Tables 2a & 2b)
python scripts/run_benchmarks.py --suite load
```

### Table 1: Real-World Groq Telemetry & Limits (`openai/gpt-oss-20b`)
- **Requests Per Minute (RPM)**: 30 RPM
- **Tokens Per Minute (TPM)**: 8,000 TPM
- **Requests Per Day (RPD)**: 1,000 RPD
- **Tokens Per Day (TPD)**: 200,000 TPD
- **Safe In-Flight Concurrency Ceiling**: 15 concurrent requests
- **HTTP 429 Recovery Window**: 2.0 seconds

### Table 2a: Gateway Load Telemetry (160 requests)
| Scenario | Total Reqs | Success % | Throughput | TTFT p50 | Latency p50 | Latency p95 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline Sequential** | 10 | 100.0% | 2.75 req/s | 16.28 ms | 101.00 ms | 2063.27 ms |
| **2. Moderate Load (C=5)** | 20 | 100.0% | 2.34 req/s | 24.81 ms | 115.30 ms | 1251.36 ms |
| **3. Concurrency Saturation (C=20)** | 25 | 100.0% | 43.08 req/s | 14.52 ms | 49.90 ms | 180.21 ms |
| **4. RPM Saturation (35 Reqs Burst)** | 35 | 100.0% | 1.76 req/s | 15.78 ms | 132.88 ms | 1148.77 ms |
| **5. Queue Overflow (70 Reqs Burst)** | 70 | 92.9% | 6.92 req/s | 16.55 ms | 231.78 ms | 1662.13 ms |

### Table 2b: Gateway Queue & Backpressure Telemetry
| Scenario | Queue Wait p50 | Peak Queue | Throttled Reqs | 503 Overflow Drops |
|:---|:---:|:---:|:---:|:---:|
| **1. Baseline** | 0.01 ms | 0 | 0 | 0 |
| **2. Moderate Load** | 0.00 ms | 0 | 0 | 0 |
| **3. Concurrency Saturation** | 0.00 ms | 10 | 0 | 0 |
| **4. RPM Saturation** | 66.01 ms | 20 | 20 | 0 |
| **5. Queue Overflow** | 115.82 ms | **50 (Capacity)** | 50 | **5 (Intentional Drops)** |

*Scenario 5 intentionally sheds 5 excess requests with HTTP 503 `QueueOverflowError` to protect system memory and prevent latency death spirals.*

### Table 3: Retrieval Quality & Ablations
| Setting | Configuration | Answer Accuracy | Citation Precision | Citation Recall | Citation F1 | Refusal Acc. |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Setting A** | Dense Vector ($k=3$) | 23.5% | 8.9% | 33.3% | 10.0% | 25.0% |
| **Setting B** | Dense Vector ($k=5$) | 28.2% | 16.7% | 33.3% | 17.8% | 50.0% |
| **Setting C** | **Cross-Encoder Reranked ($k_{\text{cand}}=8 \to k=3$)** | **25.4%** | **15.0%** | **50.0%** | **18.9%** | 25.0% |

*Setting C achieves a **50.0% Citation Recall** (a 50% relative improvement over pure dense search) while keeping prompt tokens compact to satisfy the 8,000 TPM limit.*

For detailed methodology, empirical analysis, and architecture specifications, consult [`docs/REPORT.md`](docs/REPORT.md).

---

## Running Automated Tests

Execute the complete 85-test suite:
```powershell
pytest
```
Output:
```text
============================= 85 passed in 29.74s =============================
```

All 85 tests pass with **0 warnings** and **0 failures**.

---

## License & Attribution

Developed for *The Relay* Advanced RAG Architecture. Academic papers cited are attributed under standard scientific citation conventions.
