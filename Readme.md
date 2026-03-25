# 🤖 Agentic RAG Chatbot

A FastAPI-based conversational AI chatbot with document intelligence, streaming responses, autonomous tool use (Google Search via SerpAPI + DateTime), and persistent conversation memory.

---

## 📁 Project Structure

```
agentic_chatbot/
├── app.py                      # FastAPI app entry point
├── user/
│   ├── main.py                 # API route definitions
│   ├── chatbot.py              # Core chat engine (streaming, tools, RAG, memory)
│   └── utils.py                # File ingestion, embedding, ChromaDB storage
├── DB/                         # ChromaDB vector store (auto-created on first upload)
├── file_record/
│   └── file_record.json        # Tracks uploaded files by MD5 hash (prevents duplicates)
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn openai langchain langchain-chroma langchain-huggingface \
            sentence-transformers chromadb PyPDF2 pdf2image pytesseract httpx python-multipart
```

> **Note:** `pytesseract` requires Tesseract OCR installed on your system:
> - Ubuntu/Debian: `sudo apt install tesseract-ocr`
> - macOS: `brew install tesseract`

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
export SERPAPI_API_KEY="your-serpapi-key"
```

> Get your free SerpAPI key at: https://serpapi.com/

### 3. Run the server

```bash
uvicorn app:app --reload
```

### 4. Open Swagger UI

```
http://localhost:8000/docs
```

---

## 🔌 API Endpoints

### `POST /upload_data`
Upload a PDF or TXT file to the knowledge base.

```bash
curl -X POST http://localhost:8000/upload_data \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "added_chunks": 42,
  "skipped_chunks": 0,
  "total_chunks_in_db": 42,
  "messages": ["document.pdf: processed successfully (42 chunks)"]
}
```

---

### `POST /chat`
Send a message and receive a **streaming response** (Server-Sent Events).

**Request body:**
```json
{
  "message": "What does the document say about revenue?"
}
```

**Stream format:**
```
data: {"token": "The"}
data: {"token": " revenue"}
data: {"token": " in Q3 was..."}
data: [TOOL_CALL:web_search]        ← emitted when SerpAPI is triggered
data: [TOOL_CALL:get_datetime_info] ← emitted when datetime tool is triggered
data: [DONE]
```

**curl example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about OpenAI?"}' \
  --no-buffer
```

---

### `GET /history`
Retrieve the full global conversation history.

```bash
curl http://localhost:8000/history
```

**Response:**
```json
{
  "total_messages": 4,
  "turns": 2,
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

---

### `DELETE /history`
Clear the conversation and start fresh.

```bash
curl -X DELETE http://localhost:8000/history
```

---

## 🧠 How It Works

### Document Upload & RAG Pipeline

```
User uploads PDF or TXT
         ↓
Text extracted page by page
(PyPDF2 for digital PDFs → pytesseract OCR fallback for scanned pages)
         ↓
Split into 500-character chunks with 50-character overlap
         ↓
Each chunk embedded using sentence-transformers/all-MiniLM-L6-v2
         ↓
Embeddings stored in ChromaDB (local, persistent, in ./DB/)
         ↓
MD5 hash of file saved to file_record.json (prevents re-ingestion)
```

### Chat & Streaming Pipeline

```
User sends message
         ↓
Similarity search → retrieve top-4 relevant chunks from ChromaDB
         ↓
Build message list:
  [system prompt] → [conversation history] → [user message + doc context]
         ↓
Send to GPT-4o-mini with tool definitions (tool_choice="auto")
         ↓
Model decides: answer directly OR call a tool
         ↓
If tool called:
  → web_search  : call SerpAPI Google Search, extract structured results
  → get_datetime_info : call Python datetime, return local + UTC time
  → append tool result to messages
  → re-call model for final answer
         ↓
Stream final answer token-by-token via SSE
         ↓
Append [user question, assistant answer] to global conversation history
```

### Conversation Memory

- History stored as a **global in-memory list** — no database needed.
- Capped at **20 messages (10 turns)** to stay within context limits.
- History injected **before** the current user message so the model sees full context.
- Reset anytime via `DELETE /history`.

---

## 🛠️ Tools

### Tool 1: `web_search` — powered by SerpAPI (Google Search)

SerpAPI hits the real Google Search engine and returns structured results including answer boxes, knowledge graphs, and organic snippets.

**Environment variable required:** `SERPAPI_API_KEY`  
**SerpAPI free tier:** 100 searches/month — https://serpapi.com/

**What gets extracted from each search:**
1. **Answer Box** — Google's direct answer (highest priority)
2. **Knowledge Graph** — Entity description for people, places, companies
3. **Organic Results** — Top 4 page titles + snippets + URLs
4. **People Also Ask** — Related Q&A pairs (fallback)

**Trigger scenarios:**

| User Message | Tool Query Sent to SerpAPI |
|---|---|
| `"What's the latest news about OpenAI?"` | `"latest OpenAI news 2025"` |
| `"What is the current Bitcoin price?"` | `"Bitcoin price today"` |
| `"Who won the IPL 2025?"` | `"IPL 2025 winner"` |
| `"Search for Python 3.13 features"` | `"Python 3.13 new features"` |
| `"What's the weather in Delhi?"` | `"weather in Delhi today"` |
| `"What is Tesla's stock price?"` | `"Tesla stock price today"` |

**Error handling:**
- Missing key → clear message: `SERPAPI_API_KEY is not set`
- Invalid key (401) → `Invalid SERPAPI_API_KEY`
- Rate limit (429) → `Rate limit reached, please wait`

---

### Tool 2: `get_datetime_info` — Python datetime (no API key needed)

**Trigger scenarios:**

| User Message | Response |
|---|---|
| `"What's today's date?"` | `Monday, March 24, 2025` |
| `"What time is it?"` | `02:45:30 PM` |
| `"What day of the week is it?"` | `Monday` |
| `"What week number are we in?"` | `Week 13` |
| `"What's the UTC time right now?"` | `2025-03-24 09:15:30 UTC` |

---

## 💬 Conversation Memory Scenarios

| User Message | What Happens |
|---|---|
| `"What was my last question?"` | Model reads history, quotes the previous user message verbatim |
| `"Summarize what we've discussed"` | Model synthesizes the full conversation from history |
| `"Can you elaborate on your last answer?"` | Model references its most recent assistant response |
| `"You mentioned X earlier — tell me more"` | Model finds X in history and expands |

---

## 📄 RAG Scenarios

| User Message | What Happens |
|---|---|
| `"What does the report say about Q3 revenue?"` | Retrieves matching chunks, answers from document |
| `"Summarize the uploaded document"` | Retrieves broad chunks, synthesizes a summary |
| `"Who wrote this document?"` | Retrieves header/metadata chunks |
| `"What is the capital of France?"` | No doc context needed — answers from GPT knowledge |
| `"What does the doc say about quantum physics?"` (not in doc) | Says: *"I couldn't find that in the uploaded documents."* |
| No file uploaded yet | Says: *"No documents have been uploaded yet."* |

---

## 🔒 Duplicate File Prevention

Files are hashed with MD5 before ingestion. If the same file is re-uploaded, it is **skipped entirely** — no duplicate chunks are added to the vector store. The response shows `skipped_chunks` count.

---

## 🧩 Configuration Reference

| Component | Value |
|---|---|
| LLM | `gpt-4o-mini` via OpenAI API |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector store | ChromaDB (local persistent, `./DB/`) |
| Web search | SerpAPI — Google Search engine |
| Chunk size | 500 characters |
| Chunk overlap | 50 characters |
| Top-k retrieval | 4 chunks per query |
| History cap | 20 messages (10 turns) |
| Streaming | SSE (`text/event-stream`) |

---

## 🧪 Quick Test Sequence

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/upload_data \
  -F "file=@report.pdf"

# 2. Ask about the document (RAG)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the document"}' --no-buffer

# 3. Follow-up (tests conversation memory)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What was my last question?"}' --no-buffer

# 4. Trigger SerpAPI web search
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about AI?"}' --no-buffer

# 5. Trigger datetime tool
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What day is it today?"}' --no-buffer

# 6. Check history
curl http://localhost:8000/history

# 7. Clear history
curl -X DELETE http://localhost:8000/history
```

---

## 🚨 Common Issues

| Problem | Fix |
|---|---|
| `SERPAPI_API_KEY is not set` | Run `export SERPAPI_API_KEY="your-key"` before starting uvicorn |
| `Invalid SERPAPI_API_KEY` | Check your key at https://serpapi.com/manage-api-key |
| `Rate limit reached` | Free tier is 100 searches/month — upgrade or wait |
| Embedding warning `UNEXPECTED key` | Harmless — it's a known HuggingFace/ST mismatch, safe to ignore |
| `file_record` dir not found | Auto-created on first upload — ensure write permissions in project root |
| OCR not working | Install Tesseract: `sudo apt install tesseract-ocr` |