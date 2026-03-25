import os
import json
import datetime
import httpx
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAMESPACE = "public"
PERSIST_DIR = os.path.join(BASE_DIR, "DB")
MODEL_NAME = "gpt-4o-mini"

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")  

# -----------------------------
# EMBEDDINGS
# -----------------------------

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": False}
        )
    return _embeddings

# -----------------------------
# CHAT SESSION
# -----------------------------

_conversation_history: list = []


def get_history() -> list:
    return list(_conversation_history)


def append_history(role: str, content: str):
    _conversation_history.append({"role": role, "content": content})
    # Cap at 20 messages (10 turns)
    if len(_conversation_history) > 20:
        _conversation_history[:] = _conversation_history[-20:]


def clear_history():
    _conversation_history.clear()


# -----------------------------
# SYSTEM PROMPT
# -----------------------------

SYSTEM_PROMPT = """You are an expert AI analyst and conversational assistant with access to conversation history, uploaded documents, and external tools.

## IDENTITY & STYLE
- Be precise, professional, and concise (≤120 words unless detail is required).
- Never fabricate. If unsure, say so clearly.

## CONVERSATION AWARENESS
- Always interpret short replies (e.g., "yes", "ok", "do it") using previous user intent.
- If the user confirms a previous request, CONTINUE that task instead of resetting context.
- You have full memory of this conversation — use it naturally.

## DOCUMENT HANDLING
- If documents exist:
  - Answer ONLY from them when relevant.
  - Cite filename(s).
- If answer not found:
  → "I couldn't find that in the uploaded documents."
- If no documents exist:
  - DO NOT block the user unnecessarily.
  - Only mention missing documents if the query explicitly depends on them.

## TOOL USAGE (CRITICAL)

### Use `web_search` when:
- Query needs real-time or external info (stocks, news, weather, etc.)
- User confirms a previous query requiring live data
- No document covers the query

### Use `get_datetime_info` when:
- Any date/time or duration calculation is asked

## TOOL DECISION PRIORITY
1. Check conversation history (very important)
2. Check document relevance
3. If missing → use tools (DO NOT refuse prematurely)
4. Fallback to general knowledge only if appropriate

## RESPONSE RULES
- Never ignore prior context (e.g., “yes” should trigger previous intent)
- Do NOT say “upload documents” unless explicitly needed
- Do NOT reject queries that tools can solve
- Summarize tool results naturally

## EXAMPLE BEHAVIOR
User: "What is Tesla's stock price?"
→ Use web_search

User: "yes"
→ Interpret as confirmation → call web_search

User: "What did I ask before?"
→ Answer from conversation history
"""

# -----------------------------
# TOOL DEFINITIONS
# -----------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using SerpAPI (Google Search) for current, real-time, "
                "or up-to-date information. Use for news, stock prices, sports results, "
                "weather, product releases, or anything that changes over time and is "
                "not in the uploaded documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific, concise Google search query. "
                            "E.g. 'Python 3.13 new features 2025' or 'Bitcoin price today'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime_info",
            "description": (
                "Returns current date, time, day of week, week number, and UTC time. "
                "Use whenever the user asks anything about the current date or time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What the user wants to know, e.g. 'current date and time'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# -----------------------------
# TOOL IMPLEMENTATIONS
# -----------------------------

def web_search(query: str) -> str:
    """
    Search Google via SerpAPI and return a clean, structured summary.
    Requires SERPAPI_API_KEY environment variable.
    Docs: https://serpapi.com/search-api
    """
    if not SERPAPI_KEY:
        return (
            "[web_search] SERPAPI_API_KEY is not set. "
            "Please export SERPAPI_API_KEY='your-key' and restart the server."
        )

    try:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "num": 5,           # top 5 results
            "hl": "en",         # language: English
            "gl": "us",         # country: US
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        parts = []

        # 1. Answer box (direct answer, highest priority)
        answer_box = data.get("answer_box", {})
        if answer_box:
            answer = (
                answer_box.get("answer")
                or answer_box.get("snippet")
                or answer_box.get("result")
                or ""
            )
            if answer:
                parts.append(f"Direct Answer: {answer.strip()}")

        # 2. Knowledge graph (entity info)
        kg = data.get("knowledge_graph", {})
        if kg:
            kg_desc = kg.get("description", "")
            if kg_desc:
                parts.append(f"Knowledge Graph: {kg_desc.strip()}")

        # 3. Top web pages
        organic = data.get("organic_results", [])
        if organic:
            parts.append("Top Results:")
            for i, result in enumerate(organic[:4], 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                if title or snippet:
                    parts.append(f"  {i}. {title}\n     {snippet}\n     {link}")

        # 4. Related questions
        paa = data.get("related_questions", [])
        if paa and not parts:
            parts.append("Related Info:")
            for q in paa[:2]:
                parts.append(f"  Q: {q.get('question', '')}\n  A: {q.get('snippet', '')}")

        if not parts:
            return f"[Web Search: '{query}']\nNo results found. Try a more specific query."

        result_text = "\n".join(parts)
        return f"[Web Search: '{query}']\n{result_text}"

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "[web_search] Invalid SERPAPI_API_KEY. Please check your API key."
        if e.response.status_code == 429:
            return "[web_search] SerpAPI rate limit reached. Please wait and try again."
        return f"[web_search] HTTP error {e.response.status_code}: {str(e)}"
    except Exception as e:
        return f"[web_search] Search failed: {str(e)}"


def get_datetime_info(query: str) -> str:
    """Return current date/time details."""
    now = datetime.datetime.now()
    utc_now = datetime.datetime.utcnow()
    return (
        f"[DateTime]\n"
        f"Date       : {now.strftime('%A, %B %d, %Y')}\n"
        f"Time       : {now.strftime('%I:%M:%S %p')}\n"
        f"UTC        : {utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"Day        : {now.strftime('%A')}\n"
        f"Week No.   : {now.isocalendar()[1]}"
    )


def dispatch_tool(name: str, arguments: dict) -> str:
    if name == "web_search":
        return web_search(arguments["query"])
    elif name == "get_datetime_info":
        return get_datetime_info(arguments["query"])
    return f"Unknown tool: {name}"

# -----------------------------
# RAG
# -----------------------------

def retrieve_context(query: str, k: int = 4) -> str:
    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings(),
            collection_name=NAMESPACE,
        )
        if vectordb._collection.count() == 0:
            return ""
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(query)
        if not docs:
            return ""
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
    except Exception as e:
        print(f"[RAG] Retrieval failed: {e}")
        return ""

# -----------------------------
# BUILD MESSAGES
# -----------------------------

def build_messages(user_query: str, doc_context: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Full history goes BEFORE the current user turn
    messages.extend(get_history())

    if doc_context:
        user_content = (
            f"### RELEVANT DOCUMENT CONTEXT\n"
            f"{doc_context}\n\n"
            f"---\n\n"
            f"### MY QUESTION\n"
            f"{user_query}"
        )
    else:
        user_content = user_query

    messages.append({"role": "user", "content": user_content})
    return messages

# -----------------------------
# STREAMING CHAT
# -----------------------------

def stream_response(user_query: str):
    """
    Generator yielding SSE-formatted strings:
      data: {"token": "..."}        — one streamed token
      data: [TOOL_CALL:tool_name]   — a tool was invoked
      data: [DONE]                  — stream finished
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    doc_context = retrieve_context(user_query)
    messages = build_messages(user_query, doc_context)

    try:
        # First call — detect whether tool use is needed
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=600,
        )

        msg = response.choices[0].message

        # Handle tool calls if triggered
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                yield f"data: [TOOL_CALL:{tool_name}]\n\n"

                tool_result = dispatch_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })

        # Stream final answer
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token = delta.content
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

        # Save plain query to history
        append_history("user", user_query)
        append_history("assistant", full_response)

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"