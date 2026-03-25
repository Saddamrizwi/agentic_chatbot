from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .utils import upload_files
from .chatbot import stream_response, get_history, clear_history

router = APIRouter()


# -----------------------------
# REQUEST SCHEMA 
# -----------------------------

class ChatRequest(BaseModel):
    message: str


# -----------------------------
# UPLOAD ENDPOINT
# -----------------------------

@router.post("/upload_data", tags=["DATA"])
async def upload_data(
    file: UploadFile = File(description="Upload a PDF or TXT file"),
):
    """
    Upload a PDF or TXT file.
    Splits into chunks → generates embeddings → stores in ChromaDB.
    """
    try:
        result = upload_files(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")


# -----------------------------
# STREAMING CHAT ENDPOINT
# -----------------------------

@router.post("/chat", tags=["CHAT"])
async def chat(request: ChatRequest):
    """
    Send a message and receive a streaming response (SSE).

    Response stream format:
    - `data: {"token": "..."}` — one text token
    - `data: [TOOL_CALL:web_search]` — web search was triggered
    - `data: [TOOL_CALL:get_datetime_info]` — datetime tool was triggered
    - `data: [DONE]` — stream complete

    Tool trigger examples:
    - "What's today's date?" → triggers `get_datetime_info`
    - "Search for latest AI news" → triggers `web_search`
    - "What does the document say about X?" → uses RAG context
    - "What was my last question?" → answered from conversation history
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return StreamingResponse(
        stream_response(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# -----------------------------
# HISTORY ENDPOINTS
# -----------------------------

@router.get("/history", tags=["CHAT"])
async def get_chat_history():
    """Get the full global conversation history."""
    history = get_history()
    return {
        "total_messages": len(history),
        "turns": len(history) // 2,
        "history": history
    }


@router.delete("/history", tags=["CHAT"])
async def delete_chat_history():
    """Clear the global conversation history."""
    clear_history()
    return {"message": "Conversation history cleared successfully."}