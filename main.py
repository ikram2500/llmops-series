from __future__ import annotations 


import os 
from pathlib import Path
from typing import Dict, List


from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from multi_doc_chat.model.models import UploadResponse, ChatResponse, ChatRequest
from multi_doc_chat.utils.document_ops import FastAPIFileAdapter

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor 
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG 
from langchain_core.messages import HumanMessage, AIMessage 
from multi_doc_chat.exception.custom_exception import DocumentPortalException



app = FastAPI(title="Document Chat Portal", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static and template 
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


SESSIONS: Dict[str, List[dict]] = {}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    try:
        print(f"[DEBUG] Upload started - {len(files)} file(s)")
        
        #Wrap FastAPI files to preserve filename/ext and provide a read buffer
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        ingestor = ChatIngestor(use_session_dirs=True)
        session_id = ingestor.session_id
        
        print(f"[DEBUG] Session ID created: {session_id}")

        # save , load, split, embed, and write Faiss index
        ingestor.built_retriever(uploaded_files=wrapped_files)


        # Initialize empty chat history for session
        SESSIONS[session_id] = []
        
        print(f"[DEBUG] Upload complete - Session initialized: {session_id}")
        print(f"[DEBUG] Active sessions: {list(SESSIONS.keys())}")

        return UploadResponse(session_id=session_id, indexed=True, message="indexing complete.")
    except DocumentPortalException as e:
        print(f"[ERROR] DocumentPortalException: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Upload exception: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()
    
    # Debug logging
    print(f"[DEBUG] Received chat request - Session: {session_id}, Message: {message}")
    print(f"[DEBUG] Available sessions: {list(SESSIONS.keys())}")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required. Please upload documents first.")
    
    if session_id not in SESSIONS:
        # Check if FAISS index exists for this session
        index_path = Path(f"faiss_index/{session_id}")
        if index_path.exists():
            print(f"[DEBUG] Session not in memory but FAISS index exists. Initializing session.")
            SESSIONS[session_id] = []
        else:
            raise HTTPException(status_code=400, detail=f"Invalid or expired session_id: {session_id}. Please re-upload documents.")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Build RAG and load retriever from persisted FAISS with MMR
        rag = ConversationalRAG(session_id=session_id)
        index_path = f"faiss_index/{session_id}"
        rag.load_retriever_from_faiss(
            index_path=index_path,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Use simple in-memory history and convert to BaseMessage list
        simple = SESSIONS.get(session_id, [])
        lc_history = []
        for m in simple:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        answer = rag.invoke(message, chat_history=lc_history)

        # Update history
        simple.append({"role": "user", "content": message})
        simple.append({"role": "assistant", "content": answer})
        SESSIONS[session_id] = simple

        return ChatResponse(answer=answer)
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")








if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)