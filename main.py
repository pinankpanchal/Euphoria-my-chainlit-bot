from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import Base, engine, SessionLocal
from . import crud, models
from pydantic import BaseModel
from typing import Optional
from typing import Any

def sanitize_text(text: Any) -> str:
    """Convert text to safe string for JSON storage."""
    if text is None:
        return ""
    
    text = str(text)
    text = text.replace('\x00', '')
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
    return text.strip()

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SessionCreate(BaseModel):
    user_id: str

class MessageCreate(BaseModel):
    question: str
    answer: str
    summary: Optional[str] = ""

@app.post("/new_session")
def new_session(payload: SessionCreate, db: Session = Depends(get_db)):
    try:
        session = crud.create_session(db, payload.user_id)
        return {"id": session.id, "user_id": session.user_id, "created_at": session.created_at}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_message")
def save_message(
    session_id: int,
    question: str,
    answer: str,
    summary: str = "",
    db: Session = Depends(get_db)
):
    try:
        question = sanitize_text(question)
        answer = sanitize_text(answer)
        summary = sanitize_text(summary)

        message = crud.save_message(db, session_id, question, answer, summary)
        return {
            "id": message.id,
            "session_id": message.session_id,
            "created_at": message.created_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/messages/{session_id}")
def get_messages(session_id: int, db: Session = Depends(get_db)):
    try:
        messages = db.query(models.ChatMessage).filter(
            models.ChatMessage.session_id == session_id
        ).order_by(models.ChatMessage.created_at.asc()).all()
        
        return [
            {
                "id": m.id,
                "question": sanitize_text(m.question),
                "answer": sanitize_text(m.answer),
                "summary": sanitize_text(m.summary),
                "created_at": m.created_at
            }
            for m in messages
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session_summary/{session_id}")
def get_session_summary(session_id: int, db: Session = Depends(get_db)):
    try:
        chat = crud.get_chat_by_id(db, session_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "summary": chat.summary or ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_summary")
def update_summary(session_id: int, summary: str, db: Session = Depends(get_db)):
    try:
        summary = sanitize_text(summary)
        return crud.update_session_summary(db, session_id, summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/chats")
def get_user_chats(user_id: str, db: Session = Depends(get_db)):
    try:
        chats = crud.get_user_chats(db, user_id)
        return [
            {
                "id": chat.id,
                "title": chat.title or f"Chat #{chat.id}",
                "created_at": chat.created_at.isoformat(),
                "last_updated": chat.last_message_at.isoformat(),
                "message_count": len(chat.messages),
                "preview": chat.messages[-1].question[:50] if chat.messages else "No messages"
            }
            for chat in chats
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/chats/{session_id}/title")
def update_chat_title(session_id: int, title: str, db: Session = Depends(get_db)):
    try:
        title = sanitize_text(title)
        return crud.update_chat_title(db, session_id, title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{session_id}")
def delete_chat(session_id: int, db: Session = Depends(get_db)):
    try:
        return crud.delete_chat(db, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/archived_chats")
def get_archived_chats(user_id: str, db: Session = Depends(get_db)):
    try:
        chats = crud.get_archived_chats(db, user_id)
        return [
            {
                "id": chat.id,
                "title": chat.title or f"Chat #{chat.id}",
                "created_at": chat.created_at.isoformat()
            }
            for chat in chats
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))