from sqlalchemy.orm import Session
from . import models
from datetime import datetime

def create_session(db: Session, user_id: str):
    session = models.ChatSession(user_id=user_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def save_message(db: Session, session_id: int, question: str, answer: str, summary: str = ""):
    message = models.ChatMessage(
        session_id=session_id,
        question=question.strip(),
        answer=answer.strip(),
        summary=summary.strip()
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    if session:
        session.last_message_at = datetime.utcnow()
        db.commit()

    return message

def get_user_chats(db: Session, user_id: str):
    return db.query(models.ChatSession).filter(
        models.ChatSession.user_id == user_id,
        models.ChatSession.is_archived == False
    ).order_by(models.ChatSession.last_message_at.desc()).all()

def update_chat_title(db: Session, session_id: int, title: str):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    if session:
        session.title = title.strip()
        db.commit()
        db.refresh(session)
    return session

def delete_chat(db: Session, session_id: int):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    if session:
        session.is_archived = True
        db.commit()
    return {"status": "archived"}

def get_archived_chats(db: Session, user_id: str):
    return db.query(models.ChatSession).filter(
        models.ChatSession.user_id == user_id,
        models.ChatSession.is_archived == True
    ).all()

def get_chat_by_id(db: Session, session_id: int):
    return db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

def update_session_summary(db: Session, session_id: int, summary: str):
    chat = get_chat_by_id(db, session_id)
    if not chat:
        return {"error": "Chat not found"}
    chat.summary = summary.strip()
    db.commit()
    db.refresh(chat)
    return {"message": "Summary updated", "session_id": session_id}

def get_session_summary(db: Session, session_id: int):
    chat = get_chat_by_id(db, session_id)
    return chat.summary or "" if chat else ""