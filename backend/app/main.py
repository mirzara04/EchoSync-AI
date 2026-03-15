import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .models.schema import (
    AlarmModel,
    ImageAttachmentModel,
    MeetingModel,
    NoteModel,
    SessionLocal,
    TaskModel,
    init_db,
)
from .services.asr import transcribe
from .services.intent_engine import analyze_intents
from .services.translator import translate

app = FastAPI(title="EchoSync AI Offline Backend")


@app.on_event("startup")
def _startup() -> None:
    init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AlarmRequest(BaseModel):
    label: str
    time_iso: str
    recurrence: str | None = None


class ManualTaskRequest(BaseModel):
    description: str


def _save_audio(file: UploadFile) -> Path:
    if not file.filename.endswith((".wav", ".flac", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{datetime.utcnow().timestamp()}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    temp_path = _save_audio(file)
    try:
        urdu_transcript = transcribe(str(temp_path), language="ur")
        english_text = translate(urdu_transcript)
        analysis = analyze_intents(english_text)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    summary = analysis.get("summary", "")

    note = NoteModel(content=urdu_transcript, summary=summary)
    db.add(note)
    db.commit()
    db.refresh(note)

    created_tasks = _persist_tasks(db, note.id, analysis.get("tasks", []))
    created_meetings = _persist_meetings(db, note.id, analysis.get("meetings", []))
    created_alarms = _persist_alarms(db, note.id, analysis.get("alarms", []))
    db.commit()

    return {
        "note_id": note.id,
        "transcript_urdu": urdu_transcript,
        "transcript_english": english_text,
        "summary": summary,
        "agenda": analysis.get("agenda", ""),
        "tasks": created_tasks,
        "meetings": created_meetings,
        "alarms": created_alarms,
    }


def _persist_tasks(db: Session, note_id: int, tasks: List[str]) -> List[dict]:
    stored = []
    for description in tasks:
        if not description:
            continue
        task = TaskModel(note_id=note_id, description=description)
        db.add(task)
        db.flush()
        stored.append(
            {"id": task.id, "description": task.description, "completed": task.completed}
        )
    return stored


def _persist_meetings(db: Session, note_id: int, meetings: List[dict]) -> List[dict]:
    stored = []
    for meeting in meetings or []:
        title = meeting.get("title")
        dt_raw = meeting.get("datetime_iso")
        dt_parsed = _parse_datetime(dt_raw)
        if not title or not dt_parsed:
            continue
        record = MeetingModel(note_id=note_id, title=title, time=dt_parsed)
        db.add(record)
        db.flush()
        stored.append(
            {
                "id": record.id,
                "title": record.title,
                "time": record.time.isoformat(),
            }
        )
    return stored


def _persist_alarms(db: Session, note_id: int, alarms: List[dict]) -> List[dict]:
    stored = []
    for alarm in alarms or []:
        label = alarm.get("label")
        time_raw = alarm.get("time_iso")
        time_parsed = _parse_datetime(time_raw)
        if not label or not time_parsed:
            continue
        recurrence = alarm.get("recurrence")
        record = AlarmModel(note_id=note_id, label=label, time=time_parsed, recurrence=recurrence)
        db.add(record)
        db.flush()
        stored.append(
            {
                "id": record.id,
                "label": record.label,
                "time": record.time.isoformat(),
                "recurrence": record.recurrence,
            }
        )
    return stored


@app.post("/upload_image/{note_id}")
async def upload_image(
    note_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
        raise HTTPException(status_code=400, detail="Invalid image format")
    image_dir = Path("images")
    image_dir.mkdir(exist_ok=True)
    image_path = image_dir / f"{note_id}_{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    attachment = ImageAttachmentModel(note_id=note_id, image_path=str(image_path))
    db.add(attachment)
    db.commit()

    return {"message": "Image uploaded successfully"}


@app.get("/notes")
async def get_summaries(db: Session = Depends(get_db)):
    notes = db.query(NoteModel).all()
    return [
        {
            "id": note.id,
            "summary": note.summary,
            "date": note.created_at.strftime("%Y-%m-%d"),
        }
        for note in notes
        if note.summary
    ]


@app.get("/tasks")
async def get_tasks(db: Session = Depends(get_db)):
    tasks = db.query(TaskModel).all()
    return [
        {
            "id": task.id,
            "description": task.description,
            "completed": task.completed,
            "date": task.created_at.strftime("%Y-%m-%d"),
        }
        for task in tasks
    ]


@app.post("/tasks")
async def create_task(request: ManualTaskRequest, db: Session = Depends(get_db)):
    task = TaskModel(note_id=None, description=request.description)
    db.add(task)
    db.commit()
    db.refresh(task)
    return {"id": task.id, "description": task.description, "completed": task.completed}


@app.get("/meetings")
async def get_meetings(db: Session = Depends(get_db)):
    meetings = db.query(MeetingModel).all()
    return [
        {
            "id": meeting.id,
            "title": meeting.title,
            "time": meeting.time.strftime("%Y-%m-%d %H:%M"),
        }
        for meeting in meetings
    ]


@app.get("/alarms")
async def get_alarms(db: Session = Depends(get_db)):
    alarms = db.query(AlarmModel).all()
    return [
        {
            "id": alarm.id,
            "label": alarm.label,
            "time": alarm.time.isoformat(),
            "recurrence": alarm.recurrence,
        }
        for alarm in alarms
    ]


@app.post("/alarms")
async def create_alarm(request: AlarmRequest, db: Session = Depends(get_db)):
    alarm_time = _parse_datetime(request.time_iso)
    if not alarm_time:
        raise HTTPException(status_code=400, detail="Invalid ISO8601 datetime for alarm")
    alarm = AlarmModel(label=request.label, time=alarm_time, recurrence=request.recurrence, note_id=None)
    db.add(alarm)
    db.commit()
    db.refresh(alarm)
    return {
        "id": alarm.id,
        "label": alarm.label,
        "time": alarm.time.isoformat(),
        "recurrence": alarm.recurrence,
    }


@app.get("/health")
async def healthcheck():
    return {"status": "ok", "models_ready": os.path.exists(os.getenv("LLM_MODEL_PATH", "models/llm/model.gguf"))}
