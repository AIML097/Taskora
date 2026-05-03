import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import datetime
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, ConfigDict
from contextlib import asynccontextmanager

from database import SessionLocal, engine, Task, User, UserPreference, ProcessedEmail, Base
from ai.model_utils import processor as ai_processor
from services.gmail_service import gmail_service
from services.telegram_service import telegram_service
from services.priority_service import priority_service
from services.alert_service import trigger_alert
from services.google_calendar_service import create_calendar_event


# Security configurations
SECRET_KEY = "your-secret-key-here"  # In production, use an environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

import asyncio
from contextlib import asynccontextmanager

# Create tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting background services...")
    # Start Telegram Bot in background
    telegram_task = None
    try:
        telegram_task = asyncio.create_task(telegram_service.start_polling())
    except Exception as e:
        print(f"Failed to start Telegram polling: {e}")
        
    # Start Gmail Auto-sync in background
    gmail_task = None
    try:
        gmail_task = asyncio.create_task(gmail_auto_sync_loop())
    except Exception as e:
        print(f"Failed to start Gmail auto-sync: {e}")
        
    print("Background tasks initialization attempted.")
    yield
    # Cleanup logic
    if telegram_task:
        await telegram_service.stop_polling()
        telegram_task.cancel()
    if gmail_task:
        gmail_task.cancel()

app = FastAPI(title="Task Manager API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TaskBase(BaseModel):
    task_text: str  # Renamed from title
    description: Optional[str] = None
    source: str = "MANUAL"
    status: str = "PENDING"
    priority: str = "MEDIUM"
    due_date: Optional[datetime.datetime] = None

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    status: str
    actual_duration: Optional[int] = None

class TaskResponse(TaskBase):
    id: int
    status: str
    priority: str
    created_at: datetime.datetime
    user_id: int  # Renamed from owner_id

    model_config = ConfigDict(from_attributes=True)

class PreferenceCreate(BaseModel):
    keyword: str
    priority: str

class PreferenceResponse(BaseModel):
    id: int
    keyword: str
    priority: str

    model_config = ConfigDict(from_attributes=True)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth Utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# Auth Routes
@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# Task Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is up and running"}

@app.get("/tasks", response_model=List[TaskResponse])
def get_tasks(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Task).filter(Task.user_id == current_user.id).all()

@app.post("/tasks", response_model=TaskResponse)
def create_task(task: TaskCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_task = Task(**task.dict(), user_id=current_user.id)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.patch("/tasks/{task_id}", response_model=TaskResponse)
def update_task_status(task_id: int, task_update: TaskUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    db_task.status = task_update.status

    if task_update.actual_duration is not None and task_update.status == "COMPLETED":
        db_task.actual_duration = task_update.actual_duration
        
        # Save to Feedback
        from database import TaskFeedback
        fb = TaskFeedback(
            task_text=db_task.task_text,
            predicted_duration=db_task.predicted_duration,
            actual_duration=task_update.actual_duration
        )
        db.add(fb)
        
        # Trigger retraining check asynchronously
        import asyncio
        from ai.learning_service import check_and_retrain
        asyncio.create_task(asyncio.to_thread(check_and_retrain))

    db.commit()
    db.refresh(db_task)
    return db_task

@app.get("/calendar/tasks", response_model=List[TaskResponse])
def get_calendar_tasks(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Task).filter(Task.user_id == current_user.id, Task.due_date.isnot(None)).all()

@app.get("/tasks/upcoming", response_model=List[TaskResponse])
def get_upcoming_tasks(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    now = datetime.datetime.utcnow()
    in_30_mins = now + datetime.timedelta(minutes=30)
    return db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.due_date >= now,
        Task.due_date <= in_30_mins,
        Task.status == "PENDING"
    ).all()

# ─── Preference CRUD ───────────────────────────────────────

@app.post("/preferences", response_model=PreferenceResponse)
def create_preference(pref: PreferenceCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_pref = UserPreference(user_id=current_user.id, keyword=pref.keyword, priority=pref.priority)
    db.add(db_pref)
    db.commit()
    db.refresh(db_pref)
    return db_pref

@app.get("/preferences", response_model=List[PreferenceResponse])
def get_preferences(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(UserPreference).filter(UserPreference.user_id == current_user.id).all()

@app.delete("/preferences/{pref_id}")
def delete_preference(pref_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_pref = db.query(UserPreference).filter(UserPreference.id == pref_id, UserPreference.user_id == current_user.id).first()
    if not db_pref:
        raise HTTPException(status_code=404, detail="Preference not found")
    db.delete(db_pref)
    db.commit()
    return {"message": "Preference deleted successfully"}

# ─── Process Message (with user-preference priority override) ──

@app.post("/process-message")
async def process_message(data: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    result = ai_processor.process_message(text)
    if not result:
        return {"is_task": False, "message": "Message classified as non-task"}
    
    # Step 1: Use PriorityService to handle user-preference priority override
    final_priority, predicted_duration = priority_service.get_priority(text, current_user.id, db, result.get("priority", "MEDIUM"), result.get("due_date"))
    
    # If filtered
    if final_priority is None:
        return {"is_task": False, "message": "Message filtered."}
    
    # Duplicate prevention
    existing = db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.task_text == result["title"],
        Task.source == data.get("source", "MANUAL")
    ).first()
    if existing:
        return {"is_task": False, "message": "Duplicate task already exists."}
    
    # Save task with final priority
    new_task = Task(
        task_text=result["title"],
        description=result["description"],
        source=data.get("source", "MANUAL"),
        status="PENDING",
        priority=final_priority.upper(),
        predicted_duration=predicted_duration,
        due_date=result["due_date"],
        user_id=current_user.id
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    
    # Trigger priority-based alerts
    trigger_alert(result["title"], final_priority.upper())
    
    # Auto-create Google Calendar event if task has a due date
    if result["due_date"]:
        try:
            create_calendar_event(
                user_id=str(current_user.id),
                title=result["title"],
                description=result["description"] or "",
                due_date=result["due_date"]
            )
        except Exception as cal_err:
            print(f"Calendar event creation failed: {cal_err}")
    
    return {
        "is_task": True, 
        "task": new_task,
        "message": "Task successfully extracted and saved"
    }

@app.get("/integrations/status")
async def get_integration_status(current_user: User = Depends(get_current_user)):
    # Check if any gmail account exists for this user
    gmail_connected = gmail_service.get_active_account(str(current_user.id)) is not None
    # Check if this user has linked their telegram ID
    telegram_connected = current_user.telegram_id is not None
    return {
        "gmail": gmail_connected,
        "telegram": telegram_connected
    }

@app.post("/integrations/gmail/disconnect")
async def disconnect_gmail(current_user: User = Depends(get_current_user)):
    gmail_service.disconnect_account(str(current_user.id))
    return {"message": "Gmail disconnected successfully!"}

@app.post("/integrations/telegram/disconnect")
async def disconnect_telegram(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    current_user.telegram_id = None
    db.commit()
    return {"message": "Telegram disconnected successfully!"}

class TelegramConnect(BaseModel):
    telegram_id: str

@app.post("/connect-telegram")
async def connect_telegram(data: TelegramConnect, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Check if this telegram_id is already used by another user
    existing = db.query(User).filter(User.telegram_id == data.telegram_id, User.id != current_user.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="This Telegram ID is already linked to another account.")
    current_user.telegram_id = data.telegram_id
    db.commit()
    return {"message": "Telegram connected successfully!"}

# Integration Routes
@app.get("/auth/google")
async def google_auth_initiate(current_user: User = Depends(get_current_user)):
    url = gmail_service.get_auth_url(state=str(current_user.id))
    if not url:
        raise HTTPException(status_code=400, detail="Google Client ID/Secret not configured")
    return {"url": url}

@app.get("/auth/google/callback")
async def google_auth_callback(code: str, state: Optional[str] = None, db: Session = Depends(get_db)):
    try:
        user_id = state
        
        # Fallback if state is missing or default "1"
        if not user_id or user_id == "1":
            first_user = db.query(User).first()
            if first_user:
                user_id = str(first_user.id)
            else:
                raise Exception("No users found in the database. Cannot link Gmail.")
                
        gmail_service.get_credentials_from_code(code, user_id)
        # Return a simple HTML or redirect to frontend
        return {"message": "Authentication successful! You can close this window."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

def process_user_emails(user_id: int, db: Session):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return 0, 0

    messages = gmail_service.fetch_unread_messages(str(user_id))
    if not messages:
        return 0, 0
    
    tasks_created = 0
    message_ids_to_mark_read = []
    
    for msg in messages:
        gmail_msg_id = msg['id']
        
        # Skip already-processed messages (duplicate prevention by message ID)
        already_processed = db.query(ProcessedEmail).filter(
            ProcessedEmail.message_id == gmail_msg_id,
            ProcessedEmail.user_id == user_id
        ).first()
        if already_processed:
            print(f"⏩ Skipping already-processed email: {gmail_msg_id}")
            continue
        
        msg_text = msg['text']
        result = ai_processor.process_message(msg_text)
        if result:
            # Check for exact duplicate task by content
            existing = db.query(Task).filter(
                Task.user_id == user_id,
                Task.task_text == result["title"],
                Task.source == "EMAIL"
            ).first()
            
            if not existing:
                # Use PriorityService
                final_priority, predicted_duration = priority_service.get_priority(msg_text, user_id, db, result.get("priority", "MEDIUM"), result.get("due_date"))
                
                # If filtered skip
                if final_priority is None:
                    print(f"🔇 Email filtered: {result['title'][:30]}...")
                    # Still record as processed so we don't re-check it
                    db.add(ProcessedEmail(message_id=gmail_msg_id, user_id=user_id))
                    message_ids_to_mark_read.append(gmail_msg_id)
                    continue

                new_task = Task(
                    task_text=result["title"],
                    description=result["description"],
                    source="EMAIL",
                    status="PENDING",
                    priority=final_priority.upper(),
                    predicted_duration=predicted_duration,
                    due_date=result["due_date"],
                    user_id=user_id
                )
                db.add(new_task)
                tasks_created += 1
                
                # Trigger priority-based alerts
                trigger_alert(result["title"], final_priority.upper())
                
                # Auto-create Google Calendar event if task has a due date
                if result["due_date"]:
                    try:
                        create_calendar_event(
                            user_id=str(user_id),
                            title=result["title"],
                            description=result["description"] or "",
                            due_date=result["due_date"]
                        )
                    except Exception as cal_err:
                        print(f"Calendar event creation failed: {cal_err}")
        
        # Record this message as processed
        db.add(ProcessedEmail(message_id=gmail_msg_id, user_id=user_id))
        message_ids_to_mark_read.append(gmail_msg_id)
    
    if message_ids_to_mark_read:
        gmail_service.mark_as_read(str(user_id), message_ids_to_mark_read)
        
    db.commit()
    return tasks_created, len(messages)

@app.post("/gmail/sync")
async def sync_gmail(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Enable auto-sync by setting gmail_authenticated timestamp
    if not current_user.gmail_authenticated:
        current_user.gmail_authenticated = datetime.datetime.utcnow()
        db.commit()
        print(f"✅ Auto-sync enabled for user: {current_user.email}")
    
    tasks_created, msg_count = process_user_emails(current_user.id, db)
    if msg_count == 0:
        return {"message": "No new unread messages found. Auto-sync is now enabled."}
    return {"message": f"Sync complete. {tasks_created} tasks extracted from {msg_count} messages. Auto-sync enabled."}

async def gmail_auto_sync_loop():
    """Background loop to sync emails for all active users every 60 seconds."""
    print("Starting Gmail auto-sync loop...")
    while True:
        try:
            db = SessionLocal()
            # Find all users
            users = db.query(User).all()
            for user in users:
                # Check if this user has an active Gmail session
                if gmail_service.get_active_account(str(user.id)):
                    print(f"🔄 Auto-syncing Gmail for: {user.email}")
                    process_user_emails(user.id, db)
            db.close()
        except Exception as e:
            print(f"Gmail Sync Loop Error: {e}")
        
        await asyncio.sleep(60)

        await asyncio.sleep(60)

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(db_task)
    db.commit()
    return {"message": "Task deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

