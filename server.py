import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from socketio import AsyncServer, ASGIApp
import base64
import asyncio
from PIL import Image
import io
import os
import psycopg
import time
from datetime import datetime, timedelta
import random
import logging
import tempfile
from pydantic import BaseModel

from original_code import (
    EnhancedCustomCNN, ModifiedResNet, EfficientNetModel, load_model, transform, CUDATransform,
    PPE_ITEMS, PPE_COLORS, VALID_DEPTS, MAX_FRAMES_TO_PROCESS, DETECTION_CONFIDENCE_THRESHOLD,
    DB_PARAMS, connect_db, init_database, PPEHistory, face_cascade, detect_faces,
    device, BASE_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app.mount('/socket.io', ASGIApp(sio))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginData(BaseModel):
    employee_id: str
    department: str

class SelectData(BaseModel):
    target_class: str
    model: str

state = {
    "user_id": None,
    "department": None,
    "target_class": None,
    "selected_model": None,
    "model": None,
    "num_classes": len(PPE_ITEMS),
    "ppe_history": None,
    "ppe_stats": {"Accepted": 0, "Flagged": 0},
    "cap": None,
    "is_running": False,
    "start_time": None,
    "frame_count": 0,
    "conn": connect_db(),
}

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@app.on_event("startup")
async def startup_event():
    init_database(state["conn"])
    logger.info("Application startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    if state["conn"]:
        state["conn"].close()
        logger.info("Database connection closed")

@app.post("/login")
async def login(data: LoginData):
    employee_id = data.employee_id.strip()
    department = data.department
    if not employee_id:
        logger.warning("Login attempted with empty employee ID")
        raise HTTPException(status_code=400, detail="Employee ID is required")
    if department not in VALID_DEPTS:
        logger.warning(f"Invalid department: {department}")
        raise HTTPException(status_code=400, detail="Invalid department")
    state["user_id"] = employee_id
    state["department"] = department
    try:
        with state["conn"].cursor() as cur:
            cur.execute("SELECT employee_id FROM employees WHERE employee_id = %s", (employee_id,))
            if not cur.fetchone():
                cur.execute("INSERT INTO employees (employee_id, department) VALUES (%s, %s)", (employee_id, department))
                state["conn"].commit()
                logger.info(f"New employee registered: {employee_id}, {department}")
    except psycopg.Error as e:
        logger.error(f"Database error during login: {e}")
        state["conn"].rollback()
        raise HTTPException(status_code=500, detail="Database error")
    return {"message": "Login successful"}

@app.post("/select")
async def select(data: SelectData):
    if data.target_class not in PPE_ITEMS:
        logger.warning(f"Invalid target class: {data.target_class}")
        raise HTTPException(status_code=400, detail="Invalid target class")
    if data.model not in ["CustomCNN", "ResNet18", "EfficientNet"]:
        logger.warning(f"Invalid model: {data.model}")
        raise HTTPException(status_code=400, detail="Invalid model")
    state["target_class"] = data.target_class
    state["selected_model"] = data.model
    try:
        state["model"], state["num_classes"] = load_model(state["selected_model"])
        state["ppe_history"] = PPEHistory(history_size=5)
        logger.info(f"Model selected: {data.model}, Target class: {data.target_class}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return {"message": "Selection saved"}

@sio.on("start_camera")
async def start_camera(sid):
    if state["is_running"]:
        logger.warning("Camera already running")
        return
    state["cap"] = cv2.VideoCapture(0)
    if not state["cap"].isOpened():
        logger.error("Failed to open camera")
        await sio.emit("error", {"message": "Failed to open camera"}, to=sid)
        return
    state["is_running"] = True
    state["start_time"] = time.time()
    state["frame_count"] = 0
    state["ppe_stats"] = {"Accepted": 0, "Flagged": 0}
    logger.info("Camera started")
    asyncio.create_task(process_frames(sid))

@sio.on("stop_camera")
async def stop_camera(sid):
    state["is_running"] = False
    if state["cap"]:
        state["cap"].release()
        state["cap"] = None
        logger.info("Camera stopped")
    record_session_data()

async def process_frames(sid):
    cuda_transform = CUDATransform(transform, use_edge=True)
    while state["is_running"] and state["cap"]:
        ret, frame = state["cap"].read()
        if not ret:
            logger.error("Failed to capture frame")
            await sio.emit("error", {"message": "Failed to capture frame"}, to=sid)
            break
        frame, stats, status = process_frame(frame, cuda_transform)
        _, buffer = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        session_duration = time.time() - state["start_time"]
        hours, remainder = divmod(int(session_duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        session_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        await sio.emit("frame", {
            "frame": frame_b64,
            "stats": stats,
            "sessionTime": session_time,
            "status": status.capitalize()
        }, to=sid)
        state["frame_count"] += 1
        await asyncio.sleep(0.033)

def process_frame(frame, cuda_transform):
    face_rois, face_coords = detect_faces(frame)
    stats = state["ppe_stats"]
    status = "No status detected yet"
    ppe_map = {name: idx for idx, name in enumerate(PPE_ITEMS)}
    target_idx = ppe_map[state["target_class"]]
    for face_roi, (x, y, w, h) in zip(face_rois, face_coords):
        pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        img_tensor = cuda_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = state["model"](img_tensor)
            probs = torch.sigmoid(logits).squeeze()
            detected_items = [PPE_ITEMS[i] for i in range(len(probs)) if probs[i] >= DETECTION_CONFIDENCE_THRESHOLD]
            confidence = probs[target_idx].item()
        if detected_items or confidence >= DETECTION_CONFIDENCE_THRESHOLD:
            state["ppe_history"].add_ppe(detected_items, confidence)
            status, acceptance_rate, flagged_items = state["ppe_history"].get_status(state["target_class"], state["num_classes"])
            if status == "Accepted":
                stats["Accepted"] += 1
                color = PPE_COLORS[state["target_class"]]
            else:
                stats["Flagged"] += 1
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            status_text = f"{status}: {acceptance_rate:.1f}%"
            if flagged_items:
                status_text += f" | Missing: {', '.join(flagged_items)}"
            cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame, stats, status

def record_session_data():
    if not state["start_time"] or state["frame_count"] == 0:
        logger.info("No session data to record: start_time or frame_count is invalid")
        return
    try:
        end_time = time.time()
        session_duration = end_time - state["start_time"]
        status, acceptance_rate, flagged_items = state["ppe_history"].get_status(state["target_class"], state["num_classes"])
        session_date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")
        with state["conn"].cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (employee_id, duration_seconds, status, target_class, session_date, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (state["user_id"], session_duration, status, state["target_class"], session_date, timestamp)
            )
            session_id = cur.fetchone()[0]
            logger.info(f"Session recorded: ID {session_id}")
            item = state["target_class"]
            count = 1 if item in flagged_items else 0
            percentage = 0.0 if count else 100.0
            cur.execute(
                """
                INSERT INTO ppe_details (session_id, ppe_item, count, percentage, flagged)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, item, count, percentage, bool(count))
            )
            logger.info(f"PPE details recorded for session ID {session_id}")
            state["conn"].commit()
    except psycopg.Error as e:
        logger.error(f"Database error in record_session_data: {e}")
        state["conn"].rollback()
        if "relation \"ppe_details\" does not exist" in str(e).lower():
            logger.error("ppe_details table does not exist. Please create it manually.")
    except Exception as e:
        logger.error(f"Unexpected error in record_session_data: {e}")
        state["conn"].rollback()

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        logger.warning("Invalid file type uploaded")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    max_file_size = 500 * 1024 * 1024
    if file.size > max_file_size:
        logger.warning(f"File size exceeds limit: {file.size} bytes")
        raise HTTPException(status_code=400, detail="File size exceeds 500 MB limit")
    cuda_transform = CUDATransform(transform, use_edge=True)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file_path = temp_file.name
            logger.info(f"Writing video to temporary file: {temp_file_path}")
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
        logger.info("Opening video for processing")
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            raise HTTPException(status_code=400, detail="Failed to open video file")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = min(total_frames, MAX_FRAMES_TO_PROCESS)
        state["ppe_stats"] = {"Accepted": 0, "Flagged": 0}
        state["ppe_history"] = PPEHistory(history_size=5)
        frame_count = 0
        state["start_time"] = time.time()
        ppe_map = {name: idx for idx, name in enumerate(PPE_ITEMS)}
        target_idx = ppe_map[state["target_class"]]
        logger.info(f"Processing video: {frames_to_process}/{total_frames} frames")
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video or read error")
                break
            face_rois, _ = detect_faces(frame)
            for face_roi in face_rois:
                pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                img_tensor = cuda_transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = state["model"](img_tensor)
                    probs = torch.sigmoid(logits).squeeze()
                    detected_items = [PPE_ITEMS[i] for i in range(len(probs)) if probs[i] >= DETECTION_CONFIDENCE_THRESHOLD]
                    confidence = probs[target_idx].item()
                if detected_items or confidence >= DETECTION_CONFIDENCE_THRESHOLD:
                    state["ppe_history"].add_ppe(detected_items, confidence)
                    status, _, _ = state["ppe_history"].get_status(state["target_class"], state["num_classes"])
                    state["ppe_stats"]["Accepted" if status == "Accepted" else "Flagged"] += 1
            frame_count += 1
            if frames_to_process < total_frames:
                skip_frames = max(1, int(total_frames / frames_to_process) - 1)
                for _ in range(skip_frames):
                    cap.read()
        cap.release()
        logger.info(f"Video processing complete: {frame_count} frames processed")
        record_session_data()
        total = sum(state["ppe_stats"].values())
        message = "Video Analysis Results:\n"
        if total > 0:
            for status, count in state["ppe_stats"].items():
                percentage = (count / total) * 100
                message += f"{status}: {percentage:.1f}%\n"
        else:
            message += "No PPE status detected in the video."
        logger.info("Video analysis results prepared")
        return {"message": message, "stats": state["ppe_stats"]}
    except Exception as e:
        logger.error(f"Error processing video upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Temporary file removed: {temp_file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")

@app.get("/test_samples")
async def test_samples():
    test_dir = os.path.join(BASE_DIR, "test")
    if not os.path.exists(test_dir):
        logger.warning(f"Test directory not found: {test_dir}")
        raise HTTPException(status_code=400, detail=f"Test data directory not found at {test_dir}")
    num_samples = 10
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) < num_samples:
        logger.warning(f"Not enough test images: found {len(image_files)}, need {num_samples}")
        raise HTTPException(status_code=400, detail=f"Not enough test images. Found {len(image_files)}, need {num_samples}")
    sample_indices = random.sample(range(len(image_files)), num_samples)
    state["ppe_stats"] = {"Accepted": 0, "Flagged": 0}
    state["ppe_history"] = PPEHistory(history_size=5)
    cuda_transform = CUDATransform(transform, use_edge=True)
    ppe_map = {name: idx for idx, name in enumerate(PPE_ITEMS)}
    target_idx = ppe_map[state["target_class"]]
    for idx in sample_indices:
        img_path = os.path.join(test_dir, image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img_tensor = cuda_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = state["model"](img_tensor)
            probs = torch.sigmoid(logits).squeeze()
            detected_items = [PPE_ITEMS[i] for i in range(len(probs)) if probs[i] >= DETECTION_CONFIDENCE_THRESHOLD]
            confidence = probs[target_idx].item()
        state["ppe_history"].add_ppe(detected_items, confidence)
        status, _, _ = state["ppe_history"].get_status(state["target_class"], state["num_classes"])
        state["ppe_stats"]["Accepted" if status == "Accepted" else "Flagged"] += 1
    message = "Random Sample Results:\n"
    for status, count in state["ppe_stats"].items():
        percentage = (count / num_samples) * 100 if num_samples > 0 else 0
        message += f"{status}: {percentage:.1f}%\n"
    logger.info(f"Test samples completed: {message}")
    return {"message": message, "stats": state["ppe_stats"]}

@app.get("/admin_stats")
async def admin_stats(department: str = "All", date_range: str = "Last 7 Days"):
    if department != "All" and department not in VALID_DEPTS:
        logger.warning(f"Invalid department for admin stats: {department}")
        raise HTTPException(status_code=400, detail="Invalid department")
    if date_range not in ["Last 7 Days", "Last 30 Days", "All Time"]:
        logger.warning(f"Invalid date range for admin stats: {date_range}")
        raise HTTPException(status_code=400, detail="Invalid date range")
    today = datetime.now().date()
    if date_range == "Last 7 Days":
        start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    elif date_range == "Last 30 Days":
        start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        start_date = "2000-01-01"
    dept_condition = "" if department == "All" else f"AND e.department = %s"
    try:
        with state["conn"].cursor() as cur:
            query_params = [start_date]
            if dept_condition:
                query_params.append(department)
            cur.execute(
                f"""
                SELECT COALESCE(s.status, 'No Data') AS status, COALESCE(COUNT(*), 0) AS total
                FROM sessions s
                JOIN employees e ON s.employee_id = e.employee_id
                WHERE s.session_date >= %s
                {dept_condition}
                GROUP BY s.status
                """,
                query_params
            )
            status_stats = cur.fetchall()
            cur.execute(
                f"""
                SELECT e.employee_id, e.department, COALESCE(COUNT(s.id), 0) AS session_count,
                       COALESCE(AVG(CASE WHEN s.status = 'Accepted' THEN 100.0 ELSE 0.0 END), 0.0) AS acceptance_rate,
                       COALESCE(STRING_AGG(pd.ppe_item, ', '), 'None') AS flagged_items
                FROM employees e
                LEFT JOIN sessions s ON e.employee_id = s.employee_id AND s.session_date >= %s
                LEFT JOIN ppe_details pd ON s.id = pd.session_id AND pd.flagged = TRUE
                {dept_condition}
                GROUP BY e.employee_id, e.department
                ORDER BY session_count DESC
                """,
                query_params
            )
            employee_stats = cur.fetchall()
        logger.info(f"Admin stats fetched: {len(status_stats)} status records, {len(employee_stats)} employee records")
        return {
            "compliance": [{"status": s[0], "total": s[1]} for s in status_stats] or [{"status": "No Data", "total": 0}],
            "employees": [
                {
                    "employee_id": e[0],
                    "department": e[1],
                    "session_count": e[2],
                    "acceptance_rate": float(e[3]),
                    "flagged_items": e[4]
                } for e in employee_stats
            ]
        }
    except psycopg.Error as e:
        logger.error(f"Database query error in admin_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in admin_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")