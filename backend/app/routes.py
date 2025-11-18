import os
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
from fastapi import Form, File, Depends, APIRouter, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from celery.result import AsyncResult
from app.storage import UserStorage, RunStorage, NotificationStorage
from app.schemas import (
    UserCreate, UserLogin, Token, UserResponse, 
    ModelRunCreate, ModelRunResponse, NotificationResponse,
    FeedbackRequest, NoteRequest
)
from app.auth import (
    verify_password, get_password_hash, create_access_token, get_current_user
)
from app.config import settings
from app.model_tasks import train_model
from app.send_email import send_clinician_review_email, send_model_completed_email, send_model_failed_email

router = APIRouter()

# Auth routes
@router.post("/auth/signup", response_model=UserResponse)
async def signup(user_data: UserCreate):
    # Check if user exists
    if UserStorage.get_by_email(user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = UserStorage.create({
        'email': user_data.email,
        'hashed_password': hashed_password,
        'role': user_data.role.value
    })
    return user

@router.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    user = UserStorage.get_by_email(user_data.email)
    
    if not user or not verify_password(user_data.password, user['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['email']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

# Model run routes
@router.post("/runs", response_model=ModelRunResponse)
async def create_run(
    model_data: str = Form(...), 
    # dataset_name: str = Form(...), 
    # file: UploadFile = File(...),
    # current_user: dict = Depends(get_current_user)
    # model_data: ModelRunCreate,
    dataset_file: UploadFile = File(...),
    parameters_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    
    model_data_dict = json.loads(model_data)
    print("Received model data:", model_data_dict)
        # 3. Validate into the Pydantic model (string "kmeans" becomes ModelType.kmeans)
    model_data_obj = ModelRunCreate(**model_data_dict)

    # Create run record
    print("Details of new run taken in, creating record...")
    # This will print the string "kmeans"
    print(model_data_obj.model_type.value)
    print(model_data_obj.dataset_name)
    print(model_data_obj.dataset_details)
    run = RunStorage.create({
        'user_id': current_user['id'],
        'user_email': current_user['email'],  
        'model_type': model_data_obj.model_type.value,
        'dataset_name': model_data_obj.dataset_name,
        'dataset_details': model_data_obj.dataset_details,
        'dataset_filename': dataset_file.filename,
        'parameters_filename': parameters_file.filename,
    })

    # Create folder name: <model>_run<runid>_<datetime>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_data_obj.model_type}_run{run['id']}_{timestamp}"
    # folder_path = os.path.join(settings.RESULTS_DIR, folder_name)
    
    results_dir = Path(settings.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    folder_path = results_dir / folder_name
    # Save uploaded file
    os.makedirs(folder_path, exist_ok=True)

    # Create metadata file with model details
    metadata_file = folder_path / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Dataset Metadata\n")
        f.write(f"{'='*50}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Name: {model_data_obj.dataset_name}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"Data Details:\n")
        f.write(f"{model_data_obj.dataset_details}\n")
    print(f"Metadata file created at: {metadata_file}")
    
    # dataset_path = os.path.join(absolute_folder_path, dataset_file.filename)
    abs_dataset_path = (folder_path/dataset_file.filename).resolve()
    print(abs_dataset_path)

    with open(abs_dataset_path, "wb") as buffer:
        shutil.copyfileobj(dataset_file.file, buffer)

    abs_parameters_path = (folder_path/parameters_file.filename).resolve()
    with open(abs_parameters_path, "wb") as buffer:
        shutil.copyfileobj(parameters_file.file, buffer)
    
    # Update run with folder path
    RunStorage.update(run['id'], {'folder_path': folder_path})
    
    # Start Celery task
    task = train_model.delay(
        run['id'], 
        model_data_obj.model_type.value,
        str(abs_dataset_path).replace("\\", "/"),
        str(abs_parameters_path).replace("\\", "/"),
        model_data_obj.dataset_name, 
        str(folder_path).replace("\\", "/")
    )
    
    # Update with task ID and status
    run = RunStorage.update(run['id'], {
        'celery_task_id': task.id,
        'status': 'running'
    })
    
    return run

@router.get("/runs", response_model=List[ModelRunResponse])
async def get_runs(current_user: dict = Depends(get_current_user)):
    if current_user['role'] == 'data_scientist':
        runs = RunStorage.get_all()
    if current_user['role'] == 'clinician':
        runs = RunStorage.get_all()
        runs = [r for r in runs if r['sent_to_clinician']]
        # runs = RunStorage.get_by_user(current_user['id'])
    return sorted(runs, key=lambda x: x['created_at'], reverse=True)

@router.get("/runs/{run_id}", response_model=ModelRunResponse)
async def get_run(run_id: int, current_user: dict = Depends(get_current_user)):
    run = RunStorage.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Check task status if running
    if run['status'] == 'running' and run['celery_task_id']:
        task_result = AsyncResult(run['celery_task_id'])
        if task_result.ready():
            result = task_result.result
            if result and result.get("status") == "success":
                run = RunStorage.update(run_id, {
                    'status': 'completed',
                    'optimal_clusters': result.get('optimal_clusters'),
                    'completed_at': datetime.utcnow().isoformat()
                })
                
                # Send completion email to the data scientist
                user = UserStorage.get_by_id(run['user_id'])
                if user:
                    send_model_completed_email(
                        to_email=user['email'],
                        to_name=user['email'].split('@')[0],
                        run_id=run_id,
                        model_type=run['model_type'],
                        optimal_clusters=result.get('optimal_clusters')
                    )
            else:
                run = RunStorage.update(run_id, {'status': 'failed'})
                
                # Send failure email to the data scientist
                user = UserStorage.get_by_id(run['user_id'])
                if user:
                    send_model_failed_email(
                        to_email=user['email'],
                        to_name=user['email'].split('@')[0],
                        run_id=run_id,
                        model_type=run['model_type']
                    )
    
    return run

@router.get("/runs/{run_id}/plots")
async def get_run_plots(run_id: int, current_user: dict = Depends(get_current_user)):
    run = RunStorage.get_by_id(run_id)
    # if not run or run['user_id'] != current_user['id']:
    #     raise HTTPException(status_code=404, detail="Run not found")
    
    # Get all PNG files from folder
    plots = []
    if run['folder_path'] and os.path.exists(run['folder_path']):
        for file in os.listdir(run['folder_path']):
            if file.endswith('.png'):
                plots.append(file)
    
    return {"plots": plots}

@router.get("/runs/{run_id}/plots/{filename}")
async def get_plot_file(
    run_id: int,
    filename: str,
    current_user: dict = Depends(get_current_user)
):
    run = RunStorage.get_by_id(run_id)
    # if not run or run['user_id'] != current_user['id']:
    #     raise HTTPException(status_code=404, detail="Run not found")
    
    if not run['folder_path']:
        raise HTTPException(status_code=404, detail="Run folder not found")
    
    file_path = os.path.join(run['folder_path'], filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@router.get("/runs/{run_id}/notes")
async def get_notes(run_id: int, current_user: dict = Depends(get_current_user)):
    run = RunStorage.get_by_id(run_id)
    # if not run or run['user_id'] != current_user['id']:
    #     raise HTTPException(status_code=404, detail="Run not found")
    
    if not run['folder_path']:
        raise HTTPException(status_code=404, detail="Run folder not found")
    
    notes_file = os.path.join(run['folder_path'], 'notes_feedback.txt')
    if os.path.exists(notes_file):
        with open(notes_file, 'r') as f:
            content = f.read()
        return {"content": content}
    return {"content": ""}

@router.post("/runs/{run_id}/notes")
async def add_note(
    run_id: int,
    note_data: NoteRequest,
    current_user: dict = Depends(get_current_user)
):
    run = RunStorage.get_by_id(run_id)
    if not run or run['user_id'] != current_user['id']:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if not run['folder_path']:
        raise HTTPException(status_code=404, detail="Run folder not found")
    
    print(run['folder_path'])
    notes_file = os.path.join(run['folder_path'], 'notes_feedback.txt')
    notes_directory = os.path.dirname(notes_file)
    
    # Create the directory (and any parent directories)
    # exist_ok=True prevents an error if the directory already exists
    os.makedirs(notes_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(notes_file, 'a') as f:
        f.write(f"\n[{timestamp}] Note by {current_user['email']}:\n")
        f.write(f"{note_data.note}\n")
        f.write("-" * 50 + "\n")
    
    return {"status": "success"}

@router.post("/runs/{run_id}/feedback")
async def add_feedback(
    run_id: int,
    feedback_data: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    run = RunStorage.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if not run['folder_path']:
        raise HTTPException(status_code=404, detail="Run folder not found")
    
    notes_file = os.path.join(run['folder_path'], 'notes_feedback.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(notes_file, 'a') as f:
        f.write(f"\n[{timestamp}] FEEDBACK by {current_user['email']} (Clinician):\n")
        f.write(f"{feedback_data.feedback}\n")
        f.write("=" * 50 + "\n")
    
    # Mark feedback as added
    RunStorage.update(run_id, {'feedback_added': True})
    
    # Create notification for data scientist
    data_scientist = UserStorage.get_by_id(run['user_id'])
    if data_scientist:
        NotificationStorage.create({
            'user_id': data_scientist['id'],
            'run_id': run_id,
            'message': f"Feedback added for run #{run_id}. Click to view the feedback."
        })
    
    return {"status": "success"}

@router.post("/runs/{run_id}/send-to-clinician")
async def send_to_clinician(
    run_id: int,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    run = RunStorage.get_by_id(run_id)
    if not run or run['user_id'] != current_user['id']:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Run not completed yet")
    
    # Mark as sent to clinician
    RunStorage.update(run_id, {'sent_to_clinician': True})
    hardcoded_test_email = "nathaniayeo.2022@scis.smu.edu.sg"
    
    # Create notifications for all clinicians
    all_users = UserStorage.get_all()
    clinicians = [u for u in all_users if u['role'] == 'clinician']

    message = f"The model run #{run_id} has been sent to you for review. Please log in to the platform to view the results and provide feedback."
    
    for clinician in clinicians:
        NotificationStorage.create({
            'user_id': clinician['id'],
            'run_id': run_id,
            'message': f"Model run #{run_id} completed. Waiting for your feedback."
        })

        send_clinician_review_email(
            to_email=clinician['email'],
            to_name=clinician['email'].split('@')[0],
            run_id=run_id,
            data_scientist_name=current_user['email'].split('@')[0]
        )
    
    return {"status": "success"}

# Notification routes
@router.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(current_user: dict = Depends(get_current_user)):
    notifications = NotificationStorage.get_by_user(current_user['id'])
    print("notifications:", notifications)
    return sorted(notifications, key=lambda x: x['created_at'], reverse=True)

@router.put("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: dict = Depends(get_current_user)
):
    notification = NotificationStorage.get_by_id(notification_id)
    if not notification or notification['user_id'] != current_user['id']:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    NotificationStorage.update(notification_id, {'is_read': True})
    return {"status": "success"}