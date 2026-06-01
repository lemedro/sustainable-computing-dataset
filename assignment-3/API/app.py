# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

app = FastAPI(
    title="Green Windows Scheduling API",
    description="API for cost and carbon-aware industrial scheduling using day-ahead energy price predictions.",
    version="1.0"
)

# Load the trained model (fallback to simulated data if not found for testing)
MODEL_PATH = "green_windows_rf_model.pkl"
try:
    rf_model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False
    print("Warning: ML model not found. API will use simulated baseline prices for testing.")

# --- Define API Data Schemas (Input & Output Validation) ---
class Task(BaseModel):
    name: str
    power_kw: float
    duration_hrs: int
    deadline_hr: int

class ScheduleRequest(BaseModel):
    date: str
    tasks: List[Task]

class ScheduledTask(BaseModel):
    task_name: str
    start_hour: int
    end_hour: int
    power_kw: float
    estimated_cost_eur: float

class ScheduleResponse(BaseModel):
    status: str
    date: str
    total_daily_cost_eur: float
    schedule: List[ScheduledTask]

# --- API Endpoint ---
@app.post("/api/v1/optimize-schedule", response_model=ScheduleResponse)
def optimize_schedule(request: ScheduleRequest):
    # 1. Generate 24-hour Day-Ahead Predictions
    # In a fully connected app, we would pass tomorrow's weather features into rf_model.predict()
    # For this demonstration, we simulate the predicted output curve.
    np.random.seed(42) 
    day_ahead_preds = np.random.uniform(40, 80, 24) 
    
    schedule_result = []
    total_cost = 0.0
    
    # 2. Run the Green Windows Optimization Algorithm
    for task in request.tasks:
        best_cost = float('inf')
        best_start = 0
        latest_start = task.deadline_hr - task.duration_hrs
        
        if latest_start < 0:
            raise HTTPException(status_code=400, detail=f"Task '{task.name}' duration exceeds its deadline.")
            
        for start in range(latest_start + 1):
            window_prices = day_ahead_preds[start : start + task.duration_hrs]
            cost = float(np.sum(window_prices) * task.power_kw / 1000) # Convert kW to MW
            
            if cost < best_cost:
                best_cost = cost
                best_start = start
                
        schedule_result.append(ScheduledTask(
            task_name=task.name,
            start_hour=best_start,
            end_hour=best_start + task.duration_hrs,
            power_kw=task.power_kw,
            estimated_cost_eur=round(best_cost, 2)
        ))
        total_cost += best_cost
        
    return ScheduleResponse(
        status="success",
        date=request.date,
        total_daily_cost_eur=round(total_cost, 2),
        schedule=schedule_result
    )