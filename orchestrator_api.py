"""
Master Orchestrator API
FastAPI backend for serving dynamic dashboard data and managing orchestration state
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import random
from enum import Enum

app = FastAPI(title="Master Orchestrator API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ModelStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    TRAINING = "training"
    ERROR = "error"

class ModelType(str, Enum):
    API = "api"
    LOCAL = "local"

class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class Model(BaseModel):
    id: str
    name: str
    type: ModelType
    status: ModelStatus
    cost: str
    speed: str
    specialty: str
    current_load: float = 0.0
    accuracy_score: float = 0.0
    total_requests: int = 0
    error_rate: float = 0.0

class WorkflowTemplate(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    steps: List[str]
    suggested_models: List[str]
    estimated_duration: int  # in seconds

class ContextPack(BaseModel):
    id: str
    name: str
    type: str
    size: str
    last_updated: str
    tokens: int
    status: str = "ready"

class SystemStatus(BaseModel):
    orchestrator: str
    mlx_training: str
    vector_db: str
    api_models: str

class DashboardData(BaseModel):
    workflow_templates: List[WorkflowTemplate]
    available_models: List[Model]
    context_packs: List[ContextPack]
    system_status: SystemStatus
    active_workflows: int
    training_queue: int
    memory_usage: float
    total_contexts: float

# In-memory data store (replace with database in production)
class DataStore:
    def __init__(self):
        self.models = {
            "gpt4": Model(
                id="gpt4",
                name="GPT-4",
                type=ModelType.API,
                status=ModelStatus.ONLINE,
                cost="high",
                speed="medium",
                specialty="reasoning",
                current_load=0.3,
                accuracy_score=0.94,
                total_requests=1284,
                error_rate=0.02
            ),
            "claude": Model(
                id="claude",
                name="Claude Sonnet",
                type=ModelType.API,
                status=ModelStatus.ONLINE,
                cost="medium",
                speed="fast",
                specialty="analysis",
                current_load=0.45,
                accuracy_score=0.92,
                total_requests=2156,
                error_rate=0.01
            ),
            "llama_local": Model(
                id="llama_local",
                name="Llama 3.1 (Local)",
                type=ModelType.LOCAL,
                status=ModelStatus.ONLINE,
                cost="free",
                speed="fast",
                specialty="general",
                current_load=0.15,
                accuracy_score=0.87,
                total_requests=3421,
                error_rate=0.03
            ),
            "mistral_local": Model(
                id="mistral_local",
                name="Mistral 7B (Local)",
                type=ModelType.LOCAL,
                status=ModelStatus.ONLINE,
                cost="free",
                speed="very-fast",
                specialty="speed",
                current_load=0.22,
                accuracy_score=0.82,
                total_requests=4532,
                error_rate=0.04
            ),
            "deepseek": Model(
                id="deepseek",
                name="DeepSeek Coder",
                type=ModelType.API,
                status=ModelStatus.ONLINE,
                cost="low",
                speed="medium",
                specialty="coding",
                current_load=0.68,
                accuracy_score=0.89,
                total_requests=1876,
                error_rate=0.02
            ),
            "custom_persona": Model(
                id="custom_persona",
                name="Custom Persona",
                type=ModelType.LOCAL,
                status=ModelStatus.TRAINING,
                cost="free",
                speed="medium",
                specialty="style",
                current_load=0.0,
                accuracy_score=0.78,
                total_requests=234,
                error_rate=0.05
            )
        }
        
        self.workflow_templates = [
            WorkflowTemplate(
                id="research",
                name="Deep Research",
                description="Multi-source analysis with cross-validation",
                icon="FileText",
                steps=["Context Loading", "Source Gathering", "Cross-Analysis", "Synthesis"],
                suggested_models=["GPT-4", "Claude", "Llama-Local"],
                estimated_duration=420
            ),
            WorkflowTemplate(
                id="coding",
                name="Code Development",
                description="Build, test, and document features",
                icon="Code",
                steps=["Repo Analysis", "Code Generation", "Testing", "Documentation"],
                suggested_models=["DeepSeek", "GPT-4", "Mistral-Local"],
                estimated_duration=300
            ),
            WorkflowTemplate(
                id="creative",
                name="Creative Projects",
                description="Content creation with style consistency",
                icon="Brain",
                steps=["Style Analysis", "Content Generation", "Review", "Refinement"],
                suggested_models=["Claude", "GPT-4", "Custom-Persona"],
                estimated_duration=360
            )
        ]
        
        self.context_packs = [
            ContextPack(
                id="main_repo",
                name="Main Repository",
                type="code",
                size="2.3M tokens",
                last_updated="2 hours ago",
                tokens=2300000,
                status="ready"
            ),
            ContextPack(
                id="docs",
                name="Documentation",
                type="docs",
                size="800K tokens",
                last_updated="1 day ago",
                tokens=800000,
                status="ready"
            ),
            ContextPack(
                id="research_papers",
                name="Research Papers",
                type="research",
                size="1.2M tokens",
                last_updated="3 days ago",
                tokens=1200000,
                status="ready"
            )
        ]
        
        self.system_status = SystemStatus(
            orchestrator="active",
            mlx_training="active",
            vector_db="active",
            api_models="active"
        )
        
        self.active_workflows = []
        self.training_queue = []

data_store = DataStore()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Master Orchestrator API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/api/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    """Get all dashboard data in a single request"""
    
    # Calculate dynamic metrics
    total_contexts = sum(cp.tokens for cp in data_store.context_packs) / 1_000_000
    memory_usage = random.uniform(0.6, 0.85)  # Simulate memory usage
    active_workflows = len(data_store.active_workflows)
    training_queue = random.randint(1, 5)  # Simulate training queue
    
    return DashboardData(
        workflow_templates=data_store.workflow_templates,
        available_models=list(data_store.models.values()),
        context_packs=data_store.context_packs,
        system_status=data_store.system_status,
        active_workflows=active_workflows,
        training_queue=training_queue,
        memory_usage=memory_usage,
        total_contexts=total_contexts
    )

@app.get("/api/models")
async def get_models():
    """Get all available models with their current status"""
    return list(data_store.models.values())

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model details"""
    if model_id not in data_store.models:
        raise HTTPException(status_code=404, detail="Model not found")
    return data_store.models[model_id]

@app.post("/api/models/{model_id}/update-status")
async def update_model_status(model_id: str, status: ModelStatus):
    """Update model status"""
    if model_id not in data_store.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    data_store.models[model_id].status = status
    
    # Broadcast update to all connected clients
    await manager.broadcast({
        "type": "model_update",
        "model_id": model_id,
        "status": status
    })
    
    return {"success": True, "model_id": model_id, "new_status": status}

@app.get("/api/workflows")
async def get_workflow_templates():
    """Get all workflow templates"""
    return data_store.workflow_templates

@app.post("/api/workflows/execute")
async def execute_workflow(workflow_id: str, model_preferences: Optional[Dict[str, Any]] = None):
    """Execute a workflow"""
    workflow = next((w for w in data_store.workflow_templates if w.id == workflow_id), None)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Create workflow execution record
    execution = {
        "id": f"exec_{datetime.now().timestamp()}",
        "workflow_id": workflow_id,
        "status": WorkflowStatus.RUNNING,
        "started_at": datetime.now().isoformat(),
        "current_step": 0,
        "total_steps": len(workflow.steps)
    }
    
    data_store.active_workflows.append(execution)
    
    # Start async workflow execution
    asyncio.create_task(simulate_workflow_execution(execution, workflow))
    
    return execution

async def simulate_workflow_execution(execution: dict, workflow: WorkflowTemplate):
    """Simulate workflow execution with progress updates"""
    for i, step in enumerate(workflow.steps):
        execution["current_step"] = i + 1
        
        # Broadcast progress update
        await manager.broadcast({
            "type": "workflow_progress",
            "execution_id": execution["id"],
            "current_step": i + 1,
            "total_steps": execution["total_steps"],
            "step_name": step,
            "progress": (i + 1) / execution["total_steps"] * 100
        })
        
        # Simulate step execution time
        await asyncio.sleep(random.uniform(2, 5))
    
    execution["status"] = WorkflowStatus.COMPLETED
    execution["completed_at"] = datetime.now().isoformat()
    
    # Broadcast completion
    await manager.broadcast({
        "type": "workflow_complete",
        "execution_id": execution["id"],
        "status": "completed"
    })

@app.get("/api/context-packs")
async def get_context_packs():
    """Get all context packs"""
    return data_store.context_packs

@app.post("/api/context-packs/add")
async def add_context_pack(name: str, type: str, file_path: Optional[str] = None):
    """Add a new context pack"""
    new_pack = ContextPack(
        id=f"pack_{datetime.now().timestamp()}",
        name=name,
        type=type,
        size="Processing...",
        last_updated="Just now",
        tokens=0,
        status="processing"
    )
    
    data_store.context_packs.append(new_pack)
    
    # Simulate processing
    asyncio.create_task(process_context_pack(new_pack))
    
    return new_pack

async def process_context_pack(pack: ContextPack):
    """Simulate context pack processing"""
    await asyncio.sleep(5)
    
    # Update pack with processed data
    pack.tokens = random.randint(100000, 5000000)
    pack.size = f"{pack.tokens / 1_000_000:.1f}M tokens"
    pack.status = "ready"
    
    # Broadcast update
    await manager.broadcast({
        "type": "context_pack_ready",
        "pack_id": pack.id,
        "status": "ready"
    })

@app.get("/api/system/status")
async def get_system_status():
    """Get system status"""
    return data_store.system_status

@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    return {
        "cpu_usage": random.uniform(0.3, 0.7),
        "memory_usage": random.uniform(0.5, 0.8),
        "gpu_usage": random.uniform(0.2, 0.9),
        "api_calls_today": random.randint(1000, 5000),
        "total_workflows_executed": random.randint(50, 200),
        "average_workflow_time": random.uniform(180, 420),
        "model_accuracy_average": sum(m.accuracy_score for m in data_store.models.values()) / len(data_store.models),
        "total_tokens_processed": sum(cp.tokens for cp in data_store.context_packs)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "subscribe":
                # Handle subscription to specific updates
                pass
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

# Background task to simulate model metrics updates
async def update_model_metrics():
    """Periodically update model metrics to simulate real activity"""
    while True:
        await asyncio.sleep(10)
        
        for model in data_store.models.values():
            # Simulate load changes
            model.current_load = max(0, min(1, model.current_load + random.uniform(-0.2, 0.2)))
            
            # Occasionally update request counts
            if random.random() > 0.7:
                model.total_requests += random.randint(1, 10)
            
            # Broadcast updates
            await manager.broadcast({
                "type": "metrics_update",
                "model_id": model.id,
                "current_load": model.current_load,
                "total_requests": model.total_requests
            })

# Start background tasks on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_model_metrics())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)