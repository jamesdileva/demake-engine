"""Health check — useful for deployment and debugging."""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status":    "ok",
        "service":   "demake-engine",
        "timestamp": datetime.utcnow().isoformat(),
    }