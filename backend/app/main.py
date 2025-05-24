import logging
import uvicorn
from contextlib import asynccontextmanager
from backend.app import create_app

# Set up proper logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global learning engine
learning_engine = None

@asynccontextmanager
async def lifespan(app):
    """Modern FastAPI lifespan event handler"""
    global learning_engine

    # Startup
    logger.info("Starting application initialization...")

    try:
        # Import here to avoid circular imports
        from backend.app.ml.learning_engine import LearningEngine

        logger.info("Importing Learning Engine...")
        learning_engine = LearningEngine()
        logger.info("Learning Engine initialized successfully during startup")

        # Optional: Setup learning scheduler
        setup_learning_scheduler()

    except ImportError as e:
        logger.error(f"Failed to import Learning Engine: {e}")
        logger.error("Check if all dependencies are installed and paths are correct")
    except Exception as e:
        logger.error(f"Failed to initialize Learning Engine: {e}")
        logger.error(f"Full error details: {str(e)}")
        # Continue startup even if learning engine fails
        logger.warning("Application will continue without learning engine")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application...")
    if learning_engine:
        logger.info("Learning Engine cleanup completed")
    logger.info("Application shutdown complete")

# Create the FastAPI app with lifespan
app = create_app(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Sri Lanka Tourism Chatbot API is running!"}

def setup_learning_scheduler():
    """Set up automatic learning cycles (optional)"""
    try:
        import schedule
        import threading
        import time

        def run_learning_cycle():
            """Run periodic learning cycle"""
            if learning_engine:
                try:
                    from backend.app.models.database import get_db
                    db = next(get_db())
                    result = learning_engine.continuous_learning_cycle(db)
                    logger.info(f"Scheduled learning cycle completed: {result.get('success', False)}")
                    db.close()
                except Exception as e:
                    logger.error(f"Error in scheduled learning cycle: {e}")

        def run_knowledge_update():
            """Run periodic knowledge update"""
            if learning_engine:
                try:
                    from backend.app.models.database import get_db
                    db = next(get_db())
                    result = learning_engine.update_knowledge_from_feedback(db)
                    logger.info(f"Scheduled knowledge update completed: {result.get('success', False)}")
                    db.close()
                except Exception as e:
                    logger.error(f"Error in scheduled knowledge update: {e}")

        def scheduler_worker():
            """Background scheduler worker"""
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        # Schedule learning operations
        schedule.every(6).hours.do(run_learning_cycle)
        schedule.every(2).hours.do(run_knowledge_update)

        # Start scheduler in background thread
        scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        scheduler_thread.start()

        logger.info("Learning scheduler initialized successfully")

    except ImportError:
        logger.warning("Schedule library not available - automatic learning cycles disabled")
        logger.info("Install with: pip install schedule")
    except Exception as e:
        logger.error(f"Failed to setup learning scheduler: {e}")

# Learning Engine Management Endpoints
@app.get("/api/admin/learning/status")
async def get_learning_status():
    """Get learning engine status"""
    if not learning_engine:
        return {
            "success": False,
            "error": "Learning engine not initialized",
            "initialized": False
        }

    try:
        from backend.app.models.database import get_db
        db = next(get_db())
        result = learning_engine.get_learning_status(db)
        db.close()
        return result
    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/admin/learning/cycle")
async def trigger_learning_cycle():
    """Manually trigger a learning cycle"""
    if not learning_engine:
        return {
            "success": False,
            "error": "Learning engine not initialized"
        }

    try:
        from backend.app.models.database import get_db
        db = next(get_db())
        result = learning_engine.force_learning_cycle(db)
        db.close()
        logger.info(f"Manual learning cycle triggered: {result.get('success', False)}")
        return result
    except Exception as e:
        logger.error(f"Error in manual learning cycle: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/admin/learning/update-knowledge")
async def update_knowledge():
    """Manually trigger knowledge update from feedback"""
    if not learning_engine:
        return {
            "success": False,
            "error": "Learning engine not initialized"
        }

    try:
        from backend.app.models.database import get_db
        db = next(get_db())
        result = learning_engine.update_knowledge_from_feedback(db)
        db.close()
        logger.info(f"Manual knowledge update triggered: {result.get('success', False)}")
        return result
    except Exception as e:
        logger.error(f"Error in manual knowledge update: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/health")
async def health_check():
    """Application health check"""
    return {
        "status": "healthy",
        "learning_engine_initialized": learning_engine is not None,
        "message": "Sri Lanka Tourism Chatbot API is running"
    }

@app.get("/api/health/learning")
async def learning_health_check():
    """Learning engine specific health check"""
    if not learning_engine:
        return {
            "status": "unavailable",
            "error": "Learning engine not initialized",
            "initialized": False
        }

    try:
        from backend.app.models.database import get_db
        db = next(get_db())
        status = learning_engine.get_learning_status(db)
        db.close()

        return {
            "status": "healthy" if status.get('system_status') == 'active' else "degraded",
            "initialized": True,
            "learning_status": status
        }
    except Exception as e:
        logger.error(f"Error in learning health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "initialized": True
        }

if __name__ == "__main__":
    # Add more detailed logging for development
    logging.getLogger("backend").setLevel(logging.INFO)

    logger.info("Starting Sri Lanka Tourism Chatbot API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )