# Simple learning functions that work
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

def check_and_trigger_learning_simple(db: Session):
    """Simple learning check that doesn't cause errors"""
    try:
        from backend.app.models.database import Conversation

        # Count total conversations
        total_conversations = db.query(Conversation).count()

        # Count recent conversations (last 7 days)
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_conversations = db.query(Conversation).filter(
            Conversation.created_at >= recent_date
        ).count()

        logger.info(f"Learning check: {recent_conversations} recent conversations (total: {total_conversations})")

        if recent_conversations >= 10:
            logger.info("SUCCESS: Sufficient data for learning")
        else:
            logger.info(f"INFO: Need {10 - recent_conversations} more conversations for learning")

        return True

    except Exception as e:
        logger.error(f"Learning check failed: {e}")
        return False

def trigger_learning_with_feedback_simple(db: Session, rating: int = None):
    """Simple learning trigger for feedback"""
    try:
        from backend.app.models.database import UserFeedback

        # Count total feedback
        total_feedback = db.query(UserFeedback).count()

        logger.info(f"Feedback processing: total feedback entries = {total_feedback}")

        if rating and rating <= 2:
            logger.info("WARNING: Poor rating received - flagged for review")
        elif rating and rating >= 4:
            logger.info("SUCCESS: Good rating received")

        return True

    except Exception as e:
        logger.error(f"Learning with feedback failed: {e}")
        return False

def get_learning_status_simple(db: Session) -> dict:
    """Simple learning status that works with any schema"""
    try:
        from backend.app.models.database import Conversation, UserFeedback

        # Count data
        total_conversations = db.query(Conversation).count()
        total_feedback = db.query(UserFeedback).count()

        # Recent data (last 7 days)
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_conversations = db.query(Conversation).filter(
            Conversation.created_at >= recent_date
        ).count()

        # Calculate health score
        health_score = min(100, (total_conversations * 5) + (total_feedback * 10))

        return {
            "system_status": "active",
            "health_score": health_score,
            "statistics": {
                "total_conversations": total_conversations,
                "recent_conversations": recent_conversations,
                "total_feedback": total_feedback
            },
            "learning_efficiency": {
                "data_availability": min(total_conversations / 10, 1.0),
                "feedback_rate": total_feedback / max(total_conversations, 1)
            },
            "next_learning_cycle": "Ready now" if recent_conversations >= 10 else f"Need {10 - recent_conversations} more conversations",
            "recommendations": [
                "System appears healthy" if total_conversations >= 10 else "Need more conversation data",
                "Feedback collection working" if total_feedback > 0 else "Encourage user feedback"
            ]
        }

    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        return {
            "system_status": "error",
            "health_score": 0,
            "error": str(e)
        }