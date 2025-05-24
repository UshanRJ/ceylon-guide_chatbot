# Fixed learning functions for your schema
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import logging

logger = logging.getLogger(__name__)

def get_learning_status_fixed(db: Session) -> dict:
    """Fixed learning status that works with your schema"""
    try:
        from backend.app.models.database import Conversation, UserFeedback, ModelMetrics
        
        # Get recent date (7 days ago)
        recent_date = datetime.utcnow() - timedelta(days=7)
        
        # Use created_at instead of timestamp
        total_conversations = db.query(Conversation).count()
        recent_conversations = db.query(Conversation).filter(
            Conversation.created_at >= recent_date
        ).count()
        
        # Count feedback using the correct relationship
        total_feedback = db.query(UserFeedback).count()
        recent_feedback = db.query(UserFeedback).join(Conversation).filter(
            Conversation.created_at >= recent_date
        ).count()
        
        # Count model metrics using metric_date instead of training_date
        total_metrics = db.query(ModelMetrics).count()
        recent_metrics = db.query(ModelMetrics).filter(
            ModelMetrics.metric_date >= recent_date
        ).count()
        
        # Calculate health score
        health_score = min(100, (total_conversations * 5) + (total_feedback * 10))
        
        # Calculate learning efficiency
        feedback_rate = total_feedback / max(total_conversations, 1)
        avg_rating = 4.0  # Default good rating
        
        if recent_feedback > 0:
            # Calculate average rating from UserFeedback.user_rating
            avg_rating_result = db.query(func.avg(UserFeedback.user_rating)).join(Conversation).filter(
                Conversation.created_at >= recent_date,
                UserFeedback.user_rating.isnot(None)
            ).scalar()
            if avg_rating_result:
                avg_rating = float(avg_rating_result)
        
        return {
            "system_status": "active",
            "health_score": health_score,
            "statistics": {
                "total_conversations": total_conversations,
                "recent_conversations": recent_conversations,
                "total_feedback": total_feedback,
                "recent_feedback_count": recent_feedback,
                "model_updates_last_30_days": recent_metrics
            },
            "learning_efficiency": {
                "feedback_rate": feedback_rate,
                "average_recent_rating": avg_rating,
                "learning_data_availability": min(recent_conversations / 10, 1.0)
            },
            "next_learning_cycle": "Ready now" if recent_conversations >= 10 else f"Need {10 - recent_conversations} more conversations",
            "recommendations": _generate_recommendations_fixed(total_conversations, total_feedback, recent_conversations)
        }
        
    except Exception as e:
        logger.error(f"Error in get_learning_status_fixed: {e}")
        return {
            "system_status": "error",
            "health_score": 0,
            "error": str(e),
            "statistics": {"total_conversations": 0, "recent_conversations": 0, "total_feedback": 0},
            "learning_efficiency": {"feedback_rate": 0, "average_recent_rating": 0, "learning_data_availability": 0},
            "next_learning_cycle": "Error",
            "recommendations": ["Fix database connection and schema issues"]
        }

def _generate_recommendations_fixed(total_conversations: int, total_feedback: int, recent_conversations: int) -> list:
    """Generate recommendations based on data"""
    recommendations = []
    
    if total_conversations < 10:
        recommendations.append(f"Need more conversation data - current: {total_conversations}, needed: 10")
    
    if recent_conversations < 5:
        recommendations.append("System needs recent user interactions for learning")
    
    if total_feedback == 0:
        recommendations.append("Encourage user feedback to improve learning")
    elif total_feedback / max(total_conversations, 1) < 0.1:
        recommendations.append("Low feedback rate - encourage more user ratings")
    
    if not recommendations:
        recommendations.append("System is healthy and ready for learning")
    
    return recommendations

def collect_learning_data_fixed(db: Session) -> dict:
    """Collect learning data using correct schema"""
    try:
        from backend.app.models.database import Conversation, UserFeedback
        
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        # Get conversations with the correct date field
        conversations = db.query(Conversation).filter(
            Conversation.created_at >= cutoff_date,
            Conversation.intent.isnot(None)
        ).all()
        
        # Get feedback data with proper relationship
        feedback_query = db.query(UserFeedback).join(Conversation).filter(
            Conversation.created_at >= cutoff_date
        ).all()
        
        # Organize data
        learning_data = {
            'conversations': [],
            'feedback_map': {},
            'total_samples': len(conversations)
        }
        
        # Create feedback map using conversation_id
        for feedback in feedback_query:
            learning_data['feedback_map'][feedback.conversation_id] = {
                'rating': feedback.user_rating or 3,  # Use user_rating field
                'is_helpful': feedback.user_rating >= 4 if feedback.user_rating else False,
                'feedback_text': feedback.feedback_text
            }
        
        # Process conversations
        for conv in conversations:
            conv_data = {
                'id': conv.id,
                'user_message': conv.user_message,
                'bot_response': conv.bot_response,
                'intent': conv.intent,
                'confidence_score': float(conv.confidence_score) if conv.confidence_score else 0.0,
                'entities': conv.entities or {},
                'tools_used': conv.tools_used or [],
                'created_at': conv.created_at,  # Use created_at
                'feedback': learning_data['feedback_map'].get(conv.id)
            }
            learning_data['conversations'].append(conv_data)
        
        return learning_data
        
    except Exception as e:
        logger.error(f"Error collecting learning data: {e}")
        return {'conversations': [], 'feedback_map': {}, 'total_samples': 0}
