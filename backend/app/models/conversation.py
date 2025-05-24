from sqlalchemy.orm import Session
from backend.app.models.database import Conversation, UserFeedback
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta

class ConversationManager:
    def __init__(self, db: Session):
        self.db = db

    def save_conversation(self, session_id: str, user_message: str, bot_response: str,
                          intent: str = None, entities: Dict = None, confidence_score: float = None,
                          tools_used: List[str] = None) -> int:
        """Save conversation to database"""
        try:
            conversation = Conversation(
                session_id=session_id,
                user_message=user_message,
                bot_response=bot_response,
                intent=intent,
                entities=entities or {},
                confidence_score=str(confidence_score) if confidence_score else None,
                tools_used=tools_used or []
            )
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            return conversation.id
        except Exception as e:
            self.db.rollback()
            raise e

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history for a session"""
        conversations = self.db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).order_by(Conversation.timestamp.desc()).limit(limit).all()

        return [{
            'id': conv.id,
            'user_message': conv.user_message,
            'bot_response': conv.bot_response,
            'intent': conv.intent,
            'entities': conv.entities,
            'confidence_score': conv.confidence_score,
            'tools_used': conv.tools_used,
            'timestamp': conv.timestamp
        } for conv in reversed(conversations)]

    def save_feedback(self, conversation_id: int, rating: int, feedback_text: str = None,
                      is_helpful: bool = None) -> bool:
        """Save user feedback"""
        try:
            feedback = UserFeedback(
                conversation_id=conversation_id,
                rating=rating,
                feedback_text=feedback_text,
                is_helpful=is_helpful
            )
            self.db.add(feedback)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            return False

    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics data for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Total conversations
        total_conversations = self.db.query(Conversation).filter(
            Conversation.timestamp >= cutoff_date
        ).count()

        # Most common intents
        intent_counts = self.db.query(Conversation.intent).filter(
            Conversation.timestamp >= cutoff_date,
            Conversation.intent.isnot(None)
        ).all()

        intent_stats = {}
        for intent in intent_counts:
            if intent[0]:
                intent_stats[intent[0]] = intent_stats.get(intent[0], 0) + 1

        # Average confidence scores
        confidence_scores = self.db.query(Conversation.confidence_score).filter(
            Conversation.timestamp >= cutoff_date,
            Conversation.confidence_score.isnot(None)
        ).all()

        avg_confidence = 0
        if confidence_scores:
            valid_scores = [float(score[0]) for score in confidence_scores if score[0]]
            avg_confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return {
            'total_conversations': total_conversations,
            'intent_distribution': intent_stats,
            'average_confidence': round(avg_confidence, 2),
            'period_days': days
        }