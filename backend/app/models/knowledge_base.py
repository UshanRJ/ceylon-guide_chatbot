from sqlalchemy.orm import Session
from backend.app.models.database import KnowledgeBase
from typing import List, Dict, Any
import json

class KnowledgeBaseManager:
    def __init__(self, db: Session):
        self.db = db

    def search_knowledge(self, query: str, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Search knowledge base using keywords and query similarity"""
        base_query = self.db.query(KnowledgeBase)

        if keywords:
            # Search by keywords
            results = []
            for kb_item in base_query.all():
                kb_keywords = json.loads(kb_item.keywords) if isinstance(kb_item.keywords, str) else kb_item.keywords
                if any(keyword.lower() in [kw.lower() for kw in kb_keywords] for keyword in keywords):
                    results.append({
                        'id': kb_item.id,
                        'category': kb_item.category,
                        'question': kb_item.question,
                        'answer': kb_item.answer,
                        'keywords': kb_keywords,
                        'relevance_score': self._calculate_relevance(query, kb_item.answer, keywords, kb_keywords)
                    })

            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:3]  # Return top 3 results

        # Fallback to text search
        results = base_query.filter(
            KnowledgeBase.answer.contains(query) |
            KnowledgeBase.question.contains(query)
        ).limit(3).all()

        return [{
            'id': item.id,
            'category': item.category,
            'question': item.question,
            'answer': item.answer,
            'keywords': json.loads(item.keywords) if isinstance(item.keywords, str) else item.keywords,
            'relevance_score': 0.5
        } for item in results]

    def _calculate_relevance(self, query: str, answer: str, query_keywords: List[str], kb_keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches and text similarity"""
        score = 0.0

        # Keyword matching score
        if query_keywords and kb_keywords:
            matching_keywords = len(set([kw.lower() for kw in query_keywords]).intersection(
                set([kw.lower() for kw in kb_keywords])
            ))
            score += (matching_keywords / len(query_keywords)) * 0.7

        # Text similarity score (simple word overlap)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        word_overlap = len(query_words.intersection(answer_words))
        score += (word_overlap / max(len(query_words), 1)) * 0.3

        return min(score, 1.0)

    def add_knowledge(self, category: str, question: str, answer: str, keywords: List[str]) -> bool:
        """Add new knowledge to the database"""
        try:
            new_knowledge = KnowledgeBase(
                category=category,
                question=question,
                answer=answer,
                keywords=keywords
            )
            self.db.add(new_knowledge)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            return False

    def update_knowledge(self, knowledge_id: int, **kwargs) -> bool:
        """Update existing knowledge"""
        try:
            knowledge = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_id).first()
            if knowledge:
                for key, value in kwargs.items():
                    if hasattr(knowledge, key):
                        setattr(knowledge, key, value)
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            return False