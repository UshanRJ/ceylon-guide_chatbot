# backend/app/ml/learning_engine.py
import json
import os
import pickle
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import logging
from collections import defaultdict, Counter
import threading
import time

from backend.app.models.database import get_db, Conversation, UserFeedback, ModelMetrics, KnowledgeBase, LearningData
from backend.app.models.knowledge_base import KnowledgeBaseManager
from backend.app.nlp.intent_classifier import IntentClassifier
from backend.app.nlp.entity_extractor import EntityExtractor
from backend.app.utils.helpers import extract_keywords, clean_text, ChatbotError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningEngine:
    """
    Advanced machine learning engine for continuous learning and model improvement
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.learning_threshold = 10  # Minimum interactions before retraining
        self.confidence_threshold = 0.6  # Minimum confidence for auto-learning
        self.feedback_threshold = 0.7  # Minimum positive feedback ratio
        self.learning_lock = threading.Lock()  # Thread safety for learning operations

        # Learning parameters
        self.learning_config = {
            'min_samples_for_retraining': 20,
            'feedback_weight': 0.3,
            'confidence_weight': 0.4,
            'frequency_weight': 0.3,
            'max_training_iterations': 5,
            'performance_degradation_threshold': 0.05,
            'auto_learning_enabled': True,
            'learning_rate_decay': 0.95
        }

        # Knowledge categories for structured learning
        self.knowledge_categories = {
            'destinations': {'weight': 1.0, 'priority': 'high'},
            'transportation': {'weight': 0.9, 'priority': 'high'},
            'accommodation': {'weight': 0.8, 'priority': 'medium'},
            'food': {'weight': 0.7, 'priority': 'medium'},
            'culture': {'weight': 0.8, 'priority': 'medium'},
            'weather': {'weight': 0.6, 'priority': 'low'},
            'currency': {'weight': 0.6, 'priority': 'low'},
            'practical': {'weight': 0.9, 'priority': 'high'},
            'emergency': {'weight': 1.0, 'priority': 'critical'}
        }

        logger.info("Learning Engine initialized with advanced ML capabilities")

    def update_knowledge_from_feedback(self, db: Session) -> Dict[str, Any]:
        """
        Update knowledge base based on comprehensive user feedback analysis
        """
        try:
            logger.info("Starting knowledge update from feedback analysis")

            # Get feedback data for analysis
            feedback_data = self._get_comprehensive_feedback_data(db)

            if not feedback_data:
                return {"success": True, "message": "No feedback data available", "updates": 0}

            # Analyze feedback patterns
            feedback_analysis = self._analyze_feedback_patterns(feedback_data)

            # Generate knowledge updates
            knowledge_updates = self._generate_knowledge_updates(feedback_analysis, db)

            # Apply updates to knowledge base
            update_results = self._apply_knowledge_updates(knowledge_updates, db)

            # Update learning metrics
            self._update_learning_metrics(feedback_analysis, update_results, db)

            logger.info(f"Knowledge update completed: {update_results['total_updates']} updates applied")

            return {
                "success": True,
                "feedback_analyzed": len(feedback_data),
                "updates_applied": update_results['total_updates'],
                "categories_updated": update_results['categories'],
                "learning_insights": feedback_analysis['insights']
            }

        except Exception as e:
            logger.error(f"Error in knowledge update from feedback: {e}")
            return {"success": False, "error": str(e)}

    def continuous_learning_cycle(self, db: Session) -> Dict[str, Any]:
        """
        Run comprehensive continuous learning cycle with advanced ML techniques
        """
        with self.learning_lock:
            try:
                logger.info("Starting continuous learning cycle")

                # Step 1: Data Collection and Analysis
                learning_data = self._collect_learning_data(db)

                if learning_data['total_samples'] < self.learning_config['min_samples_for_retraining']:
                    return {
                        "success": True,
                        "message": f"Insufficient data for learning cycle. Need {self.learning_config['min_samples_for_retraining']}, have {learning_data['total_samples']}",
                        "data_collected": learning_data['total_samples']
                    }

                # Step 2: Extract and Save Learning Insights
                learning_insights = self._extract_learning_insights_from_conversations(learning_data)
                insights_saved = self._save_learning_data_to_db(learning_insights, db)

                # Step 3: Performance Analysis
                performance_analysis = self._analyze_model_performance(db)

                # Step 4: Intent Classification Improvement
                intent_results = self._improve_intent_classification(learning_data, db)

                # Step 5: Entity Extraction Enhancement
                entity_results = self._enhance_entity_extraction(learning_data, db)

                # Step 6: Knowledge Base Expansion
                knowledge_results = self._expand_knowledge_base(learning_data, db)

                # Step 7: Response Quality Improvement
                response_results = self._improve_response_quality(learning_data, db)

                # Step 8: Model Validation and Testing
                validation_results = self._validate_learning_improvements(db)

                # Step 9: Learning Analytics Update
                analytics_results = self._update_learning_analytics(
                    learning_data, performance_analysis, validation_results, db
                )

                # Step 10: Update Model Metrics with actual improvements
                self._update_model_metrics_with_improvements(learning_data, db)

                # Step 11: Cleanup and Optimization
                self._cleanup_learning_data(db)

                learning_summary = {
                    "success": True,
                    "cycle_completed": datetime.now().isoformat(),
                    "data_processed": learning_data['total_samples'],
                    "learning_insights_saved": insights_saved,  # NEW: Track saved insights
                    "improvements_made": {
                        "intent_classification": intent_results,
                        "entity_extraction": entity_results,
                        "knowledge_expansion": knowledge_results,
                        "response_quality": response_results
                    },
                    "performance_metrics": validation_results,
                    "learning_analytics": analytics_results
                }

                logger.info(f"Continuous learning cycle completed successfully: {learning_summary}")
                return learning_summary

            except Exception as e:
                logger.error(f"Error in continuous learning cycle: {e}")
                return {"success": False, "error": str(e)}

    def _get_comprehensive_feedback_data(self, db: Session) -> List[Dict[str, Any]]:
        """
        Collect comprehensive feedback data for analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=30)  # Last 30 days

            # Query conversations with feedback
            query = db.query(
                Conversation.id,
                Conversation.user_message,
                Conversation.bot_response,
                Conversation.intent,
                Conversation.confidence_score,
                Conversation.entities,
                Conversation.tools_used,
                Conversation.created_at,
                UserFeedback.user_rating,
                UserFeedback.feedback_text,
                UserFeedback.would_use_again
            ).join(
                UserFeedback, Conversation.id == UserFeedback.conversation_id
            ).filter(
                Conversation.created_at >= cutoff_date
            ).all()

            feedback_data = []
            for row in query:
                feedback_data.append({
                    'conversation_id': row.id,
                    'user_message': row.user_message,
                    'bot_response': row.bot_response,
                    'intent': row.intent,
                    'confidence_score': float(row.confidence_score) if row.confidence_score else 0.0,
                    'entities': row.entities or {},
                    'tools_used': row.tools_used or [],
                    'timestamp': row.created_at,
                    'rating': row.user_rating,
                    'feedback_text': row.feedback_text,
                    'is_helpful': row.would_use_again
                })

            return feedback_data

        except Exception as e:
            logger.error(f"Error collecting feedback data: {e}")
            return []

    def _analyze_feedback_patterns(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze feedback patterns to identify learning opportunities
        """
        try:
            analysis = {
                'total_feedback': len(feedback_data),
                'rating_distribution': Counter(),
                'intent_performance': defaultdict(list),
                'confidence_correlation': [],
                'tool_performance': defaultdict(list),
                'common_issues': [],
                'improvement_areas': [],
                'insights': []
            }

            # Analyze rating distribution
            for feedback in feedback_data:
                rating = feedback['rating']
                analysis['rating_distribution'][rating] += 1

                # Intent performance analysis
                intent = feedback['intent']
                if intent:
                    analysis['intent_performance'][intent].append({
                        'rating': rating,
                        'confidence': feedback['confidence_score'],
                        'helpful': feedback['is_helpful']
                    })

                # Confidence correlation
                analysis['confidence_correlation'].append({
                    'confidence': feedback['confidence_score'],
                    'rating': rating,
                    'helpful': feedback['is_helpful']
                })

                # Tool performance
                for tool in feedback['tools_used']:
                    analysis['tool_performance'][tool].append(rating)

            # Generate insights
            analysis['insights'] = self._generate_learning_insights(analysis)

            # Identify improvement areas
            analysis['improvement_areas'] = self._identify_improvement_areas(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return {}

    def _generate_learning_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate actionable learning insights from feedback analysis
        """
        insights = []

        try:
            # Rating insights
            total_feedback = analysis['total_feedback']
            if total_feedback > 0:
                avg_rating = sum(rating * count for rating, count in analysis['rating_distribution'].items()) / total_feedback
                if avg_rating < 3.0:
                    insights.append("Overall user satisfaction is below average - need significant improvements")
                elif avg_rating < 3.5:
                    insights.append("User satisfaction is moderate - focus on key improvement areas")
                else:
                    insights.append("User satisfaction is good - focus on maintaining quality")

            # Intent performance insights
            for intent, performances in analysis['intent_performance'].items():
                if len(performances) >= 5:  # Sufficient data
                    avg_rating = sum(p['rating'] for p in performances) / len(performances)
                    avg_confidence = sum(p['confidence'] for p in performances) / len(performances)

                    if avg_rating < 3.0:
                        insights.append(f"Intent '{intent}' has poor performance - needs model retraining")
                    if avg_confidence < 0.6:
                        insights.append(f"Intent '{intent}' has low confidence - needs more training data")

            # Tool performance insights
            for tool, ratings in analysis['tool_performance'].items():
                if len(ratings) >= 3:
                    avg_rating = sum(ratings) / len(ratings)
                    if avg_rating < 3.0:
                        insights.append(f"Tool '{tool}' has poor user satisfaction - needs improvement")

            # Confidence correlation insights
            if analysis['confidence_correlation']:
                high_conf_low_rating = [
                    item for item in analysis['confidence_correlation']
                    if item['confidence'] > 0.8 and item['rating'] < 3
                ]
                if len(high_conf_low_rating) > len(analysis['confidence_correlation']) * 0.1:
                    insights.append("High confidence predictions with low ratings detected - model may be overconfident")

            return insights

        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
            return []

    def _identify_improvement_areas(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify specific areas for improvement with priorities
        """
        improvement_areas = []

        try:
            # Intent-based improvements
            for intent, performances in analysis['intent_performance'].items():
                if len(performances) >= 3:
                    avg_rating = sum(p['rating'] for p in performances) / len(performances)
                    avg_confidence = sum(p['confidence'] for p in performances) / len(performances)

                    if avg_rating < 3.5 or avg_confidence < 0.7:
                        priority = 'high' if avg_rating < 3.0 else 'medium'
                        improvement_areas.append({
                            'area': 'intent_classification',
                            'intent': intent,
                            'issue': 'low_performance',
                            'priority': priority,
                            'avg_rating': avg_rating,
                            'avg_confidence': avg_confidence,
                            'sample_count': len(performances)
                        })

            # Tool-based improvements
            for tool, ratings in analysis['tool_performance'].items():
                if len(ratings) >= 3:
                    avg_rating = sum(ratings) / len(ratings)
                    if avg_rating < 3.5:
                        priority = 'high' if avg_rating < 3.0 else 'medium'
                        improvement_areas.append({
                            'area': 'tool_integration',
                            'tool': tool,
                            'issue': 'low_user_satisfaction',
                            'priority': priority,
                            'avg_rating': avg_rating,
                            'sample_count': len(ratings)
                        })

            # Sort by priority
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            improvement_areas.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)

            return improvement_areas

        except Exception as e:
            logger.error(f"Error identifying improvement areas: {e}")
            return []

    def _collect_learning_data(self, db: Session) -> Dict[str, Any]:
        """
        Collect comprehensive data for learning algorithms
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=7)  # Last 7 days for active learning

            # Get recent conversations
            conversations = db.query(Conversation).filter(
                Conversation.created_at >= cutoff_date,
                Conversation.intent.isnot(None)
            ).all()

            # Get feedback data
            feedback_query = db.query(UserFeedback).join(
                Conversation, UserFeedback.conversation_id == Conversation.id
            ).filter(
                Conversation.created_at >= cutoff_date
            ).all()

            # Organize learning data
            learning_data = {
                'conversations': [],
                'feedback_map': {},
                'intent_distribution': Counter(),
                'entity_patterns': defaultdict(list),
                'response_patterns': defaultdict(list),
                'tool_usage_patterns': defaultdict(int),
                'total_samples': len(conversations)
            }

            # Create feedback map
            for feedback in feedback_query:
                learning_data['feedback_map'][feedback.conversation_id] = {
                    'rating': feedback.user_rating,
                    'is_helpful': feedback.would_use_again,
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
                    'timestamp': conv.created_at,
                    'feedback': learning_data['feedback_map'].get(conv.id)
                }

                learning_data['conversations'].append(conv_data)

                # Update distributions
                learning_data['intent_distribution'][conv.intent] += 1

                # Entity patterns
                for entity_type, entities in (conv.entities or {}).items():
                    for entity in entities:
                        learning_data['entity_patterns'][entity_type].append({
                            'text': entity.get('text', ''),
                            'confidence': entity.get('confidence', 0.0),
                            'context': conv.user_message
                        })

                # Tool usage patterns
                for tool in (conv.tools_used or []):
                    learning_data['tool_usage_patterns'][tool] += 1

                # Response patterns
                learning_data['response_patterns'][conv.intent].append({
                    'response': conv.bot_response,
                    'user_message': conv.user_message,
                    'tools_used': conv.tools_used or [],
                    'feedback': learning_data['feedback_map'].get(conv.id)
                })

            return learning_data

        except Exception as e:
            logger.error(f"Error collecting learning data: {e}")
            return {'total_samples': 0}

    def _improve_intent_classification(self, learning_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Improve intent classification based on learning data
        """
        try:
            logger.info("Improving intent classification model")

            # Prepare training data from recent conversations
            training_data = []
            validation_data = []

            for conv in learning_data['conversations']:
                # Filter high-confidence and well-rated conversations
                if (conv['confidence_score'] > self.confidence_threshold and
                        conv['feedback'] and
                        conv['feedback']['rating'] >= 4):

                    training_data.append({
                        'text': conv['user_message'],
                        'intent': conv['intent'],
                        'confidence': conv['confidence_score']
                    })

                # Use lower-rated conversations for validation
                elif conv['feedback'] and conv['feedback']['rating'] <= 2:
                    validation_data.append({
                        'text': conv['user_message'],
                        'intent': conv['intent'],
                        'expected_quality': 'poor'
                    })

            if len(training_data) < 10:
                return {'success': False, 'reason': 'Insufficient training data'}

            # Retrain intent classifier with new data
            old_model_path = self.intent_classifier.model_path
            backup_path = old_model_path + '.backup'

            # Backup current model
            if os.path.exists(old_model_path):
                os.rename(old_model_path, backup_path)

            try:
                # Retrain with combined data
                self.intent_classifier.train_model()

                # Validate improvement
                improvement_score = self._validate_intent_improvement(validation_data)

                if improvement_score > 0.05:  # 5% improvement threshold
                    # Keep new model
                    if os.path.exists(backup_path):
                        os.remove(backup_path)

                    # Update model metrics
                    self._save_model_metrics('intent_classifier', improvement_score, db)

                    return {
                        'success': True,
                        'training_samples': len(training_data),
                        'improvement_score': improvement_score,
                        'action': 'model_updated'
                    }
                else:
                    # Restore backup if no significant improvement
                    if os.path.exists(backup_path):
                        os.rename(backup_path, old_model_path)

                    return {
                        'success': True,
                        'training_samples': len(training_data),
                        'improvement_score': improvement_score,
                        'action': 'model_restored'
                    }

            except Exception as e:
                # Restore backup on error
                if os.path.exists(backup_path):
                    os.rename(backup_path, old_model_path)
                raise e

        except Exception as e:
            logger.error(f"Error improving intent classification: {e}")
            return {'success': False, 'error': str(e)}

    def _enhance_entity_extraction(self, learning_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Enhance entity extraction based on learning patterns
        """
        try:
            logger.info("Enhancing entity extraction capabilities")

            # Analyze entity patterns
            entity_improvements = {}

            for entity_type, patterns in learning_data['entity_patterns'].items():
                if len(patterns) >= 5:  # Sufficient data for analysis
                    # Extract common patterns
                    pattern_analysis = self._analyze_entity_patterns(patterns)

                    if pattern_analysis['new_patterns']:
                        entity_improvements[entity_type] = pattern_analysis

            # Update entity extraction rules
            if entity_improvements:
                rules_updated = self._update_entity_rules(entity_improvements)

                return {
                    'success': True,
                    'entities_analyzed': len(learning_data['entity_patterns']),
                    'entities_improved': len(entity_improvements),
                    'rules_updated': rules_updated
                }
            else:
                return {
                    'success': True,
                    'entities_analyzed': len(learning_data['entity_patterns']),
                    'entities_improved': 0,
                    'message': 'No significant entity patterns found for improvement'
                }

        except Exception as e:
            logger.error(f"Error enhancing entity extraction: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_entity_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze entity patterns to identify new extraction rules
        """
        try:
            analysis = {
                'total_patterns': len(patterns),
                'new_patterns': [],
                'confidence_distribution': [],
                'context_patterns': Counter()
            }

            # Analyze contexts where entities appear
            for pattern in patterns:
                context = pattern['context'].lower()
                entity_text = pattern['text'].lower()

                # Extract words around the entity
                words = context.split()
                if entity_text in context:
                    entity_index = context.find(entity_text)
                    # Look for patterns before and after entity
                    before_words = context[:entity_index].split()[-2:]
                    after_words = context[entity_index + len(entity_text):].split()[:2]

                    if before_words:
                        analysis['context_patterns'][f"before_{' '.join(before_words)}"] += 1
                    if after_words:
                        analysis['context_patterns'][f"after_{' '.join(after_words)}"] += 1

                analysis['confidence_distribution'].append(pattern['confidence'])

            # Identify new patterns (those appearing frequently)
            for pattern, count in analysis['context_patterns'].items():
                if count >= 3:  # Pattern appears at least 3 times
                    analysis['new_patterns'].append({
                        'pattern': pattern,
                        'frequency': count,
                        'confidence': sum(analysis['confidence_distribution']) / len(analysis['confidence_distribution'])
                    })

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing entity patterns: {e}")
            return {'new_patterns': []}

    def _update_entity_rules(self, entity_improvements: Dict[str, Any]) -> int:
        """
        Update entity extraction rules based on improvements
        """
        try:
            rules_path = "app/ml/models/entity_rules.json"
            rules_updated = 0

            # Load existing rules
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    rules = json.load(f)
            else:
                rules = {}

            # Update rules with new patterns
            for entity_type, improvements in entity_improvements.items():
                if entity_type not in rules:
                    rules[entity_type] = {'patterns': [], 'context_rules': []}

                for pattern_data in improvements['new_patterns']:
                    new_rule = {
                        'pattern': pattern_data['pattern'],
                        'frequency': pattern_data['frequency'],
                        'confidence': pattern_data['confidence'],
                        'added_date': datetime.now().isoformat()
                    }

                    # Check if rule already exists
                    existing = any(
                        rule['pattern'] == new_rule['pattern']
                        for rule in rules[entity_type]['context_rules']
                    )

                    if not existing:
                        rules[entity_type]['context_rules'].append(new_rule)
                        rules_updated += 1

            # Save updated rules
            os.makedirs(os.path.dirname(rules_path), exist_ok=True)
            with open(rules_path, 'w') as f:
                json.dump(rules, f, indent=2)

            return rules_updated

        except Exception as e:
            logger.error(f"Error updating entity rules: {e}")
            return 0

    def _expand_knowledge_base(self, learning_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Expand knowledge base with new insights from conversations
        """
        try:
            logger.info("Expanding knowledge base with learning insights")

            kb_manager = KnowledgeBaseManager(db)
            expansions_made = 0

            # Analyze conversations for knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(learning_data)

            # Generate new knowledge entries
            for gap in knowledge_gaps:
                if gap['confidence'] > 0.7 and gap['frequency'] >= 3:
                    # Create new knowledge entry
                    success = kb_manager.add_knowledge(
                        category=gap['category'],
                        question=gap['common_question'],
                        answer=gap['suggested_answer'],
                        keywords=gap['keywords']
                    )

                    if success:
                        expansions_made += 1

            return {
                'success': True,
                'knowledge_gaps_identified': len(knowledge_gaps),
                'expansions_made': expansions_made
            }

        except Exception as e:
            logger.error(f"Error expanding knowledge base: {e}")
            return {'success': False, 'error': str(e)}

    def _identify_knowledge_gaps(self, learning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify knowledge gaps from conversation patterns
        """
        try:
            gaps = []

            # Group conversations by intent
            intent_groups = defaultdict(list)
            for conv in learning_data['conversations']:
                intent_groups[conv['intent']].append(conv)

            # Analyze each intent group for patterns
            for intent, conversations in intent_groups.items():
                # Find conversations with poor feedback
                poor_conversations = [
                    conv for conv in conversations
                    if conv['feedback'] and conv['feedback']['rating'] <= 2
                ]

                if len(poor_conversations) >= 3:  # Significant pattern
                    # Extract common themes
                    common_themes = self._extract_common_themes(poor_conversations)

                    for theme in common_themes:
                        gaps.append({
                            'intent': intent,
                            'category': self._map_intent_to_category(intent),
                            'common_question': theme['question_pattern'],
                            'suggested_answer': theme['suggested_improvement'],
                            'keywords': theme['keywords'],
                            'frequency': theme['frequency'],
                            'confidence': theme['confidence']
                        })

            return gaps

        except Exception as e:
            logger.error(f"Error identifying knowledge gaps: {e}")
            return []

    def _extract_common_themes(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract common themes from problematic conversations
        """
        try:
            themes = []

            # Extract keywords from all user messages
            all_keywords = []
            for conv in conversations:
                keywords = extract_keywords(conv['user_message'])
                all_keywords.extend(keywords)

            # Find most common keywords
            keyword_counts = Counter(all_keywords)
            common_keywords = [kw for kw, count in keyword_counts.most_common(10) if count >= 2]

            if common_keywords:
                # Generate theme based on common keywords
                theme = {
                    'question_pattern': f"Questions about {', '.join(common_keywords[:3])}",
                    'suggested_improvement': f"Enhanced information needed about {', '.join(common_keywords[:5])}",
                    'keywords': common_keywords,
                    'frequency': len(conversations),
                    'confidence': min(len(conversations) / 10.0, 1.0)  # Confidence based on frequency
                }
                themes.append(theme)

            return themes

        except Exception as e:
            logger.error(f"Error extracting common themes: {e}")
            return []

    def _map_intent_to_category(self, intent: str) -> str:
        """
        Map intent to knowledge base category
        """
        intent_category_map = {
            'destination_inquiry': 'destinations',
            'transportation': 'transportation',
            'accommodation': 'accommodation',
            'food_inquiry': 'food',
            'culture_inquiry': 'culture',
            'weather_inquiry': 'weather',
            'currency_inquiry': 'currency',
            'help_request': 'practical',
            'general_greeting': 'practical'
        }

        return intent_category_map.get(intent, 'practical')

    def _improve_response_quality(self, learning_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Improve response quality based on user feedback
        """
        try:
            logger.info("Analyzing response quality for improvements")

            improvements_made = 0
            quality_insights = []

            # Analyze response patterns by intent
            for intent, responses in learning_data['response_patterns'].items():
                if len(responses) >= 5:  # Sufficient data
                    quality_analysis = self._analyze_response_quality(responses)

                    if quality_analysis['needs_improvement']:
                        quality_insights.append({
                            'intent': intent,
                            'issue': quality_analysis['main_issue'],
                            'suggestions': quality_analysis['suggestions'],
                            'affected_responses': quality_analysis['poor_responses_count']
                        })
                        improvements_made += 1

            return {
                'success': True,
                'intents_analyzed': len(learning_data['response_patterns']),
                'improvements_identified': improvements_made,
                'quality_insights': quality_insights
            }

        except Exception as e:
            logger.error(f"Error improving response quality: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_response_quality(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze response quality for a specific intent
        """
        try:
            analysis = {
                'total_responses': len(responses),
                'poor_responses_count': 0,
                'average_rating': 0.0,
                'needs_improvement': False,
                'main_issue': None,
                'suggestions': []
            }

            # Calculate metrics
            rated_responses = [r for r in responses if r['feedback'] and r['feedback']['rating']]

            if rated_responses:
                total_rating = sum(r['feedback']['rating'] for r in rated_responses)
                analysis['average_rating'] = total_rating / len(rated_responses)
                analysis['poor_responses_count'] = len([r for r in rated_responses if r['feedback']['rating'] <= 2])

                # Determine if improvement is needed
                if analysis['average_rating'] < 3.5 or analysis['poor_responses_count'] > len(rated_responses) * 0.3:
                    analysis['needs_improvement'] = True

                    # Identify main issues
                    if analysis['average_rating'] < 3.0:
                        analysis['main_issue'] = 'low_overall_satisfaction'
                        analysis['suggestions'].append('Review and improve response templates')
                        analysis['suggestions'].append('Add more contextual information')

                    if analysis['poor_responses_count'] > len(rated_responses) * 0.4:
                        analysis['main_issue'] = 'high_poor_response_rate'
                        analysis['suggestions'].append('Analyze specific failure patterns')
                        analysis['suggestions'].append('Improve tool integration')

                    # Tool-specific analysis
                    tool_usage = defaultdict(int)
                    poor_tool_usage = defaultdict(int)

                    for response in responses:
                        tools = response.get('tools_used', [])
                        rating = response.get('feedback', {}).get('rating', 0)

                        for tool in tools:
                            tool_usage[tool] += 1
                            if rating <= 2:
                                poor_tool_usage[tool] += 1

                    # Identify problematic tools
                    for tool, total_usage in tool_usage.items():
                        if total_usage >= 3:  # Significant usage
                            poor_rate = poor_tool_usage[tool] / total_usage
                            if poor_rate > 0.4:
                                analysis['suggestions'].append(f'Improve {tool} integration - high failure rate')

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing response quality: {e}")
            return {'needs_improvement': False}

    def _validate_learning_improvements(self, db: Session) -> Dict[str, Any]:
        """
        Validate that learning improvements are actually beneficial
        """
        try:
            logger.info("Validating learning improvements")

            # Get recent performance metrics
            recent_conversations = db.query(Conversation).filter(
                Conversation.created_at >= datetime.now() - timedelta(days=3)
            ).all()

            if not recent_conversations:
                return {'validation': 'insufficient_data'}

            # Calculate current performance metrics
            current_metrics = self._calculate_current_metrics(recent_conversations, db)

            # Compare with historical metrics
            historical_metrics = self._get_historical_metrics(db)

            # Calculate improvement scores
            validation_results = {
                'validation_date': datetime.now().isoformat(),
                'sample_size': len(recent_conversations),
                'current_metrics': current_metrics,
                'historical_metrics': historical_metrics,
                'improvements': {},
                'overall_improvement': 'neutral'
            }

            # Compare metrics
            if historical_metrics:
                for metric, current_value in current_metrics.items():
                    if metric in historical_metrics:
                        historical_value = historical_metrics[metric]
                        if isinstance(current_value, (int, float)) and isinstance(historical_value, (int, float)):
                            improvement = (current_value - historical_value) / historical_value if historical_value != 0 else 0
                            validation_results['improvements'][metric] = {
                                'current': current_value,
                                'historical': historical_value,
                                'improvement_percent': round(improvement * 100, 2)
                            }

                # Determine overall improvement
                avg_improvement = np.mean([
                    imp['improvement_percent'] for imp in validation_results['improvements'].values()
                ])

                if avg_improvement > 5:
                    validation_results['overall_improvement'] = 'positive'
                elif avg_improvement < -5:
                    validation_results['overall_improvement'] = 'negative'
                else:
                    validation_results['overall_improvement'] = 'neutral'

            return validation_results

        except Exception as e:
            logger.error(f"Error validating learning improvements: {e}")
            return {'validation': 'error', 'error': str(e)}

    def _calculate_current_metrics(self, conversations: List[Conversation], db: Session) -> Dict[str, float]:
        """
        Calculate current performance metrics
        """
        try:
            metrics = {}

            if not conversations:
                return metrics

            # Average confidence score - CONVERT TO PYTHON TYPES
            confidence_scores = [
                float(conv.confidence_score) for conv in conversations
                if conv.confidence_score
            ]
            if confidence_scores:
                metrics['avg_confidence'] = float(np.mean(confidence_scores))  # Convert to Python float
                metrics['confidence_std'] = float(np.std(confidence_scores))   # Convert to Python float

            # Intent distribution entropy (diversity measure) - CONVERT TO PYTHON TYPES
            intent_counts = Counter(conv.intent for conv in conversations if conv.intent)
            if intent_counts:
                total = sum(intent_counts.values())
                entropy = -sum((count/total) * np.log2(count/total) for count in intent_counts.values())
                metrics['intent_entropy'] = float(entropy)  # Convert to Python float

            # Tool usage efficiency - CONVERT TO PYTHON TYPES
            tool_usage = Counter()
            for conv in conversations:
                if conv.tools_used:
                    tool_usage.update(conv.tools_used)

            if tool_usage:
                metrics['avg_tools_per_conversation'] = float(sum(tool_usage.values()) / len(conversations))  # Convert to Python float
                metrics['unique_tools_used'] = int(len(tool_usage))  # Convert to Python int

            # Response length metrics - CONVERT TO PYTHON TYPES
            response_lengths = [len(conv.bot_response) for conv in conversations if conv.bot_response]
            if response_lengths:
                metrics['avg_response_length'] = float(np.mean(response_lengths))  # Convert to Python float
                metrics['response_length_std'] = float(np.std(response_lengths))   # Convert to Python float

            # User satisfaction (from feedback) - CONVERT TO PYTHON TYPES
            conversation_ids = [conv.id for conv in conversations]
            feedback_query = db.query(UserFeedback).filter(
                UserFeedback.conversation_id.in_(conversation_ids)
            ).all()

            if feedback_query:
                ratings = [fb.user_rating for fb in feedback_query]
                metrics['avg_user_rating'] = float(np.mean(ratings))  # Convert to Python float
                metrics['user_satisfaction_rate'] = float(len([r for r in ratings if r >= 4]) / len(ratings))  # Convert to Python float

            return metrics

        except Exception as e:
            logger.error(f"Error calculating current metrics: {e}")
            return {}

    def _get_historical_metrics(self, db: Session, days_back: int = 30) -> Dict[str, float]:
        """
        Get historical performance metrics for comparison
        """
        try:
            end_date = datetime.now() - timedelta(days=7)  # Exclude recent data
            start_date = end_date - timedelta(days=days_back)

            historical_conversations = db.query(Conversation).filter(
                and_(
                    Conversation.created_at >= start_date,
                    Conversation.created_at <= end_date
                )
            ).all()

            if not historical_conversations:
                return {}

            return self._calculate_current_metrics(historical_conversations, db)

        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return {}

    def _update_learning_analytics(self, learning_data: Dict[str, Any],
                                   performance_analysis: Dict[str, Any],
                                   validation_results: Dict[str, Any],
                                   db: Session) -> Dict[str, Any]:
        """
        Update comprehensive learning analytics
        """
        try:
            # Calculate learning velocity - CONVERT TO PYTHON TYPES
            learning_velocity = self._calculate_learning_velocity(db)

            analytics = {
                'last_update': datetime.now().isoformat(),
                'data_summary': {
                    'conversations_processed': int(learning_data['total_samples']),  # Ensure int
                    'intents_analyzed': int(len(learning_data['intent_distribution'])),  # Ensure int
                    'entities_processed': int(sum(len(patterns) for patterns in learning_data['entity_patterns'].values())),  # Ensure int
                    'tools_analyzed': int(len(learning_data['tool_usage_patterns']))  # Ensure int
                },
                'performance_trends': validation_results.get('improvements', {}),
                'learning_efficiency': {
                    'data_utilization_rate': float(min(learning_data['total_samples'] / 100, 1.0)),  # Convert to Python float
                    'improvement_rate': validation_results.get('overall_improvement', 'neutral'),
                    'learning_velocity': float(learning_velocity)  # Convert to Python float
                }
            }

            # Save analytics to database
            self._save_learning_analytics(analytics, db)

            return analytics

        except Exception as e:
            logger.error(f"Error updating learning analytics: {e}")
            return {}

    def _calculate_learning_velocity(self, db: Session) -> float:
        """
        Calculate how quickly the system is learning and improving
        """
        try:
            # Get model metrics over time
            recent_metrics = db.query(ModelMetrics).filter(
                ModelMetrics.created_at >= datetime.now() - timedelta(days=30)
            ).order_by(ModelMetrics.created_at.desc()).all()

            if len(recent_metrics) < 2:
                return 0.0

            # Calculate improvement rate over time
            accuracy_improvements = []
            for i in range(1, len(recent_metrics)):
                current_accuracy = float(recent_metrics[i-1].intent_accuracy or 0)  # Convert to Python float
                previous_accuracy = float(recent_metrics[i].intent_accuracy or 0)   # Convert to Python float
                improvement = current_accuracy - previous_accuracy
                accuracy_improvements.append(improvement)

            # Return average improvement rate - CONVERT TO PYTHON TYPES
            return float(np.mean(accuracy_improvements)) if accuracy_improvements else 0.0

        except Exception as e:
            logger.error(f"Error calculating learning velocity: {e}")
            return 0.0

    def _save_learning_analytics(self, analytics: Dict[str, Any], db: Session):
        """
        Save learning analytics to database
        """
        try:
            # Save as model metrics entry
            analytics_metric = ModelMetrics(
                metric_date=datetime.now(),
                model_version='learning_analytics',
                intent_accuracy=str(analytics['learning_efficiency']['data_utilization_rate']),
                total_conversations=str(analytics['data_summary']['conversations_processed']),
                # recall_score=str(analytics['data_summary']['intents_analyzed']),
                average_user_satisfaction=str(analytics['learning_efficiency']['learning_velocity'])
            )

            db.add(analytics_metric)
            db.commit()

        except Exception as e:
            logger.error(f"Error saving learning analytics: {e}")

    def _cleanup_learning_data(self, db: Session):
        """
        Clean up old learning data and optimize database
        """
        try:
            logger.info("Cleaning up old learning data")

            # Clean up very old conversation data (older than 6 months)
            cutoff_date = datetime.now() - timedelta(days=180)

            # Count records to be deleted
            old_conversations = db.query(Conversation).filter(
                Conversation.created_at < cutoff_date
            ).count()

            if old_conversations > 1000:  # Only cleanup if there are many old records
                # Delete in batches to avoid locking issues
                batch_size = 100
                while True:
                    old_batch = db.query(Conversation).filter(
                        Conversation.created_at < cutoff_date
                    ).limit(batch_size).all()

                    if not old_batch:
                        break

                    for conv in old_batch:
                        db.delete(conv)

                    db.commit()
                    time.sleep(0.1)  # Brief pause to avoid overwhelming the database

                logger.info(f"Cleaned up {old_conversations} old conversation records")

            # Clean up old model metrics (keep only last 50 entries per model type)
            model_versions = db.query(ModelMetrics.model_version).distinct().all()

            for (model_version,) in model_versions:
                metrics = db.query(ModelMetrics).filter(
                    ModelMetrics.model_version == model_version
                ).order_by(ModelMetrics.created_at.desc()).all()

                if len(metrics) > 50:
                    # Keep only the 50 most recent
                    to_delete = metrics[50:]
                    for metric in to_delete:
                        db.delete(metric)

                    db.commit()

        except Exception as e:
            logger.error(f"Error cleaning up learning data: {e}")

    def _validate_intent_improvement(self, validation_data: List[Dict[str, Any]]) -> float:
        """
        Validate intent classification improvements
        """
        try:
            if not validation_data:
                return 0.0

            correct_predictions = 0
            total_predictions = len(validation_data)

            for item in validation_data:
                predicted_intent, confidence = self.intent_classifier.classify_intent(item['text'])

                # For validation, we expect poor quality predictions to have low confidence
                if item['expected_quality'] == 'poor' and confidence < 0.6:
                    correct_predictions += 1
                elif item['expected_quality'] != 'poor' and confidence >= 0.6:
                    correct_predictions += 1

            accuracy = correct_predictions / total_predictions
            return accuracy - 0.5  # Return improvement over baseline (50%)

        except Exception as e:
            logger.error(f"Error validating intent improvement: {e}")
            return 0.0

    def _save_model_metrics(self, model_type: str, accuracy: float, db: Session):
        """
        Save model performance metrics to database
        """
        try:
            metrics = ModelMetrics(
                model_type=model_type,
                accuracy=str(accuracy),
                precision_score='0.0',  # Would calculate in full implementation
                recall_score='0.0',
                f1_score='0.0'
            )

            db.add(metrics)
            db.commit()

        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")

    def _analyze_model_performance(self, db: Session) -> Dict[str, Any]:
        """
        Analyze overall model performance trends
        """
        try:
            # Get recent performance data
            recent_metrics = db.query(ModelMetrics).filter(
                ModelMetrics.created_at >= datetime.now() - timedelta(days=7)
            ).all()

            # Get feedback-based performance
            recent_feedback = db.query(UserFeedback).join(
                Conversation, UserFeedback.conversation_id == Conversation.id
            ).filter(
                Conversation.created_at >= datetime.now() - timedelta(days=7)
            ).all()

            analysis = {
                'model_metrics_count': len(recent_metrics),
                'feedback_count': len(recent_feedback),
                'average_rating': 0.0,
                'satisfaction_rate': 0.0,
                'performance_trend': 'stable'
            }

            if recent_feedback:
                ratings = [fb.user_rating for fb in recent_feedback]
                analysis['average_rating'] = np.mean(ratings)
                analysis['satisfaction_rate'] = len([r for r in ratings if r >= 4]) / len(ratings)

                # Determine trend
                if analysis['average_rating'] >= 4.0:
                    analysis['performance_trend'] = 'excellent'
                elif analysis['average_rating'] >= 3.5:
                    analysis['performance_trend'] = 'good'
                elif analysis['average_rating'] >= 3.0:
                    analysis['performance_trend'] = 'moderate'
                else:
                    analysis['performance_trend'] = 'needs_improvement'

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return {}

    def get_learning_status(self, db: Session) -> Dict[str, Any]:
        """
        Get current learning system status and statistics
        """
        try:
            # Get recent learning activity
            recent_metrics = db.query(ModelMetrics).filter(
                ModelMetrics.created_at >= datetime.now() - timedelta(days=30)
            ).order_by(ModelMetrics.created_at.desc()).all()

            # Get conversation statistics
            total_conversations = db.query(Conversation).count()
            recent_conversations = db.query(Conversation).filter(
                Conversation.created_at >= datetime.now() - timedelta(days=7)
            ).count()

            # Get feedback statistics
            total_feedback = db.query(UserFeedback).count()
            recent_feedback = db.query(UserFeedback).join(
                Conversation, UserFeedback.conversation_id == Conversation.id
            ).filter(
                Conversation.created_at >= datetime.now() - timedelta(days=7)
            ).all()

            # Calculate learning health score
            health_score = self._calculate_learning_health_score(
                recent_conversations, recent_feedback, recent_metrics
            )

            status = {
                'system_status': 'active' if self.learning_config['auto_learning_enabled'] else 'inactive',
                'health_score': health_score,
                'statistics': {
                    'total_conversations': total_conversations,
                    'recent_conversations': recent_conversations,
                    'total_feedback': total_feedback,
                    'recent_feedback_count': len(recent_feedback),
                    'model_updates_last_30_days': len(recent_metrics)
                },
                'learning_efficiency': {
                    'feedback_rate': len(recent_feedback) / max(recent_conversations, 1),
                    'average_recent_rating': np.mean([fb.user_rating for fb in recent_feedback]) if recent_feedback else 0.0,
                    'learning_data_availability': min(recent_conversations / self.learning_threshold, 1.0)
                },
                'next_learning_cycle': self._estimate_next_learning_cycle(recent_conversations),
                'recommendations': self._generate_learning_recommendations(health_score, recent_conversations, recent_feedback)
            }

            return status

        except Exception as e:
            logger.error(f"Error getting learning status: {e}")
            return {'system_status': 'error', 'error': str(e)}

    def _calculate_learning_health_score(self, recent_conversations: int,
                                         recent_feedback: List[UserFeedback],
                                         recent_metrics: List[ModelMetrics]) -> float:
        """
        Calculate overall learning system health score (0-100)
        """
        try:
            score = 0.0

            # Data availability score (40% of total)
            data_score = min(recent_conversations / 50, 1.0) * 40
            score += data_score

            # Feedback quality score (30% of total)
            if recent_feedback:
                avg_rating = np.mean([fb.user_rating for fb in recent_feedback])
                feedback_score = (avg_rating / 5.0) * 30
                score += feedback_score

            # Learning activity score (20% of total)
            activity_score = min(len(recent_metrics) / 5, 1.0) * 20
            score += activity_score

            # System configuration score (10% of total)
            config_score = 10 if self.learning_config['auto_learning_enabled'] else 5
            score += config_score

            return min(score, 100.0)

        except Exception as e:
            logger.error(f"Error calculating learning health score: {e}")
            return 0.0

    def _estimate_next_learning_cycle(self, recent_conversations: int) -> str:
        """
        Estimate when the next learning cycle should occur
        """
        try:
            if recent_conversations >= self.learning_threshold:
                return "Ready now"
            elif recent_conversations >= self.learning_threshold * 0.7:
                return "Within 1-2 days"
            elif recent_conversations >= self.learning_threshold * 0.4:
                return "Within 3-5 days"
            else:
                return "More than 1 week"

        except Exception as e:
            logger.error(f"Error estimating next learning cycle: {e}")
            return "Unknown"

    def _generate_learning_recommendations(self, health_score: float,
                                           recent_conversations: int,
                                           recent_feedback: List[UserFeedback]) -> List[str]:
        """
        Generate recommendations for improving learning system
        """
        recommendations = []

        try:
            if health_score < 50:
                recommendations.append("Learning system health is low - immediate attention needed")

            if recent_conversations < self.learning_threshold:
                recommendations.append(f"Need more conversation data for learning (current: {recent_conversations}, needed: {self.learning_threshold})")

            if len(recent_feedback) < recent_conversations * 0.1:
                recommendations.append("Low feedback rate - encourage more user feedback")

            if recent_feedback:
                avg_rating = np.mean([fb.rating for fb in recent_feedback])
                if avg_rating < 3.5:
                    recommendations.append("User satisfaction is below target - focus on response quality")

            if not self.learning_config['auto_learning_enabled']:
                recommendations.append("Consider enabling auto-learning for continuous improvement")

            if not recommendations:
                recommendations.append("Learning system is operating well - continue monitoring")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating learning recommendations: {e}")
            return ["Error generating recommendations"]

    def force_learning_cycle(self, db: Session) -> Dict[str, Any]:
        """
        Force a learning cycle to run immediately (for testing/manual triggers)
        """
        try:
            logger.info("Forcing immediate learning cycle")

            # Temporarily lower thresholds for forced learning
            original_threshold = self.learning_threshold
            self.learning_threshold = 1

            # Run learning cycle
            result = self.continuous_learning_cycle(db)

            # Restore original threshold
            self.learning_threshold = original_threshold

            result['forced'] = True
            result['timestamp'] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"Error in forced learning cycle: {e}")
            return {'success': False, 'error': str(e), 'forced': True}

    def export_learning_data(self, db: Session, days: int = 30) -> Dict[str, Any]:
        """
        Export learning data for external analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get conversations with feedback
            conversations = db.query(Conversation).filter(
                Conversation.created_at >= cutoff_date
            ).all()

            feedback_map = {}
            feedback_query = db.query(UserFeedback).join(
                Conversation, UserFeedback.conversation_id == Conversation.id
            ).filter(
                Conversation.created_at >= cutoff_date
            ).all()

            for fb in feedback_query:
                feedback_map[fb.conversation_id] = {
                    'rating': fb.rating,
                    'is_helpful': fb.is_helpful,
                    'feedback_text': fb.feedback_text
                }

            # Export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'days_included': days,
                'total_conversations': len(conversations),
                'conversations_with_feedback': len(feedback_map),
                'data': []
            }

            for conv in conversations:
                export_data['data'].append({
                    'id': conv.id,
                    'timestamp': conv.created_at.isoformat(),
                    'user_message': conv.user_message,
                    'bot_response': conv.bot_response,
                    'intent': conv.intent,
                    'confidence_score': conv.confidence_score,
                    'entities': conv.entities,
                    'tools_used': conv.tools_used,
                    'feedback': feedback_map.get(conv.id)
                })

            return export_data

        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return {'success': False, 'error': str(e)}

    def get_learning_status_with_tables(self, db: Session) -> Dict[str, Any]:
        """Get learning status with database table information"""
        try:
            # Get the existing learning status
            status = self.get_learning_status(db)

            # Add table-specific information
            from backend.app.models.database import Conversation, UserFeedback, ModelMetrics, KnowledgeBase

            table_info = {
                'conversations_table': db.query(Conversation).count(),
                'feedback_table': db.query(UserFeedback).count(),
                'metrics_table': db.query(ModelMetrics).count(),
                'knowledge_base_table': db.query(KnowledgeBase).count()
            }

            status['table_statistics'] = table_info
            return status

        except Exception as e:
            logger.error(f"Error getting learning status with tables: {e}")
            return {'error': str(e)}

    def run_learning_cycle(self, db: Session) -> Dict[str, Any]:
        """Run a complete learning cycle - wrapper for continuous_learning_cycle"""
        try:
            logger.info("Running learning cycle")
            return self.continuous_learning_cycle(db)
        except Exception as e:
            logger.error(f"Error running learning cycle: {e}")
            return {'success': False, 'error': str(e)}

    def trigger_learning_with_feedback_and_updates(self, db: Session, rating: int = None) -> Dict[str, Any]:
        """Trigger learning with feedback consideration"""
        try:
            logger.info(f"Triggering learning with feedback (rating: {rating})")

            # If rating is very poor, prioritize immediate learning
            if rating and rating <= 2:
                logger.info("Poor rating detected - forcing immediate learning cycle")
                return self.force_learning_cycle(db)

            # Otherwise, run normal learning cycle if threshold is met
            return self.continuous_learning_cycle(db)

        except Exception as e:
            logger.error(f"Error triggering learning with feedback: {e}")
            return {'success': False, 'error': str(e)}

    def _save_learning_data_to_db(self, learning_insights: List[Dict[str, Any]], db: Session) -> int:
        """
        Save learning insights to the LearningData table
        """
        try:
            saved_count = 0

            for insight in learning_insights:
                learning_entry = LearningData(
                    intent=insight.get('intent', 'unknown'),
                    user_input=insight.get('user_input', ''),
                    correct_response=insight.get('suggested_response', ''),
                    feedback_score=insight.get('confidence', 0.0),
                    is_validated=False,
                    validation_source='auto',
                    domain_category=insight.get('category', 'general'),
                    language_detected='en',
                    context_data=insight.get('context', {}),
                    improvement_suggestion=insight.get('improvement_suggestion', '')
                )

                db.add(learning_entry)
                saved_count += 1

            db.commit()
            logger.info(f"Saved {saved_count} learning insights to database")
            return saved_count

        except Exception as e:
            logger.error(f"Error saving learning data to database: {e}")
            db.rollback()
            return 0

    def _extract_learning_insights_from_conversations(self, learning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract actionable learning insights from conversation data
        """
        insights = []

        try:
            # Extract insights from low-confidence conversations
            for conv in learning_data['conversations']:
                if conv['confidence_score'] < 0.4:  # Low confidence threshold
                    insight = {
                        'intent': conv['intent'],
                        'user_input': conv['user_message'],
                        'suggested_response': conv['bot_response'],
                        'confidence': conv['confidence_score'],
                        'category': self._map_intent_to_category(conv['intent']),
                        'improvement_suggestion': f"Low confidence ({conv['confidence_score']:.3f}) - needs better training data",
                        'context': {
                            'entities': conv['entities'],
                            'tools_used': conv['tools_used'],
                            'conversation_id': conv['id']
                        }
                    }

                    # Add feedback context if available
                    if conv['feedback']:
                        insight['feedback_score'] = conv['feedback']['rating']
                        insight['improvement_suggestion'] += f" - User rating: {conv['feedback']['rating']}/5"

                    insights.append(insight)

            # Extract insights from poor feedback conversations
            for conv in learning_data['conversations']:
                if conv['feedback'] and conv['feedback']['rating'] <= 2:
                    insight = {
                        'intent': conv['intent'],
                        'user_input': conv['user_message'],
                        'suggested_response': conv['bot_response'],
                        'confidence': conv['confidence_score'],
                        'category': self._map_intent_to_category(conv['intent']),
                        'improvement_suggestion': f"Poor user feedback ({conv['feedback']['rating']}/5) - response needs improvement",
                        'context': {
                            'entities': conv['entities'],
                            'tools_used': conv['tools_used'],
                            'conversation_id': conv['id'],
                            'feedback_text': conv['feedback'].get('feedback_text', '')
                        }
                    }
                    insights.append(insight)

            logger.info(f"Extracted {len(insights)} learning insights from conversations")
            return insights

        except Exception as e:
            logger.error(f"Error extracting learning insights: {e}")
            return []

    def _update_model_metrics_with_improvements(self, learning_data: Dict[str, Any], db: Session):
        """
        Update ModelMetrics table with actual learning improvements
        """
        try:
            # Calculate comprehensive metrics
            total_conversations = learning_data['total_samples']

            # Calculate success/failure rates
            conversations_with_feedback = [
                conv for conv in learning_data['conversations']
                if conv['feedback']
            ]

            successful_interactions = len([
                conv for conv in conversations_with_feedback
                if conv['feedback']['rating'] >= 4
            ])

            failed_interactions = len([
                conv for conv in conversations_with_feedback
                if conv['feedback']['rating'] <= 2
            ])

            # Calculate average metrics - CONVERT TO PYTHON NATIVE TYPES
            avg_confidence = float(np.mean([
                conv['confidence_score'] for conv in learning_data['conversations']
            ])) if learning_data['conversations'] else 0.0

            avg_response_time = float(np.mean([
                len(conv['bot_response']) / 100  # Rough estimate based on response length
                for conv in learning_data['conversations']
            ])) if learning_data['conversations'] else 0.0

            avg_user_satisfaction = float(np.mean([
                conv['feedback']['rating'] for conv in conversations_with_feedback
            ])) if conversations_with_feedback else 0.0

            # Calculate intent accuracy - CONVERT TO PYTHON NATIVE TYPES
            intent_accuracy = float(len([
                conv for conv in learning_data['conversations']
                if conv['confidence_score'] >= 0.6
            ]) / len(learning_data['conversations'])) if learning_data['conversations'] else 0.0

            # Most common intent
            most_common_intent = max(
                learning_data['intent_distribution'].items(),
                key=lambda x: x[1]
            )[0] if learning_data['intent_distribution'] else 'unknown'

            # Calculate entity extraction accuracy - CONVERT TO PYTHON NATIVE TYPES
            entity_extraction_accuracy = float(avg_confidence)  # Using confidence as proxy

            # Create comprehensive metrics entry with only essential fields
            metrics_entry = ModelMetrics(
                metric_date=datetime.now(),
                total_conversations=int(total_conversations),  # Ensure int
                successful_interactions=int(successful_interactions),  # Ensure int
                failed_interactions=int(failed_interactions),  # Ensure int
                average_response_time=float(avg_response_time),  # Ensure float
                average_user_satisfaction=float(avg_user_satisfaction),  # Ensure float
                most_common_intent=str(most_common_intent),  # Ensure string
                total_api_calls=int(sum(learning_data['tool_usage_patterns'].values())),  # Ensure int
                unique_users=int(len(set([conv['id'] for conv in learning_data['conversations']]))),  # Ensure int
                intent_accuracy=float(intent_accuracy),  # Ensure float
                entity_extraction_accuracy=float(entity_extraction_accuracy),  # Ensure float
                model_version="1.0.0",
                training_data_size=int(total_conversations)  # Ensure int
            )

            db.add(metrics_entry)
            db.commit()

            logger.info(f"Updated ModelMetrics with {total_conversations} conversations")

        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")
            db.rollback()

    def _expand_knowledge_base(self, learning_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Expand knowledge base with new insights from conversations
        """
        try:
            logger.info("Expanding knowledge base with learning insights")

            kb_manager = KnowledgeBaseManager(db)
            expansions_made = 0

            # Analyze conversations for knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(learning_data)

            # Generate new knowledge entries
            for gap in knowledge_gaps:
                if gap['confidence'] > 0.5 and gap['frequency'] >= 2:  # Lower threshold for actual expansion
                    # Create new knowledge entry
                    try:
                        new_entry = KnowledgeBase(
                            category=gap['category'],
                            question=gap['common_question'],
                            answer=gap['suggested_answer'],
                            keywords=json.dumps(gap['keywords']),
                            confidence_score=gap['confidence'],
                            location='Sri Lanka',
                            season_relevance='all',
                            difficulty_level='easy'
                        )

                        db.add(new_entry)
                        expansions_made += 1

                    except Exception as e:
                        logger.error(f"Error adding knowledge entry: {e}")
                        continue

            if expansions_made > 0:
                db.commit()
                logger.info(f"Added {expansions_made} new knowledge base entries")

            return {
                'success': True,
                'knowledge_gaps_identified': len(knowledge_gaps),
                'expansions_made': expansions_made
            }

        except Exception as e:
            logger.error(f"Error expanding knowledge base: {e}")
            db.rollback()
            return {'success': False, 'error': str(e)}