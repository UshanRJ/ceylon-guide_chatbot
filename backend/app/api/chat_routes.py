# backend/app/api/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging
import json

from backend.app.models.database import get_db
from backend.app.models.conversation import ConversationManager
from backend.app.models.knowledge_base import KnowledgeBaseManager
from backend.app.nlp.intent_classifier import IntentClassifier
from backend.app.nlp.entity_extractor import EntityExtractor
from backend.app.nlp.response_generator import ResponseGenerator
from backend.app.tools.currency_converter import CurrencyConverter
from backend.app.tools.maps_integration import MapsIntegration
from backend.app.tools.translator import Translator
from backend.app.tools.weather_checker import WeatherChecker
# # Removed problematic learning_engine import
from backend.app.ml.learning_engine import LearningEngine
from backend.app.utils.helpers import (
    generate_session_id, clean_text, format_response_for_display,
    validate_input, calculate_response_time, log_interaction
)

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a dedicated file handler for chat debugging
chat_handler = logging.FileHandler('logs/chat_debug.log')
chat_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chat_handler.setFormatter(formatter)
logger.addHandler(chat_handler)

learning_engine = None

router = APIRouter()

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int
    feedback_text: Optional[str] = None
    is_helpful: Optional[bool] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_id: int
    intent: str
    confidence: float
    entities: Dict[str, Any]
    tools_used: List[str]
    timestamp: str
    debug_info: Optional[Dict[str, Any]] = None  # Add debug info

# Initialize components with error handling
# Initialize learning engine at module level (after tool initializations)
def initialize_learning_engine():
    global learning_engine
    try:
        learning_engine = LearningEngine()
        logger.info("Learning engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize learning engine: {e}")
        learning_engine = None
        return False

# Call initialization
initialize_learning_engine()

try:
    intent_classifier = IntentClassifier()
    intent_classifier.update_confidence_threshold(0.4)
    logger.info("Intent classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize intent classifier: {e}")
    intent_classifier = None

try:
    entity_extractor = EntityExtractor()
    logger.info("Entity extractor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize entity extractor: {e}")
    entity_extractor = None

try:
    response_generator = ResponseGenerator()
    logger.info("Response generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize response generator: {e}")
    response_generator = None

try:
    learning_engine = LearningEngine()
    logger.info("Learning engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize learning engine: {e}")
    learning_engine = None

# Initialize tools
currency_converter = CurrencyConverter()
maps_integration = MapsIntegration()
translator = Translator()
weather_checker = WeatherChecker()

@router.post("/send-message", response_model=ChatResponse)
async def send_message(
        chat_message: ChatMessage,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """Process user message and return bot response with enhanced debugging"""
    start_time = datetime.now()
    tracking_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Initialize debug info
    debug_info = {
        "tracking_id": tracking_id,
        "processing_steps": [],
        "errors": [],
        "warnings": []
    }

    try:
        logger.info(f"[{tracking_id}] ==> USER INPUT: {chat_message.message}")
        debug_info["processing_steps"].append("Input received")

        # Validate input
        validation = validate_input(
            {"message": chat_message.message},
            ["message"]
        )

        if not validation['valid']:
            logger.error(f"[{tracking_id}] Input validation failed: {validation['errors']}")
            raise HTTPException(status_code=400, detail=validation['errors'])

        # Generate session ID if not provided
        session_id = chat_message.session_id or generate_session_id()
        logger.info(f"[{tracking_id}] Session ID: {session_id}")

        # Clean user message
        user_message = clean_text(chat_message.message)
        logger.info(f"[{tracking_id}] Cleaned message: {user_message}")
        debug_info["processing_steps"].append("Input cleaned")

        # Initialize managers
        conversation_manager = ConversationManager(db)
        kb_manager = KnowledgeBaseManager(db)
        debug_info["processing_steps"].append("Managers initialized")

        # Step 1: Intent Classification
        if intent_classifier is None:
            logger.error(f"[{tracking_id}] Intent classifier not available")
            debug_info["errors"].append("Intent classifier not initialized")
            intent, confidence = "unknown", 0.0
        else:
            intent, confidence = intent_classifier.classify_intent(user_message)
            logger.info(f"[{tracking_id}] INTENT: {intent} (confidence: {confidence:.3f})")

            # Get all intent probabilities for debugging
            all_probabilities = intent_classifier.get_intent_probabilities(user_message)
            logger.info(f"[{tracking_id}] ALL_INTENTS: {json.dumps(all_probabilities, indent=2)}")

            debug_info["intent_analysis"] = {
                "predicted_intent": intent,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "threshold": intent_classifier.confidence_threshold
            }

            # Check if confidence is below threshold
            if confidence < intent_classifier.confidence_threshold:
                debug_info["warnings"].append(f"Low confidence: {confidence:.3f} < {intent_classifier.confidence_threshold}")
                logger.warning(f"[{tracking_id}] Low confidence: {confidence:.3f} < {intent_classifier.confidence_threshold}")

        debug_info["processing_steps"].append("Intent classified")

        # Step 2: Entity Extraction
        if entity_extractor is None:
            logger.error(f"[{tracking_id}] Entity extractor not available")
            debug_info["errors"].append("Entity extractor not initialized")
            entities = {}
        else:
            entities = entity_extractor.extract_entities(user_message)
            logger.info(f"[{tracking_id}] ENTITIES: {json.dumps(entities, indent=2)}")
            debug_info["entities"] = entities

        debug_info["processing_steps"].append("Entities extracted")

        # Step 3: Search Knowledge Base
        try:
            keywords = [entity['text'] for entity_type in entities.values()
                        for entity in entity_type if entity.get('text')]
            knowledge_results = kb_manager.search_knowledge(user_message, keywords)
            logger.info(f"[{tracking_id}] KNOWLEDGE_RESULTS: {len(knowledge_results)} results found")

            for i, result in enumerate(knowledge_results):
                score = result.get('relevance_score', 0)
                answer_preview = result.get('answer', '')[:100]
                logger.info(f"[{tracking_id}] KB_RESULT_{i}: Score={score:.3f}, Answer={answer_preview}...")

            debug_info["knowledge_base"] = {
                "results_count": len(knowledge_results),
                "top_score": knowledge_results[0].get('relevance_score', 0) if knowledge_results else 0,
                "keywords_used": keywords
            }
        except Exception as e:
            logger.error(f"[{tracking_id}] Knowledge base search failed: {e}")
            knowledge_results = []
            debug_info["errors"].append(f"Knowledge base search failed: {str(e)}")

        debug_info["processing_steps"].append("Knowledge base searched")

        # Step 4: Tool Integration
        tools_used = []
        tool_outputs = {}
        debug_info["tool_execution"] = {}

        # Currency conversion
        if intent == 'currency_inquiry':
            try:
                logger.info(f"[{tracking_id}] Executing currency conversion tool")

                # Enhanced currency extraction
                currency_info_list = entity_extractor.get_currency_info(user_message)
                debug_info["tool_execution"]["currency_info"] = currency_info_list

                if currency_info_list:
                    currency_info = currency_info_list[0]

                    # Map currency symbols to standard codes
                    currency_mapping = {
                        'LKR': 'lkr', 'Rs.': 'lkr', 'Rs': 'lkr', 'rupees': 'lkr',
                        '$': 'usd', 'USD': 'usd', 'dollars': 'usd',
                        '€': 'eur', 'EUR': 'eur', 'euros': 'eur',
                        '£': 'gbp', 'GBP': 'gbp', 'pounds': 'gbp'
                    }

                    from_currency = currency_mapping.get(currency_info.get('currency', ''), 'usd')
                    amount_str = currency_info.get('amount', '1').replace(',', '')

                    try:
                        amount = float(amount_str)
                    except ValueError:
                        amount = 1.0
                        debug_info["warnings"].append(f"Could not parse amount: {amount_str}")

                    conversion_result = currency_converter.convert_currency(
                        amount, from_currency, 'lkr'
                    )

                    if conversion_result['success']:
                        tool_outputs['currency_converter'] = conversion_result
                        tools_used.append('currency_converter')
                        logger.info(f"[{tracking_id}] TOOL_SUCCESS: currency_converter")
                    else:
                        debug_info["errors"].append(f"Currency conversion failed: {conversion_result.get('error', 'Unknown error')}")
                        logger.error(f"[{tracking_id}] TOOL_FAILED: currency_converter - {conversion_result.get('error', 'Unknown')}")
                else:
                    debug_info["warnings"].append("No currency information found in message")
                    logger.warning(f"[{tracking_id}] No currency info found for currency inquiry")

            except Exception as e:
                logger.error(f"[{tracking_id}] Currency tool error: {e}")
                debug_info["errors"].append(f"Currency tool error: {str(e)}")

        # Weather information
        elif intent == 'weather_inquiry':
            try:
                logger.info(f"[{tracking_id}] Executing weather tool")

                primary_location = entity_extractor.get_primary_location(entities)
                location = primary_location['text'] if primary_location else 'Colombo'

                logger.info(f"[{tracking_id}] Weather location: {location}")
                debug_info["tool_execution"]["weather_location"] = location

                weather_result = weather_checker.get_current_weather(location)
                if weather_result['success']:
                    tool_outputs['weather_checker'] = weather_result
                    tools_used.append('weather_checker')
                    logger.info(f"[{tracking_id}] TOOL_SUCCESS: weather_checker")
                else:
                    debug_info["errors"].append(f"Weather check failed: {weather_result.get('error', 'Unknown error')}")
                    logger.error(f"[{tracking_id}] TOOL_FAILED: weather_checker - {weather_result.get('error', 'Unknown')}")

            except Exception as e:
                logger.error(f"[{tracking_id}] Weather tool error: {e}")
                debug_info["errors"].append(f"Weather tool error: {str(e)}")

        # Translation
        elif intent == 'translation_request':
            try:
                logger.info(f"[{tracking_id}] Executing translation tool")

                translation_info = entity_extractor.get_translation_info(user_message)
                debug_info["tool_execution"]["translation_info"] = translation_info

                if translation_info.get('target_languages'):
                    # Extract text to translate (simple approach)
                    # This needs improvement - should extract the actual text to translate
                    text_to_translate = "hello"  # Default for testing
                    target_lang = translation_info['target_languages'][0]['iso_code']

                    translation_result = translator.translate_text(
                        text_to_translate, 'en', target_lang
                    )

                    if translation_result['success']:
                        tool_outputs['translator'] = translation_result
                        tools_used.append('translator')
                        logger.info(f"[{tracking_id}] TOOL_SUCCESS: translator")
                    else:
                        debug_info["errors"].append(f"Translation failed: {translation_result.get('error', 'Unknown error')}")
                        logger.error(f"[{tracking_id}] TOOL_FAILED: translator - {translation_result.get('error', 'Unknown')}")
                else:
                    debug_info["warnings"].append("No target language found for translation")
                    logger.warning(f"[{tracking_id}] No target language found for translation")

            except Exception as e:
                logger.error(f"[{tracking_id}] Translation tool error: {e}")
                debug_info["errors"].append(f"Translation tool error: {str(e)}")

        # Maps and location
        elif intent == 'map_request':
            try:
                logger.info(f"[{tracking_id}] Executing maps tool")

                primary_location = entity_extractor.get_primary_location(entities)
                if primary_location:
                    location_name = primary_location['text']
                    logger.info(f"[{tracking_id}] Maps location: {location_name}")
                    debug_info["tool_execution"]["maps_location"] = location_name

                    location_result = maps_integration.get_location_info(location_name)
                    if location_result['success']:
                        tool_outputs['maps_integration'] = location_result
                        tools_used.append('maps_integration')
                        logger.info(f"[{tracking_id}] TOOL_SUCCESS: maps_integration")
                    else:
                        debug_info["errors"].append(f"Maps lookup failed: {location_result.get('error', 'Unknown error')}")
                        logger.error(f"[{tracking_id}] TOOL_FAILED: maps_integration - {location_result.get('error', 'Unknown')}")
                else:
                    debug_info["warnings"].append("No location found for map request")
                    logger.warning(f"[{tracking_id}] No location found for map request")

            except Exception as e:
                logger.error(f"[{tracking_id}] Maps tool error: {e}")
                debug_info["errors"].append(f"Maps tool error: {str(e)}")

        debug_info["processing_steps"].append("Tools executed")

        # Step 5: Generate Response
        if response_generator is None:
            logger.error(f"[{tracking_id}] Response generator not available")
            bot_response = "I apologize, but the response system is currently unavailable."
            debug_info["errors"].append("Response generator not initialized")
        else:
            try:
                bot_response = response_generator.generate_response(
                    intent, entities, knowledge_results, tool_outputs
                )
                logger.info(f"[{tracking_id}] RESPONSE_GENERATED: {len(bot_response)} characters")

                # Enhanced response with context
                enhanced_response = response_generator.enhance_response_with_context(
                    bot_response, entities
                )
                logger.info(f"[{tracking_id}] <== FINAL_RESPONSE: {enhanced_response}")

                debug_info["response_generation"] = {
                    "base_response_length": len(bot_response),
                    "enhanced_response_length": len(enhanced_response),
                    "knowledge_used": len(knowledge_results) > 0 and knowledge_results[0].get('relevance_score', 0) > 0.5,
                    "tools_used": tools_used
                }

            except Exception as e:
                logger.error(f"[{tracking_id}] Response generation failed: {e}")
                enhanced_response = "I apologize, but I'm having trouble generating a response right now."
                debug_info["errors"].append(f"Response generation failed: {str(e)}")

        debug_info["processing_steps"].append("Response generated")

        # Step 6: Save Conversation
        try:
            conversation_id = conversation_manager.save_conversation(
                session_id, user_message, enhanced_response, intent,
                entities, confidence, tools_used
            )
            logger.info(f"[{tracking_id}] CONVERSATION_SAVED: ID={conversation_id}")
            debug_info["conversation_id"] = conversation_id
        except Exception as e:
            logger.error(f"[{tracking_id}] Failed to save conversation: {e}")
            conversation_id = -1  # Fallback ID
            debug_info["errors"].append(f"Conversation save failed: {str(e)}")

        debug_info["processing_steps"].append("Conversation saved")

        if learning_engine is not None:
            background_tasks.add_task(
                log_interaction, session_id, user_message, enhanced_response,
                calculate_response_time(start_time), tools_used
            )

            # Trigger learning check if we have enough data
            background_tasks.add_task(check_and_trigger_learning, db)
            background_tasks.add_task(trigger_learning_with_feedback, db)
        else:
            # Just log the interaction if learning engine is not available
            background_tasks.add_task(
                log_interaction, session_id, user_message, enhanced_response,
                calculate_response_time(start_time), tools_used
            )


        # Format response
        try:
            formatted_response = format_response_for_display(enhanced_response, tool_outputs)
        except Exception as e:
            logger.error(f"[{tracking_id}] Response formatting failed: {e}")
            formatted_response = enhanced_response
            debug_info["warnings"].append(f"Response formatting failed: {str(e)}")

        debug_info["processing_steps"].append("Processing completed")
        debug_info["total_processing_time"] = (datetime.now() - start_time).total_seconds()

        response_data = ChatResponse(
            response=enhanced_response,
            session_id=session_id,
            conversation_id=conversation_id,
            intent=intent,
            confidence=confidence,
            entities=entities,
            tools_used=tools_used,
            timestamp=datetime.now().isoformat(),
            debug_info=debug_info  # Include debug info in development
        )

        logger.info(f"[{tracking_id}] Processing completed successfully in {debug_info['total_processing_time']:.2f}s")
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{tracking_id}] Unexpected error: {e}")
        debug_info["errors"].append(f"Unexpected error: {str(e)}")

        # Return error response with debug info
        return ChatResponse(
            response="I apologize, but I encountered an unexpected error while processing your request.",
            session_id=chat_message.session_id or generate_session_id(),
            conversation_id=-1,
            intent="error",
            confidence=0.0,
            entities={},
            tools_used=[],
            timestamp=datetime.now().isoformat(),
            debug_info=debug_info
        )

@router.post("/feedback")
async def submit_feedback(
        feedback: FeedbackRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """Submit user feedback for a conversation"""
    try:
        logger.info(f"Feedback received for conversation {feedback.conversation_id}: rating={feedback.rating}")

        conversation_manager = ConversationManager(db)

        # Validate rating
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

        # Save feedback
        success = conversation_manager.save_feedback(
            feedback.conversation_id,
            feedback.rating,
            feedback.feedback_text,
            feedback.is_helpful
        )

        if success:
            logger.info(f"Feedback saved successfully for conversation {feedback.conversation_id}")

            # Always trigger learning check when feedback is received - only if learning engine available
            if learning_engine is not None:
                background_tasks.add_task(trigger_learning_with_feedback, db, feedback.rating)

            return {
                "success": True,
                "message": "Feedback submitted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save feedback")

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@router.get("/debug/intent/{text}")
async def debug_intent_classification(text: str):
    """Debug endpoint to test intent classification"""
    try:
        if intent_classifier is None:
            return {"error": "Intent classifier not initialized"}

        intent, confidence = intent_classifier.classify_intent(text)
        all_probabilities = intent_classifier.get_intent_probabilities(text)
        explanation = intent_classifier.explain_classification(text)

        return {
            "input_text": text,
            "predicted_intent": intent,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "explanation": explanation
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/debug/learning-status")
async def get_learning_status(db: Session = Depends(get_db)):
    """Get current learning system status"""
    global learning_engine
    try:
        if learning_engine is None:
            return {"success": False, "error": "Learning engine not initialized"}

        status = learning_engine.get_learning_status_with_tables(db)
        return {"success": True, "status": status}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/debug/trigger-learning")
async def trigger_learning_manually(db: Session = Depends(get_db)):
    """Manually trigger learning cycle for testing"""
    try:
        from backend.app.models.database import Conversation, UserFeedback
        total_conversations = db.query(Conversation).count()
        total_feedback = db.query(UserFeedback).count()

        result = {
            "message": "Learning cycle simulated",
            "total_conversations": total_conversations,
            "total_feedback": total_feedback,
            "status": "healthy" if total_conversations > 0 else "needs_data"
        }
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Existing endpoints remain the same...
@router.get("/conversation-history/{session_id}")
async def get_conversation_history(
        session_id: str,
        limit: int = 10,
        db: Session = Depends(get_db)
):
    """Get conversation history for a session"""
    try:
        conversation_manager = ConversationManager(db)
        history = conversation_manager.get_conversation_history(session_id, limit)

        return {
            "success": True,
            "session_id": session_id,
            "conversation_count": len(history),
            "conversations": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@router.get("/session/new")
async def create_new_session():
    """Create a new chat session"""
    try:
        session_id = generate_session_id()
        return {
            "success": True,
            "session_id": session_id,
            "message": "New session created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/analytics")
async def get_analytics(
        days: int = 30,
        db: Session = Depends(get_db)
):
    """Get analytics data for the chatbot"""
    try:
        conversation_manager = ConversationManager(db)
        analytics = conversation_manager.get_analytics_data(days)

        return {
            "success": True,
            "analytics": analytics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "intent_classifier": intent_classifier is not None,
            "entity_extractor": entity_extractor is not None,
            "response_generator": response_generator is not None
        }
    }

# Enhanced background task functions
def check_and_trigger_learning(db: Session):
    """Check if learning should be triggered and do it"""
    global learning_engine
    try:
        if learning_engine is None:
            logger.warning("Learning engine not initialized - skipping learning cycle")
            return {"success": False, "error": "Learning engine not available"}

        result = learning_engine.run_learning_cycle(db)
        logger.info(f"Manual learning cycle: {result}")
        return result
    except Exception as e:
        logger.error(f"Background learning failed: {e}")
        return {"success": False, "error": str(e)}

def trigger_learning_with_feedback(db: Session, rating: int = None):
    """Trigger learning update based on feedback"""
    global learning_engine
    try:
        if learning_engine is None:
            logger.warning("Learning engine not initialized - skipping feedback learning")
            return {"success": False, "error": "Learning engine not available"}

        return learning_engine.trigger_learning_with_feedback_and_updates(db, rating)
    except Exception as e:
        logger.error(f"Learning with feedback failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/debug/run-learning-cycle")
async def run_learning_cycle_manually(db: Session = Depends(get_db)):
    """Manually run a complete learning cycle"""
    global learning_engine
    try:
        if learning_engine is None:
            return {"success": False, "error": "Learning engine not initialized"}

        result = learning_engine.run_learning_cycle(db)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}