# Create this as debug_chatbot_fixed.py in your backend/app/tools/ folder

import sys
import os
import json
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from backend.app.models.database import get_db
from backend.app.nlp.intent_classifier import IntentClassifier
from backend.app.nlp.entity_extractor import EntityExtractor
from backend.app.nlp.response_generator import ResponseGenerator
from backend.app.ml.learning_engine import LearningEngine

def inspect_database_schema():
    """Inspect the actual database schema to understand the structure"""
    print("=" * 80)
    print("DATABASE SCHEMA INSPECTION")
    print("=" * 80)

    try:
        db = next(get_db())

        # Import all models to check their attributes
        from backend.app.models.database import Conversation, UserFeedback, ModelMetrics

        print("Conversation model attributes:")
        conversation_attrs = [attr for attr in dir(Conversation) if not attr.startswith('_')]
        for attr in conversation_attrs:
            if hasattr(Conversation, attr):
                attr_type = type(getattr(Conversation, attr))
                print(f"  - {attr}: {attr_type}")

        print("\nUserFeedback model attributes:")
        feedback_attrs = [attr for attr in dir(UserFeedback) if not attr.startswith('_')]
        for attr in feedback_attrs:
            if hasattr(UserFeedback, attr):
                attr_type = type(getattr(UserFeedback, attr))
                print(f"  - {attr}: {attr_type}")

        # Check if tables exist and have data
        try:
            total_conversations = db.query(Conversation).count()
            print(f"\nTotal conversations in database: {total_conversations}")
        except Exception as e:
            print(f"Error querying conversations: {e}")

        try:
            total_feedback = db.query(UserFeedback).count()
            print(f"Total feedback in database: {total_feedback}")
        except Exception as e:
            print(f"Error querying feedback: {e}")

        # Try to get a sample conversation to see actual structure
        try:
            sample_conversation = db.query(Conversation).first()
            if sample_conversation:
                print(f"\nSample conversation attributes:")
                for attr in dir(sample_conversation):
                    if not attr.startswith('_') and not callable(getattr(sample_conversation, attr)):
                        value = getattr(sample_conversation, attr)
                        print(f"  - {attr}: {value} ({type(value)})")
            else:
                print("No conversations found in database")
        except Exception as e:
            print(f"Error getting sample conversation: {e}")

        return True

    except Exception as e:
        print(f"Database inspection failed: {e}")
        return False

def test_component_initialization():
    """Test if all components initialize properly"""
    print("\n" + "=" * 80)
    print("COMPONENT INITIALIZATION TEST")
    print("=" * 80)

    components = {}

    # Test Intent Classifier
    try:
        intent_classifier = IntentClassifier()
        model_info = intent_classifier.get_model_info()
        components['intent_classifier'] = {
            'status': 'SUCCESS',
            'model_loaded': model_info['model_loaded'],
            'confidence_threshold': model_info['confidence_threshold'],
            'supported_intents': len(model_info['supported_intents'])
        }
        print("✅ Intent Classifier: INITIALIZED")
        print(f"   Model loaded: {model_info['model_loaded']}")
        print(f"   Confidence threshold: {model_info['confidence_threshold']}")
        print(f"   Supported intents: {len(model_info['supported_intents'])}")
    except Exception as e:
        components['intent_classifier'] = {'status': 'FAILED', 'error': str(e)}
        print(f"❌ Intent Classifier: FAILED - {e}")

    # Test Entity Extractor
    try:
        entity_extractor = EntityExtractor()
        components['entity_extractor'] = {'status': 'SUCCESS'}
        print("✅ Entity Extractor: INITIALIZED")
    except Exception as e:
        components['entity_extractor'] = {'status': 'FAILED', 'error': str(e)}
        print(f"❌ Entity Extractor: FAILED - {e}")

    # Test Response Generator
    try:
        response_generator = ResponseGenerator()
        stats = response_generator.get_response_statistics()
        components['response_generator'] = {
            'status': 'SUCCESS',
            'total_intents': stats['total_intents'],
            'total_templates': stats['total_templates']
        }
        print("✅ Response Generator: INITIALIZED")
        print(f"   Total intents: {stats['total_intents']}")
        print(f"   Total templates: {stats['total_templates']}")
    except Exception as e:
        components['response_generator'] = {'status': 'FAILED', 'error': str(e)}
        print(f"❌ Response Generator: FAILED - {e}")

    # Test Learning Engine
    try:
        learning_engine = LearningEngine()
        components['learning_engine'] = {'status': 'SUCCESS'}
        print("✅ Learning Engine: INITIALIZED")
    except Exception as e:
        components['learning_engine'] = {'status': 'FAILED', 'error': str(e)}
        print(f"❌ Learning Engine: FAILED - {e}")

    return components

def test_intent_classification():
    """Test intent classification with common inputs"""
    print("\n" + "=" * 80)
    print("INTENT CLASSIFICATION TEST")
    print("=" * 80)

    test_inputs = [
        "Hello",
        "Hi there",
        "What places should I visit?",
        "Where can I go in Sri Lanka?",
        "How do I get to Kandy?",
        "Transportation to Galle",
        "Where can I stay?",
        "Hotels in Colombo",
        "What food should I try?",
        "Local cuisine",
        "Help me",
        "I need assistance"
    ]

    try:
        intent_classifier = IntentClassifier()

        results = []

        for test_input in test_inputs:
            try:
                intent, confidence = intent_classifier.classify_intent(test_input)
                all_probs = intent_classifier.get_intent_probabilities(test_input)

                # Get top 2 alternative intents
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:2]

                result = {
                    'input': test_input,
                    'predicted_intent': intent,
                    'confidence': confidence,
                    'is_unknown': intent == 'unknown',
                    'below_threshold': confidence < intent_classifier.confidence_threshold,
                    'alternatives': sorted_probs
                }

                results.append(result)

                status = "✅" if intent != "unknown" and confidence >= intent_classifier.confidence_threshold else "❌"
                print(f"{status} '{test_input}'")
                print(f"    → {intent} (confidence: {confidence:.3f})")

                if intent == "unknown" or confidence < intent_classifier.confidence_threshold:
                    print(f"    📊 Top alternatives: {sorted_probs[0][0]} ({sorted_probs[0][1]:.3f}), {sorted_probs[1][0]} ({sorted_probs[1][1]:.3f})")

            except Exception as e:
                print(f"❌ Error processing '{test_input}': {e}")
                results.append({
                    'input': test_input,
                    'error': str(e)
                })

        # Summary
        successful = len([r for r in results if r.get('predicted_intent') != 'unknown' and not r.get('below_threshold', False)])
        unknown_intents = len([r for r in results if r.get('is_unknown', False)])
        low_confidence = len([r for r in results if r.get('below_threshold', False)])

        print(f"\n📊 INTENT CLASSIFICATION SUMMARY:")
        print(f"   Total tests: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Unknown intents: {unknown_intents}")
        print(f"   Low confidence: {low_confidence}")
        print(f"   Success rate: {successful/len(results)*100:.1f}%")

        return results

    except Exception as e:
        print(f"❌ Intent classification test failed: {e}")
        return []

def test_response_generation():
    """Test response generation for different scenarios"""
    print("\n" + "=" * 80)
    print("RESPONSE GENERATION TEST")
    print("=" * 80)

    test_scenarios = [
        {
            'intent': 'general_greeting',
            'entities': {},
            'description': 'Simple greeting'
        },
        {
            'intent': 'destination_inquiry',
            'entities': {'locations': [{'text': 'Sri Lanka', 'type': 'country'}]},
            'description': 'Destination query'
        },
        {
            'intent': 'transportation',
            'entities': {'locations': [{'text': 'Kandy', 'type': 'city'}]},
            'description': 'Transportation query'
        },
        {
            'intent': 'unknown',
            'entities': {},
            'description': 'Unknown intent'
        }
    ]

    try:
        response_generator = ResponseGenerator()

        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario['description']}")
            print(f"   Intent: {scenario['intent']}")
            print(f"   Entities: {scenario['entities']}")

            try:
                response = response_generator.generate_response(
                    scenario['intent'],
                    scenario['entities'],
                    [],  # No knowledge results
                    {}   # No tool outputs
                )

                print(f"   ✅ Response: {response[:100]}...")

                # Check response quality
                if len(response) < 20:
                    print(f"   ⚠️  Very short response")
                elif "I apologize" in response or "I'm sorry" in response:
                    print(f"   ⚠️  Fallback response detected")
                else:
                    print(f"   ✅ Good response generated")

            except Exception as e:
                print(f"   ❌ Response generation failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Response generation test failed: {e}")
        return False

def check_model_files():
    """Check if required model files exist"""
    print("\n" + "=" * 80)
    print("MODEL FILES CHECK")
    print("=" * 80)

    required_files = [
        ("Intent Classifier Model", "app/ml/models/intent_classifier.pkl"),
        ("Intent Classifier Metadata", "app/ml/models/intent_classifier_metadata.json"),
        ("Entity Rules", "app/ml/models/entity_rules.json"),
        ("Response Templates", "app/ml/models/response_templates.json")
    ]

    file_status = {}

    for file_name, file_path in required_files:
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                print(f"✅ {file_name}: Found ({file_size} bytes)")
                file_status[file_name] = {'exists': True, 'size': file_size}
            except Exception as e:
                print(f"⚠️  {file_name}: Found but error reading ({e})")
                file_status[file_name] = {'exists': True, 'error': str(e)}
        else:
            print(f"❌ {file_name}: Missing ({file_path})")
            file_status[file_name] = {'exists': False, 'path': file_path}

    # Check directories
    directories = [
        "app/ml/models",
        "logs",
        "data/training_data"
    ]

    print(f"\n📁 DIRECTORIES:")
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}: Exists")
        else:
            print(f"❌ {directory}: Missing")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   ✅ Created {directory}")
            except Exception as e:
                print(f"   ❌ Failed to create {directory}: {e}")

    return file_status

def create_sample_data():
    """Create sample data for testing if needed"""
    print("\n" + "=" * 80)
    print("SAMPLE DATA CREATION")
    print("=" * 80)

    try:
        db = next(get_db())
        from backend.app.models.database import Conversation, UserFeedback

        # Check current data
        try:
            conversation_count = db.query(Conversation).count()
            print(f"Current conversations: {conversation_count}")
        except Exception as e:
            print(f"Error counting conversations: {e}")
            conversation_count = 0

        if conversation_count < 5:
            print("Creating sample conversations...")

            # We need to check what attributes the Conversation model actually has
            # Let's create a sample conversation to see what's required

            sample_conversations = [
                {
                    "user_message": "What are the best places to visit in Sri Lanka?",
                    "bot_response": "Sri Lanka offers amazing destinations like Kandy (cultural capital), Galle (historic fort), Ella (scenic hill country), and Sigiriya (ancient rock fortress).",
                    "intent": "destination_inquiry",
                    "confidence_score": 0.9,
                    "entities": {"locations": [{"text": "Sri Lanka", "type": "country"}]},
                    "tools_used": [],
                    "session_id": "sample_session_001"
                },
                {
                    "user_message": "How do I get from Colombo to Kandy?",
                    "bot_response": "You can travel from Colombo to Kandy by train (scenic 3-hour journey), bus (2.5 hours), or private car/taxi (2 hours via highway).",
                    "intent": "transportation",
                    "confidence_score": 0.85,
                    "entities": {"locations": [{"text": "Colombo", "type": "city"}, {"text": "Kandy", "type": "city"}]},
                    "tools_used": [],
                    "session_id": "sample_session_001"
                },
                {
                    "user_message": "Where can I stay in Galle?",
                    "bot_response": "Galle offers various accommodation from luxury hotels in Galle Fort to budget guesthouses. The fort area provides the best historic atmosphere.",
                    "intent": "accommodation",
                    "confidence_score": 0.8,
                    "entities": {"locations": [{"text": "Galle", "type": "city"}]},
                    "tools_used": [],
                    "session_id": "sample_session_001"
                }
            ]

            created_count = 0

            for conv_data in sample_conversations:
                try:
                    # Create conversation - we'll use only the attributes that exist
                    conversation = Conversation(
                        user_message=conv_data["user_message"],
                        bot_response=conv_data["bot_response"],
                        intent=conv_data["intent"],
                        confidence_score=conv_data["confidence_score"],
                        entities=conv_data["entities"],
                        tools_used=conv_data["tools_used"],
                        session_id=conv_data["session_id"]
                    )

                    db.add(conversation)
                    db.flush()  # Get the ID

                    # Create feedback
                    feedback = UserFeedback(
                        conversation_id=conversation.id,
                        rating=4,
                        is_helpful=True,
                        feedback_text="This was helpful for testing"
                    )

                    db.add(feedback)
                    created_count += 1

                    print(f"✅ Created: {conv_data['user_message'][:40]}...")

                except Exception as e:
                    print(f"❌ Failed to create conversation: {e}")
                    print(f"   Data: {conv_data}")

            try:
                db.commit()
                print(f"✅ Successfully created {created_count} sample conversations")
            except Exception as e:
                print(f"❌ Failed to commit to database: {e}")
                db.rollback()

        else:
            print("Sufficient sample data already exists")

        return True

    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def run_learning_test():
    """Test the learning system"""
    print("\n" + "=" * 80)
    print("LEARNING SYSTEM TEST")
    print("=" * 80)

    try:
        db = next(get_db())
        learning_engine = LearningEngine()

        # Get learning status
        print("Getting learning system status...")
        try:
            status = learning_engine.get_learning_status(db)
            print(f"✅ Learning system status retrieved")
            print(f"   System status: {status.get('system_status', 'unknown')}")
            print(f"   Health score: {status.get('health_score', 0):.1f}/100")

            stats = status.get('statistics', {})
            print(f"   Total conversations: {stats.get('total_conversations', 0)}")
            print(f"   Recent conversations: {stats.get('recent_conversations', 0)}")
            print(f"   Total feedback: {stats.get('total_feedback', 0)}")

        except Exception as e:
            print(f"❌ Failed to get learning status: {e}")

        # Test learning cycle
        print(f"\nTesting learning cycle...")
        try:
            result = learning_engine.force_learning_cycle(db)
            print(f"✅ Learning cycle completed")
            print(f"   Success: {result.get('success', False)}")
            if result.get('success'):
                print(f"   Data processed: {result.get('data_processed', 0)}")
                improvements = result.get('improvements_made', {})
                for improvement_type, details in improvements.items():
                    print(f"   {improvement_type}: {details}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Learning cycle test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Learning system test failed: {e}")
        return False

def generate_recommendations():
    """Generate specific recommendations based on findings"""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # Test components first
    components = test_component_initialization()

    # Check for failed components
    for comp_name, comp_data in components.items():
        if comp_data['status'] == 'FAILED':
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Component Failure',
                'issue': f'{comp_name} failed to initialize',
                'solution': f'Fix {comp_name}: {comp_data.get("error", "Unknown error")}',
                'action': f'Check dependencies and configuration for {comp_name}'
            })

    # Check intent classifier specifically
    if components.get('intent_classifier', {}).get('status') == 'SUCCESS':
        intent_data = components['intent_classifier']
        if not intent_data.get('model_loaded', False):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Model Training',
                'issue': 'Intent classifier model not loaded',
                'solution': 'Train the intent classification model',
                'action': 'Run: python backend/app/models/model_trainer.py'
            })

        if intent_data.get('confidence_threshold', 1.0) > 0.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Configuration',
                'issue': f'High confidence threshold ({intent_data.get("confidence_threshold")})',
                'solution': 'Lower confidence threshold to 0.4-0.5',
                'action': 'Update intent_classifier.confidence_threshold = 0.4'
            })

    # Check file existence
    file_status = check_model_files()
    missing_files = [name for name, status in file_status.items() if not status.get('exists', False)]

    if missing_files:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Missing Files',
            'issue': f'Missing model files: {", ".join(missing_files)}',
            'solution': 'Create missing model files',
            'action': 'Run model training and ensure file generation'
        })

    # Test intent classification
    intent_results = test_intent_classification()
    if intent_results:
        unknown_count = len([r for r in intent_results if r.get('is_unknown', False)])
        if unknown_count > len(intent_results) * 0.3:  # More than 30% unknown
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Intent Classification',
                'issue': f'High unknown intent rate ({unknown_count}/{len(intent_results)})',
                'solution': 'Improve intent classification training data',
                'action': 'Add more training examples and retrain model'
            })

    # Print recommendations
    if recommendations:
        print("🎯 PRIORITY ACTIONS NEEDED:")

        # Sort by priority
        priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)

        for i, rec in enumerate(recommendations, 1):
            priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(rec['priority'], '⚪')
            print(f"\n{i}. {priority_emoji} {rec['priority']} - {rec['category']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Solution: {rec['solution']}")
            print(f"   Action: {rec['action']}")

    else:
        print("✅ No critical issues found! Your system appears to be working well.")

    return recommendations

def main():
    """Run the fixed diagnostic"""
    print("🔍 CHATBOT SYSTEM DIAGNOSTIC (FIXED VERSION)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)

        # Step 1: Inspect database schema first
        inspect_database_schema()

        # Step 2: Test component initialization
        test_component_initialization()

        # Step 3: Check model files
        check_model_files()

        # Step 4: Create sample data if needed
        create_sample_data()

        # Step 5: Test intent classification
        test_intent_classification()

        # Step 6: Test response generation
        test_response_generation()

        # Step 7: Test learning system
        run_learning_test()

        # Step 8: Generate recommendations
        generate_recommendations()

        print(f"\n{'=' * 80}")
        print("🎉 DIAGNOSTIC COMPLETED")
        print("Follow the recommendations above to fix any issues.")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()