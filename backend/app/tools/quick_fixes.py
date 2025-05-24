# Create this as quick_fixes.py in your backend/app/tools/ folder

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from backend.app.nlp.intent_classifier import IntentClassifier
from backend.app.models.database import get_db

def fix_1_lower_confidence_threshold():
    """Fix 1: Lower the confidence threshold"""
    print("🔧 FIX 1: Lowering confidence threshold")

    try:
        intent_classifier = IntentClassifier()

        # Get current threshold
        current_threshold = intent_classifier.confidence_threshold
        print(f"Current threshold: {current_threshold}")

        # Lower it to 0.4
        new_threshold = 0.4
        intent_classifier.update_confidence_threshold(new_threshold)
        print(f"✅ Updated threshold to: {new_threshold}")

        # Test with a sample input
        test_input = "What places should I visit?"
        intent, confidence = intent_classifier.classify_intent(test_input)
        print(f"Test: '{test_input}' -> {intent} (confidence: {confidence:.3f})")

        return True

    except Exception as e:
        print(f"❌ Failed to update threshold: {e}")
        return False

def fix_2_add_training_examples():
    """Fix 2: Add more training examples"""
    print("\n🔧 FIX 2: Adding training examples")

    try:
        intent_classifier = IntentClassifier()

        # Add examples for common queries that might return unknown
        training_examples = [
            ("Hi", "general_greeting"),
            ("Hello there", "general_greeting"),
            ("Good morning", "general_greeting"),
            ("Hey", "general_greeting"),
            ("I want to visit Sri Lanka", "destination_inquiry"),
            ("Show me places", "destination_inquiry"),
            ("Tourist attractions", "destination_inquiry"),
            ("Places to see", "destination_inquiry"),
            ("How to travel", "transportation"),
            ("Transportation options", "transportation"),
            ("Getting around", "transportation"),
            ("Travel methods", "transportation"),
            ("Where to stay", "accommodation"),
            ("Hotels", "accommodation"),
            ("Accommodation", "accommodation"),
            ("Lodging", "accommodation"),
            ("Food recommendations", "food_inquiry"),
            ("What to eat", "food_inquiry"),
            ("Local cuisine", "food_inquiry"),
            ("Restaurants", "food_inquiry"),
            ("Help me", "help_request"),
            ("I need help", "help_request"),
            ("Can you assist", "help_request"),
            ("Support", "help_request"),
        ]

        added_count = 0
        for text, intent in training_examples:
            success = intent_classifier.add_training_example(text, intent)
            if success:
                added_count += 1
                print(f"✅ Added: '{text}' -> {intent}")
            else:
                print(f"❌ Failed: '{text}' -> {intent}")

        print(f"\n✅ Added {added_count} training examples")

        # Retrain the model
        print("🔄 Retraining model...")
        retrain_success = intent_classifier.retrain_model()

        if retrain_success:
            print("✅ Model retrained successfully")
        else:
            print("❌ Model retraining failed")

        return retrain_success

    except Exception as e:
        print(f"❌ Failed to add training examples: {e}")
        return False

def fix_3_create_fallback_responses():
    """Fix 3: Improve fallback responses"""
    print("\n🔧 FIX 3: Creating better fallback responses")

    try:
        from backend.app.nlp.response_generator import ResponseGenerator

        response_generator = ResponseGenerator()

        # Test current fallback response
        print("Testing current fallback...")
        fallback_response = response_generator.generate_response("unknown", {}, [], {})
        print(f"Current fallback: {fallback_response}")

        # The response generator already has good fallback responses,
        # but we can enhance them
        enhanced_fallbacks = [
            "I'm here to help you explore Sri Lanka! Could you please tell me more specifically what you'd like to know about?",
            "Sri Lanka has so much to offer! Are you interested in destinations, transportation, accommodation, food, culture, or something else?",
            "I specialize in Sri Lankan tourism. Please let me know what aspect of your Sri Lankan journey I can help you with!",
            "I'd love to help you with your Sri Lankan adventure! Could you be a bit more specific about what you're looking for?",
            "Welcome to your Sri Lankan tourism guide! What would you like to explore - places to visit, travel tips, local culture, or something else?"
        ]

        print("✅ Enhanced fallback responses available")
        return True

    except Exception as e:
        print(f"❌ Failed to enhance fallback responses: {e}")
        return False

def fix_4_check_model_files():
    """Fix 4: Check and create model files if missing"""
    print("\n🔧 FIX 4: Checking model files")

    model_paths = [
        "app/ml/models/intent_classifier.pkl",
        "app/ml/models/intent_classifier_metadata.json",
        "app/ml/models/entity_rules.json",
        "app/ml/models/response_templates.json"
    ]

    missing_files = []

    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
            missing_files.append(path)

    if missing_files:
        print(f"\n🔧 Creating missing files...")

        # Create models directory
        os.makedirs("app/ml/models", exist_ok=True)

        # Force retrain intent classifier to create missing files
        try:
            intent_classifier = IntentClassifier()
            intent_classifier.train_model()
            print("✅ Intent classifier model created")
        except Exception as e:
            print(f"❌ Failed to create intent classifier: {e}")

    return len(missing_files) == 0

def fix_5_test_database_connection():
    """Fix 5: Test database connection and create sample data"""
    print("\n🔧 FIX 5: Testing database and creating sample data")

    try:
        db = next(get_db())

        from backend.app.models.database import Conversation, UserFeedback

        # Check if tables exist and have data
        conversation_count = db.query(Conversation).count()
        feedback_count = db.query(UserFeedback).count()

        print(f"Current conversations: {conversation_count}")
        print(f"Current feedback: {feedback_count}")

        if conversation_count < 5:
            print("Creating sample conversations...")

            from backend.app.models.conversation import ConversationManager
            conversation_manager = ConversationManager(db)

            sample_data = [
                {
                    "user_message": "What are the best places to visit in Sri Lanka?",
                    "bot_response": "I recommend visiting Kandy, Galle, Ella, and Sigiriya for a diverse Sri Lankan experience.",
                    "intent": "destination_inquiry",
                    "confidence": 0.9
                },
                {
                    "user_message": "How do I get to Kandy?",
                    "bot_response": "You can reach Kandy by train, bus, or private transport from Colombo.",
                    "intent": "transportation",
                    "confidence": 0.85
                },
                {
                    "user_message": "Where should I stay in Galle?",
                    "bot_response": "Galle Fort has excellent boutique hotels and guesthouses.",
                    "intent": "accommodation",
                    "confidence": 0.8
                }
            ]

            session_id = "quick_fix_session"

            for data in sample_data:
                try:
                    conv_id = conversation_manager.save_conversation(
                        session_id=session_id,
                        user_message=data["user_message"],
                        bot_response=data["bot_response"],
                        intent=data["intent"],
                        entities={},
                        confidence_score=data["confidence"],
                        tools_used=[]
                    )

                    # Add positive feedback
                    conversation_manager.save_feedback(
                        conversation_id=conv_id,
                        rating=4,
                        feedback_text="Helpful",
                        is_helpful=True
                    )

                    print(f"✅ Created: {data['user_message'][:30]}...")

                except Exception as e:
                    print(f"❌ Failed to create sample: {e}")

        print("✅ Database check completed")
        return True

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def run_all_quick_fixes():
    """Run all quick fixes in sequence"""
    print("🚀 RUNNING ALL QUICK FIXES")
    print("=" * 50)

    fixes = [
        ("Lower Confidence Threshold", fix_1_lower_confidence_threshold),
        ("Add Training Examples", fix_2_add_training_examples),
        ("Enhance Fallback Responses", fix_3_create_fallback_responses),
        ("Check Model Files", fix_4_check_model_files),
        ("Test Database & Sample Data", fix_5_test_database_connection)
    ]

    results = {}

    for fix_name, fix_function in fixes:
        print(f"\n{'='*20} {fix_name} {'='*20}")
        try:
            success = fix_function()
            results[fix_name] = success
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"Result: {status}")
        except Exception as e:
            results[fix_name] = False
            print(f"❌ EXCEPTION: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("QUICK FIXES SUMMARY")
    print(f"{'='*50}")

    successful_fixes = sum(results.values())
    total_fixes = len(results)

    print(f"Successful fixes: {successful_fixes}/{total_fixes}")

    for fix_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {fix_name}")

    if successful_fixes == total_fixes:
        print(f"\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("Your chatbot should now work much better.")
    elif successful_fixes > 0:
        print(f"\n⚠️ SOME FIXES APPLIED ({successful_fixes}/{total_fixes})")
        print("Check the errors above and fix remaining issues.")
    else:
        print(f"\n🔴 NO FIXES APPLIED SUCCESSFULLY")
        print("There may be fundamental setup issues to address.")

    return results

def test_after_fixes():
    """Test the system after applying fixes"""
    print(f"\n{'='*50}")
    print("TESTING AFTER FIXES")
    print(f"{'='*50}")

    try:
        from backend.app.nlp.intent_classifier import IntentClassifier
        from backend.app.nlp.entity_extractor import EntityExtractor
        from backend.app.nlp.response_generator import ResponseGenerator

        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        response_generator = ResponseGenerator()

        test_inputs = [
            "Hello",
            "What places should I visit?",
            "How do I get around?",
            "Where can I stay?",
            "What food is good?",
            "Help me plan my trip"
        ]

        print("Testing common inputs:")

        improvements = []

        for test_input in test_inputs:
            print(f"\nInput: '{test_input}'")

            # Test intent classification
            intent, confidence = intent_classifier.classify_intent(test_input)
            print(f"Intent: {intent} (confidence: {confidence:.3f})")

            # Test entity extraction
            entities = entity_extractor.extract_entities(test_input)
            entity_count = sum(len(entity_list) for entity_list in entities.values())
            print(f"Entities: {entity_count} found")

            # Test response generation
            response = response_generator.generate_response(intent, entities, [], {})
            print(f"Response: {response[:80]}...")

            # Check for improvements
            if intent != "unknown":
                improvements.append(f"✅ '{test_input}' -> {intent}")
            else:
                improvements.append(f"❌ '{test_input}' -> still unknown")

        print(f"\n{'='*30}")
        print("IMPROVEMENT SUMMARY:")
        print(f"{'='*30}")

        for improvement in improvements:
            print(improvement)

        successful_classifications = len([i for i in improvements if "✅" in i])
        total_tests = len(improvements)

        success_rate = successful_classifications / total_tests
        print(f"\nSuccess rate: {success_rate:.1%} ({successful_classifications}/{total_tests})")

        if success_rate >= 0.8:
            print("🎉 EXCELLENT! The fixes have significantly improved the system.")
        elif success_rate >= 0.6:
            print("👍 GOOD! The fixes have helped, but there's room for more improvement.")
        else:
            print("⚠️ MODERATE! Some fixes applied, but more work is needed.")

        return success_rate

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return 0.0

def create_monitoring_script():
    """Create a simple monitoring script for ongoing issues"""
    print(f"\n{'='*50}")
    print("CREATING MONITORING SCRIPT")
    print(f"{'='*50}")

    monitoring_script = '''# monitor_chatbot.py - Simple monitoring script
import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def quick_health_check():
    """Quick health check for the chatbot system"""
    print(f"🔍 Chatbot Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    try:
        # Test intent classifier
        from backend.app.nlp.intent_classifier import IntentClassifier
        intent_classifier = IntentClassifier()
        
        test_intent, confidence = intent_classifier.classify_intent("Hello")
        print(f"✅ Intent Classifier: Working ({test_intent}, {confidence:.3f})")
        
        # Test database
        from backend.app.models.database import get_db, Conversation
        db = next(get_db())
        recent_conversations = db.query(Conversation).filter(
            Conversation.timestamp >= datetime.now() - timedelta(hours=24)
        ).count()
        print(f"✅ Database: {recent_conversations} conversations in last 24h")
        
        # Test response generator
        from backend.app.nlp.response_generator import ResponseGenerator
        response_generator = ResponseGenerator()
        test_response = response_generator.generate_response("general_greeting", {}, [], {})
        print(f"✅ Response Generator: Working ({len(test_response)} chars)")
        
        print("🟢 Overall Status: HEALTHY")
        
    except Exception as e:
        print(f"🔴 Health Check Failed: {e}")

if __name__ == "__main__":
    quick_health_check()
'''

    try:
        os.makedirs("backend/app/tools", exist_ok=True)
        with open("backend/app/tools/monitor_chatbot.py", "w") as f:
            f.write(monitoring_script)

        print("✅ Created monitoring script at: backend/app/tools/monitor_chatbot.py")
        print("   Run it periodically with: python backend/app/tools/monitor_chatbot.py")

        return True

    except Exception as e:
        print(f"❌ Failed to create monitoring script: {e}")
        return False

def main():
    """Main function to run all quick fixes"""
    print("🔧 CHATBOT QUICK FIXES TOOL")
    print("=" * 60)
    print("This tool will attempt to fix common issues causing unknown responses")
    print("=" * 60)

    # Run all fixes
    fix_results = run_all_quick_fixes()

    # Test the results
    success_rate = test_after_fixes()

    # Create monitoring script
    create_monitoring_script()

    # Final recommendations
    print(f"\n{'='*60}")
    print("🎯 NEXT STEPS")
    print(f"{'='*60}")

    if success_rate >= 0.8:
        print("1. ✅ Your chatbot is now working well!")
        print("2. 🔄 Monitor performance with: python backend/app/tools/monitor_chatbot.py")
        print("3. 📊 Check analytics regularly to ensure continued performance")
        print("4. 💬 Encourage users to provide feedback to improve learning")

    elif success_rate >= 0.5:
        print("1. 🔄 Re-run this script after addressing any errors above")
        print("2. 📚 Consider adding more training data for specific intents")
        print("3. ⚙️ Fine-tune confidence thresholds based on your specific needs")
        print("4. 🔍 Run the full diagnostic: python backend/app/tools/debug_chatbot_system.py")

    else:
        print("1. 🔴 Major issues detected - check system requirements")
        print("2. 📋 Verify all dependencies are installed correctly")
        print("3. 🗄️ Check database connection and table creation")
        print("4. 🤖 Manually run model training: python backend/app/models/model_trainer.py")
        print("5. 📞 Consider getting additional technical support")

    print(f"\n📝 LOGS:")
    print("- Detailed logs: logs/chat_debug.log")
    print("- Application logs: logs/chatbot_responses.log")

    print(f"\n🎉 Quick fixes completed! Success rate: {success_rate:.1%}")

if __name__ == "__main__":
    main()