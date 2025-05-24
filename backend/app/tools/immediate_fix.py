# Create this as immediate_fix.py in your backend/app/tools/ folder
# This will quickly test and fix the most common issues

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def test_current_system():
    """Test the current system with common inputs"""
    print("🔍 TESTING CURRENT SYSTEM")
    print("=" * 50)

    try:
        from backend.app.nlp.intent_classifier import IntentClassifier
        from backend.app.nlp.entity_extractor import EntityExtractor
        from backend.app.nlp.response_generator import ResponseGenerator

        # Initialize components
        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        response_generator = ResponseGenerator()

        # Test inputs that users commonly try
        test_inputs = [
            "Hello",
            "Hi there",
            "What places should I visit in Sri Lanka?",
            "Best destinations in Sri Lanka",
            "How do I get to Kandy?",
            "Transportation to Galle",
            "Where can I stay in Colombo?",
            "Hotels in Sri Lanka",
            "What food should I try?",
            "Sri Lankan cuisine"
        ]

        print("Testing with common user inputs:\n")

        unknown_count = 0
        low_confidence_count = 0

        for test_input in test_inputs:
            # Test intent classification
            intent, confidence = intent_classifier.classify_intent(test_input)

            # Test entity extraction
            entities = entity_extractor.extract_entities(test_input)

            # Test response generation
            response = response_generator.generate_response(intent, entities, [], {})

            # Analyze results
            is_unknown = intent == "unknown"
            is_low_confidence = confidence < 0.6
            is_fallback = "apologize" in response.lower() or "sorry" in response.lower()

            if is_unknown:
                unknown_count += 1
                status = "❌ UNKNOWN"
            elif is_low_confidence:
                low_confidence_count += 1
                status = "⚠️ LOW CONF"
            else:
                status = "✅ GOOD"

            print(f"{status} '{test_input}'")
            print(f"     Intent: {intent} (confidence: {confidence:.3f})")
            print(f"     Response: {response[:60]}...")

            if is_unknown or is_low_confidence:
                # Get alternatives to see what the classifier is thinking
                all_probs = intent_classifier.get_intent_probabilities(test_input)
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"     Top 3 alternatives: {[(intent, f'{prob:.3f}') for intent, prob in sorted_probs]}")

            print()

        print(f"📊 SUMMARY:")
        print(f"   Total tests: {len(test_inputs)}")
        print(f"   Unknown responses: {unknown_count}")
        print(f"   Low confidence: {low_confidence_count}")
        print(f"   Success rate: {(len(test_inputs) - unknown_count) / len(test_inputs) * 100:.1f}%")

        return {
            'total': len(test_inputs),
            'unknown': unknown_count,
            'low_confidence': low_confidence_count,
            'threshold': intent_classifier.confidence_threshold
        }

    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_immediate_fixes():
    """Apply immediate fixes for unknown responses"""
    print("\n🔧 APPLYING IMMEDIATE FIXES")
    print("=" * 50)

    fixes_applied = []

    # Fix 1: Lower confidence threshold
    try:
        from backend.app.nlp.intent_classifier import IntentClassifier
        intent_classifier = IntentClassifier()

        current_threshold = intent_classifier.confidence_threshold
        print(f"Current confidence threshold: {current_threshold}")

        if current_threshold > 0.5:
            new_threshold = 0.4
            intent_classifier.update_confidence_threshold(new_threshold)
            print(f"✅ Lowered confidence threshold to {new_threshold}")
            fixes_applied.append("Lowered confidence threshold")
        else:
            print(f"✅ Confidence threshold already reasonable: {current_threshold}")

    except Exception as e:
        print(f"❌ Failed to adjust confidence threshold: {e}")

    # Fix 2: Check if model file exists and is loaded
    try:
        model_info = intent_classifier.get_model_info()
        print(f"Model loaded: {model_info['model_loaded']}")
        print(f"Supported intents: {len(model_info['supported_intents'])}")

        if not model_info['model_loaded']:
            print("⚠️ Model not loaded - trying to retrain...")
            try:
                intent_classifier.train_model()
                print("✅ Model retrained successfully")
                fixes_applied.append("Retrained model")
            except Exception as e:
                print(f"❌ Model retraining failed: {e}")

    except Exception as e:
        print(f"❌ Failed to check model status: {e}")

    # Fix 3: Add some common training examples
    try:
        print("Adding common training examples...")

        common_examples = [
            ("hello", "general_greeting"),
            ("hi", "general_greeting"),
            ("hey there", "general_greeting"),
            ("good morning", "general_greeting"),
            ("places to visit", "destination_inquiry"),
            ("tourist attractions", "destination_inquiry"),
            ("what to see", "destination_inquiry"),
            ("sightseeing", "destination_inquiry"),
            ("how to get there", "transportation"),
            ("travel options", "transportation"),
            ("getting around", "transportation"),
            ("where to stay", "accommodation"),
            ("hotels", "accommodation"),
            ("food recommendations", "food_inquiry"),
            ("what to eat", "food_inquiry"),
            ("help me", "help_request"),
            ("i need help", "help_request")
        ]

        added_count = 0
        for text, intent in common_examples:
            try:
                if intent_classifier.add_training_example(text, intent):
                    added_count += 1
            except:
                pass  # Some might already exist

        if added_count > 0:
            print(f"✅ Added {added_count} training examples")
            # Retrain with new examples
            intent_classifier.retrain_model()
            print("✅ Retrained model with new examples")
            fixes_applied.append(f"Added {added_count} training examples")

    except Exception as e:
        print(f"❌ Failed to add training examples: {e}")

    return fixes_applied

def test_after_fixes():
    """Test the system after applying fixes"""
    print("\n🧪 TESTING AFTER FIXES")
    print("=" * 50)

    # Run the same tests again
    results_after = test_current_system()

    if results_after:
        improvement_msg = ""
        if results_after['unknown'] == 0:
            improvement_msg = "🎉 EXCELLENT! No more unknown responses!"
        elif results_after['unknown'] <= 2:
            improvement_msg = "👍 GOOD! Significantly reduced unknown responses!"
        elif results_after['unknown'] <= 4:
            improvement_msg = "⚠️ MODERATE improvement, but more work needed"
        else:
            improvement_msg = "❌ Minimal improvement - deeper issues need fixing"

        print(f"\n{improvement_msg}")

        # Specific recommendations
        if results_after['unknown'] > 0:
            print(f"\n💡 NEXT STEPS:")
            print(f"   1. Check if specific user inputs are still failing")
            print(f"   2. Add more training data for problematic intents")
            print(f"   3. Consider further lowering confidence threshold to {results_after['threshold'] - 0.1:.1f}")
            print(f"   4. Review intent definitions and keywords")

        return results_after

    return None

def create_test_endpoint():
    """Create a simple test endpoint you can call"""
    print("\n🔗 CREATING TEST ENDPOINT")
    print("=" * 50)

    test_endpoint_code = '''# Add this to your chat_routes.py for testing

@router.get("/test-intent/{text}")
async def test_intent_classification(text: str):
    """Test endpoint to debug intent classification"""
    try:
        from backend.app.nlp.intent_classifier import IntentClassifier
        from backend.app.nlp.entity_extractor import EntityExtractor
        from backend.app.nlp.response_generator import ResponseGenerator
        
        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        response_generator = ResponseGenerator()
        
        # Test intent classification
        intent, confidence = intent_classifier.classify_intent(text)
        all_probabilities = intent_classifier.get_intent_probabilities(text)
        
        # Test entity extraction
        entities = entity_extractor.extract_entities(text)
        
        # Test response generation
        response = response_generator.generate_response(intent, entities, [], {})
        
        return {
            "input_text": text,
            "intent": intent,
            "confidence": confidence,
            "all_probabilities": dict(sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)),
            "entities": entities,
            "response": response,
            "is_unknown": intent == "unknown",
            "below_threshold": confidence < intent_classifier.confidence_threshold,
            "threshold": intent_classifier.confidence_threshold
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    print("✅ Test endpoint code generated")
    print("Add this to your chat_routes.py, then test with:")
    print("GET http://localhost:8000/test-intent/hello")
    print("GET http://localhost:8000/test-intent/what%20places%20should%20I%20visit")

    return test_endpoint_code

def main():
    """Run immediate fixes"""
    print("⚡ IMMEDIATE CHATBOT FIXES")
    print("=" * 60)
    print("This will quickly test and fix common 'unknown response' issues")
    print("=" * 60)

    # Test current state
    print("1️⃣ Testing current system...")
    results_before = test_current_system()

    if results_before and results_before['unknown'] > 0:
        # Apply fixes
        print("2️⃣ Applying fixes...")
        fixes_applied = apply_immediate_fixes()

        # Test after fixes
        print("3️⃣ Testing after fixes...")
        results_after = test_after_fixes()

        # Create test endpoint
        print("4️⃣ Creating test tools...")
        create_test_endpoint()

        # Summary
        print(f"\n{'=' * 60}")
        print("📋 SUMMARY")
        print(f"{'=' * 60}")
        print(f"Fixes applied: {len(fixes_applied)}")
        for fix in fixes_applied:
            print(f"  ✅ {fix}")

        if results_before and results_after:
            improvement = results_before['unknown'] - results_after['unknown']
            print(f"\nImprovement: Reduced unknown responses by {improvement}")
            print(f"Before: {results_before['unknown']}/{results_before['total']} unknown")
            print(f"After: {results_after['unknown']}/{results_after['total']} unknown")

        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Test your chatbot with common user inputs")
        print(f"2. If still getting unknowns, run the full diagnostic:")
        print(f"   python backend/app/tools/debug_chatbot_fixed.py")
        print(f"3. Add the test endpoint to your chat_routes.py for debugging")
        print(f"4. Monitor logs at: logs/chat_debug.log")

    else:
        print("✅ System appears to be working well!")
        print("If you're still having issues, run the full diagnostic.")

if __name__ == "__main__":
    main()