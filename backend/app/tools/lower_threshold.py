# Create this as lower_threshold.py in your backend/app/tools/ folder

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def lower_confidence_threshold():
    """Lower the confidence threshold and test the results"""
    print("🔧 LOWERING CONFIDENCE THRESHOLD")
    print("=" * 50)

    try:
        from backend.app.nlp.intent_classifier import IntentClassifier

        # Initialize intent classifier
        intent_classifier = IntentClassifier()

        # Check current threshold
        current_threshold = intent_classifier.confidence_threshold
        print(f"Current confidence threshold: {current_threshold}")

        # Lower the threshold
        new_threshold = 0.4
        intent_classifier.update_confidence_threshold(new_threshold)
        print(f"✅ Updated confidence threshold to: {new_threshold}")

        # Test with common inputs that might have been returning "unknown"
        test_inputs = [
            "Hello",
            "Hi there",
            "What places should I visit?",
            "Where can I go?",
            "How do I get there?",
            "Where should I stay?",
            "What food is good?",
            "Help me plan",
            "I need assistance"
        ]

        print(f"\n🧪 Testing with new threshold:")
        print("-" * 40)

        improved_count = 0

        for test_input in test_inputs:
            intent, confidence = intent_classifier.classify_intent(test_input)

            if intent != "unknown" and confidence >= new_threshold:
                status = "✅ GOOD"
                improved_count += 1
            elif intent != "unknown" and confidence < new_threshold:
                status = "⚠️ CLASSIFIED"
                improved_count += 1
            else:
                status = "❌ UNKNOWN"

            print(f"{status} '{test_input}' -> {intent} ({confidence:.3f})")

        improvement_rate = improved_count / len(test_inputs)
        print(f"\n📊 RESULTS:")
        print(f"Improved responses: {improved_count}/{len(test_inputs)} ({improvement_rate:.1%})")

        if improvement_rate >= 0.8:
            print("🎉 EXCELLENT! The threshold change significantly improved results.")
        elif improvement_rate >= 0.6:
            print("👍 GOOD! Notable improvement, but may need more training data.")
        else:
            print("⚠️ MODERATE improvement. Consider adding more training examples.")

        # Save the new threshold permanently
        print(f"\n💾 Saving new threshold permanently...")

        # Update the metadata file if it exists
        try:
            import json
            metadata_path = "app/ml/models/intent_classifier_metadata.json"

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                metadata['confidence_threshold'] = new_threshold
                metadata['last_updated'] = str(datetime.now())

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                print("✅ Updated metadata file with new threshold")
            else:
                print("⚠️ Metadata file not found, threshold will reset on restart")

        except Exception as e:
            print(f"⚠️ Could not save to metadata file: {e}")

        return new_threshold, improvement_rate

    except Exception as e:
        print(f"❌ Failed to update confidence threshold: {e}")
        return None, 0

def test_with_real_examples():
    """Test with real user examples that might have been failing"""
    print(f"\n🔍 TESTING WITH REAL USER EXAMPLES")
    print("-" * 50)

    try:
        from backend.app.nlp.intent_classifier import IntentClassifier
        from backend.app.nlp.entity_extractor import EntityExtractor
        from backend.app.nlp.response_generator import ResponseGenerator

        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        response_generator = ResponseGenerator()

        # Real examples that users might type
        real_examples = [
            "hi",
            "hello there",
            "places to visit",
            "tourist spots",
            "what to see",
            "best destinations",
            "how to travel",
            "getting around",
            "where to stay",
            "good hotels",
            "local food",
            "what to eat",
            "sri lankan culture",
            "temples to visit",
            "help me",
            "i need info"
        ]

        print("Testing complete pipeline with lowered threshold:")

        success_count = 0

        for example in real_examples:
            try:
                # Full pipeline test
                intent, confidence = intent_classifier.classify_intent(example)
                entities = entity_extractor.extract_entities(example)
                response = response_generator.generate_response(intent, entities, [], {})

                # Check if it's a good result
                is_good = (intent != "unknown" and
                           len(response) > 20 and
                           "apologize" not in response.lower())

                if is_good:
                    success_count += 1
                    status = "✅"
                else:
                    status = "❌"

                print(f"{status} '{example}' -> {intent} ({confidence:.3f})")

            except Exception as e:
                print(f"❌ '{example}' -> ERROR: {e}")

        success_rate = success_count / len(real_examples)
        print(f"\n📈 PIPELINE SUCCESS RATE: {success_rate:.1%} ({success_count}/{len(real_examples)})")

        return success_rate

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return 0

def create_config_file():
    """Create a configuration file for easy threshold management"""
    print(f"\n📝 CREATING CONFIGURATION FILE")
    print("-" * 50)

    try:
        config = {
            "intent_classifier": {
                "confidence_threshold": 0.4,
                "learning_threshold": 10,
                "auto_learning_enabled": True
            },
            "entity_extractor": {
                "confidence_threshold": 0.7
            },
            "response_generator": {
                "fallback_enabled": True,
                "enhancement_enabled": True
            },
            "learning_engine": {
                "min_samples_for_retraining": 20,
                "feedback_weight": 0.3,
                "confidence_weight": 0.4,
                "frequency_weight": 0.3
            },
            "last_updated": str(datetime.now()),
            "version": "1.0"
        }

        # Create config directory if it doesn't exist
        os.makedirs("config", exist_ok=True)

        # Save config file
        import json
        with open("config/chatbot_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("✅ Created config/chatbot_config.json")
        print("You can edit this file to adjust thresholds in the future")

        return True

    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
        return False

def main():
    """Main function to lower threshold and test"""
    from datetime import datetime

    print("🎯 CONFIDENCE THRESHOLD ADJUSTMENT TOOL")
    print("=" * 60)
    print("This will lower the confidence threshold to reduce 'unknown' responses")
    print("=" * 60)

    # Step 1: Lower the threshold
    new_threshold, improvement_rate = lower_confidence_threshold()

    if new_threshold:
        # Step 2: Test with real examples
        pipeline_success = test_with_real_examples()

        # Step 3: Create config file for future adjustments
        create_config_file()

        # Summary
        print(f"\n{'=' * 60}")
        print("📋 SUMMARY")
        print(f"{'=' * 60}")
        print(f"✅ Confidence threshold lowered to: {new_threshold}")
        print(f"📈 Intent classification improvement: {improvement_rate:.1%}")
        print(f"🔄 Full pipeline success rate: {pipeline_success:.1%}")

        if pipeline_success >= 0.8:
            print(f"\n🎉 EXCELLENT! Your chatbot should now work much better!")
            print(f"   • Restart your FastAPI server")
            print(f"   • Test with real user inputs")
            print(f"   • Monitor for any remaining issues")
        elif pipeline_success >= 0.6:
            print(f"\n👍 GOOD improvement! To get even better results:")
            print(f"   • Add more training examples")
            print(f"   • Consider lowering threshold further to 0.3")
            print(f"   • Ensure all model files are present")
        else:
            print(f"\n⚠️ MODERATE improvement. Additional steps needed:")
            print(f"   • Run: python create_missing_files.py")
            print(f"   • Add more training data")
            print(f"   • Consider retraining the model")

        print(f"\n📁 FILES TO CHECK:")
        print(f"   • config/chatbot_config.json (new configuration)")
        print(f"   • app/ml/models/intent_classifier_metadata.json (threshold saved)")

    else:
        print(f"\n❌ Failed to lower confidence threshold")
        print(f"   Check for errors above and try manual approach")

if __name__ == "__main__":
    main()