# Create this as permanent_threshold_fix.py in your backend/app/tools/ folder

import sys
import os
import json
import pickle
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def fix_threshold_in_metadata():
    """Fix the threshold in the metadata file"""
    print("🔧 FIXING THRESHOLD IN METADATA FILE")
    print("=" * 50)

    metadata_path = "app/ml/models/intent_classifier_metadata.json"

    try:
        if os.path.exists(metadata_path):
            # Read existing metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            print(f"Current metadata threshold: {metadata.get('confidence_threshold', 'not set')}")

            # Update threshold
            metadata['confidence_threshold'] = 0.4
            metadata['last_updated'] = datetime.now().isoformat()

            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print("✅ Updated metadata file with threshold 0.4")
            return True
        else:
            print("❌ Metadata file not found, creating new one...")

            # Create new metadata file
            metadata = {
                "model_type": "intent_classifier",
                "confidence_threshold": 0.4,
                "created_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }

            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print("✅ Created new metadata file with threshold 0.4")
            return True

    except Exception as e:
        print(f"❌ Failed to update metadata: {e}")
        return False

def fix_threshold_in_model_file():
    """Fix the threshold in the actual model pickle file"""
    print("\n🔧 FIXING THRESHOLD IN MODEL FILE")
    print("=" * 50)

    model_path = "app/ml/models/intent_classifier.pkl"

    try:
        if os.path.exists(model_path):
            # Load the model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            print(f"Model file found, updating threshold...")

            # Update threshold in model data
            if isinstance(model_data, dict):
                model_data['confidence_threshold'] = 0.4
                model_data['last_updated'] = datetime.now().isoformat()

            # Save updated model
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print("✅ Updated model file with threshold 0.4")
            return True
        else:
            print("❌ Model file not found")
            return False

    except Exception as e:
        print(f"❌ Failed to update model file: {e}")
        return False

def modify_intent_classifier_code():
    """Modify the IntentClassifier source code directly"""
    print("\n🔧 MODIFYING INTENT CLASSIFIER SOURCE CODE")
    print("=" * 50)

    try:
        intent_classifier_path = "backend/app/nlp/intent_classifier.py"

        if not os.path.exists(intent_classifier_path):
            print(f"❌ IntentClassifier file not found at {intent_classifier_path}")
            return False

        # Read the file
        with open(intent_classifier_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check current threshold setting
        if "self.confidence_threshold = 0.6" in content:
            print("Found threshold = 0.6 in source code")

            # Replace with 0.4
            updated_content = content.replace(
                "self.confidence_threshold = 0.6",
                "self.confidence_threshold = 0.4"
            )

            # Create backup
            backup_path = intent_classifier_path + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created backup at {backup_path}")

            # Write updated content
            with open(intent_classifier_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            print("✅ Updated source code with threshold 0.4")
            return True

        elif "self.confidence_threshold = 0.4" in content:
            print("✅ Source code already has threshold 0.4")
            return True

        else:
            print("⚠️ Could not find threshold setting in source code")
            print("Looking for other patterns...")

            # Look for other possible patterns
            if "confidence_threshold" in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "confidence_threshold" in line and "self." in line:
                        print(f"Found at line {i+1}: {line.strip()}")

            return False

    except Exception as e:
        print(f"❌ Failed to modify source code: {e}")
        return False

def create_environment_config():
    """Create environment configuration for threshold"""
    print("\n🔧 CREATING ENVIRONMENT CONFIGURATION")
    print("=" * 50)

    try:
        # Check if .env file exists
        env_path = ".env"
        env_content = ""

        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_content = f.read()

        # Add or update threshold setting
        threshold_line = "INTENT_CONFIDENCE_THRESHOLD=0.4"

        if "INTENT_CONFIDENCE_THRESHOLD" in env_content:
            # Replace existing line
            lines = env_content.split('\n')
            updated_lines = []
            for line in lines:
                if line.startswith("INTENT_CONFIDENCE_THRESHOLD"):
                    updated_lines.append(threshold_line)
                else:
                    updated_lines.append(line)
            env_content = '\n'.join(updated_lines)
        else:
            # Add new line
            if env_content and not env_content.endswith('\n'):
                env_content += '\n'
            env_content += threshold_line + '\n'

        # Write updated .env file
        with open(env_path, 'w') as f:
            f.write(env_content)

        print(f"✅ Updated .env file with INTENT_CONFIDENCE_THRESHOLD=0.4")

        # Also create a sample environment modification for the IntentClassifier
        sample_code = '''
# Add this to your IntentClassifier.__init__() method:
import os
self.confidence_threshold = float(os.getenv('INTENT_CONFIDENCE_THRESHOLD', 0.4))
'''

        print("\n📝 To use environment variables, add this to IntentClassifier:")
        print(sample_code)

        return True

    except Exception as e:
        print(f"❌ Failed to create environment config: {e}")
        return False

def verify_fix():
    """Verify that the fix worked"""
    print("\n🧪 VERIFYING THE FIX")
    print("=" * 50)

    try:
        from backend.app.nlp.intent_classifier import IntentClassifier

        # Create new instance to see if threshold changed
        intent_classifier = IntentClassifier()

        print(f"Current threshold: {intent_classifier.confidence_threshold}")

        if intent_classifier.confidence_threshold == 0.4:
            print("✅ SUCCESS! Threshold is now 0.4")

            # Test with a few examples
            test_cases = [
                "hello",
                "what places to visit",
                "help me"
            ]

            print(f"\nTesting with lowered threshold:")
            unknown_count = 0

            for test in test_cases:
                intent, confidence = intent_classifier.classify_intent(test)
                if intent == "unknown":
                    unknown_count += 1
                    status = "❌"
                else:
                    status = "✅"

                print(f"{status} '{test}' -> {intent} ({confidence:.3f})")

            if unknown_count == 0:
                print(f"\n🎉 PERFECT! All test cases now return valid intents")
            elif unknown_count < len(test_cases):
                print(f"\n👍 IMPROVED! Reduced unknown responses ({unknown_count}/{len(test_cases)} still unknown)")
            else:
                print(f"\n⚠️ Still getting unknown responses - may need additional fixes")

            return True

        else:
            print(f"❌ FAILED! Threshold is still {intent_classifier.confidence_threshold}")
            return False

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Apply permanent threshold fix"""
    print("🔧 PERMANENT CONFIDENCE THRESHOLD FIX")
    print("=" * 60)
    print("This will permanently lower the confidence threshold to 0.4")
    print("=" * 60)

    fixes_applied = []

    # Try multiple approaches
    if fix_threshold_in_metadata():
        fixes_applied.append("Metadata file")

    if fix_threshold_in_model_file():
        fixes_applied.append("Model file")

    if modify_intent_classifier_code():
        fixes_applied.append("Source code")

    if create_environment_config():
        fixes_applied.append("Environment config")

    print(f"\n📊 FIXES APPLIED:")
    for fix in fixes_applied:
        print(f"✅ {fix}")

    if not fixes_applied:
        print("❌ No fixes could be applied")
        return

    print(f"\n🔄 RESTARTING COMPONENTS...")
    print("You need to restart your FastAPI server for changes to take effect")

    # Verify if possible (this will only work if source code was modified)
    if "Source code" in fixes_applied:
        print(f"\n🧪 TESTING AFTER SOURCE CODE MODIFICATION...")

        try:
            # Force reload of the module
            import importlib
            from backend.app.nlp import intent_classifier
            importlib.reload(intent_classifier)

            verify_fix()

        except Exception as e:
            print(f"Cannot test immediately due to module caching: {e}")
            print("Restart your server and run verification")

    print(f"\n📋 NEXT STEPS:")
    print(f"1. 🔄 Restart your FastAPI server")
    print(f"2. 🧪 Test your chatbot with common inputs")
    print(f"3. 🔍 Run diagnostic again to verify:")
    print(f"   python backend/app/tools/debug_chatbot_fixed.py")
    print(f"4. 📊 Check that threshold now shows 0.4")

    print(f"\n💡 IF THRESHOLD IS STILL 0.6 AFTER RESTART:")
    print(f"   The threshold might be hardcoded in __init__. Manually edit:")
    print(f"   backend/app/nlp/intent_classifier.py")
    print(f"   Find: self.confidence_threshold = 0.6")
    print(f"   Change to: self.confidence_threshold = 0.4")

if __name__ == "__main__":
    main()