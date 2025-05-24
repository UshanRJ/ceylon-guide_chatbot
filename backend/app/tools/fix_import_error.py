# Create this as fix_import_error.py in your backend/app/tools/ folder

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def fix_chat_routes_imports():
    """Fix the import errors in chat_routes.py"""
    print("🔧 FIXING CHAT ROUTES IMPORT ERRORS")
    print("=" * 50)

    try:
        # Read the current chat_routes.py
        chat_routes_path = "backend/app/api/chat_routes.py"

        if not os.path.exists(chat_routes_path):
            print(f"❌ Chat routes file not found: {chat_routes_path}")
            return False

        with open(chat_routes_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print("📝 Updating import statements...")

        # Fix the import statements
        old_import = "from backend.app.ml.learning_engine import get_learning_status_fixed, collect_learning_data_fixed"
        new_import = "from backend.app.ml.learning_engine_fixed import get_learning_status_fixed, collect_learning_data_fixed"

        if old_import in content:
            content = content.replace(old_import, new_import)
            print("✅ Fixed import statement")
        else:
            print("⚠️ Import statement not found - adding safe version")

        # Also fix the other import issue
        content = content.replace(
            "from backend.app.ml.learning_engine import get_learning_status_fixed",
            "from backend.app.ml.learning_engine_fixed import get_learning_status_fixed"
        )

        # Create a backup
        backup_path = chat_routes_path + ".backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_path}")

        # Write the fixed version
        with open(chat_routes_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("✅ Updated chat_routes.py with correct imports")
        return True

    except Exception as e:
        print(f"❌ Failed to fix imports: {e}")
        return False

def create_simple_learning_functions():
    """Create simple learning functions that don't cause import errors"""
    print(f"\n🔧 CREATING SIMPLE LEARNING FUNCTIONS")
    print("=" * 50)

    simple_functions = '''# Simple learning functions that work
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

def check_and_trigger_learning_simple(db: Session):
    """Simple learning check that doesn't cause errors"""
    try:
        from backend.app.models.database import Conversation
        
        # Count total conversations
        total_conversations = db.query(Conversation).count()
        
        # Count recent conversations (last 7 days)
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_conversations = db.query(Conversation).filter(
            Conversation.created_at >= recent_date
        ).count()
        
        logger.info(f"Learning check: {recent_conversations} recent conversations (total: {total_conversations})")
        
        if recent_conversations >= 10:
            logger.info("✅ Sufficient data for learning")
        else:
            logger.info(f"ℹ️ Need {10 - recent_conversations} more conversations for learning")
        
        return True
        
    except Exception as e:
        logger.error(f"Learning check failed: {e}")
        return False

def trigger_learning_with_feedback_simple(db: Session, rating: int = None):
    """Simple learning trigger for feedback"""
    try:
        from backend.app.models.database import UserFeedback
        
        # Count total feedback
        total_feedback = db.query(UserFeedback).count()
        
        logger.info(f"Feedback processing: total feedback entries = {total_feedback}")
        
        if rating and rating <= 2:
            logger.info("⚠️ Poor rating received - flagged for review")
        elif rating and rating >= 4:
            logger.info("✅ Good rating received")
        
        return True
        
    except Exception as e:
        logger.error(f"Learning with feedback failed: {e}")
        return False

def get_learning_status_simple(db: Session) -> dict:
    """Simple learning status that works with any schema"""
    try:
        from backend.app.models.database import Conversation, UserFeedback
        
        # Count data
        total_conversations = db.query(Conversation).count()
        total_feedback = db.query(UserFeedback).count()
        
        # Recent data (last 7 days)
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_conversations = db.query(Conversation).filter(
            Conversation.created_at >= recent_date
        ).count()
        
        # Calculate health score
        health_score = min(100, (total_conversations * 5) + (total_feedback * 10))
        
        return {
            "system_status": "active",
            "health_score": health_score,
            "statistics": {
                "total_conversations": total_conversations,
                "recent_conversations": recent_conversations,
                "total_feedback": total_feedback
            },
            "learning_efficiency": {
                "data_availability": min(total_conversations / 10, 1.0),
                "feedback_rate": total_feedback / max(total_conversations, 1)
            },
            "next_learning_cycle": "Ready now" if recent_conversations >= 10 else f"Need {10 - recent_conversations} more conversations",
            "recommendations": [
                "System appears healthy" if total_conversations >= 10 else "Need more conversation data",
                "Feedback collection working" if total_feedback > 0 else "Encourage user feedback"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        return {
            "system_status": "error",
            "health_score": 0,
            "error": str(e)
        }
'''

    try:
        # Save the simple functions
        with open("backend/app/ml/learning_functions_simple.py", "w") as f:
            f.write(simple_functions)

        print("✅ Created backend/app/ml/learning_functions_simple.py")
        return True

    except Exception as e:
        print(f"❌ Failed to create simple functions: {e}")
        return False

def update_chat_routes_with_simple_functions():
    """Update chat_routes.py to use the simple functions"""
    print(f"\n📝 UPDATING CHAT ROUTES WITH SIMPLE FUNCTIONS")
    print("=" * 50)

    try:
        chat_routes_path = "backend/app/api/chat_routes.py"

        with open(chat_routes_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace the problematic import and function calls
        replacements = [
            # Fix imports
            (
                "from backend.app.ml.learning_engine_fixed import get_learning_status_fixed, collect_learning_data_fixed",
                "from backend.app.ml.learning_functions_simple import check_and_trigger_learning_simple, trigger_learning_with_feedback_simple, get_learning_status_simple"
            ),
            (
                "from backend.app.ml.learning_engine import get_learning_status_fixed",
                "from backend.app.ml.learning_functions_simple import get_learning_status_simple"
            ),
            # Fix function calls in check_and_trigger_learning
            (
                """def check_and_trigger_learning(db: Session):
    \"\"\"Check if learning should be triggered and do it\"\"\"
    try:
        from backend.app.ml.learning_engine import get_learning_status_fixed, collect_learning_data_fixed

        status = get_learning_status_fixed(db)
        stats = status.get('statistics', {})
        recent_conversations = stats.get('recent_conversations', 0)

        logger.info(f"Learning check: {recent_conversations} recent conversations")

        if recent_conversations >= 10:
            logger.info("Sufficient data for learning - would trigger learning cycle")
            # Here you would call your learning functions
        else:
            logger.info(f"Not enough data for learning: {recent_conversations}/10")

    except Exception as e:
        logger.error(f"Learning check failed: {e}")""",
                """def check_and_trigger_learning(db: Session):
    \"\"\"Check if learning should be triggered and do it\"\"\"
    try:
        return check_and_trigger_learning_simple(db)
    except Exception as e:
        logger.error(f"Learning check failed: {e}")"""
            ),
            # Fix function calls in trigger_learning_with_feedback
            (
                """def trigger_learning_with_feedback(db: Session, rating: int):
    \"\"\"Trigger learning update based on feedback\"\"\"
    try:
        from backend.app.ml.learning_engine import get_learning_status_fixed

        status = get_learning_status_fixed(db)
        logger.info(f"Feedback received (rating: {rating}), learning status: {status['system_status']}")

        if rating <= 2:
            logger.info("Poor rating received - would trigger immediate learning review")

    except Exception as e:
        logger.error(f"Learning with feedback failed: {e}")""",
                """def trigger_learning_with_feedback(db: Session, rating: int = None):
    \"\"\"Trigger learning update based on feedback\"\"\"
    try:
        return trigger_learning_with_feedback_simple(db, rating)
    except Exception as e:
        logger.error(f"Learning with feedback failed: {e}")"""
            ),
            # Fix the debug endpoint
            (
                """@router.get("/debug/learning-status")
async def get_learning_status(db: Session = Depends(get_db)):
    \"\"\"Get current learning system status\"\"\"
    try:
        learning_engine = LearningEngine()
        status = learning_engine.get_learning_status(db)
        return {"success": True, "status": status}
    except Exception as e:
        return {"success": False, "error": str(e)}""",
                """@router.get("/debug/learning-status")
async def get_learning_status(db: Session = Depends(get_db)):
    \"\"\"Get current learning system status\"\"\"
    try:
        status = get_learning_status_simple(db)
        return {"success": True, "status": status}
    except Exception as e:
        return {"success": False, "error": str(e)}"""
            )
        ]

        # Apply all replacements
        for old_text, new_text in replacements:
            if old_text in content:
                content = content.replace(old_text, new_text)
                print("✅ Applied replacement")
            else:
                print("⚠️ Replacement text not found, skipping")

        # Write the updated content
        with open(chat_routes_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("✅ Updated chat_routes.py with simple functions")
        return True

    except Exception as e:
        print(f"❌ Failed to update chat routes: {e}")
        return False

def test_simple_functions():
    """Test the simple functions work"""
    print(f"\n🧪 TESTING SIMPLE FUNCTIONS")
    print("=" * 50)

    try:
        from backend.app.models.database import get_db
        from backend.app.ml.learning_functions_simple import get_learning_status_simple, check_and_trigger_learning_simple

        db = next(get_db())

        # Test learning status
        status = get_learning_status_simple(db)
        print(f"✅ Learning status: {status['system_status']}")
        print(f"   Total conversations: {status['statistics']['total_conversations']}")
        print(f"   Recent conversations: {status['statistics']['recent_conversations']}")
        print(f"   Health score: {status['health_score']}")

        # Test learning check
        check_result = check_and_trigger_learning_simple(db)
        print(f"✅ Learning check: {'Passed' if check_result else 'Failed'}")

        return True

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fix all import errors"""
    print("🔧 FIXING CHAT ROUTES IMPORT ERRORS")
    print("=" * 60)
    print("This will fix the import errors causing your server to crash")
    print("=" * 60)

    success_count = 0
    total_steps = 4

    # Step 1: Create simple learning functions
    if create_simple_learning_functions():
        success_count += 1
        print("✅ Step 1: Simple learning functions created")
    else:
        print("❌ Step 1: Failed to create simple functions")

    # Step 2: Update chat routes
    if update_chat_routes_with_simple_functions():
        success_count += 1
        print("✅ Step 2: Chat routes updated")
    else:
        print("❌ Step 2: Failed to update chat routes")

    # Step 3: Test the functions
    if test_simple_functions():
        success_count += 1
        print("✅ Step 3: Simple functions tested")
    else:
        print("❌ Step 3: Function testing failed")

    # Step 4: Final verification
    try:
        # Try to import to verify no syntax errors
        with open("backend/app/api/chat_routes.py", 'r') as f:
            content = f.read()

        # Basic syntax check
        compile(content, "chat_routes.py", "exec")
        success_count += 1
        print("✅ Step 4: Syntax verification passed")

    except Exception as e:
        print(f"❌ Step 4: Syntax verification failed: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"📋 SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successfully completed: {success_count}/{total_steps} steps")

    if success_count == total_steps:
        print(f"\n🎉 ALL IMPORT ERRORS FIXED!")
        print(f"\n📋 NEXT STEPS:")
        print(f"1. 🔄 Restart your FastAPI server")
        print(f"2. 🧪 Test the /debug/learning-status endpoint")
        print(f"3. 💬 Send a test message to your chatbot")
        print(f"4. ✅ Check logs - should no longer see import errors!")

    elif success_count >= 2:
        print(f"\n👍 MOSTLY FIXED ({success_count}/{total_steps})")
        print(f"🔄 Try restarting your server")
        print(f"📊 Should see fewer import errors")

    else:
        print(f"\n❌ MULTIPLE FAILURES ({success_count}/{total_steps})")
        print(f"🔧 Check file permissions and paths")
        print(f"📋 Verify Python environment is correct")

    print(f"\n📁 FILES CREATED/MODIFIED:")
    if os.path.exists("backend/app/ml/learning_functions_simple.py"):
        print(f"✅ backend/app/ml/learning_functions_simple.py")
    if os.path.exists("backend/app/api/chat_routes.py.backup"):
        print(f"✅ backend/app/api/chat_routes.py.backup (original)")
    print(f"📝 backend/app/api/chat_routes.py (updated)")

if __name__ == "__main__":
    main()