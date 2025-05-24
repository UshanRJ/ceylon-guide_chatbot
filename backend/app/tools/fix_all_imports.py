# fix_all_imports.py - Updated with correct path
import re
import os

def fix_all_learning_imports():
    """Fix ALL learning engine imports in chat_routes.py"""

    # Get the correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chat_routes_path = os.path.join(script_dir, "..", "api", "chat_routes.py")
    chat_routes_path = os.path.normpath(chat_routes_path)

    print(f"Looking for file at: {chat_routes_path}")

    if not os.path.exists(chat_routes_path):
        print(f"❌ File not found: {chat_routes_path}")
        print("Current directory:", os.getcwd())
        print("Script directory:", script_dir)
        return False

    with open(chat_routes_path, "r", encoding='utf-8') as f:
        content = f.read()

    print("✅ File found, original content length:", len(content))

    # 1. Comment out main LearningEngine import
    content = re.sub(
        r"from backend\.app\.ml\.learning_engine import LearningEngine",
        "# from backend.app.ml.learning_engine import LearningEngine  # Commented out",
        content
    )

    # 2. Remove any other learning_engine imports
    content = re.sub(
        r"from backend\.app\.ml\.learning_engine import.*",
        "# Removed problematic learning_engine import",
        content
    )

    # 3. Make sure the simple import is there
    if "from backend.app.ml.learning_functions_simple import" not in content:
        # Add the import after other imports
        import_position = content.find("from backend.app.utils.helpers import")
        if import_position != -1:
            insert_position = content.find("\n", import_position) + 1
            new_import = "from backend.app.ml.learning_functions_simple import check_and_trigger_learning_simple, trigger_learning_with_feedback_simple, get_learning_status_simple\n"
            content = content[:insert_position] + new_import + content[insert_position:]

    # 4. Replace problematic functions
    replacements = [
        # Fix check_and_trigger_learning function
        (
            r"def check_and_trigger_learning\(db: Session\):.*?(?=def|\n@|\nclass|\Z)",
            '''def check_and_trigger_learning(db: Session):
    """Check if learning should be triggered and do it"""
    try:
        return check_and_trigger_learning_simple(db)
    except Exception as e:
        logger.error(f"Learning check failed: {e}")

'''
        ),
        # Fix trigger_learning_with_feedback function
        (
            r"def trigger_learning_with_feedback\(db: Session.*?\):.*?(?=def|\n@|\nclass|\Z)",
            '''def trigger_learning_with_feedback(db: Session, rating: int = None):
    """Trigger learning update based on feedback"""
    try:
        return trigger_learning_with_feedback_simple(db, rating)
    except Exception as e:
        logger.error(f"Learning with feedback failed: {e}")

'''
        ),
        # Fix trigger_learning function
        (
            r"def trigger_learning\(db: Session\):.*?(?=def|\n@|\nclass|\Z)",
            '''def trigger_learning(db: Session):
    """Original trigger learning function for compatibility"""
    try:
        from backend.app.models.database import Conversation
        total_conversations = db.query(Conversation).count()
        logger.info(f"Background learning triggered: {total_conversations} total conversations")
    except Exception as e:
        logger.error(f"Background learning failed: {e}")

'''
        )
    ]

    # Apply all replacements
    for pattern, replacement in replacements:
        old_content = content
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        if content != old_content:
            print(f"✅ Applied replacement for pattern: {pattern[:50]}...")

    # Write the fixed content
    with open(chat_routes_path, "w", encoding='utf-8') as f:
        f.write(content)

    print("Fixed all learning engine imports")
    print("New content length:", len(content))
    return True

if __name__ == "__main__":
    fix_all_learning_imports()