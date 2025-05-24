from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(100), nullable=False, index=True)  # destinations, transportation, accommodation, food, activities, culture
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    keywords = Column(JSON)  # JSON string of keywords for search
    confidence_score = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    location = Column(String(100))  # Specific location/destination
    season_relevance = Column(String(50))  # all, summer, winter, monsoon
    difficulty_level = Column(String(20))  # easy, moderate, difficult

    # Indexes for better query performance
    __table_args__ = (
        Index('idx_category_location', 'category', 'location'),
        Index('idx_keywords_active', 'keywords', 'is_active'),
    )

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    intent = Column(String(100), index=True)  # booking_inquiry, destination_info, travel_advice, etc.
    entities = Column(JSON)  # JSON string - destinations, dates, budget, etc.
    tools_used = Column(JSON)  # JSON string of tools used (weather_api, booking_api, etc.)
    satisfaction_score = Column(Float)
    response_time = Column(Float)  # Response time in seconds
    conversation_type = Column(String(50))  # planning, booking, support, information
    user_location = Column(String(100))  # User's current location
    travel_dates = Column(JSON)  # JSON string for travel date ranges
    budget_range = Column(String(50))  # budget, mid-range, luxury
    party_size = Column(Integer)  # Number of travelers
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class LearningData(Base):
    __tablename__ = 'learning_data'

    id = Column(Integer, primary_key=True, index=True)
    intent = Column(String(100), nullable=False, index=True)
    user_input = Column(Text, nullable=False)
    correct_response = Column(Text)
    feedback_score = Column(Float)
    is_validated = Column(Boolean, default=False)
    validation_source = Column(String(50))  # 'user', 'admin', 'auto', 'expert'
    domain_category = Column(String(50))  # travel, accommodation, transport, activities
    language_detected = Column(String(10))  # Language of user input
    context_data = Column(JSON)  # JSON string for conversation context
    improvement_suggestion = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserSession(Base):
    __tablename__ = 'user_sessions'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_preferences = Column(Text)  # JSON string for user preferences
    language_preference = Column(String(10), default='en')
    travel_interests = Column(Text)  # JSON array: adventure, culture, relaxation, food, etc.
    preferred_budget = Column(String(20))  # budget, mid-range, luxury
    travel_style = Column(String(30))  # solo, couple, family, group, business
    accessibility_needs = Column(Text)  # JSON string for special requirements
    previous_destinations = Column(Text)  # JSON array of visited places
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    total_messages = Column(Integer, default=0)
    total_bookings = Column(Integer, default=0)
    preferred_communication_style = Column(String(20))  # formal, casual, detailed, brief

class ToolUsage(Base):
    __tablename__ = 'tool_usage'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    tool_name = Column(String(50), nullable=False, index=True)  # weather_api, booking_api, maps_api, etc.
    input_data = Column(Text)  # JSON string
    output_data = Column(Text)  # JSON string
    execution_time = Column(Float)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    api_endpoint = Column(String(200))  # Specific API endpoint used
    api_cost = Column(Float)  # Cost of API call if applicable
    data_source = Column(String(50))  # External data source name
    created_at = Column(DateTime, default=datetime.utcnow)

class FeedbackData(Base):
    __tablename__ = 'feedback_data'

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    rating = Column(Integer)  # 1-5 rating
    feedback_text = Column(Text)
    feedback_type = Column(String(50))  # 'helpful', 'accurate', 'relevant', 'booking_success', etc.
    feedback_category = Column(String(50))  # information_quality, response_speed, booking_assistance
    would_recommend = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation")

class UserFeedback(Base):
    __tablename__ = 'user_feedback'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    user_rating = Column(Integer)  # 1-5 overall experience rating
    response_accuracy = Column(Integer)  # 1-5 accuracy rating
    response_helpfulness = Column(Integer)  # 1-5 helpfulness rating
    response_speed = Column(Integer)  # 1-5 speed rating
    booking_satisfaction = Column(Integer)  # 1-5 booking process satisfaction
    recommendation_quality = Column(Integer)  # 1-5 recommendation quality
    ease_of_use = Column(Integer)  # 1-5 chatbot ease of use
    feedback_text = Column(Text)
    improvement_suggestions = Column(Text)
    favorite_features = Column(Text)  # JSON array of liked features
    problematic_areas = Column(Text)  # JSON array of issues encountered
    would_use_again = Column(Boolean)
    would_recommend_to_others = Column(Boolean)
    user_expertise_level = Column(String(20))  # novice, intermediate, expert traveler
    interaction_type = Column(String(50))  # planning, booking, support, general_inquiry
    destination_discussed = Column(String(100))
    travel_date_range = Column(String(100))
    feedback_sentiment = Column(String(20))  # positive, negative, neutral
    follow_up_needed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversation = relationship("Conversation")

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'

    id = Column(Integer, primary_key=True, index=True)
    metric_date = Column(DateTime, nullable=False, index=True)
    total_conversations = Column(Integer, default=0)
    successful_interactions = Column(Integer, default=0)
    failed_interactions = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    average_user_satisfaction = Column(Float, default=0.0)
    total_bookings_assisted = Column(Integer, default=0)
    booking_conversion_rate = Column(Float, default=0.0)
    most_common_intent = Column(String(100))
    most_requested_destination = Column(String(100))
    total_api_calls = Column(Integer, default=0)
    api_success_rate = Column(Float, default=0.0)
    total_api_cost = Column(Float, default=0.0)
    unique_users = Column(Integer, default=0)
    returning_users = Column(Integer, default=0)
    average_conversation_length = Column(Float, default=0.0)  # Average messages per conversation
    peak_usage_hour = Column(Integer)  # Hour of day with most activity
    language_distribution = Column(Text)  # JSON object with language usage stats
    intent_accuracy = Column(Float, default=0.0)  # Intent classification accuracy
    entity_extraction_accuracy = Column(Float, default=0.0)
    knowledge_base_hit_rate = Column(Float, default=0.0)  # % of queries answered from KB
    escalation_rate = Column(Float, default=0.0)  # % of conversations escalated to human
    user_retention_rate = Column(Float, default=0.0)  # % of users who return
    average_session_duration = Column(Float, default=0.0)  # Minutes
    top_feedback_categories = Column(Text)  # JSON array of most common feedback types
    improvement_areas = Column(Text)  # JSON array of areas needing improvement
    model_version = Column(String(50))  # Version of the chatbot model
    training_data_size = Column(Integer)  # Number of training examples
    last_model_update = Column(DateTime)
    accuracy_by_category = Column(Text)  # JSON object with accuracy per travel category
    seasonal_performance = Column(Text)  # JSON object with performance by season
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite index for time-series queries
    __table_args__ = (
        Index('idx_metrics_date_version', 'metric_date', 'model_version'),
    )

# Database connection setup
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/travel_chatbot_db')

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database utility functions
def init_database():
    """Initialize database with sample data for travel and tourism"""
    create_tables()

    # Add sample knowledge base data for Sri Lanka tourism
    db = SessionLocal()
    try:
        # Check if data already exists
        existing_data = db.query(KnowledgeBase).first()
        if not existing_data:
            sample_knowledge = [
                {
                    'category': 'destinations',
                    'question': 'What are the best places to visit in Sri Lanka?',
                    'answer': 'Sri Lanka offers amazing destinations including: Sigiriya Rock Fortress (ancient palace and fortress with stunning frescoes), Kandy Temple of the Tooth (sacred Buddhist temple), Galle Fort (UNESCO World Heritage Dutch colonial fortress), Ella Hill Country (scenic mountain town perfect for hiking), Yala National Park (best wildlife safari destination for leopards), Anuradhapura Ancient City (ancient capital with sacred ruins), Polonnaruwa (medieval capital), and beautiful beaches in Mirissa, Unawatuna, and Arugam Bay.',
                    'keywords': '["destinations", "places", "visit", "tourist", "attractions", "sigiriya", "kandy", "galle", "ella", "yala", "beaches", "unesco"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'all',
                    'difficulty_level': 'easy'
                },
                {
                    'category': 'transportation',
                    'question': 'How to travel around Sri Lanka?',
                    'answer': 'Transportation options in Sri Lanka: Scenic train journeys (especially Colombo-Kandy-Ella route with mountain views), public buses (affordable but crowded), private buses (more comfortable), tuk-tuks (perfect for short city trips), taxi services and ride-hailing apps, car rentals with driver (recommended for tourists), and domestic flights for longer distances. The train from Kandy to Ella is considered one of the most beautiful train rides in the world.',
                    'keywords': '["transportation", "travel", "bus", "train", "taxi", "tuk-tuk", "car rental", "scenic route", "ella", "kandy"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'all',
                    'difficulty_level': 'easy'
                },
                {
                    'category': 'accommodation',
                    'question': 'Where to stay in Sri Lanka?',
                    'answer': 'Sri Lanka accommodation options: Luxury beach resorts (Amanwella, Shangri-La Hambantota), heritage hotels in cultural cities, eco-lodges in national parks, mountain retreats in hill country, boutique hotels in Galle Fort, beachside villas, homestays for cultural immersion, budget hostels for backpackers, and tea estate bungalows. Popular areas: Colombo (business hub), Kandy (cultural triangle), Ella (mountains), Galle (history), Mirissa/Unawatuna (beaches), Sigiriya (heritage).',
                    'keywords': '["accommodation", "hotels", "stay", "lodging", "resorts", "guesthouses", "homestays", "luxury", "budget", "eco-lodge"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'all',
                    'difficulty_level': 'easy'
                },
                {
                    'category': 'food',
                    'question': 'What is Sri Lankan cuisine like?',
                    'answer': 'Sri Lankan cuisine is a flavorful blend featuring: Rice and curry (the staple meal with various curries), hoppers and string hoppers (fermented rice flour pancakes), kottu roti (chopped roti stir-fry), fish ambul thiyal (sour fish curry), dhal curry, coconut sambol, wood apple curry, and tropical fruits. Key ingredients include coconut, curry leaves, pandan, cinnamon, cardamom, and chilies. Must-try drinks: King coconut water, Ceylon tea, and arrack. The cuisine varies by region with Tamil influences in the north and Malay influences in coastal areas.',
                    'keywords': '["food", "cuisine", "rice", "curry", "hoppers", "kottu", "local dishes", "spicy", "coconut", "tea", "street food"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'all',
                    'difficulty_level': 'easy'
                },
                {
                    'category': 'culture',
                    'question': 'Tell me about Sri Lankan culture and customs',
                    'answer': 'Sri Lankan culture is deeply rooted in Buddhism (70% of population), Hinduism, Islam, and Christianity. Key aspects: Respect for elders and religious sites, traditional arts like Kandyan dance and drumming, colorful festivals (Vesak, Kandy Perahera, Thai Pusam), ancient crafts (wood carving, batik, gem cutting), and renowned hospitality. Temple etiquette: Remove shoes and hats, dress modestly covering shoulders and knees, no pointing feet toward Buddha statues. The island has a rich literary tradition and is famous for Ceylon tea culture.',
                    'keywords': '["culture", "traditions", "festivals", "buddhist", "hindu", "dance", "temples", "customs", "respect", "arts"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'all',
                    'difficulty_level': 'easy'
                },
                {
                    'category': 'weather',
                    'question': 'What is the weather and best time to visit Sri Lanka?',
                    'answer': 'Sri Lanka has a tropical climate with two monsoon seasons: Southwest monsoon (May-September) affects west and south coasts with heavy rains; Northeast monsoon (December-February) brings rain to north and east. Best travel times: West/South coast and hill country (December-March), East coast (April-September). Temperature: 26-30°C year-round in coastal areas, cooler in mountains (15-20°C in Nuwara Eliya). Humidity is high, so pack light, breathable clothing and rain gear.',
                    'keywords': '["weather", "climate", "monsoon", "temperature", "best time", "seasons", "rain", "humidity", "when to visit"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'all',
                    'difficulty_level': 'easy'
                },
                {
                    'category': 'activities',
                    'question': 'What activities and adventures can I do in Sri Lanka?',
                    'answer': 'Sri Lanka adventure activities: Wildlife safaris (Yala, Udawalawe for elephants, Wilpattu), whale watching in Mirissa (blue whales, sperm whales), surfing in Arugam Bay and Hikkaduwa, hiking (Ella Rock, Little Adam\'s Peak, World\'s End in Horton Plains), white water rafting in Kitulgala, scuba diving and snorkeling, hot air ballooning over cultural sites, rock climbing, tea plantation tours in Nuwara Eliya, bird watching (over 400 species), and cycling through countryside.',
                    'keywords': '["activities", "adventure", "safari", "wildlife", "whale watching", "surfing", "hiking", "diving", "tea tours", "cycling"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'varies',
                    'difficulty_level': 'moderate'
                },
                {
                    'category': 'budget',
                    'question': 'How much does it cost to travel in Sri Lanka?',
                    'answer': 'Sri Lanka travel costs (per day): Budget travelers: $15-25 (hostels, local buses, street food), Mid-range: $30-60 (guesthouses, private transport, restaurants), Luxury: $100+ (hotels, private tours, fine dining). Key expenses: Accommodation ($5-200/night), meals ($2-20), transportation ($10-50/day), entrance fees ($5-30), activities ($20-100). Money-saving tips: Use public transport, eat at local places, book accommodations in advance, travel during shoulder season. Tipping: 10% at restaurants, $2-5 for guides.',
                    'keywords': '["budget", "cost", "money", "expensive", "cheap", "prices", "accommodation", "food", "transport", "activities"]',
                    'location': 'Sri Lanka',
                    'season_relevance': 'all',
                    'difficulty_level': 'easy'
                }
            ]

            for item in sample_knowledge:
                kb_entry = KnowledgeBase(**item)
                db.add(kb_entry)

            # Add sample model metrics
            sample_metrics = ModelMetrics(
                metric_date=datetime.utcnow(),
                total_conversations=0,
                model_version="1.0.0",
                training_data_size=len(sample_knowledge)
            )
            db.add(sample_metrics)

            db.commit()
            print("Sample travel and tourism knowledge base data added successfully!")
            print("Database tables created with tourism-specific schema!")

    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()