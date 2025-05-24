import os
import random
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from datetime import timedelta

load_dotenv()

class Config:
    """Travel and Tourism Chatbot Configuration Class"""

    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/chatbot_db')

    # API Keys - Travel & Tourism Services
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
    GOOGLE_PLACES_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY')
    GOOGLE_TRANSLATE_API_KEY = os.getenv('GOOGLE_TRANSLATE_API_KEY')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

    # Travel Booking APIs
    BOOKING_COM_API_KEY = os.getenv('BOOKING_COM_API_KEY')
    EXPEDIA_API_KEY = os.getenv('EXPEDIA_API_KEY')
    SKYSCANNER_API_KEY = os.getenv('SKYSCANNER_API_KEY')
    TRIPADVISOR_API_KEY = os.getenv('TRIPADVISOR_API_KEY')

    # Additional Travel Services
    UBER_API_KEY = os.getenv('UBER_API_KEY')
    PICKUP_API_KEY = os.getenv('PICKUP_API_KEY')  # Local Sri Lankan ride service
    AIRBNB_API_KEY = os.getenv('AIRBNB_API_KEY')

    # AI/ML Services
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

    # Cache and Database
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')

    # Application Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-this')
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')

    # Environment Settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')  # development, staging, production
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # NLP and AI Configuration
    SPACY_MODEL = os.getenv('SPACY_MODEL', 'en_core_web_lg')
    INTENT_CONFIDENCE_THRESHOLD = float(os.getenv('INTENT_CONFIDENCE_THRESHOLD', '0.65'))
    ENTITY_CONFIDENCE_THRESHOLD = float(os.getenv('ENTITY_CONFIDENCE_THRESHOLD', '0.7'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.75'))
    MAX_TOKENS_PER_REQUEST = int(os.getenv('MAX_TOKENS_PER_REQUEST', '4000'))

    # Machine Learning Configuration
    MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL', '500'))  # Number of interactions
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    AUTO_RETRAIN_THRESHOLD = float(os.getenv('AUTO_RETRAIN_THRESHOLD', '0.8'))  # Accuracy threshold

    # API Request Configuration
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '10'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))

    # Session and User Management
    SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '3600'))  # 1 hour
    MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', '100'))
    USER_SESSION_CLEANUP_INTERVAL = int(os.getenv('USER_SESSION_CLEANUP_INTERVAL', '86400'))  # 24 hours
    MAX_CONCURRENT_SESSIONS = int(os.getenv('MAX_CONCURRENT_SESSIONS', '1000'))

    # File Upload Configuration
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '10485760'))  # 10MB
    ALLOWED_FILE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf', '.txt', '.csv'}
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')

    # Supported Languages (Sri Lanka focused)
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'si': 'Sinhala (සිංහල)',
        'ta': 'Tamil (தமிழ்)',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'ru': 'Russian',
        'ja': 'Japanese'
    }

    # Sri Lankan Geographic Data
    SRI_LANKAN_PROVINCES = [
        'Western Province', 'Central Province', 'Southern Province', 'Northern Province',
        'Eastern Province', 'North Western Province', 'North Central Province',
        'Uva Province', 'Sabaragamuwa Province'
    ]

    SRI_LANKAN_CITIES = [
        'colombo', 'kandy', 'galle', 'jaffna', 'negombo', 'anuradhapura',
        'polonnaruwa', 'ella', 'nuwara eliya', 'sigiriya', 'dambulla',
        'trincomalee', 'batticaloa', 'matara', 'hikkaduwa', 'mirissa',
        'unawatuna', 'bentota', 'kalutara', 'ratnapura', 'kurunegala',
        'badulla', 'kegalle', 'puttalam', 'monaragala', 'hambantota',
        'vavuniya', 'mullaitivu', 'kilinochchi', 'mannar', 'ampara',
        'chilaw', 'kalmunai', 'beruwala', 'tangalle', 'arugam bay'
    ]

    SRI_LANKAN_AIRPORTS = {
        'BIA': 'Bandaranaike International Airport (Colombo)',
        'HRI': 'Mattala Rajapaksa International Airport (Hambantota)',
        'RML': 'Ratmalana Airport (Colombo Domestic)'
    }

    # Currency Configuration
    SUPPORTED_CURRENCIES = [
        'USD', 'EUR', 'GBP', 'LKR', 'INR', 'JPY', 'AUD', 'CAD',
        'CHF', 'CNY', 'SGD', 'AED', 'SAR', 'THB', 'MYR'
    ]
    BASE_CURRENCY = 'LKR'  # Sri Lankan Rupee

    # Travel Intent Categories with Enhanced Keywords
    INTENT_CATEGORIES = {
        'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening', 'greetings', 'namaste', 'ayubowan'],
        'destination_inquiry': [
            'places to visit', 'tourist attractions', 'destinations', 'where to go', 'sightseeing',
            'must see', 'recommended places', 'top attractions', 'hidden gems', 'unesco sites'
        ],
        'accommodation': [
            'hotels', 'where to stay', 'accommodation', 'lodging', 'resorts', 'guesthouses',
            'homestays', 'villas', 'apartments', 'booking', 'rooms', 'luxury hotels', 'budget stays'
        ],
        'transportation': [
            'how to travel', 'transport', 'bus', 'train', 'taxi', 'getting around', 'uber',
            'tuk tuk', 'car rental', 'domestic flights', 'airport transfer', 'pickup'
        ],
        'food_inquiry': [
            'food', 'cuisine', 'restaurants', 'local dishes', 'what to eat', 'street food',
            'rice and curry', 'hoppers', 'kottu', 'traditional food', 'vegetarian', 'halal'
        ],
        'weather_request': [
            'weather', 'temperature', 'climate', 'forecast', 'rain', 'monsoon', 'humidity',
            'best time to visit', 'seasonal weather', 'typhoon', 'sunshine'
        ],
        'currency_conversion': [
            'currency', 'exchange rate', 'convert money', 'price', 'cost', 'budget',
            'rupees', 'dollars', 'euros', 'exchange', 'atm', 'money changer'
        ],
        'translation_request': [
            'translate', 'meaning', 'sinhala', 'tamil', 'language', 'how to say',
            'pronunciation', 'local language', 'phrase book'
        ],
        'map_request': [
            'map', 'directions', 'route', 'location', 'how to get', 'distance',
            'gps coordinates', 'navigation', 'nearby', 'address'
        ],
        'cultural_inquiry': [
            'culture', 'traditions', 'festivals', 'customs', 'religion', 'buddhism',
            'temples', 'etiquette', 'dress code', 'local customs', 'history'
        ],
        'booking_assistance': [
            'book', 'reservation', 'booking', 'availability', 'rates', 'packages',
            'tours', 'activities', 'tickets', 'safari', 'diving', 'hiking'
        ],
        'emergency': [
            'help', 'emergency', 'hospital', 'police', 'urgent', 'medical',
            'tourist police', 'embassy', 'consulate', 'lost passport'
        ],
        'travel_planning': [
            'itinerary', 'travel plan', 'trip planning', 'duration', 'how many days',
            'route planning', 'budget planning', 'travel tips', 'advice'
        ],
        'activities': [
            'activities', 'things to do', 'adventure', 'safari', 'diving', 'surfing',
            'hiking', 'whale watching', 'tea plantation', 'ayurveda', 'spa'
        ],
        'goodbye': ['bye', 'goodbye', 'see you', 'thanks', 'thank you', 'farewell', 'take care']
    }

    # Enhanced Response Templates
    RESPONSE_TEMPLATES = {
        'greeting': [
            "🇱🇰 Ayubowan! Welcome to Sri Lanka Tourism Assistant. I can help you discover the pearl of the Indian Ocean - from ancient temples to pristine beaches, wildlife safaris to cultural experiences. How can I assist your Sri Lankan adventure?",
            "Hello! I'm your personal Sri Lanka travel guide. Whether you're looking for destinations, weather updates, currency conversion, translations, or booking assistance, I'm here to help. What would you like to explore?",
            "Greetings, fellow traveler! Ready to discover Sri Lanka's wonders? I can assist with travel planning, cultural insights, local recommendations, and much more. Where shall we start?"
        ],
        'goodbye': [
            "🙏 Thank you for using Sri Lanka Tourism Assistant! May your Sri Lankan journey be filled with wonderful memories. Safe travels and Ayubowan!",
            "Farewell! I hope I've helped make your Sri Lanka trip planning easier. Feel free to return anytime for more assistance. Have an amazing adventure!",
            "Goodbye and thank you! Wishing you an incredible experience in beautiful Sri Lanka. Travel safely and enjoy every moment!"
        ],
        'error': [
            "I apologize for the confusion. Could you please rephrase your question? I'm here to help with all things Sri Lanka!",
            "Sorry, I'm having trouble understanding that request. Could you try asking in a different way? I want to provide you with the best assistance possible.",
            "I'm not quite sure what you're looking for. Could you provide a bit more detail? I'm eager to help with your Sri Lankan travel needs."
        ],
        'no_data': [
            "I don't have specific information about that right now, but I'm constantly learning! Is there something else about Sri Lanka I can help you with?",
            "That's not in my current knowledge base, but I'd love to help you with other Sri Lankan travel information. What else can I assist you with?",
            "I'm sorry I don't have that particular information. However, I can help you with destinations, weather, culture, food, and much more about Sri Lanka!"
        ],
        'booking_help': [
            "I'd be happy to help you with booking information! I can assist with hotels, activities, transportation, and tours. What would you like to book?",
            "Great choice! I can help you find and compare options for accommodation, activities, and transport. What are you looking to book?",
            "Let me help you with your booking needs! I have access to various travel services. What type of reservation are you interested in?"
        ]
    }

    # Emergency Contact Information (Sri Lanka)
    EMERGENCY_INFO = {
        'police': '119',
        'ambulance': '110',
        'fire_brigade': '111',
        'tourist_police': '1912',
        'tourist_hotline': '+94 11 242 1052',
        'accident_service': '1990',
        'disaster_management': '117',
        'coastguard': '+94 11 252 1101'
    }

    # Embassy Information
    EMBASSY_INFO = {
        'us_embassy': '+94 11 249 8500',
        'uk_embassy': '+94 11 539 0639',
        'indian_embassy': '+94 11 232 7587',
        'chinese_embassy': '+94 11 269 4491',
        'german_embassy': '+94 11 258 0431',
        'french_embassy': '+94 11 263 9400'
    }

    # Enhanced Popular Destinations
    POPULAR_DESTINATIONS = {
        'sigiriya': {
            'name': 'Sigiriya Rock Fortress',
            'description': 'Ancient palace and fortress complex built on a 200m high rock formation',
            'location': 'Central Province',
            'coordinates': {'lat': 7.9569, 'lng': 80.7598},
            'best_time': 'December to April',
            'entry_fee': 'USD 30 for foreigners, LKR 60 for locals',
            'duration': '3-4 hours',
            'difficulty': 'Moderate (climbing required)',
            'highlights': ['Ancient frescoes', 'Mirror wall', 'Lion gate', 'Royal gardens']
        },
        'kandy': {
            'name': 'Kandy - Cultural Capital',
            'description': 'Last kingdom of Sri Lanka, home to the sacred Temple of the Tooth',
            'location': 'Central Province',
            'coordinates': {'lat': 7.2906, 'lng': 80.6337},
            'best_time': 'December to April',
            'highlights': ['Temple of the Tooth', 'Kandy Lake', 'Royal Botanical Gardens', 'Cultural shows'],
            'festivals': ['Kandy Esala Perahera (July/August)']
        },
        'ella': {
            'name': 'Ella Hill Country',
            'description': 'Scenic mountain town perfect for hiking and tea plantation visits',
            'location': 'Uva Province',
            'coordinates': {'lat': 6.8667, 'lng': 81.0467},
            'best_time': 'January to March',
            'altitude': '1,041m above sea level',
            'highlights': ['Nine Arch Bridge', 'Ella Rock hike', 'Little Adams Peak', 'Tea factories'],
            'activities': ['Hiking', 'Train rides', 'Tea tasting', 'Zip-lining']
        },
        'galle': {
            'name': 'Galle Fort',
            'description': 'UNESCO World Heritage Dutch colonial fortress by the sea',
            'location': 'Southern Province',
            'coordinates': {'lat': 6.0367, 'lng': 80.2170},
            'best_time': 'December to March',
            'highlights': ['Historic ramparts', 'Dutch architecture', 'Lighthouse', 'Museums'],
            'activities': ['Walking tours', 'Shopping', 'Dining', 'Photography']
        }
    }

    # Travel Seasons
    TRAVEL_SEASONS = {
        'west_south_coast': {
            'best_season': 'December to March',
            'monsoon': 'May to September',
            'description': 'Ideal for beaches, cultural sites, and hill country'
        },
        'east_coast': {
            'best_season': 'April to September',
            'monsoon': 'October to January',
            'description': 'Perfect for surfing and beach activities'
        },
        'hill_country': {
            'best_season': 'December to March',
            'cool_season': 'April to August',
            'description': 'Great for hiking, tea plantations, and cooler weather'
        }
    }

    # Budget Guidelines (per person per day in USD)
    BUDGET_GUIDELINES = {
        'budget': {
            'range': '15-30 USD',
            'accommodation': 'Hostels, guesthouses',
            'food': 'Local restaurants, street food',
            'transport': 'Public buses, trains'
        },
        'mid_range': {
            'range': '30-80 USD',
            'accommodation': 'Mid-range hotels, boutique stays',
            'food': 'Mix of local and international restaurants',
            'transport': 'Private transport, domestic flights'
        },
        'luxury': {
            'range': '100+ USD',
            'accommodation': 'Luxury resorts, heritage hotels',
            'food': 'Fine dining, hotel restaurants',
            'transport': 'Private chauffeur, helicopter transfers'
        }
    }

    @classmethod
    def get_api_key(cls, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        api_keys = {
            'google_maps': cls.GOOGLE_MAPS_API_KEY,
            'google_places': cls.GOOGLE_PLACES_API_KEY,
            'google_translate': cls.GOOGLE_TRANSLATE_API_KEY,
            'openweather': cls.OPENWEATHER_API_KEY,
            'exchange_rate': cls.EXCHANGE_RATE_API_KEY,
            'booking_com': cls.BOOKING_COM_API_KEY,
            'expedia': cls.EXPEDIA_API_KEY,
            'skyscanner': cls.SKYSCANNER_API_KEY,
            'tripadvisor': cls.TRIPADVISOR_API_KEY,
            'uber': cls.UBER_API_KEY,
            'pickup': cls.PICKUP_API_KEY,
            'airbnb': cls.AIRBNB_API_KEY,
            'openai': cls.OPENAI_API_KEY,
            'anthropic': cls.ANTHROPIC_API_KEY,
            'hugging_face': cls.HUGGING_FACE_API_KEY
        }
        return api_keys.get(service.lower())

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Comprehensive configuration validation"""
        missing_keys = []
        warning_keys = []

        # Critical API keys (required for core functionality)
        critical_keys = [
            ('GOOGLE_MAPS_API_KEY', 'Maps and location services'),
            ('OPENWEATHER_API_KEY', 'Weather information'),
            ('EXCHANGE_RATE_API_KEY', 'Currency conversion'),
            ('GOOGLE_TRANSLATE_API_KEY', 'Translation services')
        ]

        # Optional but recommended API keys
        optional_keys = [
            ('BOOKING_COM_API_KEY', 'Hotel booking'),
            ('OPENAI_API_KEY', 'Enhanced AI responses')
        ]

        for key, description in critical_keys:
            if not getattr(cls, key):
                missing_keys.append(f"{key} ({description})")

        for key, description in optional_keys:
            if not getattr(cls, key):
                warning_keys.append(f"{key} ({description})")

        # Validate numeric configurations
        config_errors = []
        if cls.INTENT_CONFIDENCE_THRESHOLD < 0 or cls.INTENT_CONFIDENCE_THRESHOLD > 1:
            config_errors.append("INTENT_CONFIDENCE_THRESHOLD must be between 0 and 1")

        if cls.SESSION_TIMEOUT < 300:  # 5 minutes minimum
            config_errors.append("SESSION_TIMEOUT should be at least 300 seconds")

        return {
            'valid': len(missing_keys) == 0 and len(config_errors) == 0,
            'critical_missing': missing_keys,
            'optional_missing': warning_keys,
            'config_errors': config_errors,
            'database_url': cls.DATABASE_URL,
            'environment': cls.ENVIRONMENT,
            'debug_mode': cls.DEBUG,
            'supported_languages': len(cls.SUPPORTED_LANGUAGES),
            'supported_currencies': len(cls.SUPPORTED_CURRENCIES)
        }

    @classmethod
    def get_intent_keywords(cls, intent: str) -> List[str]:
        """Get keywords for a specific intent"""
        return cls.INTENT_CATEGORIES.get(intent, [])

    @classmethod
    def get_random_response(cls, template_type: str) -> str:
        """Get a random response template"""
        templates = cls.RESPONSE_TEMPLATES.get(template_type, cls.RESPONSE_TEMPLATES['error'])
        return random.choice(templates)

    @classmethod
    def get_destination_info(cls, destination: str) -> Optional[Dict[str, Any]]:
        """Get information about a popular destination"""
        return cls.POPULAR_DESTINATIONS.get(destination.lower())

    @classmethod
    def is_sri_lankan_city(cls, city: str) -> bool:
        """Check if a city is in Sri Lanka"""
        return city.lower() in [c.lower() for c in cls.SRI_LANKAN_CITIES]

    @classmethod
    def get_emergency_info(cls, emergency_type: str = None) -> Dict[str, str]:
        """Get emergency contact information"""
        if emergency_type:
            return {emergency_type: cls.EMERGENCY_INFO.get(emergency_type, 'Not available')}
        return cls.EMERGENCY_INFO

    @classmethod
    def get_budget_info(cls, budget_type: str) -> Optional[Dict[str, Any]]:
        """Get budget guidelines for travel planning"""
        return cls.BUDGET_GUIDELINES.get(budget_type.lower())

    @classmethod
    def get_seasonal_info(cls, region: str) -> Optional[Dict[str, Any]]:
        """Get seasonal travel information"""
        return cls.TRAVEL_SEASONS.get(region.lower())

# Create global configuration instance
config = Config()

# Validate configuration on import
validation_result = config.validate_config()

if not validation_result['valid']:
    print("⚠️  Configuration Validation Results:")
    if validation_result['critical_missing']:
        print(f"❌ Critical missing keys: {validation_result['critical_missing']}")
    if validation_result['optional_missing']:
        print(f"⚠️  Optional missing keys: {validation_result['optional_missing']}")
    if validation_result['config_errors']:
        print(f"❌ Configuration errors: {validation_result['config_errors']}")
    print("💡 Please check your .env file and add the missing API keys.")
    print("   Some features may not work properly without proper configuration.")
else:
    print("✅ Configuration validated successfully!")
    print(f"🌍 Environment: {validation_result['environment']}")
    print(f"🗣️  Languages supported: {validation_result['supported_languages']}")
    print(f"💱 Currencies supported: {validation_result['supported_currencies']}")

def get_config() -> Config:
    """Get the configuration instance"""
    return config

def get_api_key(service: str) -> Optional[str]:
    """Convenience function to get API key for a service"""
    return config.get_api_key(service)

def is_production() -> bool:
    """Check if running in production environment"""
    return config.ENVIRONMENT.lower() == 'production'

def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return config.DEBUG

# Export commonly used configurations
__all__ = [
    'Config', 'config', 'get_config', 'get_api_key',
    'is_production', 'is_debug_mode'
]