# Create this as create_missing_files.py in your backend/app/tools/ folder

import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def create_entity_rules_file():
    """Create the missing entity_rules.json file"""
    print("🔧 Creating entity_rules.json...")

    try:
        # Create the entity rules based on your EntityExtractor class
        entity_rules = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "sri_lankan_locations": {
                "colombo": {
                    "lat": 6.9271,
                    "lng": 79.8612,
                    "type": "city",
                    "province": "Western",
                    "aliases": ["colombo city", "commercial capital"],
                    "importance": "high"
                },
                "kandy": {
                    "lat": 7.2906,
                    "lng": 80.6337,
                    "type": "city",
                    "province": "Central",
                    "aliases": ["kandy city", "cultural capital"],
                    "importance": "high"
                },
                "galle": {
                    "lat": 6.0535,
                    "lng": 80.2210,
                    "type": "city",
                    "province": "Southern",
                    "aliases": ["galle city", "galle fort"],
                    "importance": "high"
                },
                "ella": {
                    "lat": 6.8721,
                    "lng": 81.0465,
                    "type": "town",
                    "province": "Uva",
                    "aliases": ["ella town", "ella rock"],
                    "importance": "high"
                },
                "sigiriya": {
                    "lat": 7.9568,
                    "lng": 80.7603,
                    "type": "archaeological_site",
                    "province": "Central",
                    "aliases": ["sigiriya rock", "lion rock", "sigiriya fortress"],
                    "importance": "high"
                },
                "negombo": {
                    "lat": 7.2084,
                    "lng": 79.8358,
                    "type": "city",
                    "province": "Western",
                    "aliases": ["negombo city"],
                    "importance": "medium"
                },
                "nuwara eliya": {
                    "lat": 6.9497,
                    "lng": 80.7891,
                    "type": "hill_station",
                    "province": "Central",
                    "aliases": ["nuwara eliya city", "little england"],
                    "importance": "high"
                },
                "mirissa": {
                    "lat": 5.9487,
                    "lng": 80.4501,
                    "type": "beach_town",
                    "province": "Southern",
                    "aliases": ["mirissa beach"],
                    "importance": "high"
                },
                "anuradhapura": {
                    "lat": 8.3114,
                    "lng": 80.4037,
                    "type": "ancient_city",
                    "province": "North Central",
                    "aliases": ["anuradhapura ancient city"],
                    "importance": "high"
                },
                "polonnaruwa": {
                    "lat": 7.9403,
                    "lng": 81.0188,
                    "type": "ancient_city",
                    "province": "North Central",
                    "aliases": ["polonnaruwa ancient city"],
                    "importance": "high"
                }
            },
            "currency_patterns": {
                "usd": {
                    "patterns": [
                        "\\$\\s*([0-9,]+(?:\\.[0-9]{1,2})?)",
                        "([0-9,]+(?:\\.[0-9]{1,2})?)\\s*(?:us\\s*)?dollars?",
                        "([0-9,]+(?:\\.[0-9]{1,2})?)\\s*usd"
                    ]
                },
                "lkr": {
                    "patterns": [
                        "rs\\.?\\s*([0-9,]+(?:\\.[0-9]{1,2})?)",
                        "([0-9,]+(?:\\.[0-9]{1,2})?)\\s*(?:sri\\s*lankan\\s*)?rupees?",
                        "([0-9,]+(?:\\.[0-9]{1,2})?)\\s*lkr"
                    ]
                },
                "eur": {
                    "patterns": [
                        "€\\s*([0-9,]+(?:\\.[0-9]{1,2})?)",
                        "([0-9,]+(?:\\.[0-9]{1,2})?)\\s*euros?"
                    ]
                },
                "gbp": {
                    "patterns": [
                        "£\\s*([0-9,]+(?:\\.[0-9]{1,2})?)",
                        "([0-9,]+(?:\\.[0-9]{1,2})?)\\s*pounds?"
                    ]
                }
            },
            "tourism_entities": {
                "attractions": {
                    "temples": ["temple", "kovil", "devale", "dagoba", "stupa", "vihara", "monastery"],
                    "historical": ["fort", "palace", "ruins", "archaeological site", "ancient city", "heritage site"],
                    "natural": ["beach", "waterfall", "mountain", "hill", "lake", "river", "forest", "national park"],
                    "modern": ["museum", "gallery", "shopping mall", "market", "zoo", "aquarium"]
                },
                "activities": {
                    "adventure": ["safari", "trekking", "hiking", "climbing", "diving", "snorkeling", "surfing"],
                    "cultural": ["cultural show", "dance performance", "cooking class", "meditation", "yoga"],
                    "wildlife": ["whale watching", "bird watching", "elephant watching", "leopard safari"],
                    "leisure": ["spa", "massage", "beach relaxation", "sunset viewing", "photography"]
                },
                "transportation": {
                    "public": ["bus", "train", "tuk tuk", "three wheeler"],
                    "private": ["taxi", "car", "van", "motorbike", "bicycle"],
                    "special": ["boat", "helicopter", "seaplane", "domestic flight"]
                },
                "accommodation": {
                    "luxury": ["resort", "luxury hotel", "boutique hotel", "5 star hotel"],
                    "standard": ["hotel", "inn", "lodge"],
                    "budget": ["guesthouse", "hostel", "homestay", "budget hotel"],
                    "unique": ["eco lodge", "treehouse", "villa", "bungalow", "camp"]
                }
            }
        }

        # Create directory if it doesn't exist
        os.makedirs("app/ml/models", exist_ok=True)

        # Save the file
        with open("app/ml/models/entity_rules.json", "w", encoding="utf-8") as f:
            json.dump(entity_rules, f, indent=2, ensure_ascii=False)

        print("✅ Created app/ml/models/entity_rules.json")
        return True

    except Exception as e:
        print(f"❌ Failed to create entity_rules.json: {e}")
        return False

def create_response_templates_file():
    """Create the missing response_templates.json file"""
    print("🔧 Creating response_templates.json...")

    try:
        # Create comprehensive response templates
        response_templates = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "destination_inquiry": {
                "templates": [
                    "Here are some amazing places to visit in Sri Lanka: {content}",
                    "I'd recommend these fantastic destinations: {content}",
                    "Sri Lanka has wonderful places like: {content}",
                    "You'll love these destinations: {content}",
                    "For sightseeing, I suggest: {content}",
                    "These are must-visit places in Sri Lanka: {content}",
                    "Let me recommend some beautiful locations: {content}",
                    "Perfect destinations for your Sri Lankan adventure: {content}"
                ],
                "contextual_modifiers": {
                    "family": "family-friendly destinations",
                    "adventure": "exciting adventure spots",
                    "culture": "culturally rich locations",
                    "nature": "scenic natural attractions",
                    "history": "historically significant sites",
                    "beach": "beautiful coastal destinations",
                    "mountain": "stunning hill country locations"
                },
                "follow_up_suggestions": [
                    "Would you like specific information about any of these places?",
                    "I can provide more details about transportation or accommodation for these destinations.",
                    "Would you like to know the best time to visit these places?",
                    "I can help you plan an itinerary for these locations."
                ]
            },
            "transportation": {
                "templates": [
                    "For getting around Sri Lanka: {content}",
                    "Here are your transportation options: {content}",
                    "You can travel in Sri Lanka by: {content}",
                    "Transportation methods available: {content}",
                    "To reach your destination: {content}",
                    "Travel options include: {content}",
                    "Getting there is easy with: {content}",
                    "Your journey options: {content}"
                ],
                "contextual_modifiers": {
                    "budget": "affordable transportation options",
                    "comfort": "comfortable travel methods",
                    "fast": "quickest routes available",
                    "scenic": "scenic travel routes",
                    "local": "authentic local transport experiences"
                },
                "follow_up_suggestions": [
                    "Would you like specific schedules or pricing information?",
                    "I can help you book transportation if needed.",
                    "Need directions or route planning assistance?",
                    "Would you prefer door-to-door or public transport options?"
                ]
            },
            "accommodation": {
                "templates": [
                    "For accommodation in Sri Lanka: {content}",
                    "Here are some great places to stay: {content}",
                    "Your lodging options include: {content}",
                    "Perfect stays for your trip: {content}",
                    "Comfortable accommodation choices: {content}",
                    "I recommend these places to stay: {content}",
                    "Excellent lodging options: {content}",
                    "For your comfort, consider: {content}"
                ],
                "contextual_modifiers": {
                    "luxury": "premium luxury accommodations",
                    "budget": "affordable and comfortable options",
                    "family": "family-friendly accommodations",
                    "couples": "romantic getaway options",
                    "business": "business-friendly hotels",
                    "unique": "unique and memorable stays"
                },
                "follow_up_suggestions": [
                    "Would you like me to check availability and pricing?",
                    "I can provide contact details for booking.",
                    "Need information about amenities or location?",
                    "Would you prefer beachfront, city center, or hill country locations?"
                ]
            },
            "food_inquiry": {
                "templates": [
                    "Sri Lankan cuisine is amazing! {content}",
                    "You'll love the local food: {content}",
                    "Here's what you should know about Sri Lankan food: {content}",
                    "Delicious culinary experiences await: {content}",
                    "Sri Lankan gastronomy offers: {content}",
                    "Food lovers will enjoy: {content}",
                    "The local cuisine features: {content}",
                    "Taste the authentic flavors: {content}"
                ],
                "contextual_modifiers": {
                    "spicy": "wonderfully spiced dishes",
                    "mild": "milder flavor options",
                    "vegetarian": "excellent vegetarian cuisine",
                    "seafood": "fresh seafood specialties",
                    "street": "authentic street food experiences",
                    "fine": "upscale dining experiences"
                },
                "follow_up_suggestions": [
                    "Would you like restaurant recommendations in specific areas?",
                    "I can suggest dishes based on your spice tolerance.",
                    "Need information about food allergies or dietary restrictions?",
                    "Would you like to know about cooking classes or food tours?"
                ]
            },
            "culture_inquiry": {
                "templates": [
                    "Sri Lankan culture is rich and diverse: {content}",
                    "Here's what makes Sri Lankan culture special: {content}",
                    "Let me tell you about Sri Lankan traditions: {content}",
                    "The cultural heritage includes: {content}",
                    "Sri Lankan cultural experiences: {content}",
                    "Discover the rich traditions: {content}",
                    "Cultural insights for your visit: {content}",
                    "Immerse yourself in: {content}"
                ],
                "contextual_modifiers": {
                    "religious": "sacred and spiritual traditions",
                    "arts": "traditional arts and crafts",
                    "festivals": "colorful festivals and celebrations",
                    "music": "traditional music and dance",
                    "history": "ancient historical heritage",
                    "modern": "contemporary cultural expressions"
                },
                "follow_up_suggestions": [
                    "Would you like to visit specific cultural sites?",
                    "I can recommend cultural shows or performances.",
                    "Interested in participating in local festivals?",
                    "Would you like information about cultural etiquette?"
                ]
            },
            "weather_inquiry": {
                "templates": [
                    "Here's the weather information: {content}",
                    "Current weather conditions: {content}",
                    "Weather update: {content}",
                    "Climate information: {content}",
                    "Weather forecast: {content}",
                    "Atmospheric conditions: {content}",
                    "Today's weather: {content}",
                    "Weather outlook: {content}"
                ],
                "contextual_modifiers": {
                    "planning": "perfect weather for trip planning",
                    "outdoor": "great conditions for outdoor activities",
                    "beach": "ideal beach weather",
                    "hiking": "perfect hiking conditions",
                    "sightseeing": "excellent weather for sightseeing"
                },
                "follow_up_suggestions": [
                    "Would you like extended weather forecasts?",
                    "I can suggest weather-appropriate activities.",
                    "Need packing recommendations based on weather?",
                    "Would you like information about seasonal variations?"
                ]
            },
            "currency_inquiry": {
                "templates": [
                    "Here's the currency information: {content}",
                    "Currency conversion: {content}",
                    "Exchange rate details: {content}",
                    "Money matters: {content}",
                    "Financial information: {content}",
                    "Currency exchange: {content}",
                    "Current rates: {content}",
                    "Monetary conversion: {content}"
                ],
                "contextual_modifiers": {
                    "travel": "travel-friendly exchange information",
                    "shopping": "shopping budget considerations",
                    "dining": "restaurant pricing guidance",
                    "tourism": "tourist-focused currency tips"
                },
                "follow_up_suggestions": [
                    "Need information about where to exchange money?",
                    "Would you like ATM locations and banking tips?",
                    "I can provide cost estimates for various activities.",
                    "Need advice on payment methods accepted locally?"
                ]
            },
            "translation_request": {
                "templates": [
                    "Here's the translation: {content}",
                    "Translation result: {content}",
                    "In the requested language: {content}",
                    "Language conversion: {content}",
                    "Translated text: {content}",
                    "Linguistic conversion: {content}",
                    "Language assistance: {content}",
                    "Translation service: {content}"
                ],
                "contextual_modifiers": {
                    "basic": "essential phrases for travelers",
                    "polite": "courteous expressions",
                    "emergency": "emergency communication phrases",
                    "shopping": "useful shopping phrases",
                    "dining": "restaurant communication help"
                },
                "follow_up_suggestions": [
                    "Would you like more phrases in this language?",
                    "I can teach you pronunciation tips.",
                    "Need help with cultural context for these phrases?",
                    "Would you like a list of common travel phrases?"
                ]
            },
            "map_request": {
                "templates": [
                    "Here's the location information: {content}",
                    "Map details: {content}",
                    "Location and directions: {content}",
                    "Geographic information: {content}",
                    "Spatial details: {content}",
                    "Navigation assistance: {content}",
                    "Location guidance: {content}",
                    "Directional information: {content}"
                ],
                "contextual_modifiers": {
                    "walking": "pedestrian-friendly routes",
                    "driving": "vehicle navigation details",
                    "public": "public transport directions",
                    "scenic": "scenic route options",
                    "fastest": "quickest route available"
                },
                "follow_up_suggestions": [
                    "Need detailed turn-by-turn directions?",
                    "Would you like alternative route options?",
                    "I can provide nearby landmarks for reference.",
                    "Need information about parking or transportation hubs?"
                ]
            },
            "general_greeting": {
                "templates": [
                    "🙏 Ayubowan! Welcome to your Sri Lanka tourism assistant! {content}",
                    "Hello! I'm here to help you explore Sri Lanka. {content}",
                    "Greetings! Ready to discover the Pearl of the Indian Ocean? {content}",
                    "Welcome! Your Sri Lankan adventure starts here. {content}",
                    "Namaste! Let's explore beautiful Sri Lanka together. {content}",
                    "Hello there! Sri Lanka awaits your discovery. {content}",
                    "Warm greetings! I'm your guide to Sri Lankan wonders. {content}",
                    "Welcome to Sri Lanka's tourism hub! {content}"
                ],
                "content_variations": [
                    "How can I help you today?",
                    "What would you like to know about Sri Lanka?",
                    "Ready to plan your perfect Sri Lankan getaway?",
                    "What Sri Lankan adventure can I help you plan?",
                    "Where would you like to start your journey?",
                    "What brings you to the Pearl of the Indian Ocean?",
                    "How can I make your Sri Lankan experience unforgettable?",
                    "What would you like to explore first?"
                ],
                "follow_up_suggestions": [
                    "I can help with destinations, transportation, accommodation, food, culture, weather, and more!",
                    "Feel free to ask about anything related to Sri Lankan tourism.",
                    "I'm here to make your Sri Lankan journey amazing!"
                ]
            },
            "help_request": {
                "templates": [
                    "I'm here to help! {content}",
                    "Absolutely! I can assist you with: {content}",
                    "I'm happy to help! Here's what I can do: {content}",
                    "Of course! My assistance includes: {content}",
                    "Sure thing! I specialize in: {content}",
                    "I'd be delighted to help! I can provide: {content}",
                    "Perfect! I'm your Sri Lankan tourism expert for: {content}",
                    "Certainly! I offer comprehensive help with: {content}"
                ],
                "content_variations": [
                    "destinations and attractions, transportation options, accommodation recommendations, local cuisine, cultural information, weather updates, currency conversion, language translation, and map services",
                    "planning your perfect Sri Lankan itinerary with personalized recommendations",
                    "everything from must-see attractions to hidden gems throughout Sri Lanka",
                    "comprehensive travel assistance for your Sri Lankan adventure"
                ],
                "follow_up_suggestions": [
                    "What specific aspect of Sri Lankan tourism interests you most?",
                    "Would you like to start with destinations, or do you have other questions?",
                    "I can provide detailed information on any topic you're curious about.",
                    "What would make your Sri Lankan experience most memorable?"
                ]
            },
            "unknown": {
                "templates": [
                    "I'd be happy to help with Sri Lankan tourism! {content}",
                    "Let me assist you with Sri Lankan travel information. {content}",
                    "I'm here for all your Sri Lankan tourism needs. {content}",
                    "I specialize in Sri Lankan travel assistance. {content}",
                    "For Sri Lankan tourism, I'm your go-to resource. {content}",
                    "Let's explore Sri Lanka together! {content}",
                    "Sri Lankan tourism is my specialty. {content}",
                    "I'm your dedicated Sri Lankan travel companion. {content}"
                ],
                "content_variations": [
                    "Could you please be more specific about what you'd like to know?",
                    "What aspect of Sri Lankan tourism interests you?",
                    "I can help with destinations, travel, accommodation, food, culture, and more.",
                    "What would you like to explore about Sri Lanka?",
                    "Please let me know how I can make your Sri Lankan journey amazing.",
                    "What specific information about Sri Lanka would be helpful?",
                    "I'm ready to assist with any Sri Lankan travel questions."
                ],
                "follow_up_suggestions": [
                    "Try asking about popular destinations, transportation, local cuisine, or cultural attractions.",
                    "I can help with practical information like weather, currency, or language assistance.",
                    "Feel free to ask about anything related to visiting Sri Lanka!"
                ]
            }
        }

        # Save the file
        with open("app/ml/models/response_templates.json", "w", encoding="utf-8") as f:
            json.dump(response_templates, f, indent=2, ensure_ascii=False)

        print("✅ Created app/ml/models/response_templates.json")
        return True

    except Exception as e:
        print(f"❌ Failed to create response_templates.json: {e}")
        return False

def update_entity_extractor():
    """Update EntityExtractor to load the new rules"""
    print("🔧 Updating EntityExtractor to use new rules...")

    try:
        from backend.app.nlp.entity_extractor import EntityExtractor

        # Initialize EntityExtractor which will load the new rules
        entity_extractor = EntityExtractor()

        # Test that it can extract entities
        test_text = "I want to visit Kandy and Galle in Sri Lanka"
        entities = entity_extractor.extract_entities(test_text)

        location_count = len(entities.get('locations', []))
        print(f"✅ EntityExtractor loaded and tested ({location_count} locations found in test)")

        return True

    except Exception as e:
        print(f"❌ Failed to update EntityExtractor: {e}")
        return False

def update_response_generator():
    """Update ResponseGenerator to load the new templates"""
    print("🔧 Updating ResponseGenerator to use new templates...")

    try:
        from backend.app.nlp.response_generator import ResponseGenerator

        # Initialize ResponseGenerator which will load the new templates
        response_generator = ResponseGenerator()

        # Test that it can generate responses
        test_response = response_generator.generate_response(
            "general_greeting",
            {},
            [],
            {}
        )

        print(f"✅ ResponseGenerator loaded and tested (response: {len(test_response)} chars)")

        return True

    except Exception as e:
        print(f"❌ Failed to update ResponseGenerator: {e}")
        return False

def test_system_after_file_creation():
    """Test the system after creating missing files"""
    print("🧪 Testing system after creating missing files...")

    try:
        from backend.app.nlp.intent_classifier import IntentClassifier
        from backend.app.nlp.entity_extractor import EntityExtractor
        from backend.app.nlp.response_generator import ResponseGenerator

        # Initialize all components
        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        response_generator = ResponseGenerator()

        # Test with common inputs
        test_cases = [
            "Hello",
            "What places should I visit in Sri Lanka?",
            "How do I get to Kandy?",
            "Where can I stay in Galle?"
        ]

        print(f"\nTesting with {len(test_cases)} common inputs:")

        all_good = True

        for test_input in test_cases:
            try:
                # Test full pipeline
                intent, confidence = intent_classifier.classify_intent(test_input)
                entities = entity_extractor.extract_entities(test_input)
                response = response_generator.generate_response(intent, entities, [], {})

                is_unknown = intent == "unknown"
                is_good_response = len(response) > 20 and "apologize" not in response.lower()

                status = "✅" if not is_unknown and is_good_response else "⚠️"
                print(f"  {status} '{test_input}' -> {intent} ({confidence:.3f})")

                if is_unknown or not is_good_response:
                    all_good = False

            except Exception as e:
                print(f"  ❌ '{test_input}' -> ERROR: {e}")
                all_good = False

        if all_good:
            print(f"\n🎉 ALL TESTS PASSED! Your system is now working properly.")
        else:
            print(f"\n⚠️ Some tests still failing. You may need to:")
            print(f"   1. Retrain the intent classifier")
            print(f"   2. Lower the confidence threshold")
            print(f"   3. Add more training data")

        return all_good

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Create missing model files"""
    print("🔧 CREATING MISSING MODEL FILES")
    print("=" * 50)
    print("This will create the missing Entity Rules and Response Templates files")
    print("=" * 50)

    success_count = 0
    total_tasks = 4

    # Create missing files
    if create_entity_rules_file():
        success_count += 1

    if create_response_templates_file():
        success_count += 1

    # Update components to use new files
    if update_entity_extractor():
        success_count += 1

    if update_response_generator():
        success_count += 1

    print(f"\n📊 SUMMARY:")
    print(f"Successfully completed: {success_count}/{total_tasks} tasks")

    if success_count == total_tasks:
        print(f"✅ ALL FILES CREATED SUCCESSFULLY!")

        # Test the system
        print(f"\n🧪 TESTING SYSTEM...")
        if test_system_after_file_creation():
            print(f"\n🎉 PERFECT! Your chatbot should now work properly.")
            print(f"\n📋 NEXT STEPS:")
            print(f"1. Restart your FastAPI server")
            print(f"2. Test with common user inputs")
            print(f"3. If still getting 'unknown' responses, run:")
            print(f"   python backend/app/tools/immediate_fix.py")
        else:
            print(f"\n⚠️ Files created but system still has issues.")
            print(f"Run the immediate fix script:")
            print(f"python backend/app/tools/immediate_fix.py")

    else:
        print(f"❌ Some tasks failed. Check the errors above.")
        print(f"You may need to manually create the missing files.")

    print(f"\n📁 FILES CREATED:")
    if os.path.exists("app/ml/models/entity_rules.json"):
        print(f"✅ app/ml/models/entity_rules.json")
    else:
        print(f"❌ app/ml/models/entity_rules.json")

    if os.path.exists("app/ml/models/response_templates.json"):
        print(f"✅ app/ml/models/response_templates.json")
    else:
        print(f"❌ app/ml/models/response_templates.json")

if __name__ == "__main__":
    main()