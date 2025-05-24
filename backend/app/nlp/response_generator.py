# backend/app/nlp/response_generator.py
import random
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Advanced response generation system with contextual awareness,
    personalization, and multi-modal content support for Sri Lankan tourism
    """

    def __init__(self):
        # Initialize response templates with sophisticated variations
        self.response_templates = {
            'destination_inquiry': {
                'templates': [
                    "Here are some amazing places to visit in Sri Lanka: {content}",
                    "I'd recommend these fantastic destinations: {content}",
                    "Sri Lanka has wonderful places like: {content}",
                    "You'll love these destinations: {content}",
                    "For sightseeing, I suggest: {content}",
                    "These are must-visit places in Sri Lanka: {content}",
                    "Let me recommend some beautiful locations: {content}",
                    "Perfect destinations for your Sri Lankan adventure: {content}"
                ],
                'contextual_modifiers': {
                    'family': "family-friendly destinations",
                    'adventure': "exciting adventure spots",
                    'culture': "culturally rich locations",
                    'nature': "scenic natural attractions",
                    'history': "historically significant sites",
                    'beach': "beautiful coastal destinations",
                    'mountain': "stunning hill country locations"
                },
                'follow_up_suggestions': [
                    "Would you like specific information about any of these places?",
                    "I can provide more details about transportation or accommodation for these destinations.",
                    "Would you like to know the best time to visit these places?",
                    "I can help you plan an itinerary for these locations."
                ]
            },
            'transportation': {
                'templates': [
                    "For getting around Sri Lanka: {content}",
                    "Here are your transportation options: {content}",
                    "You can travel in Sri Lanka by: {content}",
                    "Transportation methods available: {content}",
                    "To reach your destination: {content}",
                    "Travel options include: {content}",
                    "Getting there is easy with: {content}",
                    "Your journey options: {content}"
                ],
                'contextual_modifiers': {
                    'budget': "affordable transportation options",
                    'comfort': "comfortable travel methods",
                    'fast': "quickest routes available",
                    'scenic': "scenic travel routes",
                    'local': "authentic local transport experiences"
                },
                'follow_up_suggestions': [
                    "Would you like specific schedules or pricing information?",
                    "I can help you book transportation if needed.",
                    "Need directions or route planning assistance?",
                    "Would you prefer door-to-door or public transport options?"
                ]
            },
            'accommodation': {
                'templates': [
                    "For accommodation in Sri Lanka: {content}",
                    "Here are some great places to stay: {content}",
                    "Your lodging options include: {content}",
                    "Perfect stays for your trip: {content}",
                    "Comfortable accommodation choices: {content}",
                    "I recommend these places to stay: {content}",
                    "Excellent lodging options: {content}",
                    "For your comfort, consider: {content}"
                ],
                'contextual_modifiers': {
                    'luxury': "premium luxury accommodations",
                    'budget': "affordable and comfortable options",
                    'family': "family-friendly accommodations",
                    'couples': "romantic getaway options",
                    'business': "business-friendly hotels",
                    'unique': "unique and memorable stays"
                },
                'follow_up_suggestions': [
                    "Would you like me to check availability and pricing?",
                    "I can provide contact details for booking.",
                    "Need information about amenities or location?",
                    "Would you prefer beachfront, city center, or hill country locations?"
                ]
            },
            'food_inquiry': {
                'templates': [
                    "Sri Lankan cuisine is amazing! {content}",
                    "You'll love the local food: {content}",
                    "Here's what you should know about Sri Lankan food: {content}",
                    "Delicious culinary experiences await: {content}",
                    "Sri Lankan gastronomy offers: {content}",
                    "Food lovers will enjoy: {content}",
                    "The local cuisine features: {content}",
                    "Taste the authentic flavors: {content}"
                ],
                'contextual_modifiers': {
                    'spicy': "wonderfully spiced dishes",
                    'mild': "milder flavor options",
                    'vegetarian': "excellent vegetarian cuisine",
                    'seafood': "fresh seafood specialties",
                    'street': "authentic street food experiences",
                    'fine': "upscale dining experiences"
                },
                'follow_up_suggestions': [
                    "Would you like restaurant recommendations in specific areas?",
                    "I can suggest dishes based on your spice tolerance.",
                    "Need information about food allergies or dietary restrictions?",
                    "Would you like to know about cooking classes or food tours?"
                ]
            },
            'culture_inquiry': {
                'templates': [
                    "Sri Lankan culture is rich and diverse: {content}",
                    "Here's what makes Sri Lankan culture special: {content}",
                    "Let me tell you about Sri Lankan traditions: {content}",
                    "The cultural heritage includes: {content}",
                    "Sri Lankan cultural experiences: {content}",
                    "Discover the rich traditions: {content}",
                    "Cultural insights for your visit: {content}",
                    "Immerse yourself in: {content}"
                ],
                'contextual_modifiers': {
                    'religious': "sacred and spiritual traditions",
                    'arts': "traditional arts and crafts",
                    'festivals': "colorful festivals and celebrations",
                    'music': "traditional music and dance",
                    'history': "ancient historical heritage",
                    'modern': "contemporary cultural expressions"
                },
                'follow_up_suggestions': [
                    "Would you like to visit specific cultural sites?",
                    "I can recommend cultural shows or performances.",
                    "Interested in participating in local festivals?",
                    "Would you like information about cultural etiquette?"
                ]
            },
            'weather_inquiry': {
                'templates': [
                    "Here's the weather information: {content}",
                    "Current weather conditions: {content}",
                    "Weather update: {content}",
                    "Climate information: {content}",
                    "Weather forecast: {content}",
                    "Atmospheric conditions: {content}",
                    "Today's weather: {content}",
                    "Weather outlook: {content}"
                ],
                'contextual_modifiers': {
                    'planning': "perfect weather for trip planning",
                    'outdoor': "great conditions for outdoor activities",
                    'beach': "ideal beach weather",
                    'hiking': "perfect hiking conditions",
                    'sightseeing': "excellent weather for sightseeing"
                },
                'follow_up_suggestions': [
                    "Would you like extended weather forecasts?",
                    "I can suggest weather-appropriate activities.",
                    "Need packing recommendations based on weather?",
                    "Would you like information about seasonal variations?"
                ]
            },
            'currency_inquiry': {
                'templates': [
                    "Here's the currency information: {content}",
                    "Currency conversion: {content}",
                    "Exchange rate details: {content}",
                    "Money matters: {content}",
                    "Financial information: {content}",
                    "Currency exchange: {content}",
                    "Current rates: {content}",
                    "Monetary conversion: {content}"
                ],
                'contextual_modifiers': {
                    'travel': "travel-friendly exchange information",
                    'shopping': "shopping budget considerations",
                    'dining': "restaurant pricing guidance",
                    'tourism': "tourist-focused currency tips"
                },
                'follow_up_suggestions': [
                    "Need information about where to exchange money?",
                    "Would you like ATM locations and banking tips?",
                    "I can provide cost estimates for various activities.",
                    "Need advice on payment methods accepted locally?"
                ]
            },
            'translation_request': {
                'templates': [
                    "Here's the translation: {content}",
                    "Translation result: {content}",
                    "In the requested language: {content}",
                    "Language conversion: {content}",
                    "Translated text: {content}",
                    "Linguistic conversion: {content}",
                    "Language assistance: {content}",
                    "Translation service: {content}"
                ],
                'contextual_modifiers': {
                    'basic': "essential phrases for travelers",
                    'polite': "courteous expressions",
                    'emergency': "emergency communication phrases",
                    'shopping': "useful shopping phrases",
                    'dining': "restaurant communication help"
                },
                'follow_up_suggestions': [
                    "Would you like more phrases in this language?",
                    "I can teach you pronunciation tips.",
                    "Need help with cultural context for these phrases?",
                    "Would you like a list of common travel phrases?"
                ]
            },
            'map_request': {
                'templates': [
                    "Here's the location information: {content}",
                    "Map details: {content}",
                    "Location and directions: {content}",
                    "Geographic information: {content}",
                    "Spatial details: {content}",
                    "Navigation assistance: {content}",
                    "Location guidance: {content}",
                    "Directional information: {content}"
                ],
                'contextual_modifiers': {
                    'walking': "pedestrian-friendly routes",
                    'driving': "vehicle navigation details",
                    'public': "public transport directions",
                    'scenic': "scenic route options",
                    'fastest': "quickest route available"
                },
                'follow_up_suggestions': [
                    "Need detailed turn-by-turn directions?",
                    "Would you like alternative route options?",
                    "I can provide nearby landmarks for reference.",
                    "Need information about parking or transportation hubs?"
                ]
            },
            'general_greeting': {
                'templates': [
                    "🙏 Ayubowan! Welcome to your Sri Lanka tourism assistant! {content}",
                    "Hello! I'm here to help you explore Sri Lanka. {content}",
                    "Greetings! Ready to discover the Pearl of the Indian Ocean? {content}",
                    "Welcome! Your Sri Lankan adventure starts here. {content}",
                    "Namaste! Let's explore beautiful Sri Lanka together. {content}",
                    "Hello there! Sri Lanka awaits your discovery. {content}",
                    "Warm greetings! I'm your guide to Sri Lankan wonders. {content}",
                    "Welcome to Sri Lanka's tourism hub! {content}"
                ],
                'content_variations': [
                    "How can I help you today?",
                    "What would you like to know about Sri Lanka?",
                    "Ready to plan your perfect Sri Lankan getaway?",
                    "What Sri Lankan adventure can I help you plan?",
                    "Where would you like to start your journey?",
                    "What brings you to the Pearl of the Indian Ocean?",
                    "How can I make your Sri Lankan experience unforgettable?",
                    "What would you like to explore first?"
                ],
                'follow_up_suggestions': [
                    "I can help with destinations, transportation, accommodation, food, culture, weather, and more!",
                    "Feel free to ask about anything related to Sri Lankan tourism.",
                    "I'm here to make your Sri Lankan journey amazing!"
                ]
            },
            'help_request': {
                'templates': [
                    "I'm here to help! {content}",
                    "Absolutely! I can assist you with: {content}",
                    "I'm happy to help! Here's what I can do: {content}",
                    "Of course! My assistance includes: {content}",
                    "Sure thing! I specialize in: {content}",
                    "I'd be delighted to help! I can provide: {content}",
                    "Perfect! I'm your Sri Lankan tourism expert for: {content}",
                    "Certainly! I offer comprehensive help with: {content}"
                ],
                'content_variations': [
                    "destinations and attractions, transportation options, accommodation recommendations, local cuisine, cultural information, weather updates, currency conversion, language translation, and map services",
                    "planning your perfect Sri Lankan itinerary with personalized recommendations",
                    "everything from must-see attractions to hidden gems throughout Sri Lanka",
                    "comprehensive travel assistance for your Sri Lankan adventure"
                ],
                'follow_up_suggestions': [
                    "What specific aspect of Sri Lankan tourism interests you most?",
                    "Would you like to start with destinations, or do you have other questions?",
                    "I can provide detailed information on any topic you're curious about.",
                    "What would make your Sri Lankan experience most memorable?"
                ]
            },
            'unknown': {
                'templates': [
                    "I'd be happy to help with Sri Lankan tourism! {content}",
                    "Let me assist you with Sri Lankan travel information. {content}",
                    "I'm here for all your Sri Lankan tourism needs. {content}",
                    "I specialize in Sri Lankan travel assistance. {content}",
                    "For Sri Lankan tourism, I'm your go-to resource. {content}",
                    "Let's explore Sri Lanka together! {content}",
                    "Sri Lankan tourism is my specialty. {content}",
                    "I'm your dedicated Sri Lankan travel companion. {content}"
                ],
                'content_variations': [
                    "Could you please be more specific about what you'd like to know?",
                    "What aspect of Sri Lankan tourism interests you?",
                    "I can help with destinations, travel, accommodation, food, culture, and more.",
                    "What would you like to explore about Sri Lanka?",
                    "Please let me know how I can make your Sri Lankan journey amazing.",
                    "What specific information about Sri Lanka would be helpful?",
                    "I'm ready to assist with any Sri Lankan travel questions."
                ],
                'follow_up_suggestions': [
                    "Try asking about popular destinations, transportation, local cuisine, or cultural attractions.",
                    "I can help with practical information like weather, currency, or language assistance.",
                    "Feel free to ask about anything related to visiting Sri Lanka!"
                ]
            }
        }

        # Fallback responses for when other methods fail
        self.fallback_responses = [
            "I understand you're asking about Sri Lanka. Let me search for relevant information for you.",
            "That's an interesting question about Sri Lankan tourism. Let me find the best answer for you.",
            "I'm here to help with Sri Lankan travel information. Let me look that up for you.",
            "Great question about Sri Lanka! Let me provide you with helpful information.",
            "I love helping with Sri Lankan tourism questions. Let me get you the details you need.",
            "Sri Lanka has so much to offer! Let me find specific information for your question.",
            "Your interest in Sri Lanka is wonderful! Let me provide you with accurate information."
        ]

        # Contextual modifiers for different scenarios
        self.contextual_modifiers = {
            'time_of_day': {
                'morning': ['Good morning!', 'Starting your day with great questions!', 'Morning greetings!'],
                'afternoon': ['Good afternoon!', 'Hope your day is going well!', 'Afternoon greetings!'],
                'evening': ['Good evening!', 'Perfect timing for travel planning!', 'Evening greetings!'],
                'night': ['Good evening!', 'Planning your next adventure?', 'Evening greetings!']
            },
            'weather_context': {
                'sunny': ['Perfect weather for sightseeing!', 'Great day for outdoor activities!'],
                'rainy': ['Good weather for indoor cultural experiences!', 'Perfect for visiting museums and temples!'],
                'cloudy': ['Comfortable weather for exploring!', 'Nice day for walking tours!']
            },
            'season_context': {
                'peak_season': ['Great timing for your visit!', 'Perfect season for Sri Lankan tourism!'],
                'off_season': ['Fewer crowds and special experiences await!', 'Authentic Sri Lankan experiences during quieter times!'],
                'shoulder_season': ['Excellent balance of weather and crowd levels!']
            }
        }

        # Personality traits for response variation
        self.personality_traits = {
            'enthusiastic': ['Amazing!', 'Fantastic!', 'Wonderful!', 'Excellent!', 'Perfect!'],
            'helpful': ['I\'m happy to help!', 'Let me assist you!', 'I\'d be delighted to help!'],
            'knowledgeable': ['Based on my knowledge...', 'From my experience...', 'As a tourism expert...'],
            'friendly': ['Hope this helps!', 'Feel free to ask more!', 'I\'m here for you!'],
            'local_expert': ['As someone familiar with Sri Lanka...', 'From a local perspective...', 'Sri Lanka insider tip:']
        }

        # Load custom templates if available
        self._load_custom_templates()

        logger.info("ResponseGenerator initialized with comprehensive templates")

    def _load_custom_templates(self):
        """Load custom response templates from file if available"""
        try:
            templates_path = Path("app/ml/models/response_templates.json")
            if templates_path.exists():
                with open(templates_path, 'r', encoding='utf-8') as f:
                    custom_templates = json.load(f)

                # Merge custom templates with default ones
                for intent, templates in custom_templates.items():
                    if intent in self.response_templates:
                        if 'templates' in templates:
                            self.response_templates[intent]['templates'].extend(templates['templates'])
                        if 'contextual_modifiers' in templates:
                            self.response_templates[intent]['contextual_modifiers'].update(templates['contextual_modifiers'])

                logger.info("Custom response templates loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load custom response templates: {e}")

    def generate_response(self, intent: str, entities: Dict[str, Any],
                          knowledge_results: List[Dict] = None,
                          tool_outputs: Dict[str, Any] = None,
                          context: Dict[str, Any] = None) -> str:
        try:
            if context is None:
                context = {}

            # Special handling for greetings and help requests
            if intent in ['general_greeting', 'help_request']:
                return self._generate_special_response(intent, context)

            # 1. First try to use knowledge results if available and relevant
            if knowledge_results and knowledge_results[0]['relevance_score'] > 0.5:
                best_knowledge = knowledge_results[0]
                response = best_knowledge['answer']

                # Skip template formatting if knowledge answer is comprehensive
                if len(best_knowledge['answer']) > 50:  # If answer is substantial
                    return self._enhance_knowledge_response(response, intent, entities)

            # 2. If no good knowledge result, use templates with any available knowledge
            content_parts = []
            if knowledge_results:
                knowledge_content = self._process_knowledge_results(knowledge_results, intent)
                if knowledge_content:
                    content_parts.extend(knowledge_content)

            if tool_outputs:
                tool_content = self._process_tool_outputs(tool_outputs, intent, entities)
                if tool_content:
                    content_parts.extend(tool_content)

            if content_parts:
                content = self._combine_content_parts(content_parts)
                response = self._format_response_with_template(intent, content, context, entities)
            else:
                # Only use fallback if absolutely no content available
                response = self._generate_fallback_response(intent, entities, context)

            return self._enhance_response_with_context(response, entities, context)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return random.choice(self.fallback_responses)


    def _generate_special_response(self, intent: str, context: Dict[str, Any]) -> str:
        """Generate special responses for greetings and help requests"""
        if intent not in self.response_templates:
            return random.choice(self.fallback_responses)

        template_data = self.response_templates[intent]

        # Select template
        template = random.choice(template_data['templates'])

        # Select content variation
        if 'content_variations' in template_data:
            content = random.choice(template_data['content_variations'])
        else:
            content = ""

        # Format response
        response = template.format(content=content)

        # Add follow-up suggestion
        if 'follow_up_suggestions' in template_data and random.random() < 0.7:
            follow_up = random.choice(template_data['follow_up_suggestions'])
            response += f"\n\n{follow_up}"

        return response

    def _process_knowledge_results(self, knowledge_results: List[Dict], intent: str) -> List[str]:
        """Process knowledge base results into content parts"""
        content_parts = []

        try:
            # Sort by relevance score
            sorted_results = sorted(knowledge_results, key=lambda x: x.get('relevance_score', 0), reverse=True)

            # Use top results with good relevance scores
            for result in sorted_results[:3]:  # Top 3 results
                if result.get('relevance_score', 0) > 0.3:
                    # Format the knowledge answer
                    answer = result.get('answer', '')

                    # Add context-specific formatting
                    if intent == 'destination_inquiry':
                        # Format destination information nicely
                        content_parts.append(f"🏛️ {answer}")
                    elif intent == 'food_inquiry':
                        # Add food emoji for cuisine
                        content_parts.append(f"🍽️ {answer}")
                    elif intent == 'culture_inquiry':
                        # Add cultural context
                        content_parts.append(f"🎭 {answer}")
                    elif intent == 'transportation':
                        # Add transport context
                        content_parts.append(f"🚌 {answer}")
                    else:
                        content_parts.append(answer)

        except Exception as e:
            logger.error(f"Error processing knowledge results: {e}")

        return content_parts

    def _process_tool_outputs(self, tool_outputs: Dict[str, Any], intent: str, entities: Dict[str, Any]) -> List[str]:
        """Process tool outputs into formatted content parts"""
        content_parts = []

        try:
            for tool_name, output in tool_outputs.items():
                if output and isinstance(output, dict) and output.get('success'):
                    formatted_output = self._format_tool_output(tool_name, output, intent)
                    if formatted_output:
                        content_parts.append(formatted_output)

        except Exception as e:
            logger.error(f"Error processing tool outputs: {e}")

        return content_parts

    def _format_tool_output(self, tool_name: str, output: Dict[str, Any], intent: str) -> Optional[str]:
        """Format individual tool output for display"""
        try:
            if tool_name == 'currency_converter':
                if 'converted_amount' in output:
                    from_curr = output.get('from_currency', '').upper()
                    to_curr = output.get('to_currency', '').upper()
                    amount = output.get('amount', 1)
                    converted = output.get('converted_amount', 0)

                    # Format with currency symbols if available
                    if from_curr == 'USD':
                        from_symbol = '$'
                    elif from_curr == 'EUR':
                        from_symbol = '€'
                    elif from_curr == 'GBP':
                        from_symbol = '£'
                    elif from_curr == 'LKR':
                        from_symbol = 'Rs. '
                    else:
                        from_symbol = ''

                    if to_curr == 'LKR':
                        to_symbol = 'Rs. '
                    elif to_curr == 'USD':
                        to_symbol = '$'
                    elif to_curr == 'EUR':
                        to_symbol = '€'
                    elif to_curr == 'GBP':
                        to_symbol = '£'
                    else:
                        to_symbol = ''

                    formatted = f"💱 **Currency Conversion**: {from_symbol}{amount:,.2f} {from_curr} = {to_symbol}{converted:,.2f} {to_curr}"

                    if 'exchange_rate' in output:
                        rate = output['exchange_rate']
                        formatted += f"\n📊 Exchange Rate: 1 {from_curr} = {rate:.4f} {to_curr}"

                    if output.get('note'):
                        formatted += f"\n💡 *{output['note']}*"

                    return formatted

            elif tool_name == 'weather_checker':
                if 'temperature' in output:
                    city = output.get('city', 'Location')
                    temp = output.get('temperature', 0)
                    description = output.get('description', '').title()

                    # Weather emoji mapping
                    weather_emojis = {
                        'sunny': '☀️', 'clear': '☀️', 'hot': '🌡️',
                        'cloudy': '☁️', 'overcast': '☁️', 'partly cloudy': '⛅',
                        'rainy': '🌧️', 'rain': '🌧️', 'shower': '🌦️',
                        'stormy': '⛈️', 'thunderstorm': '⛈️',
                        'windy': '💨', 'breezy': '💨'
                    }

                    emoji = '🌤️'  # default
                    for weather_type, weather_emoji in weather_emojis.items():
                        if weather_type in description.lower():
                            emoji = weather_emoji
                            break

                    formatted = f"{emoji} **Weather in {city}**: {temp}°C, {description}"

                    if 'humidity' in output:
                        formatted += f"\n💧 Humidity: {output['humidity']}%"

                    if 'wind_speed' in output:
                        formatted += f"\n🌬️ Wind: {output['wind_speed']} km/h"

                    if output.get('note'):
                        formatted += f"\n💡 *{output['note']}*"

                    return formatted

            elif tool_name == 'translator':
                if 'translated_text' in output:
                    original = output.get('original_text', '')
                    translated = output.get('translated_text', '')
                    from_lang = output.get('from_language', '').title()
                    to_lang = output.get('to_language', '').title()

                    # Language flag emojis
                    flag_emojis = {
                        'English': '🇺🇸', 'Sinhala': '🇱🇰', 'Tamil': '🇱🇰'
                    }

                    from_flag = flag_emojis.get(from_lang, '🗣️')
                    to_flag = flag_emojis.get(to_lang, '🗣️')

                    formatted = f"🔤 **Translation**:\n"
                    formatted += f"{from_flag} **{from_lang}**: {original}\n"
                    formatted += f"{to_flag} **{to_lang}**: {translated}"

                    if output.get('note'):
                        formatted += f"\n💡 *{output['note']}*"

                    return formatted

            elif tool_name == 'maps_integration':
                if 'location' in output:
                    location = output.get('location', '')
                    formatted = f"📍 **Location**: {location}"

                    if 'coordinates' in output:
                        coords = output['coordinates']
                        formatted += f"\n🗺️ Coordinates: {coords.get('lat', 0):.4f}, {coords.get('lng', 0):.4f}"

                    if 'distance_km' in output:
                        formatted += f"\n📏 Distance: {output['distance_km']} km"

                    if output.get('map_url'):
                        formatted += f"\n🔗 [View on Map]({output['map_url']})"

                    return formatted

            # Generic formatting for other tools
            return f"🔧 **{tool_name.replace('_', ' ').title()}**: {str(output)}"

        except Exception as e:
            logger.error(f"Error formatting tool output for {tool_name}: {e}")
            return None

    def _combine_content_parts(self, content_parts: List[str]) -> str:
        """Combine multiple content parts into coherent content"""
        if not content_parts:
            return ""

        if len(content_parts) == 1:
            return content_parts[0]

        # Combine with appropriate separators
        combined = content_parts[0]

        for i, part in enumerate(content_parts[1:], 1):
            if i == len(content_parts) - 1:
                # Last part - use "and" or "finally"
                if part.startswith(('🔧', '💱', '🌤️', '🔤', '📍')):
                    combined += f"\n\n**Additionally**: {part}"
                else:
                    combined += f"\n\n{part}"
            else:
                # Middle parts
                combined += f"\n\n{part}"

        return combined

    def _format_response_with_template(self, intent: str, content: str,
                                       context: Dict[str, Any], entities: Dict[str, Any]) -> str:
        """Format response using appropriate template"""
        if intent not in self.response_templates:
            return content

        template_data = self.response_templates[intent]

        # Select appropriate template based on context
        template = self._select_contextual_template(template_data, context, entities)

        # Format with content
        response = template.format(content=content)

        # Add follow-up suggestions based on probability
        if 'follow_up_suggestions' in template_data and random.random() < 0.4:
            follow_up = random.choice(template_data['follow_up_suggestions'])
            response += f"\n\n{follow_up}"

        return response

    def _select_contextual_template(self, template_data: Dict, context: Dict[str, Any],
                                    entities: Dict[str, Any]) -> str:
        """Select the most appropriate template based on context"""
        templates = template_data.get('templates', [])

        if not templates:
            return "{content}"

        # For now, select randomly - could be enhanced with ML
        # TODO: Implement contextual template selection based on:
        # - User preferences
        # - Time of day
        # - Previous conversations
        # - Entity types present

        return random.choice(templates)

    def _generate_fallback_response(self, intent: str, entities: Dict[str, Any],
                                    context: Dict[str, Any]) -> str:
        """Generate fallback response when no content is available"""
        # Try to use intent-specific unknown responses
        if intent in self.response_templates:
            template_data = self.response_templates[intent]
            if 'templates' in template_data:
                template = random.choice(template_data['templates'])
                content = "I'm looking for the most relevant information for you."
                return template.format(content=content)

        # Use generic fallback
        base_response = random.choice(self.fallback_responses)

        # Add intent-specific context if available
        if intent != 'unknown':
            intent_context = {
                'destination_inquiry': "about Sri Lankan destinations and attractions",
                'transportation': "about transportation and travel options in Sri Lanka",
                'accommodation': "about hotels and places to stay",
                'food_inquiry': "about Sri Lankan cuisine and restaurants",
                'culture_inquiry': "about Sri Lankan culture and traditions",
                'weather_inquiry': "about weather conditions",
                'currency_inquiry': "about currency exchange and costs",
                'translation_request': "with language translation",
                'map_request': "with location and navigation information"
            }

            if intent in intent_context:
                base_response += f" I specialize in helping {intent_context[intent]}."

        return base_response

    def _enhance_response_with_context(self, response: str, entities: Dict[str, Any],
                                       context: Dict[str, Any]) -> str:
        """Enhance response with contextual information"""
        enhanced_response = response

        try:
            # Add location-specific tips
            primary_location = self._get_primary_location(entities)
            if primary_location:
                location_tip = self._get_location_tip(primary_location)
                if location_tip:
                    enhanced_response += f"\n\n **Insider Tip**: {location_tip}"

            # Add seasonal information
            seasonal_info = self._get_seasonal_info()
            if seasonal_info and random.random() < 0.3:  # 30% chance
                enhanced_response += f"\n\n **Seasonal Note**: {seasonal_info}"

            # Add cultural etiquette tips
            if self._has_cultural_context(entities) and random.random() < 0.2:  # 20% chance
                etiquette_tip = self._get_cultural_etiquette_tip()
                if etiquette_tip:
                    enhanced_response += f"\n\n **Cultural Tip**: {etiquette_tip}"

            # Add practical information
            practical_info = self._get_practical_info(entities, context)
            if practical_info:
                enhanced_response += f"\n\n **Practical Info**: {practical_info}"

        except Exception as e:
            logger.error(f"Error enhancing response with context: {e}")

        return enhanced_response

    def _get_primary_location(self, entities: Dict[str, Any]) -> Optional[str]:
        """Extract primary location from entities"""
        locations = entities.get('locations', [])
        for location in locations:
            if location.get('type') == 'sri_lankan_location':
                return location['text']
        return locations[0]['text'] if locations else None

    def _get_location_tip(self, location: str) -> Optional[str]:
        """Get location-specific insider tips"""
        location_tips = {
            'colombo': "The city comes alive in the evening! Visit Galle Face Green for sunset and street food.",
            'kandy': "Don't miss the daily ceremony at the Temple of the Tooth at 6:30 PM.",
            'galle': "Walk the fort walls during golden hour for the most spectacular photos.",
            'ella': "Take the early morning train from Kandy for the most breathtaking mountain views.",
            'sigiriya': "Start your climb by 6 AM to avoid crowds and heat. Bring plenty of water!",
            'negombo': "The fish market is most active early morning - perfect for authentic local experiences.",
            'mirissa': "Whale watching season is November to April - book tours in advance!",
            'nuwara eliya': "Pack warm clothes even in summer - it gets surprisingly cool in the evenings.",
            'anuradhapura': "Rent a bicycle to explore the ancient ruins efficiently and enjoyably.",
            'polonnaruwa': "The archaeological museum provides great context before exploring the ruins.",
            'trincomalee': "Pigeon Island is perfect for snorkeling - visit during calm weather.",
            'yala': "Early morning (6 AM) safari drives offer the best wildlife viewing opportunities."
        }

        return location_tips.get(location.lower())

    def _get_seasonal_info(self) -> Optional[str]:
        """Get current seasonal information"""
        month = datetime.now().month

        seasonal_tips = {
            1: "January is peak season with perfect weather on the west and south coasts!",
            2: "February offers excellent beach weather and is ideal for whale watching in Mirissa.",
            3: "March is still dry season - great time for hill country visits before the heat.",
            4: "April marks the Sinhala/Tamil New Year - experience vibrant cultural celebrations!",
            5: "May begins the southwest monsoon - perfect time to explore the east coast instead.",
            6: "June to August is monsoon season on the west coast but ideal for Trincomalee and Arugam Bay.",
            7: "July brings the Esala Perahera in Kandy - book accommodation early for this spectacular festival!",
            8: "August is great for east coast beaches and cultural triangle sites.",
            9: "September sees the end of southwest monsoon - transitional weather across the island.",
            10: "October can be unpredictable - check weather forecasts and have flexible plans.",
            11: "November starts the northeast monsoon but whale watching season begins!",
            12: "December kicks off peak tourist season with perfect weather returning to most areas."
        }

        return seasonal_tips.get(month)

    def _has_cultural_context(self, entities: Dict[str, Any]) -> bool:
        """Check if response has cultural context that warrants etiquette tips"""
        # Check for cultural entities or temple-related content
        cultural_indicators = ['temple', 'buddhist', 'cultural', 'festival', 'tradition', 'religious']

        for entity_list in entities.values():
            for entity in entity_list:
                entity_text = entity.get('text', '').lower()
                if any(indicator in entity_text for indicator in cultural_indicators):
                    return True

        return False

    def _get_cultural_etiquette_tip(self) -> str:
        """Get random cultural etiquette tip"""
        etiquette_tips = [
            "Remove shoes before entering temples and homes - it's a sign of respect.",
            "Dress modestly when visiting religious sites - cover shoulders and knees.",
            "Use your right hand for eating, greeting, and giving/receiving items.",
            "Don't point your feet toward people or Buddha statues - it's considered disrespectful.",
            "Ask permission before photographing people, especially monks or in religious settings.",
            "Learn basic greetings: 'Ayubowan' (Sinhala) or 'Vanakkam' (Tamil) - locals appreciate the effort!",
            "When visiting temples, maintain a quiet, respectful demeanor and follow local customs.",
            "Tipping is not mandatory but appreciated - 10% at restaurants is generous."
        ]

        return random.choice(etiquette_tips)

    def _get_practical_info(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Get practical travel information based on context"""
        practical_tips = []

        # Check for transportation entities
        if any(entity.get('type', '').startswith('transport_') for entity_list in entities.values() for entity in entity_list):
            practical_tips.append("Always negotiate tuk-tuk fares beforehand or use the meter.")

        # Check for accommodation entities
        if any(entity.get('type', '').startswith('accommodation_') for entity_list in entities.values() for entity in entity_list):
            practical_tips.append("Book accommodation in advance during peak season (December-March).")

        # Check for currency mentions
        if entities.get('currencies') or entities.get('amounts'):
            practical_tips.append("ATMs are widely available, and most hotels accept major credit cards.")

        # Check for food-related queries
        if any(entity.get('type', '').startswith('food_') for entity_list in entities.values() for entity in entity_list):
            practical_tips.append("Try local restaurants for authentic flavors - they're often the best value too!")

        return random.choice(practical_tips) if practical_tips else None

    def _add_personality_and_local_flavor(self, response: str, intent: str, context: Dict[str, Any]) -> str:
        """Add personality and local Sri Lankan flavor to responses"""
        enhanced_response = response

        try:
            # Add enthusiasm for certain intents
            if intent in ['destination_inquiry', 'culture_inquiry'] and random.random() < 0.3:
                enthusiasm = random.choice(self.personality_traits['enthusiastic'])
                enhanced_response = f"{enthusiasm} {enhanced_response}"

            # Add local expressions occasionally
            if random.random() < 0.15:  # 15% chance
                local_expressions = [
                    "🌺 Sri Lanka truly is the Pearl of the Indian Ocean!",
                    "🙏 May your journey be filled with the warmth of Sri Lankan hospitality!",
                    "🏝️ There's so much beauty waiting to be discovered here!",
                    "☀️ Sri Lanka's magic lies in its diversity - from beaches to mountains!",
                    "🐘 Every corner of our island has a unique story to tell!"
                ]

                if not enhanced_response.endswith('.'):
                    enhanced_response += '.'
                enhanced_response += f"\n\n{random.choice(local_expressions)}"

            # Add helpful closing for certain intents
            if intent in ['help_request', 'unknown'] and random.random() < 0.4:
                helpful_closings = [
                    "Feel free to ask me anything else about Sri Lanka!",
                    "I'm here to make your Sri Lankan adventure unforgettable!",
                    "Don't hesitate to ask for more details about anything that interests you!",
                    "I love sharing the wonders of Sri Lanka - ask me anything!"
                ]
                enhanced_response += f"\n\n{random.choice(helpful_closings)}"

        except Exception as e:
            logger.error(f"Error adding personality and local flavor: {e}")

        return enhanced_response

    def enhance_response_with_context(self, response: str, entities: Dict[str, Any],
                                      conversation_history: List[Dict] = None) -> str:
        """Enhanced version with conversation history"""
        enhanced_response = response

        try:
            # Apply base context enhancement
            enhanced_response = self._enhance_response_with_context(enhanced_response, entities, {})

            # Add conversation history context
            if conversation_history:
                history_context = self._analyze_conversation_history(conversation_history)
                if history_context:
                    enhanced_response = self._apply_history_context(enhanced_response, history_context)

        except Exception as e:
            logger.error(f"Error in enhanced context processing: {e}")

        return enhanced_response

    def _analyze_conversation_history(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze conversation history for context"""
        context = {
            'previous_intents': [],
            'mentioned_locations': set(),
            'user_interests': set(),
            'conversation_length': len(history)
        }

        try:
            # Analyze last 5 exchanges
            for exchange in history[-5:]:
                if 'intent' in exchange:
                    context['previous_intents'].append(exchange['intent'])

                if 'entities' in exchange:
                    # Extract locations mentioned
                    locations = exchange['entities'].get('locations', [])
                    for loc in locations:
                        context['mentioned_locations'].add(loc.get('text', '').lower())

                # Infer interests from intents
                intent = exchange.get('intent', '')
                if intent in ['destination_inquiry', 'culture_inquiry']:
                    context['user_interests'].add('culture_history')
                elif intent in ['food_inquiry']:
                    context['user_interests'].add('culinary')
                elif intent in ['transportation', 'map_request']:
                    context['user_interests'].add('practical_travel')

        except Exception as e:
            logger.error(f"Error analyzing conversation history: {e}")

        return context

    def _apply_history_context(self, response: str, history_context: Dict[str, Any]) -> str:
        """Apply conversation history context to response"""
        enhanced_response = response

        try:
            # Add continuity for returning topics
            previous_intents = history_context.get('previous_intents', [])
            if len(previous_intents) > 1:
                if previous_intents[-1] == previous_intents[-2]:
                    # User asking similar questions
                    enhanced_response = f"Building on what we discussed, {enhanced_response.lower()}"

            # Reference previously mentioned locations
            mentioned_locations = history_context.get('mentioned_locations', set())
            if mentioned_locations and random.random() < 0.2:
                locations_list = list(mentioned_locations)
                if len(locations_list) == 1:
                    enhanced_response += f"\n\nSince you're interested in {locations_list[0].title()}, you might also enjoy exploring nearby attractions!"
                elif len(locations_list) > 1:
                    enhanced_response += f"\n\nWith your interest in {', '.join([loc.title() for loc in locations_list])}, you're planning quite an adventure!"

            # Acknowledge user's interests
            interests = history_context.get('user_interests', set())
            if 'culinary' in interests and 'culture_history' in interests and random.random() < 0.3:
                enhanced_response += "\n\n🍛 Given your interest in both food and culture, don't miss trying traditional meals at local temples during meal times!"

        except Exception as e:
            logger.error(f"Error applying history context: {e}")

        return enhanced_response

    def generate_personalized_response(self, intent: str, entities: Dict[str, Any],
                                       knowledge_results: List[Dict] = None,
                                       tool_outputs: Dict[str, Any] = None,
                                       user_profile: Dict[str, Any] = None,
                                       context: Dict[str, Any] = None) -> str:
        """Generate personalized response based on user profile"""

        # Initialize user profile context
        if user_profile:
            context = context or {}
            context.update({
                'user_preferences': user_profile.get('preferences', {}),
                'travel_style': user_profile.get('travel_style', 'general'),
                'budget_range': user_profile.get('budget_range', 'medium'),
                'interests': user_profile.get('interests', [])
            })

        # Generate base response
        response = self.generate_response(intent, entities, knowledge_results, tool_outputs, context)

        # Apply personalization
        if user_profile:
            response = self._apply_personalization(response, user_profile, intent)

        return response

    def _apply_personalization(self, response: str, user_profile: Dict[str, Any], intent: str) -> str:
        """Apply personalization based on user profile"""
        personalized_response = response

        try:
            travel_style = user_profile.get('travel_style', 'general')
            budget_range = user_profile.get('budget_range', 'medium')
            interests = user_profile.get('interests', [])

            # Add travel style specific suggestions
            if intent == 'destination_inquiry':
                if travel_style == 'adventure':
                    personalized_response += "\n\n🏔️ **For Adventure Seekers**: Consider hiking Ella Rock, white water rafting in Kitulgala, or diving in Trincomalee!"
                elif travel_style == 'luxury':
                    personalized_response += "\n\n✨ **Luxury Experience**: Look into boutique hotels in Galle Fort or luxury resorts in Bentota for premium comfort."
                elif travel_style == 'budget':
                    personalized_response += "\n\n💰 **Budget-Friendly**: Guesthouses and homestays offer authentic experiences at great prices!"
                elif travel_style == 'family':
                    personalized_response += "\n\n👨‍👩‍👧‍👦 **Family-Friendly**: Beaches in Negombo and cultural sites in Kandy are perfect for families with children."

            # Add budget-conscious suggestions
            if budget_range == 'budget' and intent in ['accommodation', 'food_inquiry', 'transportation']:
                budget_tips = {
                    'accommodation': "Look for homestays and guesthouses - they're affordable and offer authentic local experiences!",
                    'food_inquiry': "Local 'rice and curry' restaurants offer delicious meals at very reasonable prices.",
                    'transportation': "Public buses are the most economical way to travel, and trains offer scenic routes at low cost."
                }
                if intent in budget_tips:
                    personalized_response += f"\n\n💡 **Budget Tip**: {budget_tips[intent]}"

            # Add interest-based recommendations
            if 'photography' in interests:
                personalized_response += "\n\n📸 **Photography Tip**: Golden hour at your chosen locations will give you the most stunning shots!"

            if 'wildlife' in interests and intent == 'destination_inquiry':
                personalized_response += "\n\n🐘 **Wildlife Enthusiast**: Don't miss Yala National Park for leopards and Udawalawe for elephants!"

            if 'spiritual' in interests:
                personalized_response += "\n\n🙏 **Spiritual Journey**: Consider visiting during Poya (full moon) days when temples have special significance."

        except Exception as e:
            logger.error(f"Error applying personalization: {e}")

        return personalized_response

    def generate_follow_up_questions(self, intent: str, entities: Dict[str, Any],
                                     response: str) -> List[str]:
        """Generate relevant follow-up questions to continue the conversation"""
        follow_ups = []

        try:
            # Intent-specific follow-ups
            if intent == 'destination_inquiry':
                follow_ups = [
                    "Would you like detailed information about any specific destination?",
                    "Are you interested in outdoor activities or cultural experiences?",
                    "What's your preferred travel style - adventure, relaxation, or cultural exploration?",
                    "Would you like recommendations for a specific region of Sri Lanka?"
                ]

            elif intent == 'transportation':
                follow_ups = [
                    "Would you like specific schedules and booking information?",
                    "Are you looking for the most economical or most comfortable option?",
                    "Do you need help with route planning between multiple destinations?",
                    "Would you prefer public transport or private transportation?"
                ]

            elif intent == 'accommodation':
                follow_ups = [
                    "What's your preferred budget range for accommodation?",
                    "Are you looking for beachfront, city center, or hill country locations?",
                    "Would you like luxury resorts, boutique hotels, or local guesthouses?",
                    "Do you need family-friendly facilities or romantic getaway options?"
                ]

            elif intent == 'food_inquiry':
                follow_ups = [
                    "Are you interested in fine dining or authentic local experiences?",
                    "Do you have any dietary restrictions I should know about?",
                    "Would you like to try cooking classes or food tours?",
                    "How adventurous are you with spicy food?"
                ]

            elif intent == 'culture_inquiry':
                follow_ups = [
                    "Are you interested in ancient history or contemporary culture?",
                    "Would you like to participate in any cultural activities or festivals?",
                    "Are you planning to visit during any specific cultural events?",
                    "Would you like information about cultural etiquette and customs?"
                ]

            # Entity-based follow-ups
            if entities.get('locations'):
                location_name = entities['locations'][0].get('text', '')
                follow_ups.append(f"Would you like more specific information about {location_name}?")
                follow_ups.append(f"Are you planning to stay overnight in {location_name}?")

            # General follow-ups
            follow_ups.extend([
                "Is there anything specific you'd like to know more about?",
                "Would you like me to help you create an itinerary?",
                "Do you have any other questions about Sri Lankan tourism?"
            ])

            # Randomize and limit to 3-4 follow-ups
            random.shuffle(follow_ups)
            return follow_ups[:4]

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return ["What else would you like to know about Sri Lanka?"]

    def get_response_variations(self, intent: str, entities: Dict[str, Any],
                                knowledge_results: List[Dict] = None,
                                tool_outputs: Dict[str, Any] = None,
                                num_variations: int = 3) -> List[str]:
        """Generate multiple variations of the same response"""
        variations = []

        try:
            for _ in range(num_variations):
                variation = self.generate_response(intent, entities, knowledge_results, tool_outputs)
                if variation not in variations:
                    variations.append(variation)

            # Ensure we have enough variations
            while len(variations) < num_variations:
                fallback = self._generate_fallback_response(intent, entities, {})
                if fallback not in variations:
                    variations.append(fallback)

        except Exception as e:
            logger.error(f"Error generating response variations: {e}")
            variations = [self.generate_response(intent, entities, knowledge_results, tool_outputs)]

        return variations

    def evaluate_response_quality(self, response: str, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of generated response"""
        evaluation = {
            'length_score': 0.0,
            'relevance_score': 0.0,
            'helpfulness_score': 0.0,
            'engagement_score': 0.0,
            'overall_score': 0.0,
            'suggestions': []
        }

        try:
            # Length evaluation (optimal range: 100-500 characters)
            length = len(response)
            if 100 <= length <= 500:
                evaluation['length_score'] = 1.0
            elif length < 100:
                evaluation['length_score'] = length / 100.0
                evaluation['suggestions'].append("Response could be more detailed")
            else:
                evaluation['length_score'] = max(0.5, 1.0 - (length - 500) / 1000.0)
                evaluation['suggestions'].append("Response might be too lengthy")

            # Relevance evaluation (check for intent-related keywords)
            intent_keywords = {
                'destination_inquiry': ['destination', 'place', 'visit', 'attraction', 'location'],
                'transportation': ['transport', 'travel', 'bus', 'train', 'taxi', 'journey'],
                'accommodation': ['hotel', 'stay', 'accommodation', 'lodge', 'resort'],
                'food_inquiry': ['food', 'restaurant', 'cuisine', 'dish', 'meal'],
                'culture_inquiry': ['culture', 'tradition', 'festival', 'temple', 'heritage'],
                'weather_inquiry': ['weather', 'climate', 'temperature', 'forecast'],
                'currency_inquiry': ['currency', 'exchange', 'money', 'rate', 'cost'],
                'translation_request': ['translation', 'language', 'translate'],
                'map_request': ['map', 'location', 'direction', 'route']
            }

            relevant_keywords = intent_keywords.get(intent, [])
            response_lower = response.lower()
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword in response_lower)
            evaluation['relevance_score'] = min(1.0, keyword_matches / max(len(relevant_keywords), 1))

            # Helpfulness evaluation (check for actionable information)
            helpful_indicators = ['can', 'will', 'recommend', 'suggest', 'try', 'visit', 'contact', 'book']
            helpful_count = sum(1 for indicator in helpful_indicators if indicator in response_lower)
            evaluation['helpfulness_score'] = min(1.0, helpful_count / 3.0)

            # Engagement evaluation (check for engaging elements)
            engaging_elements = ['!', '?', '💡', '🌟', '✨', 'amazing', 'wonderful', 'perfect', 'great']
            engagement_count = sum(1 for element in engaging_elements if element in response)
            evaluation['engagement_score'] = min(1.0, engagement_count / 2.0)

            # Calculate overall score
            evaluation['overall_score'] = (
                    evaluation['length_score'] * 0.2 +
                    evaluation['relevance_score'] * 0.4 +
                    evaluation['helpfulness_score'] * 0.3 +
                    evaluation['engagement_score'] * 0.1
            )

            # Generate improvement suggestions
            if evaluation['overall_score'] < 0.7:
                evaluation['suggestions'].append("Consider adding more specific, actionable information")
            if evaluation['engagement_score'] < 0.5:
                evaluation['suggestions'].append("Add more engaging language and enthusiasm")
            if evaluation['relevance_score'] < 0.6:
                evaluation['suggestions'].append("Include more intent-specific keywords and information")

        except Exception as e:
            logger.error(f"Error evaluating response quality: {e}")

        return evaluation

    def save_response_templates(self, filepath: str = None):
        """Save current response templates to file"""
        try:
            if not filepath:
                filepath = Path("app/ml/models/response_templates.json")

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.response_templates, f, indent=2, ensure_ascii=False)

            logger.info(f"Response templates saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving response templates: {e}")

    def get_response_statistics(self) -> Dict[str, Any]:
        """Get statistics about response templates and generation"""
        stats = {
            'total_intents': len(self.response_templates),
            'total_templates': sum(len(intent_data.get('templates', [])) for intent_data in self.response_templates.values()),
            'intents_with_contextual_modifiers': sum(1 for intent_data in self.response_templates.values() if 'contextual_modifiers' in intent_data),
            'intents_with_follow_ups': sum(1 for intent_data in self.response_templates.values() if 'follow_up_suggestions' in intent_data),
            'average_templates_per_intent': 0,
            'template_distribution': {}
        }

        # Calculate template distribution
        for intent, intent_data in self.response_templates.items():
            template_count = len(intent_data.get('templates', []))
            stats['template_distribution'][intent] = template_count

        # Calculate average
        if stats['total_intents'] > 0:
            stats['average_templates_per_intent'] = stats['total_templates'] / stats['total_intents']

        return stats

# Utility functions for external use
def quick_response(intent: str, content: str = "") -> str:
    """Generate quick response without full context"""
    generator = ResponseGenerator()
    return generator.generate_response(intent, {}, [{'answer': content, 'relevance_score': 1.0}] if content else None)

def generate_with_tools(intent: str, entities: Dict[str, Any], tool_outputs: Dict[str, Any]) -> str:
    """Generate response with tool outputs"""
    generator = ResponseGenerator()
    return generator.generate_response(intent, entities, tool_outputs=tool_outputs)

# Main execution for testing
if __name__ == "__main__":
    # Test the response generator
    generator = ResponseGenerator()

    # Test cases
    test_cases = [
        {
            'intent': 'destination_inquiry',
            'entities': {'locations': [{'text': 'Kandy', 'type': 'sri_lankan_location'}]},
            'knowledge_results': [{'answer': 'Kandy is the cultural capital with the Temple of the Tooth', 'relevance_score': 0.9}]
        },
        {
            'intent': 'weather_inquiry',
            'entities': {'locations': [{'text': 'Colombo', 'type': 'sri_lankan_location'}]},
            'tool_outputs': {'weather_checker': {'success': True, 'temperature': 28, 'description': 'Sunny', 'city': 'Colombo'}}
        },
        {
            'intent': 'general_greeting',
            'entities': {},
            'knowledge_results': None
        }
    ]

    print("Testing Response Generator:")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Intent: {test_case['intent']}")

        response = generator.generate_response(
            test_case['intent'],
            test_case['entities'],
            test_case.get('knowledge_results'),
            test_case.get('tool_outputs')
        )

        print(f"Response: {response}")
        print("-" * 40)

    # Show statistics
    stats = generator.get_response_statistics()
    print(f"\nResponse Generator Statistics:")
    print(f"Total intents: {stats['total_intents']}")
    print(f"Total templates: {stats['total_templates']}")
    print(f"Average templates per intent: {stats['average_templates_per_intent']:.1f}")