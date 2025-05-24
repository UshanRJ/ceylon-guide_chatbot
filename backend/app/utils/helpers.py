# backend/app/utils/helpers.py
import uuid
import hashlib
import json
import re
import string
import secrets
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import unicodedata
import logging
from urllib.parse import urlparse, parse_qs
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    """Generate a unique session ID using UUID4"""
    return str(uuid.uuid4())

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token"""
    return secrets.token_urlsafe(length)

def hash_text(text: str, algorithm: str = 'md5') -> str:
    """Generate hash for text using specified algorithm"""
    if not text:
        return ""

    text_bytes = text.encode('utf-8')

    if algorithm == 'md5':
        return hashlib.md5(text_bytes).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(text_bytes).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(text_bytes).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def clean_text(text: str, max_length: int = 1000) -> str:
    """Clean and normalize text input"""
    if not text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Remove control characters but keep newlines and tabs
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')

    # Remove extra whitespace while preserving paragraph breaks
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = ' '.join(line.split())
        cleaned_lines.append(cleaned_line)
    text = '\n'.join(cleaned_lines)

    # Remove excessive line breaks (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove special characters but keep basic punctuation and unicode letters
    # Allow: letters, numbers, basic punctuation, whitespace, and common symbols
    allowed_pattern = r'[^\w\s\.\?\!\,\-\'\"\:\;\(\)\[\]\{\}\/\\\@\#\$\%\^\&\*\+\=\~\`\|\<\>\n\t]'
    text = re.sub(allowed_pattern, '', text)

    # Limit length
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + '...'

    return text.strip()

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename to remove dangerous characters"""
    if not filename:
        return "unnamed_file"

    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)

    # Limit length
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        filename = name[:max_name_length] + ('.' + ext if ext else '')

    return filename or "unnamed_file"

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 15) -> List[str]:
    """Extract keywords from text using improved NLP techniques"""
    if not text:
        return []

    # Extended stop words for better filtering
    stop_words = {
        # English stop words
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'around',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'shall', 'ought', 'need', 'want', 'like', 'get', 'got',
        'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when',
        'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
        'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine',
        'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself',
        'herself', 'itself', 'ourselves', 'yourselves', 'themselves',
        'some', 'any', 'many', 'much', 'few', 'little', 'most', 'more', 'less',
        'all', 'both', 'each', 'every', 'other', 'another', 'such', 'same',
        'different', 'new', 'old', 'first', 'last', 'next', 'previous',
        'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low',
        'right', 'wrong', 'true', 'false', 'yes', 'no', 'not', 'now', 'then',
        'today', 'tomorrow', 'yesterday', 'very', 'quite', 'really', 'just',
        'only', 'also', 'too', 'even', 'still', 'yet', 'already', 'again',
        'back', 'away', 'down', 'out', 'off', 'over', 'under', 'across'
    }

    # Clean and normalize text
    clean_text_str = clean_text(text.lower())

    # Extract words using better pattern (include hyphenated words)
    words = re.findall(r'\b[a-zA-Z][\w\-]*[a-zA-Z]\b|\b[a-zA-Z]\b', clean_text_str)

    # Filter and score keywords
    keyword_scores = {}

    for word in words:
        # Skip if too short, too long, or is stop word
        if (len(word) < min_length or
                len(word) > 25 or
                word in stop_words or
                word.isdigit()):
            continue

        # Skip if mostly numbers or special characters
        if sum(c.isalpha() for c in word) < len(word) * 0.6:
            continue

        # Calculate frequency
        keyword_scores[word] = keyword_scores.get(word, 0) + 1

    # Sort by frequency and get top keywords
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_keywords[:max_keywords]]

    return keywords

def extract_entities_simple(text: str) -> Dict[str, List[str]]:
    """Simple entity extraction using regex patterns"""
    entities = {
        'emails': [],
        'urls': [],
        'phone_numbers': [],
        'currencies': [],
        'dates': [],
        'times': [],
        'locations': [],
        'mentions': [],
        'hashtags': []
    }

    if not text:
        return entities

    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)

    # URL pattern
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    entities['urls'] = re.findall(url_pattern, text)

    # Phone number pattern (international and local formats)
    phone_pattern = r'(?:\+94|0)?[\s\-]?(?:\d{2,3}[\s\-]?\d{3}[\s\-]?\d{4}|\d{3}[\s\-]?\d{4})'
    entities['phone_numbers'] = re.findall(phone_pattern, text)

    # Currency patterns
    currency_patterns = [
        r'\$\d+(?:\.\d{2})?',  # USD
        r'€\d+(?:\.\d{2})?',   # EUR
        r'£\d+(?:\.\d{2})?',   # GBP
        r'Rs\.?\s*\d+(?:\.\d{2})?',  # LKR
        r'₹\d+(?:\.\d{2})?'    # INR
    ]
    for pattern in currency_patterns:
        entities['currencies'].extend(re.findall(pattern, text))

    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
    ]
    for pattern in date_patterns:
        entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))

    # Time patterns
    time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?\b'
    entities['times'] = re.findall(time_pattern, text)

    # Social media mentions and hashtags
    entities['mentions'] = re.findall(r'@\w+', text)
    entities['hashtags'] = re.findall(r'#\w+', text)

    # Remove duplicates while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))

    return entities

def format_response_for_display(response: str, tool_outputs: Dict[str, Any] = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format response for frontend display with enhanced metadata"""
    formatted_response = {
        'text': response,
        'timestamp': datetime.now().isoformat(),
        'tools_used': [],
        'attachments': [],
        'metadata': metadata or {},
        'display_config': {
            'show_typing_animation': True,
            'enable_markdown': True,
            'highlight_entities': True
        }
    }

    if tool_outputs:
        for tool_name, output in tool_outputs.items():
            if output and isinstance(output, dict) and output.get('success'):
                tool_info = {
                    'name': tool_name,
                    'display_name': tool_name.replace('_', ' ').title(),
                    'data': output,
                    'icon': get_tool_icon(tool_name),
                    'color': get_tool_color(tool_name)
                }

                # Enhanced formatting for different tools
                if tool_name == 'currency_converter':
                    tool_info.update({
                        'display_type': 'currency',
                        'summary': f"{output.get('amount', 0)} {output.get('from_currency', '').upper()} = {output.get('converted_amount', 0)} {output.get('to_currency', '').upper()}",
                        'primary_value': output.get('converted_amount'),
                        'secondary_value': output.get('exchange_rate'),
                        'unit': output.get('to_currency', '').upper()
                    })

                elif tool_name == 'weather_checker':
                    tool_info.update({
                        'display_type': 'weather',
                        'summary': f"{output.get('temperature', 0)}°C, {output.get('description', '')}",
                        'primary_value': output.get('temperature'),
                        'secondary_value': output.get('humidity'),
                        'unit': '°C',
                        'location': output.get('city', ''),
                        'icon_code': get_weather_icon_code(output.get('description', ''))
                    })

                elif tool_name == 'translator':
                    tool_info.update({
                        'display_type': 'translation',
                        'summary': f"Translated to {output.get('to_language', '').title()}",
                        'original_text': output.get('original_text'),
                        'translated_text': output.get('translated_text'),
                        'source_lang': output.get('from_language'),
                        'target_lang': output.get('to_language')
                    })

                elif tool_name == 'maps_integration':
                    tool_info.update({
                        'display_type': 'map',
                        'summary': f"Location: {output.get('location', '')}",
                        'location_name': output.get('location'),
                        'coordinates': output.get('coordinates'),
                        'map_url': output.get('map_url')
                    })

                formatted_response['tools_used'].append(tool_info)

    return formatted_response

def get_tool_icon(tool_name: str) -> str:
    """Get icon name for tool"""
    icon_map = {
        'currency_converter': 'dollar-sign',
        'weather_checker': 'cloud',
        'translator': 'globe',
        'maps_integration': 'map-pin',
        'knowledge_search': 'book',
        'image_analyzer': 'image',
        'sentiment_analyzer': 'heart'
    }
    return icon_map.get(tool_name, 'tool')

def get_tool_color(tool_name: str) -> str:
    """Get color scheme for tool"""
    color_map = {
        'currency_converter': '#28a745',
        'weather_checker': '#007bff',
        'translator': '#6f42c1',
        'maps_integration': '#fd7e14',
        'knowledge_search': '#17a2b8',
        'image_analyzer': '#e83e8c',
        'sentiment_analyzer': '#ffc107'
    }
    return color_map.get(tool_name, '#6c757d')

def get_weather_icon_code(description: str) -> str:
    """Get weather icon code based on description"""
    description_lower = description.lower()

    if 'sunny' in description_lower or 'clear' in description_lower:
        return 'sun'
    elif 'cloudy' in description_lower or 'overcast' in description_lower:
        return 'cloud'
    elif 'rain' in description_lower or 'shower' in description_lower:
        return 'cloud-rain'
    elif 'storm' in description_lower or 'thunder' in description_lower:
        return 'cloud-lightning'
    elif 'snow' in description_lower:
        return 'cloud-snow'
    elif 'fog' in description_lower or 'mist' in description_lower:
        return 'cloud-fog'
    else:
        return 'cloud'

def validate_input(data: Dict[str, Any], required_fields: List[str],
                   field_validators: Dict[str, callable] = None) -> Dict[str, Any]:
    """Enhanced input validation with custom validators"""
    errors = []
    warnings = []

    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not data[field] and data[field] != 0:  # Allow 0 as valid value
            errors.append(f"Empty value for required field: {field}")

    # Apply custom field validators
    if field_validators:
        for field, validator in field_validators.items():
            if field in data and data[field] is not None:
                try:
                    is_valid, error_msg = validator(data[field])
                    if not is_valid:
                        errors.append(f"Invalid {field}: {error_msg}")
                except Exception as e:
                    errors.append(f"Validation error for {field}: {str(e)}")

    # General validations
    if 'message' in data and data['message']:
        message = data['message']

        # Length validation
        if len(message) > 5000:
            errors.append("Message too long (max 5000 characters)")
        elif len(message) > 1000:
            warnings.append("Message is quite long, consider breaking it into smaller parts")

        # Security validation - check for potentially harmful content
        harmful_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript protocol
            r'eval\s*\(',                 # eval function
            r'exec\s*\(',                 # exec function
            r'<iframe[^>]*>',             # iframe tags
            r'<object[^>]*>',             # object tags
            r'on\w+\s*=',                 # event handlers
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                errors.append("Message contains potentially harmful content")
                break

        # Spam detection - simple heuristics
        if len(re.findall(r'http[s]?://', message)) > 5:
            warnings.append("Message contains many URLs, might be flagged as spam")

        if len(message) > 0 and len(set(message.lower())) / len(message) < 0.3:
            warnings.append("Message has low character diversity, might be spam")

    # Email validation
    if 'email' in data and data['email']:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data['email']):
            errors.append("Invalid email format")

    # URL validation
    if 'url' in data and data['url']:
        try:
            parsed = urlparse(data['url'])
            if not parsed.scheme or not parsed.netloc:
                errors.append("Invalid URL format")
        except Exception:
            errors.append("Invalid URL format")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'error_count': len(errors),
        'warning_count': len(warnings)
    }

def create_custom_validators() -> Dict[str, callable]:
    """Create custom field validators"""
    def validate_session_id(value):
        try:
            uuid.UUID(value)
            return True, ""
        except ValueError:
            return False, "Invalid session ID format"

    def validate_rating(value):
        try:
            rating = int(value)
            if 1 <= rating <= 5:
                return True, ""
            else:
                return False, "Rating must be between 1 and 5"
        except (ValueError, TypeError):
            return False, "Rating must be a number"

    def validate_language_code(value):
        valid_codes = ['en', 'si', 'ta', 'english', 'sinhala', 'tamil']
        if value.lower() in valid_codes:
            return True, ""
        else:
            return False, f"Unsupported language code. Supported: {', '.join(valid_codes)}"

    def validate_currency_code(value):
        valid_codes = ['USD', 'EUR', 'GBP', 'LKR', 'INR', 'AUD', 'CAD', 'JPY']
        if value.upper() in valid_codes:
            return True, ""
        else:
            return False, f"Unsupported currency code. Supported: {', '.join(valid_codes)}"

    def validate_coordinates(value):
        try:
            lat, lng = map(float, str(value).split(','))
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                return True, ""
            else:
                return False, "Coordinates out of valid range"
        except (ValueError, TypeError):
            return False, "Invalid coordinate format (expected: lat,lng)"

    return {
        'session_id': validate_session_id,
        'rating': validate_rating,
        'language': validate_language_code,
        'currency': validate_currency_code,
        'coordinates': validate_coordinates
    }

def calculate_response_time(start_time: datetime) -> float:
    """Calculate response time in seconds with high precision"""
    return (datetime.now() - start_time).total_seconds()

def calculate_processing_stats(start_time: datetime,
                               components_used: List[str]) -> Dict[str, Any]:
    """Calculate detailed processing statistics"""
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    return {
        'total_response_time_seconds': round(total_time, 3),
        'response_time_ms': round(total_time * 1000, 1),
        'processing_speed': 'fast' if total_time < 1.0 else 'medium' if total_time < 3.0 else 'slow',
        'components_used': components_used,
        'component_count': len(components_used),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'performance_score': min(100, max(0, int(100 - (total_time * 20))))  # Score out of 100
    }

def log_interaction(session_id: str, user_message: str, bot_response: str,
                    response_time: float, tools_used: List[str] = None,
                    metadata: Dict[str, Any] = None):
    """Enhanced interaction logging with structured data"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'interaction_data': {
            'user_message_length': len(user_message),
            'bot_response_length': len(bot_response),
            'user_message_words': len(user_message.split()),
            'bot_response_words': len(bot_response.split()),
        },
        'performance_metrics': {
            'response_time_seconds': round(response_time, 3),
            'response_time_ms': round(response_time * 1000, 1),
            'processing_speed_rating': get_speed_rating(response_time)
        },
        'tools_data': {
            'tools_used': tools_used or [],
            'tools_count': len(tools_used) if tools_used else 0,
            'has_external_api_calls': bool(tools_used)
        },
        'metadata': metadata or {},
        'log_level': 'INFO'
    }

    # Add performance warnings
    if response_time > 5.0:
        log_entry['log_level'] = 'WARNING'
        log_entry['performance_warning'] = 'Slow response time detected'

    # Log to different outputs based on environment
    try:
        # Console logging (development)
        logger.info(f"INTERACTION_LOG: {json.dumps(log_entry, ensure_ascii=False)}")

        # Could add additional logging sinks here:
        # - File logging
        # - External logging services (e.g., ELK stack)
        # - Metrics collection (e.g., Prometheus)

    except Exception as e:
        logger.error(f"Failed to log interaction: {str(e)}")

def get_speed_rating(response_time: float) -> str:
    """Get human-readable speed rating"""
    if response_time < 0.5:
        return 'excellent'
    elif response_time < 1.0:
        return 'good'
    elif response_time < 2.0:
        return 'acceptable'
    elif response_time < 5.0:
        return 'slow'
    else:
        return 'very_slow'

def get_sri_lankan_context() -> Dict[str, Any]:
    """Get comprehensive Sri Lankan tourism context information"""
    return {
        'basic_info': {
            'currency': 'LKR',
            'currency_symbol': 'Rs.',
            'languages': ['Sinhala', 'Tamil', 'English'],
            'time_zone': 'Asia/Colombo',
            'country_code': 'LK',
            'calling_code': '+94'
        },
        'geography': {
            'capital': 'Colombo',
            'administrative_capital': 'Sri Jayawardenepura Kotte',
            'largest_city': 'Colombo',
            'area_km2': 65610,
            'coordinates': {'lat': 7.8731, 'lng': 80.7718}
        },
        'popular_destinations': {
            'cultural_triangle': ['Anuradhapura', 'Polonnaruwa', 'Sigiriya', 'Dambulla'],
            'hill_country': ['Kandy', 'Ella', 'Nuwara Eliya', 'Badulla'],
            'southern_coast': ['Galle', 'Mirissa', 'Unawatuna', 'Tangalle'],
            'western_coast': ['Colombo', 'Negombo', 'Kalutara', 'Bentota'],
            'eastern_coast': ['Trincomalee', 'Batticaloa', 'Arugam Bay'],
            'northern_region': ['Jaffna', 'Mannar', 'Vavuniya']
        },
        'climate_seasons': {
            'dry_season_west_south': {
                'months': 'December to March',
                'description': 'Best time for west and south coasts'
            },
            'wet_season_west_south': {
                'months': 'May to September',
                'description': 'Southwest monsoon affects west and south'
            },
            'dry_season_east_north': {
                'months': 'May to September',
                'description': 'Best time for east and north coasts'
            },
            'wet_season_east_north': {
                'months': 'October to January',
                'description': 'Northeast monsoon affects east and north'
            },
            'inter_monsoon': {
                'months': 'April and October-November',
                'description': 'Transition periods with unpredictable weather'
            }
        },
        'transportation': {
            'international_airports': ['Bandaranaike International Airport (CMB)', 'Mattala Rajapaksa International Airport (HRI)'],
            'railway_network': 'Scenic train routes available',
            'bus_network': 'Extensive government and private bus services',
            'tuk_tuk': 'Three-wheeler taxis available everywhere',
            'car_rental': 'Available with or without driver'
        },
        'cultural_info': {
            'major_religions': ['Buddhism (70%)', 'Hinduism (12%)', 'Islam (10%)', 'Christianity (8%)'],
            'festivals': {
                'vesak': 'Buddha\'s birthday celebration in May',
                'esala_perahera': 'Kandy cultural pageant in July/August',
                'tamil_new_year': 'April celebration',
                'christmas': 'December 25th',
                'deepavali': 'Hindu festival of lights'
            },
            'cultural_etiquette': [
                'Remove shoes before entering temples and homes',
                'Dress modestly at religious sites',
                'Use right hand for eating and greeting',
                'Avoid pointing feet at people or sacred objects',
                'Be respectful when photographing people'
            ]
        },
        'emergency_numbers': {
            'police': '119',
            'ambulance': '110',
            'fire_brigade': '111',
            'tourist_hotline': '1912',
            'accident_service': '1938',
            'directory_inquiries': '161'
        },
        'useful_phrases': {
            'sinhala': {
                'hello': 'ආයුබෝවන් (Ayubowan)',
                'thank_you': 'ස්තුතියි (Stuthi)',
                'please': 'කරුණාකර (Karunakara)',
                'excuse_me': 'සමාවන්න (Samawanna)',
                'how_much': 'කීයද? (Kiyada?)',
                'where_is': 'කොහෙද? (Koheda?)'
            },
            'tamil': {
                'hello': 'வணக்கம் (Vanakkam)',
                'thank_you': 'நன்றி (Nandri)',
                'please': 'தயவுசெய்து (Dayavu seydu)',
                'excuse_me': 'மன்னிக்கவும் (Mannikavum)',
                'how_much': 'எவ்வளவு? (Evvalavu?)',
                'where_is': 'எங்கே? (Enge?)'
            }
        },
        'practical_tips': {
            'currency_exchange': 'Banks and authorized dealers offer best rates',
            'tipping': '10% at restaurants, small amounts for services',
            'bargaining': 'Common in markets and with tuk-tuk drivers',
            'water': 'Drink bottled water, avoid tap water',
            'electricity': '230V, Type D/G/M plugs',
            'internet': 'Good 4G coverage, WiFi widely available',
            'health': 'No special vaccinations required for most travelers'
        }
    }

def format_sri_lankan_phone(phone: str) -> str:
    """Format Sri Lankan phone number to standard format"""
    if not phone:
        return ""

    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)

    # Handle different formats
    if digits.startswith('94'):
        # International format starting with 94
        if len(digits) == 11:
            return f"+94 {digits[2:4]} {digits[4:7]} {digits[7:]}"
        elif len(digits) == 12:
            return f"+94 {digits[2:5]} {digits[5:8]} {digits[8:]}"
    elif digits.startswith('0'):
        # Local format starting with 0
        if len(digits) == 10:
            return f"{digits[0:3]} {digits[3:6]} {digits[6:]}"
        elif len(digits) == 11:
            return f"{digits[0:4]} {digits[4:7]} {digits[7:]}"
    elif len(digits) == 9:
        # Format without leading 0
        return f"0{digits[0:2]} {digits[2:5]} {digits[5:]}"

    # Return original if no standard format matches
    return phone

def validate_sri_lankan_phone(phone: str) -> Tuple[bool, str]:
    """Validate Sri Lankan phone number format"""
    if not phone:
        return False, "Phone number is required"

    # Remove all non-digit characters for validation
    digits = re.sub(r'\D', '', phone)

    # Valid patterns for Sri Lankan numbers
    valid_patterns = [
        r'^94[1-9]\d{8},      # International format: +94XXXXXXXXX'
        r'^0[1-9]\d{8},       # Local format: 0XXXXXXXXX'
        r'^[1-9]\d{8}         # Without country/area code: XXXXXXXXX'
    ]

    for pattern in valid_patterns:
        if re.match(pattern, digits):
            return True, ""

    return False, "Invalid Sri Lankan phone number format"

def extract_location_keywords(text: str) -> List[str]:
    """Extract location-related keywords specific to Sri Lanka"""
    sri_lankan_locations = [
        # Major cities
        'colombo', 'kandy', 'galle', 'jaffna', 'negombo', 'anuradhapura',
        'polonnaruwa', 'trincomalee', 'batticaloa', 'matara', 'badulla',
        'ratnapura', 'kurunegala', 'puttalam', 'kalutara', 'gampaha',

        # Tourist destinations
        'ella', 'sigiriya', 'dambulla', 'mirissa', 'unawatuna', 'hikkaduwa',
        'bentota', 'nuwara eliya', 'haputale', 'bandarawela', 'tissamaharama',
        'yala', 'udawalawe', 'sinharaja', 'horton plains', 'adams peak',

        # Regions and provinces
        'western province', 'central province', 'southern province',
        'northern province', 'eastern province', 'north western province',
        'north central province', 'uva province', 'sabaragamuwa province',

        # Geographic features
        'hill country', 'tea country', 'cultural triangle', 'southern coast',
        'eastern coast', 'western coast', 'northern plains'
    ]

    found_locations = []
    text_lower = text.lower()

    for location in sri_lankan_locations:
        if location in text_lower:
            found_locations.append(location.title())

    return found_locations

def calculate_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula"""
    import math

    # Convert latitude and longitude from degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r

def get_nearest_major_city(lat: float, lng: float) -> Dict[str, Any]:
    """Find the nearest major Sri Lankan city to given coordinates"""
    major_cities = {
        'Colombo': {'lat': 6.9271, 'lng': 79.8612},
        'Kandy': {'lat': 7.2906, 'lng': 80.6337},
        'Galle': {'lat': 6.0535, 'lng': 80.2210},
        'Jaffna': {'lat': 9.6615, 'lng': 80.0255},
        'Trincomalee': {'lat': 8.5874, 'lng': 81.2152},
        'Negombo': {'lat': 7.2084, 'lng': 79.8358},
        'Anuradhapura': {'lat': 8.3114, 'lng': 80.4037},
        'Batticaloa': {'lat': 7.7102, 'lng': 81.6924},
        'Matara': {'lat': 5.9549, 'lng': 80.5550},
        'Badulla': {'lat': 6.9895, 'lng': 81.0557}
    }

    nearest_city = None
    min_distance = float('inf')

    for city, coords in major_cities.items():
        distance = calculate_distance_km(lat, lng, coords['lat'], coords['lng'])
        if distance < min_distance:
            min_distance = distance
            nearest_city = {
                'name': city,
                'coordinates': coords,
                'distance_km': round(distance, 2)
            }

    return nearest_city

def format_currency_lkr(amount: float, include_symbol: bool = True) -> str:
    """Format currency amount in Sri Lankan Rupees"""
    if amount == 0:
        return "Rs. 0.00" if include_symbol else "0.00"

    # Format with commas for thousands
    if amount >= 1000000:
        # Format in millions
        formatted = f"{amount/1000000:.1f}M"
    elif amount >= 1000:
        # Format with commas
        formatted = f"{amount:,.2f}"
    else:
        formatted = f"{amount:.2f}"

    if include_symbol:
        return f"Rs. {formatted}"
    else:
        return formatted

def parse_currency_amount(text: str) -> Dict[str, Any]:
    """Parse currency amount from text"""
    result = {
        'amount': None,
        'currency': None,
        'original_text': text
    }

    # Currency patterns with their codes
    patterns = [
        (r'Rs\.?\s*([0-9,]+(?:\.[0-9]{2})?)', 'LKR'),
        (r'\$\s*([0-9,]+(?:\.[0-9]{2})?)', 'USD'),
        (r'€\s*([0-9,]+(?:\.[0-9]{2})?)', 'EUR'),
        (r'£\s*([0-9,]+(?:\.[0-9]{2})?)', 'GBP'),
        (r'₹\s*([0-9,]+(?:\.[0-9]{2})?)', 'INR'),
        (r'([0-9,]+(?:\.[0-9]{2})?)\s*(rupees?|lkr)', 'LKR'),
        (r'([0-9,]+(?:\.[0-9]{2})?)\s*(dollars?|usd)', 'USD'),
        (r'([0-9,]+(?:\.[0-9]{2})?)\s*(euros?|eur)', 'EUR'),
        (r'([0-9,]+(?:\.[0-9]{2})?)\s*(pounds?|gbp)', 'GBP')
    ]

    for pattern, currency in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                result['amount'] = float(amount_str)
                result['currency'] = currency
                break
            except ValueError:
                continue

    return result

def generate_session_summary(messages: List[Dict], session_duration: float) -> Dict[str, Any]:
    """Generate a summary of the chat session"""
    if not messages:
        return {'error': 'No messages to summarize'}

    user_messages = [msg for msg in messages if msg.get('type') == 'user']
    bot_messages = [msg for msg in messages if msg.get('type') == 'bot']

    # Extract topics discussed
    all_text = ' '.join([msg.get('message', '') for msg in messages])
    keywords = extract_keywords(all_text, max_keywords=10)
    locations = extract_location_keywords(all_text)

    # Calculate statistics
    avg_response_time = session_duration / len(bot_messages) if bot_messages else 0

    # Extract tools used
    tools_used = set()
    for msg in bot_messages:
        if msg.get('tools_used'):
            tools_used.update(msg['tools_used'])

    # Determine primary topics
    topic_keywords = {
        'destinations': ['place', 'visit', 'destination', 'attraction', 'sight'],
        'transportation': ['travel', 'bus', 'train', 'taxi', 'transport'],
        'accommodation': ['hotel', 'stay', 'accommodation', 'lodge'],
        'food': ['food', 'restaurant', 'cuisine', 'eat', 'meal'],
        'culture': ['culture', 'tradition', 'festival', 'temple'],
        'weather': ['weather', 'climate', 'temperature', 'rain'],
        'currency': ['currency', 'exchange', 'money', 'cost', 'price']
    }

    topics_discussed = []
    for topic, topic_words in topic_keywords.items():
        if any(word in all_text.lower() for word in topic_words):
            topics_discussed.append(topic)

    return {
        'session_stats': {
            'total_messages': len(messages),
            'user_messages': len(user_messages),
            'bot_messages': len(bot_messages),
            'session_duration_seconds': round(session_duration, 2),
            'average_response_time': round(avg_response_time, 2)
        },
        'content_analysis': {
            'main_keywords': keywords,
            'locations_mentioned': locations,
            'topics_discussed': topics_discussed,
            'tools_utilized': list(tools_used)
        },
        'interaction_quality': {
            'engagement_level': 'high' if len(user_messages) > 10 else 'medium' if len(user_messages) > 5 else 'low',
            'session_completion': 'complete' if session_duration > 300 else 'partial',  # 5+ minutes considered complete
            'tool_usage': 'active' if tools_used else 'minimal'
        }
    }

def escape_html(text: str) -> str:
    """Escape HTML characters in text"""
    if not text:
        return ""

    html_escape_table = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "/": "&#x2F;"
    }

    return "".join(html_escape_table.get(c, c) for c in text)

def unescape_html(text: str) -> str:
    """Unescape HTML characters in text"""
    if not text:
        return ""

    html_unescape_table = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#x27;": "'",
        "&#x2F;": "/"
    }

    result = text
    for escaped, original in html_unescape_table.items():
        result = result.replace(escaped, original)

    return result

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix"""
    if not text or len(text) <= max_length:
        return text

    # Try to truncate at word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')

    if last_space > max_length * 0.7:  # If we can truncate at a reasonable word boundary
        truncated = truncated[:last_space]

    return truncated + suffix

def is_question(text: str) -> bool:
    """Determine if text is likely a question"""
    if not text:
        return False

    text = text.strip().lower()

    # Check for question words
    question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose', 'whom']
    starts_with_question_word = any(text.startswith(word) for word in question_words)

    # Check for question mark
    ends_with_question_mark = text.endswith('?')

    # Check for question patterns
    question_patterns = [
        r'^(is|are|was|were|do|does|did|will|would|could|should|can|may|might)',
        r'^(tell me|show me|explain|describe|help)',
        r'(please|can you|could you)',
    ]

    matches_pattern = any(re.search(pattern, text) for pattern in question_patterns)

    return starts_with_question_word or ends_with_question_mark or matches_pattern

def extract_intent_hints(text: str) -> List[str]:
    """Extract hints about user intent from text"""
    hints = []
    text_lower = text.lower()

    # Intent patterns
    intent_patterns = {
        'seeking_information': ['what', 'tell me', 'explain', 'describe', 'information about'],
        'asking_for_recommendations': ['recommend', 'suggest', 'best', 'top', 'should i'],
        'planning_trip': ['plan', 'itinerary', 'schedule', 'visit', 'trip'],
        'comparing_options': ['vs', 'versus', 'compare', 'difference', 'better'],
        'seeking_directions': ['how to get', 'directions', 'route', 'way to'],
        'asking_about_cost': ['cost', 'price', 'expensive', 'cheap', 'budget'],
        'requesting_translation': ['translate', 'mean', 'say in', 'language'],
        'weather_inquiry': ['weather', 'temperature', 'rain', 'climate'],
        'expressing_gratitude': ['thank', 'thanks', 'appreciate', 'grateful'],
        'greeting': ['hello', 'hi', 'good morning', 'good evening']
    }

    for intent, patterns in intent_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            hints.append(intent)

    return hints

def rate_limit_key(identifier: str, action: str = 'general') -> str:
    """Generate rate limiting key"""
    return f"rate_limit:{action}:{hash_text(identifier, 'sha256')[:16]}"

def is_sri_lankan_context(text: str) -> bool:
    """Determine if text is related to Sri Lankan context"""
    sri_lankan_indicators = [
        # Country names
        'sri lanka', 'ceylon', 'srilanka',

        # Major cities and locations
        'colombo', 'kandy', 'galle', 'jaffna', 'negombo', 'ella', 'sigiriya',
        'anuradhapura', 'polonnaruwa', 'trincomalee', 'nuwara eliya',

        # Cultural/Religious terms
        'buddhist', 'temple', 'dagoba', 'stupa', 'perahera', 'vesak',
        'ayubowan', 'sinhala', 'tamil', 'sinhalese',

        # Food and culture
        'rice and curry', 'hoppers', 'kottu', 'roti', 'sambol', 'dhal',
        'ceylon tea', 'tea plantation', 'spices',

        # Currency and practical
        'rupees', 'lkr', 'rs.', 'tuk tuk', 'three wheeler',

        # Natural features
        'monsoon', 'tropical', 'indian ocean', 'pearl of the indian ocean'
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in sri_lankan_indicators)

def get_time_greeting() -> str:
    """Get appropriate greeting based on current time"""
    current_hour = datetime.now().hour

    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 21:
        return "Good evening"
    else:
        return "Hello"

def format_list_response(items: List[str], max_items: int = 10,
                         conjunction: str = 'and') -> str:
    """Format a list of items into a readable string"""
    if not items:
        return ""

    # Limit items if too many
    if len(items) > max_items:
        items = items[:max_items]
        truncated = True
    else:
        truncated = False

    if len(items) == 1:
        result = items[0]
    elif len(items) == 2:
        result = f"{items[0]} {conjunction} {items[1]}"
    else:
        result = f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"

    if truncated:
        result += f", and {len(items) - max_items} more"

    return result

# Error handling utilities
class ChatbotError(Exception):
    """Base exception class for chatbot errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ValidationError(ChatbotError):
    """Exception for validation errors"""
    pass

class ToolError(ChatbotError):
    """Exception for tool-related errors"""
    pass

class DatabaseError(ChatbotError):
    """Exception for database-related errors"""
    pass

def handle_error_gracefully(func):
    """Decorator for graceful error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ChatbotError:
            raise  # Re-raise custom errors
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise ChatbotError(
                message="An unexpected error occurred",
                error_code="INTERNAL_ERROR",
                details={"original_error": str(e), "function": func.__name__}
            )
    return wrapper

# Configuration helpers
def get_environment() -> str:
    """Get current environment (development, staging, production)"""
    import os
    return os.getenv('ENVIRONMENT', 'development').lower()

def is_production() -> bool:
    """Check if running in production environment"""
    return get_environment() == 'production'

def is_development() -> bool:
    """Check if running in development environment"""
    return get_environment() == 'development'