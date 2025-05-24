# backend/app/tools/translator.py

import logging
import requests
from typing import Dict, Optional
from backend.app.utils.config import get_api_key

logger = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        # Get Google Translate API key
        self.api_key = get_api_key('GOOGLE_TRANSLATE_API_KEY')
        self.base_url = "https://translation.googleapis.com/language/translate/v2"

        # Initialize with default phrases
        self.common_phrases = self.get_common_phrases()

        # Create reverse mapping safely
        self.reverse_phrases = {
            'sinhala_to_english': {}
        }

        # Only create reverse mapping if common_phrases exists and has the key
        if self.common_phrases and 'english_to_sinhala' in self.common_phrases:
            self.reverse_phrases['sinhala_to_english'] = {
                v: k for k, v in self.common_phrases['english_to_sinhala'].items()
            }

        # Language code mapping for Google Translate
        self.language_codes = {
            'english': 'en',
            'sinhala': 'si',
            'tamil': 'ta'
        }

        logger.info("Translator initialized successfully")
        if self.api_key:
            logger.info("Google Translate API key found")
        else:
            logger.warning("Google Translate API key not found, using fallback translations only")

    def get_common_phrases(self) -> Dict[str, Dict[str, str]]:
        """Get common phrases dictionary"""
        try:
            # Return hardcoded phrases for Ceylon/Sri Lanka tourism
            phrases = {
                'english_to_sinhala': {
                    # Greetings
                    'hello': 'ආයුබෝවන්',
                    'hi': 'ආයුබෝවන්',
                    'good morning': 'සුබ උදෑසනක්',
                    'good afternoon': 'සුබ දවසක්',
                    'good evening': 'සුබ සන්ධ්‍යාවක්',
                    'good night': 'සුබ රාත්‍රියක්',
                    'goodbye': 'ආයුබෝවන්',
                    'bye': 'ආයුබෝවන්',

                    # Common phrases
                    'thank you': 'ස්තූතියි',
                    'thanks': 'ස්තූතියි',
                    'please': 'කරුණාකර',
                    'sorry': 'සමාවෙන්න',
                    'excuse me': 'සමාවෙන්න',
                    'yes': 'ඔව්',
                    'no': 'නැහැ',
                    'welcome': 'සාදරයෙන් පිළිගනිමු',
                    'how are you': 'කොහොමද?',
                    'i am fine': 'මම හොඳින්',
                    'what is your name': 'ඔබේ නම මොකක්ද?',
                    'my name is': 'මගේ නම',

                    # Tourism related
                    'how much': 'කීයද?',
                    'where': 'කොහෙද?',
                    'when': 'කවදාද?',
                    'what': 'මොකක්ද?',
                    'why': 'ඇයි?',
                    'who': 'කවුද?',
                    'hotel': 'හෝටලය',
                    'restaurant': 'අවන්හල',
                    'food': 'ආහාර',
                    'water': 'වතුර',
                    'tea': 'තේ',
                    'coffee': 'කෝපි',
                    'help': 'උදව්',
                    'taxi': 'ටැක්සි',
                    'bus': 'බස්',
                    'train': 'දුම්රිය',
                    'airport': 'ගුවන්තොටුපළ',
                    'beach': 'වෙරළ',
                    'temple': 'පන්සල',
                    'money': 'සල්ලි',
                    'bank': 'බැංකුව',
                    'hospital': 'රෝහල',
                    'police': 'පොලිසිය',
                    'emergency': 'හදිසි',
                    'toilet': 'වැසිකිළිය',
                    'room': 'කාමරය',
                    'beautiful': 'ලස්සනයි',
                    'big': 'ලොකු',
                    'small': 'පොඩි',
                    'hot': 'උණුසුම්',
                    'cold': 'සීතල',
                    'good': 'හොඳයි',
                    'bad': 'නරකයි',
                    'expensive': 'මිල අධිකයි',
                    'cheap': 'මිල අඩුයි',
                    'open': 'විවෘතයි',
                    'closed': 'වසා ඇත',
                    'today': 'අද',
                    'tomorrow': 'හෙට',
                    'yesterday': 'ඊයේ'
                },
                'english_to_tamil': {
                    'hello': 'வணக்கம்',
                    'thank you': 'நன்றி',
                    'please': 'தயவுசெய்து',
                    'yes': 'ஆம்',
                    'no': 'இல்லை',
                    'how are you': 'எப்படி இருக்கிறீர்கள்?',
                    'goodbye': 'போய் வருகிறேன்'
                }
            }
            return phrases
        except Exception as e:
            logger.error(f"Error loading common phrases: {e}")
            # Return default structure if loading fails
            return {'english_to_sinhala': {}}

    def translate(self, text: str, source_lang: str = 'english', target_lang: str = 'sinhala') -> str:
        """
        Translate text between languages

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            Translated text or original text if translation not found
        """
        try:
            text_lower = text.lower().strip()

            # Check if we have direct translation
            if source_lang == 'english' and target_lang == 'sinhala':
                if text_lower in self.common_phrases.get('english_to_sinhala', {}):
                    return self.common_phrases['english_to_sinhala'][text_lower]
            elif source_lang == 'english' and target_lang == 'tamil':
                if text_lower in self.common_phrases.get('english_to_tamil', {}):
                    return self.common_phrases['english_to_tamil'][text_lower]
            elif source_lang == 'sinhala' and target_lang == 'english':
                if text_lower in self.reverse_phrases.get('sinhala_to_english', {}):
                    return self.reverse_phrases['sinhala_to_english'][text_lower]

            # If no common phrase found and API key is available, use Google Translate
            if self.api_key:
                translated_text = self._translate_with_google(text, source_lang, target_lang)
                if translated_text:
                    return translated_text

            # If no translation found, return original text
            logger.debug(f"No translation found for '{text}' from {source_lang} to {target_lang}")
            return text

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def _translate_with_google(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using Google Translate API

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            Translated text or None if translation fails
        """
        try:
            # Convert language names to codes
            source_code = self.language_codes.get(source_lang, source_lang)
            target_code = self.language_codes.get(target_lang, target_lang)

            # Prepare request
            params = {
                'q': text,
                'source': source_code,
                'target': target_code,
                'key': self.api_key,
                'format': 'text'
            }

            # Make API request
            response = requests.post(
                self.base_url,
                params=params,
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'translations' in data['data']:
                    translations = data['data']['translations']
                    if translations and len(translations) > 0:
                        return translations[0]['translatedText']
            else:
                logger.error(f"Google Translate API error: {response.status_code} - {response.text}")

            return None

        except requests.exceptions.Timeout:
            logger.error("Google Translate API timeout")
            return None
        except Exception as e:
            logger.error(f"Google Translate API error: {e}")
            return None

    def detect_language(self, text: str) -> str:
        """
        Detect language based on character sets or Google API

        Args:
            text: Text to detect language for

        Returns:
            Detected language ('english', 'sinhala', 'tamil', or 'unknown')
        """
        # Check for Sinhala Unicode range
        if any('\u0D80' <= char <= '\u0DFF' for char in text):
            return 'sinhala'

        # Check for Tamil Unicode range
        elif any('\u0B80' <= char <= '\u0BFF' for char in text):
            return 'tamil'

        # Check if mostly English characters
        elif text.isascii():
            return 'english'

        # If uncertain and API key is available, use Google's language detection
        if self.api_key:
            detected_lang = self._detect_language_with_google(text)
            if detected_lang:
                # Map Google language codes back to our names
                for name, code in self.language_codes.items():
                    if code == detected_lang:
                        return name

        return 'unknown'

    def _detect_language_with_google(self, text: str) -> Optional[str]:
        """
        Detect language using Google Translate API

        Args:
            text: Text to detect language for

        Returns:
            Language code or None if detection fails
        """
        try:
            detect_url = "https://translation.googleapis.com/language/translate/v2/detect"

            params = {
                'q': text,
                'key': self.api_key
            }

            response = requests.post(detect_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'detections' in data['data']:
                    detections = data['data']['detections']
                    if detections and len(detections) > 0 and len(detections[0]) > 0:
                        return detections[0][0]['language']

            return None

        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return None

    def get_greeting(self, language: str = 'english') -> str:
        """Get appropriate greeting based on language"""
        greetings = {
            'english': 'Hello! Welcome to Ceylon Guide.',
            'sinhala': 'ආයුබෝවන්! Ceylon Guide වෙත සාදරයෙන් පිළිගනිමු.',
            'tamil': 'வணக்கம்! Ceylon Guide-க்கு வரவேற்கிறோம்.'
        }
        return greetings.get(language, greetings['english'])

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return {
            'english': 'English',
            'sinhala': 'සිංහල (Sinhala)',
            'tamil': 'தமிழ் (Tamil)'
        }