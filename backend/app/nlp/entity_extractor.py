# backend/app/nlp/entity_extractor.py
import spacy
import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from dataclasses import dataclass
import unicodedata
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityMatch:
    """Data class for entity matches"""
    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'type': self.entity_type,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }

class EntityExtractor:
    """
    Advanced entity extraction system with rule-based patterns,
    spaCy NER, and domain-specific knowledge for Sri Lankan tourism
    """

    def __init__(self):
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy English model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found. Using fallback extraction only.")
            self.nlp = None

        # Initialize geocoder for location validation
        try:
            self.geocoder = Nominatim(user_agent="sri_lanka_tourism_bot", timeout=3)
        except Exception as e:
            logger.warning(f"Geocoder initialization failed: {e}")
            self.geocoder = None

        # Sri Lankan specific locations with coordinates and metadata
        self.sri_lankan_locations = {
            # Major cities
            'colombo': {
                'lat': 6.9271, 'lng': 79.8612, 'type': 'city', 'province': 'Western',
                'aliases': ['colombo city', 'commercial capital'], 'importance': 'high'
            },
            'kandy': {
                'lat': 7.2906, 'lng': 80.6337, 'type': 'city', 'province': 'Central',
                'aliases': ['kandy city', 'cultural capital'], 'importance': 'high'
            },
            'galle': {
                'lat': 6.0535, 'lng': 80.2210, 'type': 'city', 'province': 'Southern',
                'aliases': ['galle city', 'galle fort'], 'importance': 'high'
            },
            'jaffna': {
                'lat': 9.6615, 'lng': 80.0255, 'type': 'city', 'province': 'Northern',
                'aliases': ['jaffna city'], 'importance': 'medium'
            },
            'negombo': {
                'lat': 7.2084, 'lng': 79.8358, 'type': 'city', 'province': 'Western',
                'aliases': ['negombo city'], 'importance': 'medium'
            },

            # Tourist destinations
            'ella': {
                'lat': 6.8721, 'lng': 81.0465, 'type': 'town', 'province': 'Uva',
                'aliases': ['ella town', 'ella rock'], 'importance': 'high'
            },
            'sigiriya': {
                'lat': 7.9568, 'lng': 80.7603, 'type': 'archaeological_site', 'province': 'Central',
                'aliases': ['sigiriya rock', 'lion rock', 'sigiriya fortress'], 'importance': 'high'
            },
            'anuradhapura': {
                'lat': 8.3114, 'lng': 80.4037, 'type': 'ancient_city', 'province': 'North Central',
                'aliases': ['anuradhapura ancient city'], 'importance': 'high'
            },
            'polonnaruwa': {
                'lat': 7.9403, 'lng': 81.0188, 'type': 'ancient_city', 'province': 'North Central',
                'aliases': ['polonnaruwa ancient city'], 'importance': 'high'
            },
            'dambulla': {
                'lat': 7.8731, 'lng': 80.6516, 'type': 'town', 'province': 'Central',
                'aliases': ['dambulla cave temple', 'golden temple'], 'importance': 'medium'
            },
            'trincomalee': {
                'lat': 8.5874, 'lng': 81.2152, 'type': 'city', 'province': 'Eastern',
                'aliases': ['trinco', 'trincomalee city'], 'importance': 'medium'
            },
            'batticaloa': {
                'lat': 7.7102, 'lng': 81.6924, 'type': 'city', 'province': 'Eastern',
                'aliases': ['batti'], 'importance': 'low'
            },
            'matara': {
                'lat': 5.9549, 'lng': 80.5550, 'type': 'city', 'province': 'Southern',
                'aliases': ['matara city'], 'importance': 'medium'
            },
            'mirissa': {
                'lat': 5.9487, 'lng': 80.4501, 'type': 'beach_town', 'province': 'Southern',
                'aliases': ['mirissa beach'], 'importance': 'high'
            },
            'unawatuna': {
                'lat': 6.0108, 'lng': 80.2493, 'type': 'beach_town', 'province': 'Southern',
                'aliases': ['unawatuna beach'], 'importance': 'medium'
            },
            'hikkaduwa': {
                'lat': 6.1391, 'lng': 80.0997, 'type': 'beach_town', 'province': 'Southern',
                'aliases': ['hikkaduwa beach'], 'importance': 'medium'
            },
            'bentota': {
                'lat': 6.4269, 'lng': 79.9951, 'type': 'beach_town', 'province': 'Southern',
                'aliases': ['bentota beach'], 'importance': 'medium'
            },
            'nuwara eliya': {
                'lat': 6.9497, 'lng': 80.7891, 'type': 'hill_station', 'province': 'Central',
                'aliases': ['nuwara eliya city', 'little england'], 'importance': 'high'
            },
            'badulla': {
                'lat': 6.9895, 'lng': 81.0557, 'type': 'city', 'province': 'Uva',
                'aliases': ['badulla city'], 'importance': 'low'
            },
            'ratnapura': {
                'lat': 6.6828, 'lng': 80.3992, 'type': 'city', 'province': 'Sabaragamuwa',
                'aliases': ['gem city'], 'importance': 'low'
            },
            'kurunegala': {
                'lat': 7.4818, 'lng': 80.3653, 'type': 'city', 'province': 'North Western',
                'aliases': ['kurunegala city'], 'importance': 'low'
            },
            'puttalam': {
                'lat': 8.0362, 'lng': 79.8283, 'type': 'city', 'province': 'North Western',
                'aliases': ['puttalam city'], 'importance': 'low'
            },
            'kalutara': {
                'lat': 6.5831, 'lng': 79.9593, 'type': 'city', 'province': 'Western',
                'aliases': ['kalutara city'], 'importance': 'medium'
            },
            'gampaha': {
                'lat': 7.0873, 'lng': 79.9971, 'type': 'city', 'province': 'Western',
                'aliases': ['gampaha city'], 'importance': 'low'
            },
            'haputale': {
                'lat': 6.7678, 'lng': 80.9503, 'type': 'town', 'province': 'Uva',
                'aliases': ['haputale town'], 'importance': 'medium'
            },
            'bandarawela': {
                'lat': 6.8326, 'lng': 80.9857, 'type': 'town', 'province': 'Uva',
                'aliases': ['bandarawela town'], 'importance': 'medium'
            },
            'tissamaharama': {
                'lat': 6.2742, 'lng': 81.2870, 'type': 'town', 'province': 'Southern',
                'aliases': ['tissa', 'tissamaharama town'], 'importance': 'medium'
            },
            'hambantota': {
                'lat': 6.1241, 'lng': 81.1185, 'type': 'city', 'province': 'Southern',
                'aliases': ['hambantota city'], 'importance': 'medium'
            }
        }

        # Currency patterns with comprehensive regex
        self.currency_patterns = {
            'usd': {
                'patterns': [
                    r'\$\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:us\s*)?dollars?',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*usd',
                    r'usd\s*([0-9,]+(?:\.[0-9]{1,2})?)'
                ],
                'symbol': '$',
                'name': 'US Dollar',
                'code': 'USD'
            },
            'eur': {
                'patterns': [
                    r'€\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*euros?',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*eur',
                    r'eur\s*([0-9,]+(?:\.[0-9]{1,2})?)'
                ],
                'symbol': '€',
                'name': 'Euro',
                'code': 'EUR'
            },
            'gbp': {
                'patterns': [
                    r'£\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*pounds?',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*gbp',
                    r'gbp\s*([0-9,]+(?:\.[0-9]{1,2})?)'
                ],
                'symbol': '£',
                'name': 'British Pound',
                'code': 'GBP'
            },
            'lkr': {
                'patterns': [
                    r'rs\.?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:sri\s*lankan\s*)?rupees?',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*lkr',
                    r'lkr\s*([0-9,]+(?:\.[0-9]{1,2})?)'
                ],
                'symbol': 'Rs.',
                'name': 'Sri Lankan Rupee',
                'code': 'LKR'
            },
            'inr': {
                'patterns': [
                    r'₹\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:indian\s*)?rupees?',
                    r'([0-9,]+(?:\.[0-9]{1,2})?)\s*inr',
                    r'inr\s*([0-9,]+(?:\.[0-9]{1,2})?)'
                ],
                'symbol': '₹',
                'name': 'Indian Rupee',
                'code': 'INR'
            }
        }

        # Time and date patterns
        self.time_patterns = {
            'today': {
                'patterns': [r'\btoday\b', r'\bthis\s+day\b'],
                'type': 'relative_date',
                'offset_days': 0
            },
            'tomorrow': {
                'patterns': [r'\btomorrow\b', r'\bnext\s+day\b'],
                'type': 'relative_date',
                'offset_days': 1
            },
            'yesterday': {
                'patterns': [r'\byesterday\b', r'\blast\s+day\b'],
                'type': 'relative_date',
                'offset_days': -1
            },
            'now': {
                'patterns': [r'\bnow\b', r'\bcurrent(?:ly)?\b', r'\bpresent(?:ly)?\b', r'\bright\s+now\b'],
                'type': 'current_time'
            },
            'date': {
                'patterns': [
                    r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
                    r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',
                    r'\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\b'
                ],
                'type': 'absolute_date'
            },
            'time': {
                'patterns': [
                    r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b',
                    r'\b(\d{1,2})\s*(am|pm)\b'
                ],
                'type': 'time'
            },
            'month': {
                'patterns': [
                    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
                ],
                'type': 'month'
            },
            'season': {
                'patterns': [
                    r'\b(spring|summer|autumn|fall|winter|monsoon|dry\s+season|wet\s+season)\b'
                ],
                'type': 'season'
            }
        }

        # Language patterns
        self.language_patterns = {
            'sinhala': {
                'patterns': [r'\bsinhala\b', r'\bsinhalese\b', r'\bsi\b'],
                'native_name': 'සිංහල',
                'iso_code': 'si'
            },
            'tamil': {
                'patterns': [r'\btamil\b', r'\bta\b'],
                'native_name': 'தமிழ்',
                'iso_code': 'ta'
            },
            'english': {
                'patterns': [r'\benglish\b', r'\ben\b'],
                'native_name': 'English',
                'iso_code': 'en'
            }
        }

        # Tourism-specific entities
        self.tourism_entities = {
            'attractions': {
                'temples': ['temple', 'kovil', 'devale', 'dagoba', 'stupa', 'vihara', 'monastery'],
                'historical': ['fort', 'palace', 'ruins', 'archaeological site', 'ancient city', 'heritage site'],
                'natural': ['beach', 'waterfall', 'mountain', 'hill', 'lake', 'river', 'forest', 'national park'],
                'modern': ['museum', 'gallery', 'shopping mall', 'market', 'zoo', 'aquarium']
            },
            'activities': {
                'adventure': ['safari', 'trekking', 'hiking', 'climbing', 'diving', 'snorkeling', 'surfing'],
                'cultural': ['cultural show', 'dance performance', 'cooking class', 'meditation', 'yoga'],
                'wildlife': ['whale watching', 'bird watching', 'elephant watching', 'leopard safari'],
                'leisure': ['spa', 'massage', 'beach relaxation', 'sunset viewing', 'photography']
            },
            'transportation': {
                'public': ['bus', 'train', 'tuk tuk', 'three wheeler'],
                'private': ['taxi', 'car', 'van', 'motorbike', 'bicycle'],
                'special': ['boat', 'helicopter', 'seaplane', 'domestic flight']
            },
            'accommodation': {
                'luxury': ['resort', 'luxury hotel', 'boutique hotel', '5 star hotel'],
                'standard': ['hotel', 'inn', 'lodge'],
                'budget': ['guesthouse', 'hostel', 'homestay', 'budget hotel'],
                'unique': ['eco lodge', 'treehouse', 'villa', 'bungalow', 'camp']
            }
        }

        # Contact information patterns
        self.contact_patterns = {
            'phone': {
                'patterns': [
                    r'(?:\+94|0)?[\s\-]?(?:7[01245678]\d|11|2[1-8]|3[1-8]|4[1-7]|5[1-8]|6[1-8]|9[1-8])[\s\-]?\d{3}[\s\-]?\d{4}',
                    r'(?:\+94|0)?[\s\-]?\d{2,3}[\s\-]?\d{3}[\s\-]?\d{4}'
                ],
                'type': 'sri_lankan_phone'
            },
            'email': {
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ],
                'type': 'email'
            },
            'url': {
                'patterns': [
                    r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
                ],
                'type': 'url'
            }
        }

        # Load custom entity rules if available
        self._load_custom_entity_rules()

        logger.info("EntityExtractor initialized with comprehensive pattern matching")

    def _load_custom_entity_rules(self):
        """Load custom entity rules from file if available"""
        try:
            rules_path = "app/ml/models/entity_rules.json"
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    custom_rules = json.load(f)

                # Merge custom rules with default patterns
                if 'sri_lankan_locations' in custom_rules:
                    self.sri_lankan_locations.update(custom_rules['sri_lankan_locations'])

                if 'currency_patterns' in custom_rules:
                    for currency, patterns in custom_rules['currency_patterns'].items():
                        if currency in self.currency_patterns:
                            self.currency_patterns[currency]['patterns'].extend(patterns)

                logger.info("Custom entity rules loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load custom entity rules: {e}")

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from text using multiple extraction methods
        """
        if not text or not isinstance(text, str):
            return self._empty_entities_dict()

        # Normalize text
        normalized_text = self._normalize_text(text)

        # Initialize entity containers
        entities = self._empty_entities_dict()

        # Extract using different methods
        try:
            # 1. Rule-based extraction
            rule_entities = self._extract_rule_based_entities(normalized_text)
            self._merge_entities(entities, rule_entities)

            # 2. spaCy NER extraction
            if self.nlp:
                spacy_entities = self._extract_spacy_entities(normalized_text)
                self._merge_entities(entities, spacy_entities)

            # 3. Pattern-based extraction
            pattern_entities = self._extract_pattern_entities(normalized_text)
            self._merge_entities(entities, pattern_entities)

            # 4. Context-based extraction
            context_entities = self._extract_context_entities(normalized_text)
            self._merge_entities(entities, context_entities)

            # 5. Post-processing and validation
            entities = self._post_process_entities(entities, normalized_text)

            # 6. Remove duplicates and rank by confidence
            entities = self._deduplicate_and_rank_entities(entities)

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")

        return entities

    def _empty_entities_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create empty entities dictionary with all categories"""
        return {
            'locations': [],
            'currencies': [],
            'amounts': [],
            'dates': [],
            'times': [],
            'languages': [],
            'persons': [],
            'organizations': [],
            'attractions': [],
            'activities': [],
            'transportation': [],
            'accommodation': [],
            'contacts': [],
            'miscellaneous': []
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better entity extraction"""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Basic cleaning while preserving structure
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()

        return text

    def _extract_rule_based_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using rule-based patterns"""
        entities = self._empty_entities_dict()
        text_lower = text.lower()

        # Extract Sri Lankan locations
        for location, location_data in self.sri_lankan_locations.items():
            # Check main location name
            if location in text_lower:
                match_start = text_lower.find(location)
                entities['locations'].append({
                    'text': location.title(),
                    'start': match_start,
                    'end': match_start + len(location),
                    'type': 'sri_lankan_location',
                    'confidence': 0.95,
                    'metadata': location_data
                })

            # Check aliases
            for alias in location_data.get('aliases', []):
                if alias.lower() in text_lower:
                    match_start = text_lower.find(alias.lower())
                    entities['locations'].append({
                        'text': alias.title(),
                        'start': match_start,
                        'end': match_start + len(alias),
                        'type': 'sri_lankan_location_alias',
                        'confidence': 0.9,
                        'metadata': {**location_data, 'primary_name': location}
                    })

        # Extract currencies
        for currency_code, currency_data in self.currency_patterns.items():
            for pattern in currency_data['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    amount_str = match.group(1) if match.lastindex else match.group(0)
                    try:
                        amount = float(amount_str.replace(',', ''))
                        entities['currencies'].append({
                            'text': match.group(0),
                            'start': match.start(),
                            'end': match.end(),
                            'type': currency_code,
                            'confidence': 0.9,
                            'metadata': {
                                'amount': amount,
                                'currency_name': currency_data['name'],
                                'currency_code': currency_data['code'],
                                'symbol': currency_data['symbol']
                            }
                        })

                        # Also add to amounts
                        entities['amounts'].append({
                            'text': amount_str,
                            'start': match.start(),
                            'end': match.end(),
                            'type': 'currency_amount',
                            'confidence': 0.9,
                            'metadata': {
                                'value': amount,
                                'currency': currency_code
                            }
                        })
                    except ValueError:
                        pass

        # Extract time-related entities
        for time_type, time_data in self.time_patterns.items():
            for pattern in time_data['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['times'].append({
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'type': time_type,
                        'confidence': 0.85,
                        'metadata': time_data
                    })

        # Extract languages
        for language, lang_data in self.language_patterns.items():
            for pattern in lang_data['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['languages'].append({
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'type': language,
                        'confidence': 0.9,
                        'metadata': lang_data
                    })

        return entities

    def _extract_spacy_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using spaCy NER"""
        entities = self._empty_entities_dict()

        if not self.nlp:
            return entities

        try:
            doc = self.nlp(text)

            for ent in doc.ents:
                entity_data = {
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8,
                    'metadata': {
                        'spacy_label': ent.label_,
                        'spacy_description': spacy.explain(ent.label_)
                    }
                }

                # Map spaCy entity types to our categories
                if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entities, locations
                    entity_data['type'] = 'general_location'
                    entities['locations'].append(entity_data)

                elif ent.label_ == 'PERSON':
                    entity_data['type'] = 'person'
                    entities['persons'].append(entity_data)

                elif ent.label_ == 'ORG':
                    entity_data['type'] = 'organization'
                    entities['organizations'].append(entity_data)

                elif ent.label_ in ['DATE', 'TIME']:
                    entity_data['type'] = 'temporal'
                    entities['dates'].append(entity_data)

                elif ent.label_ == 'MONEY':
                    entity_data['type'] = 'money'
                    entities['currencies'].append(entity_data)

                elif ent.label_ in ['QUANTITY', 'CARDINAL']:
                    entity_data['type'] = 'number'
                    entities['amounts'].append(entity_data)

                else:
                    entity_data['type'] = ent.label_.lower()
                    entities['miscellaneous'].append(entity_data)

        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")

        return entities

    def _extract_pattern_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using regex patterns"""
        entities = self._empty_entities_dict()

        # Extract contact information
        for contact_type, contact_data in self.contact_patterns.items():
            for pattern in contact_data['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['contacts'].append({
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'type': contact_type,
                        'confidence': 0.85,
                        'metadata': contact_data
                    })

        # Extract numbers (amounts without currency)
        number_pattern = r'\b(?<![\d.])\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
        matches = re.finditer(number_pattern, text)
        for match in matches:
            try:
                value = float(match.group(0).replace(',', ''))
                entities['amounts'].append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'number',
                    'confidence': 0.7,
                    'metadata': {'value': value}
                })
            except ValueError:
                pass

        return entities

    def _extract_context_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities based on context and tourism domain knowledge"""
        entities = self._empty_entities_dict()
        text_lower = text.lower()

        # Extract tourism attractions
        for attraction_category, attraction_list in self.tourism_entities['attractions'].items():
            for attraction in attraction_list:
                if attraction in text_lower:
                    match_start = text_lower.find(attraction)
                    entities['attractions'].append({
                        'text': attraction,
                        'start': match_start,
                        'end': match_start + len(attraction),
                        'type': f'attraction_{attraction_category}',
                        'confidence': 0.8,
                        'metadata': {
                            'category': attraction_category,
                            'attraction_type': 'tourism_site'
                        }
                    })

        # Extract activities
        for activity_category, activity_list in self.tourism_entities['activities'].items():
            for activity in activity_list:
                if activity in text_lower:
                    match_start = text_lower.find(activity)
                    entities['activities'].append({
                        'text': activity,
                        'start': match_start,
                        'end': match_start + len(activity),
                        'type': f'activity_{activity_category}',
                        'confidence': 0.8,
                        'metadata': {
                            'category': activity_category,
                            'activity_type': 'tourism_activity'
                        }
                    })

        # Extract transportation
        for transport_category, transport_list in self.tourism_entities['transportation'].items():
            for transport in transport_list:
                if transport in text_lower:
                    match_start = text_lower.find(transport)
                    entities['transportation'].append({
                        'text': transport,
                        'start': match_start,
                        'end': match_start + len(transport),
                        'type': f'transport_{transport_category}',
                        'confidence': 0.8,
                        'metadata': {
                            'category': transport_category,
                            'transport_type': 'vehicle_mode'
                        }
                    })

        # Extract accommodation
        for accommodation_category, accommodation_list in self.tourism_entities['accommodation'].items():
            for accommodation in accommodation_list:
                if accommodation in text_lower:
                    match_start = text_lower.find(accommodation)
                    entities['accommodation'].append({
                        'text': accommodation,
                        'start': match_start,
                        'end': match_start + len(accommodation),
                        'type': f'accommodation_{accommodation_category}',
                        'confidence': 0.8,
                        'metadata': {
                            'category': accommodation_category,
                            'accommodation_type': 'lodging'
                        }
                    })

        return entities

    def _merge_entities(self, main_entities: Dict[str, List], new_entities: Dict[str, List]):
        """Merge new entities into main entities dictionary"""
        for category, entity_list in new_entities.items():
            if category in main_entities:
                main_entities[category].extend(entity_list)

    def _post_process_entities(self, entities: Dict[str, List], text: str) -> Dict[str, List]:
        """Post-process entities for validation and enhancement"""
        processed_entities = self._empty_entities_dict()

        for category, entity_list in entities.items():
            for entity in entity_list:
                # Validate entity bounds
                if self._validate_entity_bounds(entity, text):
                    # Enhance entity with additional metadata
                    enhanced_entity = self._enhance_entity(entity, text)
                    processed_entities[category].append(enhanced_entity)

        return processed_entities

    def _validate_entity_bounds(self, entity: Dict, text: str) -> bool:
        """Validate that entity boundaries are correct"""
        try:
            start = entity.get('start', 0)
            end = entity.get('end', 0)

            if start < 0 or end > len(text) or start >= end:
                return False

            # Check if extracted text matches the reported bounds
            extracted_text = text[start:end]
            if extracted_text.lower().strip() != entity.get('text', '').lower().strip():
                return False

            return True
        except Exception:
            return False

    def _enhance_entity(self, entity: Dict, text: str) -> Dict:
        """Enhance entity with additional metadata and validation"""
        enhanced = entity.copy()

        # Add context information
        start = entity.get('start', 0)
        end = entity.get('end', 0)

        # Extract surrounding context (10 characters before and after)
        context_start = max(0, start - 10)
        context_end = min(len(text), end + 10)
        context = text[context_start:context_end]

        if 'metadata' not in enhanced:
            enhanced['metadata'] = {}

        enhanced['metadata']['context'] = context
        enhanced['metadata']['extraction_method'] = 'rule_based'

        # Add location validation for Sri Lankan locations
        if entity.get('type') in ['sri_lankan_location', 'sri_lankan_location_alias']:
            enhanced = self._validate_location_entity(enhanced)

        return enhanced

    def _validate_location_entity(self, entity: Dict) -> Dict:
        """Validate and enhance location entities using geocoding"""
        if not self.geocoder:
            return entity

        try:
            location_text = entity.get('text', '')
            # Try to geocode the location
            location = self.geocoder.geocode(f"{location_text}, Sri Lanka", timeout=2)

            if location:
                if 'metadata' not in entity:
                    entity['metadata'] = {}

                entity['metadata']['geocoded'] = {
                    'lat': location.latitude,
                    'lng': location.longitude,
                    'address': location.address,
                    'validated': True
                }
                entity['confidence'] = min(entity.get('confidence', 0.8) + 0.1, 1.0)
            else:
                if 'metadata' not in entity:
                    entity['metadata'] = {}
                entity['metadata']['geocoded'] = {'validated': False}

        except (GeocoderTimedOut, GeocoderUnavailable, Exception):
            # Geocoding failed, but we'll keep the entity
            pass

        return entity

    def _deduplicate_and_rank_entities(self, entities: Dict[str, List]) -> Dict[str, List]:
        """Remove duplicate entities and rank by confidence"""
        deduplicated = self._empty_entities_dict()

        for category, entity_list in entities.items():
            # Remove duplicates based on text and position
            unique_entities = []
            seen_entities = set()

            for entity in entity_list:
                # Create a unique key based on text, start, and end positions
                key = (
                    entity.get('text', '').lower().strip(),
                    entity.get('start', 0),
                    entity.get('end', 0)
                )

                if key not in seen_entities:
                    seen_entities.add(key)
                    unique_entities.append(entity)
                else:
                    # If duplicate, keep the one with higher confidence
                    for i, existing_entity in enumerate(unique_entities):
                        existing_key = (
                            existing_entity.get('text', '').lower().strip(),
                            existing_entity.get('start', 0),
                            existing_entity.get('end', 0)
                        )

                        if existing_key == key:
                            if entity.get('confidence', 0) > existing_entity.get('confidence', 0):
                                unique_entities[i] = entity
                            break

            # Sort by confidence (highest first)
            unique_entities.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            deduplicated[category] = unique_entities

        return deduplicated

    def get_location_suggestions(self, partial_location: str) -> List[Dict[str, Any]]:
        """Get location suggestions based on partial input"""
        suggestions = []
        partial_lower = partial_location.lower().strip()

        if len(partial_lower) < 2:
            return suggestions

        for location, location_data in self.sri_lankan_locations.items():
            # Check if location name starts with or contains partial input
            if location.startswith(partial_lower) or partial_lower in location:
                suggestions.append({
                    'name': location.title(),
                    'type': location_data.get('type', 'location'),
                    'province': location_data.get('province', ''),
                    'importance': location_data.get('importance', 'medium'),
                    'coordinates': {
                        'lat': location_data.get('lat'),
                        'lng': location_data.get('lng')
                    }
                })

            # Check aliases
            for alias in location_data.get('aliases', []):
                if alias.lower().startswith(partial_lower) or partial_lower in alias.lower():
                    suggestions.append({
                        'name': alias.title(),
                        'primary_name': location.title(),
                        'type': location_data.get('type', 'location'),
                        'province': location_data.get('province', ''),
                        'importance': location_data.get('importance', 'medium'),
                        'coordinates': {
                            'lat': location_data.get('lat'),
                            'lng': location_data.get('lng')
                        }
                    })

        # Sort by importance and alphabetically
        importance_order = {'high': 3, 'medium': 2, 'low': 1}
        suggestions.sort(key=lambda x: (
            importance_order.get(x.get('importance', 'medium'), 2),
            x.get('name', '')
        ), reverse=True)

        # Limit to top 10 suggestions
        return suggestions[:10]

    def extract_entities_by_type(self, text: str, entity_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract only specific types of entities"""
        all_entities = self.extract_entities(text)

        filtered_entities = {}
        for entity_type in entity_types:
            if entity_type in all_entities:
                filtered_entities[entity_type] = all_entities[entity_type]
            else:
                filtered_entities[entity_type] = []

        return filtered_entities

    # Add this method inside the EntityExtractor class

    def get_primary_location(self, entities: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        Identifies and returns the most prominent or primary location from the extracted entities.
        Prioritizes Sri Lankan specific locations with high importance.
        """
        locations = entities.get('locations', [])

        if not locations:
            return None

        # Prioritize Sri Lankan specific locations by importance
        sri_lankan_locations = [
            loc for loc in locations
            if loc.get('type') in ['sri_lankan_location', 'sri_lankan_location_alias']
        ]

        if sri_lankan_locations:
            # Sort by confidence and then by importance (if available and higher is better)
            sri_lankan_locations.sort(
                key=lambda x: (
                    x.get('confidence', 0),
                    x.get('metadata', {}).get('importance', 'low') # Assuming importance can be ranked (e.g., high > medium > low)
                ),
                reverse=True
            )
            return sri_lankan_locations[0]

        # If no specific Sri Lankan locations, try to find a general location
        general_locations = [loc for loc in locations if loc.get('type') == 'general_location']
        if general_locations:
            general_locations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return general_locations[0]

        return None

    # Add this method inside the EntityExtractor class

    def get_translation_info(self, text: str) -> Dict[str, Any]:
        """
        Extracts information related to translation requests from the input text,
        such as target languages.
        """
        translation_info = {
            'target_languages': []
        }

        # Look for phrases indicating translation requests and target languages
        text_lower = text.lower()

        # Common patterns for "translate to X"
        translation_patterns = [
            r"translate to (\w+)",
            r"in (\w+)",
            r"to (\w+) language",
            r"can you say this in (\w+)",
            r"what is this in (\w+)"
        ]

        for pattern in translation_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                language_word = match.group(1)
                # Check against known languages
                for lang_code, lang_data in self.language_patterns.items():
                    # Check the main language name or its aliases/iso codes
                    if language_word in [lang_code, lang_data['native_name'].lower(), lang_data['iso_code']]:
                        if lang_data['iso_code'] not in [lang['iso_code'] for lang in translation_info['target_languages']]:
                            translation_info['target_languages'].append({
                                'name': lang_data['native_name'],
                                'iso_code': lang_data['iso_code'],
                                'text_found': match.group(0)
                            })
                        break # Found a language, move to next match

        # Consider single language mentions if they are prominent and not part of another entity
        # This needs careful consideration to avoid false positives (e.g., "I speak English")
        # For simplicity, we'll stick to explicit "translate to" patterns for now.
        # A more advanced approach might use intent detection before calling this.

        return translation_info

    def get_entity_statistics(self, text: str) -> Dict[str, Any]:
        """Get statistics about entities found in text"""
        entities = self.extract_entities(text)

        stats = {
            'total_entities': 0,
            'entities_by_category': {},
            'confidence_distribution': {
                'high': 0,    # > 0.8
                'medium': 0,  # 0.6 - 0.8
                'low': 0      # < 0.6
            },
            'unique_locations': 0,
            'currencies_found': set(),
            'languages_detected': set()
        }

        for category, entity_list in entities.items():
            count = len(entity_list)
            stats['entities_by_category'][category] = count
            stats['total_entities'] += count

            # Analyze confidence levels
            for entity in entity_list:
                confidence = entity.get('confidence', 0)
                if confidence > 0.8:
                    stats['confidence_distribution']['high'] += 1
                elif confidence > 0.6:
                    stats['confidence_distribution']['medium'] += 1
                else:
                    stats['confidence_distribution']['low'] += 1

            # Special handling for specific categories
            if category == 'locations':
                stats['unique_locations'] = len(set(
                    entity.get('text', '').lower() for entity in entity_list
                ))

            elif category == 'currencies':
                for entity in entity_list:
                    currency_type = entity.get('type', '')
                    if currency_type:
                        stats['currencies_found'].add(currency_type.upper())

            elif category == 'languages':
                for entity in entity_list:
                    lang_type = entity.get('type', '')
                    if lang_type:
                        stats['languages_detected'].add(lang_type)

        # Convert sets to lists for JSON serialization
        stats['currencies_found'] = list(stats['currencies_found'])
        stats['languages_detected'] = list(stats['languages_detected'])

        return stats

    def validate_entity_extraction(self, text: str, expected_entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Validate entity extraction against expected results"""
        extracted_entities = self.extract_entities(text)

        validation_results = {
            'accuracy_by_category': {},
            'overall_accuracy': 0,
            'precision': 0,
            'recall': 0,
            'missing_entities': {},
            'extra_entities': {},
            'correctly_identified': {}
        }

        total_expected = 0
        total_found = 0
        total_correct = 0

        for category, expected_list in expected_entities.items():
            expected_set = set(item.lower().strip() for item in expected_list)

            if category in extracted_entities:
                found_set = set(
                    entity.get('text', '').lower().strip()
                    for entity in extracted_entities[category]
                )
            else:
                found_set = set()

            # Calculate metrics for this category
            correct = expected_set.intersection(found_set)
            missing = expected_set - found_set
            extra = found_set - expected_set

            category_precision = len(correct) / len(found_set) if found_set else 0
            category_recall = len(correct) / len(expected_set) if expected_set else 0
            category_f1 = (2 * category_precision * category_recall) / (category_precision + category_recall) if (category_precision + category_recall) > 0 else 0

            validation_results['accuracy_by_category'][category] = {
                'precision': category_precision,
                'recall': category_recall,
                'f1_score': category_f1,
                'expected_count': len(expected_set),
                'found_count': len(found_set),
                'correct_count': len(correct)
            }

            validation_results['missing_entities'][category] = list(missing)
            validation_results['extra_entities'][category] = list(extra)
            validation_results['correctly_identified'][category] = list(correct)

            # Update totals
            total_expected += len(expected_set)
            total_found += len(found_set)
            total_correct += len(correct)

        # Calculate overall metrics
        validation_results['overall_accuracy'] = total_correct / total_expected if total_expected > 0 else 0
        validation_results['precision'] = total_correct / total_found if total_found > 0 else 0
        validation_results['recall'] = total_correct / total_expected if total_expected > 0 else 0

        return validation_results

    def update_entity_rules(self, new_rules: Dict[str, Any]) -> bool:
        """Update entity extraction rules with new patterns"""
        try:
            # Update location data
            if 'locations' in new_rules:
                for location, location_data in new_rules['locations'].items():
                    self.sri_lankan_locations[location.lower()] = location_data

            # Update currency patterns
            if 'currencies' in new_rules:
                for currency, currency_data in new_rules['currencies'].items():
                    if currency in self.currency_patterns:
                        self.currency_patterns[currency]['patterns'].extend(
                            currency_data.get('patterns', [])
                        )
                    else:
                        self.currency_patterns[currency] = currency_data

            # Update tourism entities
            if 'tourism_entities' in new_rules:
                for category, subcategories in new_rules['tourism_entities'].items():
                    if category in self.tourism_entities:
                        for subcategory, items in subcategories.items():
                            if subcategory in self.tourism_entities[category]:
                                self.tourism_entities[category][subcategory].extend(items)
                            else:
                                self.tourism_entities[category][subcategory] = items
                    else:
                        self.tourism_entities[category] = subcategories

            # Save updated rules
            self._save_custom_entity_rules()

            logger.info("Entity extraction rules updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating entity rules: {e}")
            return False

    def _save_custom_entity_rules(self):
        """Save custom entity rules to file"""
        try:
            rules_data = {
                'sri_lankan_locations': self.sri_lankan_locations,
                'currency_patterns': {
                    currency: {'patterns': data['patterns']}
                    for currency, data in self.currency_patterns.items()
                },
                'tourism_entities': self.tourism_entities,
                'version': '1.0',
                'last_updated': datetime.now().isoformat()
            }

            # Ensure directory exists
            os.makedirs("app/ml/models", exist_ok=True)

            rules_path = "app/ml/models/entity_rules.json"
            with open(rules_path, 'w') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Entity rules saved to {rules_path}")

        except Exception as e:
            logger.error(f"Error saving entity rules: {e}")

    def get_currency_info(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts currency values and symbols (e.g., "$100", "LKR 2000", "€50") from text.
        Returns a list of dictionaries with 'currency' and 'amount'.
        """
        currency_patterns = [
            r"(?i)(LKR|Rs\.?)\s?(\d+(?:,\d{3})*(?:\.\d{1,2})?)",  # Sri Lankan rupees
            r"\$ ?(\d+(?:,\d{3})*(?:\.\d{1,2})?)",               # Dollar values
            r"€ ?(\d+(?:,\d{3})*(?:\.\d{1,2})?)",                # Euro values
            r"£ ?(\d+(?:,\d{3})*(?:\.\d{1,2})?)"                 # Pound values
        ]

        results = []

        for pattern in currency_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    currency, amount = match
                else:
                    # For symbols like "$100", the match might be just the amount
                    currency = pattern[0]  # crude fallback
                    amount = match

                results.append({
                    "currency": currency.strip(),
                    "amount": amount.strip()
                })

        return results

# Usage example and testing functions
def test_entity_extraction():
    """Test entity extraction with sample text"""
    extractor = EntityExtractor()

    test_texts = [
        "I want to visit Sigiriya and Kandy next month. The cost is around $500.",
        "Can you translate 'hello' to Sinhala? I'm staying in Colombo.",
        "What's the weather like in Ella today? I need to book a hotel.",
        "Show me directions to Galle Fort. My phone number is +94 77 123 4567.",
        "I'm planning a safari in Yala National Park tomorrow.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text}")

        entities = extractor.extract_entities(text)

        for category, entity_list in entities.items():
            if entity_list:
                print(f"\n{category.upper()}:")
                for entity in entity_list:
                    print(f"  - {entity['text']} (type: {entity['type']}, confidence: {entity['confidence']:.2f})")

        # Get statistics
        stats = extractor.get_entity_statistics(text)
        print(f"\nTotal entities found: {stats['total_entities']}")

if __name__ == "__main__":
    test_entity_extraction()