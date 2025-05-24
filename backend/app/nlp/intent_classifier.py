# backend/app/nlp/intent_classifier.py
import spacy
import pickle
import os
import json
import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom text preprocessor for sklearn pipeline"""

    def __init__(self, lowercase=True, remove_punctuation=True,
                 remove_stopwords=True, lemmatize=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # Initialize tools
        self._initialize_nltk_tools()

    def _initialize_nltk_tools(self):
        """Initialize NLTK tools"""
        try:
            # Download required NLTK data
            nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for item in nltk_downloads:
                try:
                    nltk.download(item, quiet=True)
                except Exception:
                    pass

            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()

        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
            self.stop_words = set()
            self.lemmatizer = None
            self.stemmer = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        """Preprocess individual text"""
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove punctuation but keep important ones
        if self.remove_punctuation:
            # Keep question marks and exclamation marks as they indicate intent
            text = re.sub(r'[^\w\s\?\!]', ' ', text)

        # Tokenize
        try:
            tokens = word_tokenize(text) if 'word_tokenize' in globals() else text.split()
        except:
            tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]

        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            except:
                pass

        return ' '.join(tokens)

class IntentClassifier:
    """
    Advanced intent classification system with multi-model ensemble,
    confidence scoring, and continuous learning capabilities
    """

    def __init__(self):
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy English model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found. Using fallback processing.")
            self.nlp = None

        # Model configuration
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.confidence_threshold = 0.4

        # File paths
        self.model_dir = Path("app/ml/models")
        self.model_path = self.model_dir / "intent_classifier.pkl"
        self.metadata_path = self.model_dir / "intent_classifier_metadata.json"

        # Create directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Intent definitions and patterns
        self.intent_definitions = {
            'destination_inquiry': {
                'description': 'Questions about places to visit and tourist destinations',
                'keywords': ['places', 'visit', 'destinations', 'attractions', 'sights', 'locations', 'go', 'see', 'tourist'],
                'patterns': [
                    r'\b(?:where|what)\s+(?:should|can|to)\s+(?:i|we)\s+(?:visit|go|see)\b',
                    r'\b(?:best|top|good)\s+(?:places|destinations|attractions)\b',
                    r'\b(?:tourist|sightseeing)\s+(?:spots|places|attractions)\b',
                    r'\b(?:recommend|suggest)\s+(?:places|destinations)\b'
                ],
                'examples': [
                    "What are the best places to visit?",
                    "Where should I go for sightseeing?",
                    "Recommend tourist attractions"
                ]
            },
            'transportation': {
                'description': 'Questions about travel and transportation methods',
                'keywords': ['travel', 'transport', 'bus', 'train', 'taxi', 'flight', 'car', 'get', 'reach', 'how'],
                'patterns': [
                    r'\b(?:how|way)\s+(?:to|can)\s+(?:get|go|travel|reach)\b',
                    r'\b(?:bus|train|taxi|flight|car|transport)\b',
                    r'\b(?:travel|journey)\s+(?:from|to|between)\b'
                ],
                'examples': [
                    "How to get from Colombo to Kandy?",
                    "What transportation is available?",
                    "Bus schedule to Galle"
                ]
            },
            'accommodation': {
                'description': 'Questions about hotels, stays, and lodging',
                'keywords': ['hotel', 'stay', 'accommodation', 'lodge', 'guesthouse', 'resort', 'book', 'room'],
                'patterns': [
                    r'\b(?:where|good)\s+(?:to|can)\s+stay\b',
                    r'\b(?:hotel|accommodation|lodge|resort|guesthouse)\b',
                    r'\b(?:book|booking)\s+(?:room|hotel)\b'
                ],
                'examples': [
                    "Where can I stay in Colombo?",
                    "Best hotels in Kandy",
                    "Budget accommodation options"
                ]
            },
            'food_inquiry': {
                'description': 'Questions about food, restaurants, and cuisine',
                'keywords': ['food', 'restaurant', 'cuisine', 'eat', 'meal', 'dishes', 'local', 'cooking'],
                'patterns': [
                    r'\b(?:what|where)\s+(?:to|can)\s+eat\b',
                    r'\b(?:food|cuisine|dishes|meal)\b',
                    r'\b(?:restaurant|dining)\b',
                    r'\b(?:local|traditional)\s+food\b'
                ],
                'examples': [
                    "What is Sri Lankan food like?",
                    "Best restaurants in Galle",
                    "Local dishes to try"
                ]
            },
            'culture_inquiry': {
                'description': 'Questions about culture, traditions, and history',
                'keywords': ['culture', 'tradition', 'festival', 'history', 'temple', 'heritage', 'customs'],
                'patterns': [
                    r'\b(?:culture|cultural|tradition|traditional)\b',
                    r'\b(?:festival|celebration|ceremony)\b',
                    r'\b(?:history|historical|heritage)\b',
                    r'\b(?:temple|religious|spiritual)\b'
                ],
                'examples': [
                    "Tell me about Sri Lankan culture",
                    "What festivals are celebrated?",
                    "Buddhist temples to visit"
                ]
            },
            'weather_inquiry': {
                'description': 'Questions about weather and climate',
                'keywords': ['weather', 'climate', 'temperature', 'rain', 'sunny', 'forecast', 'season'],
                'patterns': [
                    r'\b(?:weather|climate|temperature)\b',
                    r'\b(?:rain|sunny|hot|cold|warm|cool)\b',
                    r'\b(?:forecast|season|monsoon)\b'
                ],
                'examples': [
                    "What's the weather like?",
                    "Climate in Sri Lanka",
                    "When is monsoon season?"
                ]
            },
            'currency_inquiry': {
                'description': 'Questions about currency, exchange rates, and money',
                'keywords': ['currency', 'exchange', 'money', 'rate', 'cost', 'price', 'dollar', 'rupee'],
                'patterns': [
                    r'\b(?:currency|exchange|money)\b',
                    r'\b(?:rate|cost|price|expensive|cheap)\b',
                    r'\b(?:dollar|euro|pound|rupee)\b',
                    r'\b(?:convert|conversion)\b'
                ],
                'examples': [
                    "What's the exchange rate?",
                    "Convert dollars to rupees",
                    "How much does this cost?"
                ]
            },
            'translation_request': {
                'description': 'Requests for translation between languages',
                'keywords': ['translate', 'sinhala', 'tamil', 'language', 'meaning', 'say'],
                'patterns': [
                    r'\b(?:translate|translation)\b',
                    r'\b(?:sinhala|tamil|english)\b',
                    r'\b(?:how\s+do\s+you\s+say|what\s+does.*mean)\b',
                    r'\b(?:language|meaning)\b'
                ],
                'examples': [
                    "Translate hello to Sinhala",
                    "How do you say thank you in Tamil?",
                    "What does Ayubowan mean?"
                ]
            },
            'map_request': {
                'description': 'Requests for maps, directions, and location information',
                'keywords': ['map', 'location', 'directions', 'route', 'navigate', 'where', 'show'],
                'patterns': [
                    r'\b(?:map|location|directions|route)\b',
                    r'\b(?:where\s+is|show\s+me)\b',
                    r'\b(?:navigate|navigation|gps)\b'
                ],
                'examples': [
                    "Show me on the map",
                    "Where is Sigiriya?",
                    "Directions to Galle Fort"
                ]
            },
            'general_greeting': {
                'description': 'General greetings and pleasantries',
                'keywords': ['hello', 'hi', 'good', 'morning', 'evening', 'thanks', 'thank', 'bye'],
                'patterns': [
                    r'\b(?:hello|hi|hey)\b',
                    r'\b(?:good\s+(?:morning|afternoon|evening|day))\b',
                    r'\b(?:thank|thanks|appreciate)\b',
                    r'\b(?:bye|goodbye|farewell)\b'
                ],
                'examples': [
                    "Hello",
                    "Good morning",
                    "Thank you for your help"
                ]
            },
            'help_request': {
                'description': 'Requests for help and assistance',
                'keywords': ['help', 'assist', 'support', 'guide', 'how', 'can', 'please'],
                'patterns': [
                    r'\b(?:help|assist|support|guide)\b',
                    r'\b(?:can\s+you|could\s+you|please)\b',
                    r'\b(?:how\s+(?:do|can|to))\b',
                    r'\b(?:what\s+(?:can|should))\b'
                ],
                'examples': [
                    "Can you help me?",
                    "I need assistance",
                    "How do I do this?"
                ]
            },
            'system_check': {
                'description': 'Queries related to system functionality, testing, or status checks',
                'keywords': ['test', 'check', 'status', 'working', 'function', 'try', 'is it'],
                'patterns': [
                    r'\b(?:test(?:ing)?)\b',
                    r'\b(?:check\s+(?:status|if\s+it\'?s\s+working))\b',
                    r'\b(?:is\s+it\s+working)\b',
                    r'\b(?:try\s+this)\b',
                    r'\b(?:how\s+are\s+you)\b'
                ],
                'examples': [
                    "Just testing.",
                    "Is the system working?",
                    "Check status.",
                    "How are you functioning?",
                    "Can I try this out?",
                    "Is it operational?",
                    "Test, test."
                ]
            }
        }

        # Load or train model
        self.load_or_train_model()

        logger.info("IntentClassifier initialized successfully")

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify intent of user input with confidence score
        """
        if not text or not isinstance(text, str):
            return "unknown", 0.0

        try:
            # Preprocess text
            processed_text = self._preprocess_for_classification(text)

            # If no model is loaded, use rule-based classification
            if not self.model:
                return self._rule_based_classification(text)

            # Use trained model for classification
            prediction, confidence = self._model_based_classification(processed_text)

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                # Fallback to rule-based if confidence is too low
                rule_prediction, rule_confidence = self._rule_based_classification(text)
                if rule_confidence > confidence:
                    return rule_prediction, rule_confidence

            return prediction, confidence

        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return "unknown", 0.0

    def _preprocess_for_classification(self, text: str) -> str:
        """Preprocess text for classification"""
        # Basic preprocessing
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Handle common contractions
        contractions = {
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "i'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def _model_based_classification(self, text: str) -> Tuple[str, float]:
        """Use trained model for classification"""
        try:
            # Transform text using the saved pipeline
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict([text])[0]

                # Get confidence score
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba([text])[0]
                    confidence = max(probabilities)
                elif hasattr(self.model, 'decision_function'):
                    # For SVM
                    decision_scores = self.model.decision_function([text])[0]
                    if isinstance(decision_scores, np.ndarray):
                        confidence = max(decision_scores)
                    else:
                        confidence = abs(decision_scores)
                    # Normalize confidence to 0-1 range
                    confidence = min(1.0, abs(confidence) / 3.0)
                else:
                    confidence = 0.7  # Default confidence

                return prediction, confidence
            else:
                return "unknown", 0.0

        except Exception as e:
            logger.error(f"Error in model-based classification: {e}")
            return "unknown", 0.0

    def _rule_based_classification(self, text: str) -> Tuple[str, float]:
        """Fallback rule-based classification"""
        text_lower = text.lower()
        best_intent = "unknown"
        best_score = 0.0

        for intent, definition in self.intent_definitions.items():
            score = 0.0

            # Check keyword matches
            keywords = definition.get('keywords', [])
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_score = keyword_matches / len(keywords) if keywords else 0
            score += keyword_score * 0.6

            # Check pattern matches
            patterns = definition.get('patterns', [])
            pattern_matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            pattern_score = pattern_matches / len(patterns) if patterns else 0
            score += pattern_score * 0.4

            # Update best match
            if score > best_score:
                best_score = score
                best_intent = intent

        # Apply minimum confidence threshold for rule-based classification
        if best_score < 0.3:
            return "unknown", best_score

        return best_intent, min(best_score, 0.9)  # Cap at 0.9 for rule-based

    def load_or_train_model(self):
        """Load existing model or train a new one"""
        if self.model_path.exists():
            try:
                self.load_model()
                logger.info("Intent classifier model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Error loading model: {e}")

        # Train new model if loading failed or no model exists
        logger.info("Training new intent classifier model")
        self.train_model()

    def load_model(self):
        """Load trained model from file"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.vectorizer = model_data.get('vectorizer')
                self.label_encoder = model_data.get('label_encoder')
            else:
                # Legacy format
                self.model = model_data

            # Load metadata if available
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.confidence_threshold = metadata.get('confidence_threshold', 0.4)

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def train_model(self):
        """Train intent classification model"""
        try:
            logger.info("Starting intent classifier training")

            # Generate training data
            training_data = self._generate_training_data()

            if len(training_data) < 10:
                logger.warning("Insufficient training data. Using rule-based classification only.")
                return

            # Prepare data
            X = [item['text'] for item in training_data]
            y = [item['intent'] for item in training_data]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Create pipeline with preprocessing and multiple models
            models = self._get_model_candidates()
            best_model = None
            best_score = 0

            for model_name, model in models.items():
                try:
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', TextPreprocessor()),
                        ('vectorizer', TfidfVectorizer(
                            max_features=5000,
                            ngram_range=(1, 2),
                            min_df=1,
                            max_df=0.95
                        )),
                        ('classifier', model)
                    ])

                    # Train model
                    pipeline.fit(X_train, y_train)

                    # Evaluate
                    score = pipeline.score(X_test, y_test)
                    logger.info(f"{model_name} accuracy: {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_model = pipeline
                        best_model_name = model_name

                except Exception as e:
                    logger.warning(f"Error training {model_name}: {e}")

            if best_model:
                self.model = best_model

                # Evaluate on test set
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                logger.info(f"Best model: {best_model_name}")
                logger.info(f"Test accuracy: {accuracy:.4f}")

                # Print classification report
                report = classification_report(y_test, y_pred)
                logger.info(f"Classification Report:\n{report}")

                # Save model
                self._save_model(accuracy, best_model_name)

            else:
                logger.error("No successful model training")

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise e

    def _get_model_candidates(self) -> Dict[str, Any]:
        """Get candidate models for training"""
        return {
            'svm_linear': SVC(kernel='linear', probability=True, random_state=42),
            'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'complement_nb': ComplementNB()
        }

    def _generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate comprehensive training data"""
        training_data = []

        # Add examples from intent definitions
        for intent, definition in self.intent_definitions.items():
            examples = definition.get('examples', [])
            for example in examples:
                training_data.append({
                    'text': example,
                    'intent': intent
                })

        # Generate synthetic training data
        synthetic_data = self._generate_synthetic_training_data()
        training_data.extend(synthetic_data)

        # Load external training data if available
        external_data = self._load_external_training_data()
        training_data.extend(external_data)

        # Remove duplicates
        seen = set()
        unique_data = []
        for item in training_data:
            key = (item['text'].lower(), item['intent'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)

        logger.info(f"Generated {len(unique_data)} unique training examples")
        return unique_data

    def _generate_synthetic_training_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data using templates and variations"""
        synthetic_data = []

        # Templates for each intent
        templates = {
            'destination_inquiry': [
                "What are the best {adjective} places to visit in Sri Lanka?",
                "Where should I go for {activity}?",
                "Recommend {adjective} destinations in {location}",
                "What {attractions} should I see?",
                "Tell me about {adjective} places to visit",
                "{question_word} are good tourist spots?",
                "I want to visit {adjective} places",
                "Show me {attractions} in Sri Lanka"
            ],
            'transportation': [
                "How to get from {location1} to {location2}?",
                "What {transport} options are available?",
                "How can I travel to {location}?",
                "Is there a {transport} to {location}?",
                "{transport} schedule to {location}",
                "Best way to reach {location}",
                "Transportation from {location1} to {location2}",
                "How much does {transport} cost?"
            ],
            'accommodation': [
                "Where can I stay in {location}?",
                "Best {accommodation} in {location}",
                "Recommend {adjective} {accommodation}",
                "Budget {accommodation} options",
                "Luxury {accommodation} in {location}",
                "Book {accommodation} in {location}",
                "Good places to stay near {location}",
                "Cheap {accommodation} recommendations"
            ],
            'food_inquiry': [
                "What is {adjective} Sri Lankan food?",
                "Best {food_type} in {location}",
                "Where to eat {adjective} food?",
                "Local {food_type} to try",
                "Traditional Sri Lankan {food_type}",
                "Food recommendations in {location}",
                "What should I eat in Sri Lanka?",
                "Popular {food_type} dishes"
            ],
            'culture_inquiry': [
                "Tell me about Sri Lankan {cultural_aspect}",
                "What {cultural_events} are celebrated?",
                "Buddhist {cultural_places} to visit",
                "Cultural {activities} in {location}",
                "History of {location}",
                "Traditional {cultural_aspect} in Sri Lanka",
                "Religious {cultural_places} worth visiting",
                "Cultural experiences in {location}"
            ],
            'weather_inquiry': [
                "What's the weather like in {location}?",
                "Climate in {location}",
                "Best time to visit {location}",
                "Is it {weather_condition} in {location}?",
                "Weather forecast for {location}",
                "When is {season} season?",
                "Temperature in {location}",
                "Rainy season in Sri Lanka"
            ],
            'currency_inquiry': [
                "Exchange rate for {currency}",
                "Convert {amount} {currency1} to {currency2}",
                "How much is {amount} {currency}?",
                "Currency exchange in {location}",
                "What's the current rate?",
                "Cost of {item} in Sri Lanka",
                "Price of {item} in {currency}",
                "Money exchange rates"
            ],
            'translation_request': [
                "Translate {phrase} to {language}",
                "How do you say {phrase} in {language}?",
                "What does {phrase} mean?",
                "{language} translation for {phrase}",
                "Meaning of {phrase}",
                "How to say {phrase}",
                "Translate to {language}",
                "{phrase} in {language}"
            ],
            'map_request': [
                "Show me {location} on map",
                "Where is {location}?",
                "Directions to {location}",
                "Map of {location}",
                "Location of {location}",
                "How to reach {location}?",
                "GPS coordinates for {location}",
                "Navigate to {location}"
            ],
            'general_greeting': [
                "Hello",
                "Hi there",
                "Good {time_of_day}",
                "Thank you",
                "Thanks for {service}",
                "Goodbye",
                "Nice to meet you",
                "Have a great day"
            ],
            'help_request': [
                "Can you help me?",
                "I need help with {topic}",
                "How do I {action}?",
                "What can you do?",
                "Please help me",
                "I need assistance",
                "Guide me through {process}",
                "Support needed"
            ]
        }

        # Variables for template filling
        variables = {
            'adjective': ['beautiful', 'amazing', 'best', 'popular', 'famous', 'historic', 'scenic'],
            'activity': ['sightseeing', 'photography', 'adventure', 'relaxation', 'culture'],
            'location': ['Colombo', 'Kandy', 'Galle', 'Ella', 'Sigiriya', 'Negombo'],
            'location1': ['Colombo', 'Kandy', 'Galle', 'Airport'],
            'location2': ['Kandy', 'Ella', 'Galle', 'Sigiriya'],
            'attractions': ['temples', 'beaches', 'mountains', 'forts', 'parks'],
            'question_word': ['What', 'Which', 'Where'],
            'transport': ['bus', 'train', 'taxi', 'tuk tuk'],
            'accommodation': ['hotels', 'guesthouses', 'resorts', 'hostels'],
            'food_type': ['restaurants', 'dishes', 'meals', 'cuisine'],
            'cultural_aspect': ['culture', 'traditions', 'customs', 'heritage'],
            'cultural_events': ['festivals', 'celebrations', 'ceremonies'],
            'cultural_places': ['temples', 'sites', 'monuments'],
            'activities': ['shows', 'performances', 'tours'],
            'weather_condition': ['sunny', 'rainy', 'hot', 'cool'],
            'season': ['monsoon', 'dry', 'rainy'],
            'currency': ['USD', 'EUR', 'GBP'],
            'currency1': ['dollars', 'euros', 'pounds'],
            'currency2': ['rupees', 'LKR'],
            'amount': ['100', '50', '200'],
            'item': ['food', 'hotel', 'transport'],
            'phrase': ['hello', 'thank you', 'goodbye', 'please'],
            'language': ['Sinhala', 'Tamil'],
            'time_of_day': ['morning', 'afternoon', 'evening'],
            'service': ['help', 'information', 'assistance'],
            'topic': ['travel', 'booking', 'directions'],
            'action': ['book', 'find', 'get'],
            'process': ['booking', 'planning', 'traveling']
        }

        # Generate variations for each template
        for intent, template_list in templates.items():
            for template in template_list:
                # Find variables in template
                template_vars = re.findall(r'\{(\w+)\}', template)

                if template_vars:
                    # Generate multiple variations
                    for _ in range(2):  # 2 variations per template
                        filled_template = template
                        for var in template_vars:
                            if var in variables:
                                replacement = np.random.choice(variables[var])
                                filled_template = filled_template.replace(f'{{{var}}}', replacement)

                        if filled_template != template:  # Only add if successfully filled
                            synthetic_data.append({
                                'text': filled_template,
                                'intent': intent
                            })
                else:
                    # Template has no variables
                    synthetic_data.append({
                        'text': template,
                        'intent': intent
                    })

        return synthetic_data

    def _load_external_training_data(self) -> List[Dict[str, Any]]:
        """Load external training data from files"""
        external_data = []

        try:
            # Load from intents.json if available
            intents_file = Path("data/training_data/intents.json")
            if intents_file.exists():
                with open(intents_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        external_data.extend(data)
                    elif isinstance(data, dict) and 'intents' in data:
                        external_data.extend(data['intents'])

                logger.info(f"Loaded {len(external_data)} examples from external file")

            # Load from learned intents if available
            learned_file = Path("data/training_data/learned_intents.json")
            if learned_file.exists():
                with open(learned_file, 'r') as f:
                    learned_data = json.load(f)
                    if isinstance(learned_data, list):
                        external_data.extend(learned_data)

                logger.info(f"Loaded additional learned intents")

        except Exception as e:
            logger.warning(f"Error loading external training data: {e}")

        return external_data

    def _save_model(self, accuracy: float, model_name: str):
        """Save trained model and metadata"""
        try:
            # Save model
            model_data = {
                'model': self.model,
                'model_type': model_name,
                'accuracy': accuracy,
                'confidence_threshold': self.confidence_threshold,
                'created_date': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            # Save metadata
            metadata = {
                'model_type': model_name,
                'accuracy': accuracy,
                'confidence_threshold': self.confidence_threshold,
                'created_date': datetime.now().isoformat(),
                'intent_categories': list(self.intent_definitions.keys()),
                'total_intents': len(self.intent_definitions),
                'version': '1.0'
            }

            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved successfully with accuracy: {accuracy:.4f}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def get_intent_probabilities(self, text: str) -> Dict[str, float]:
        """Get probabilities for all intents"""
        if not text or not isinstance(text, str):
            return {}

        try:
            processed_text = self._preprocess_for_classification(text)

            if self.model and hasattr(self.model, 'predict_proba'):
                # Get probabilities from trained model
                probabilities = self.model.predict_proba([processed_text])[0]
                classes = self.model.classes_

                return dict(zip(classes, probabilities))
            else:
                # Fallback to rule-based scoring
                return self._get_rule_based_probabilities(text)

        except Exception as e:
            logger.error(f"Error getting intent probabilities: {e}")
            return {}

    def _get_rule_based_probabilities(self, text: str) -> Dict[str, float]:
        """Get rule-based probability scores for all intents"""
        text_lower = text.lower()
        probabilities = {}

        for intent, definition in self.intent_definitions.items():
            score = 0.0

            # Check keyword matches
            keywords = definition.get('keywords', [])
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_score = keyword_matches / len(keywords) if keywords else 0
            score += keyword_score * 0.4

            # Check pattern matches
            patterns = definition.get('patterns', [])
            pattern_matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            pattern_score = pattern_matches / len(patterns) if patterns else 0
            score += pattern_score * 0.4

            probabilities[intent] = min(score, 0.9)  # Cap at 0.9

        # Normalize probabilities
        total = sum(probabilities.values()) or 1
        normalized_probs = {intent: prob/total for intent, prob in probabilities.items()}

        return normalized_probs

    def classify_with_alternatives(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k intent classifications with confidence scores"""
        probabilities = self.get_intent_probabilities(text)

        if not probabilities:
            return [("unknown", 0.0)]

        # Sort by probability
        sorted_intents = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        return sorted_intents[:top_k]

    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold for classifications"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Confidence threshold updated to {new_threshold}")
        else:
            logger.warning("Confidence threshold must be between 0.0 and 1.0")

    def add_training_example(self, text: str, intent: str) -> bool:
        """Add a new training example for continuous learning"""
        try:
            if intent not in self.intent_definitions:
                logger.warning(f"Unknown intent: {intent}")
                return False

            # Load existing learned examples
            learned_file = Path("data/training_data/learned_intents.json")
            learned_data = []

            if learned_file.exists():
                try:
                    with open(learned_file, 'r') as f:
                        learned_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading learned intents: {e}")

            # Add new example
            new_example = {
                'text': text,
                'intent': intent,
                'added_date': datetime.now().isoformat(),
                'source': 'user_feedback'
            }

            learned_data.append(new_example)

            # Save updated data
            learned_file.parent.mkdir(parents=True, exist_ok=True)
            with open(learned_file, 'w') as f:
                json.dump(learned_data, f, indent=2)

            logger.info(f"Added new training example for intent: {intent}")
            return True

        except Exception as e:
            logger.error(f"Error adding training example: {e}")
            return False

    def retrain_model(self) -> bool:
        """Retrain the model with new examples"""
        try:
            logger.info("Retraining intent classifier with updated data")
            self.train_model()
            return True
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            'model_loaded': self.model is not None,
            'confidence_threshold': self.confidence_threshold,
            'total_intents': len(self.intent_definitions),
            'supported_intents': list(self.intent_definitions.keys())
        }

        # Load metadata if available
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    info.update(metadata)
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")

        return info

    def validate_classification(self, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Validate model performance on test data"""
        if not test_data:
            return {'error': 'No test data provided'}

        results = {
            'total_samples': len(test_data),
            'correct_predictions': 0,
            'accuracy': 0.0,
            'per_intent_accuracy': {},
            'confusion_matrix': defaultdict(lambda: defaultdict(int)),
            'misclassified_examples': []
        }

        intent_counts = defaultdict(int)
        intent_correct = defaultdict(int)

        for item in test_data:
            text = item.get('text', '')
            expected_intent = item.get('intent', '')

            if not text or not expected_intent:
                continue

            predicted_intent, confidence = self.classify_intent(text)

            intent_counts[expected_intent] += 1

            if predicted_intent == expected_intent:
                results['correct_predictions'] += 1
                intent_correct[expected_intent] += 1
            else:
                results['misclassified_examples'].append({
                    'text': text,
                    'expected': expected_intent,
                    'predicted': predicted_intent,
                    'confidence': confidence
                })

            # Update confusion matrix
            results['confusion_matrix'][expected_intent][predicted_intent] += 1

        # Calculate accuracies
        if results['total_samples'] > 0:
            results['accuracy'] = results['correct_predictions'] / results['total_samples']

        for intent in intent_counts:
            if intent_counts[intent] > 0:
                results['per_intent_accuracy'][intent] = intent_correct[intent] / intent_counts[intent]

        return results

    def explain_classification(self, text: str) -> Dict[str, Any]:
        """Explain why a particular classification was made"""
        if not text:
            return {'error': 'No text provided'}

        # Get classification and probabilities
        predicted_intent, confidence = self.classify_intent(text)
        all_probabilities = self.get_intent_probabilities(text)

        # Get rule-based analysis
        text_lower = text.lower()
        rule_analysis = {}

        for intent, definition in self.intent_definitions.items():
            analysis = {
                'keyword_matches': [],
                'pattern_matches': [],
                'score': 0.0
            }

            # Check keywords
            keywords = definition.get('keywords', [])
            for keyword in keywords:
                if keyword in text_lower:
                    analysis['keyword_matches'].append(keyword)

            # Check patterns
            patterns = definition.get('patterns', [])
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    analysis['pattern_matches'].append(pattern)

            # Calculate score
            keyword_score = len(analysis['keyword_matches']) / len(keywords) if keywords else 0
            pattern_score = len(analysis['pattern_matches']) / len(patterns) if patterns else 0
            analysis['score'] = keyword_score * 0.6 + pattern_score * 0.4

            rule_analysis[intent] = analysis

        explanation = {
            'input_text': text,
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'rule_based_analysis': rule_analysis,
            'classification_method': 'model_based' if self.model else 'rule_based'
        }

        return explanation

    def get_similar_intents(self, intent: str) -> List[str]:
        """Get intents similar to the given intent based on keywords"""
        if intent not in self.intent_definitions:
            return []

        target_keywords = set(self.intent_definitions[intent].get('keywords', []))
        similarities = []

        for other_intent, definition in self.intent_definitions.items():
            if other_intent == intent:
                continue

            other_keywords = set(definition.get('keywords', []))

            # Calculate Jaccard similarity
            intersection = len(target_keywords.intersection(other_keywords))
            union = len(target_keywords.union(other_keywords))

            if union > 0:
                similarity = intersection / union
                if similarity > 0.1:  # Minimum similarity threshold
                    similarities.append((other_intent, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [intent for intent, _ in similarities[:3]]  # Top 3 similar intents

# Utility functions for testing and evaluation
def test_intent_classifier():
    """Test the intent classifier with sample inputs"""
    classifier = IntentClassifier()

    test_cases = [
        "What are the best places to visit in Sri Lanka?",
        "How to get from Colombo to Kandy?",
        "Where can I stay in Galle?",
        "What is Sri Lankan food like?",
        "Tell me about Buddhist temples",
        "What's the weather like today?",
        "Exchange rate for USD",
        "Translate hello to Sinhala",
        "Show me Sigiriya on the map",
        "Hello, good morning",
        "Can you help me plan my trip?"
    ]

    print("Intent Classification Test Results:")
    print("=" * 50)

    for text in test_cases:
        intent, confidence = classifier.classify_intent(text)
        alternatives = classifier.classify_with_alternatives(text, top_k=2)

        print(f"\nInput: {text}")
        print(f"Predicted Intent: {intent} (confidence: {confidence:.3f})")

        if len(alternatives) > 1:
            print(f"Alternative: {alternatives[1][0]} (confidence: {alternatives[1][1]:.3f})")

        # Get explanation
        explanation = classifier.explain_classification(text)
        if explanation['rule_based_analysis'][intent]['keyword_matches']:
            print(f"Keyword matches: {explanation['rule_based_analysis'][intent]['keyword_matches']}")

def evaluate_intent_classifier():
    """Evaluate intent classifier performance"""
    classifier = IntentClassifier()

    # Create test dataset
    test_data = [
        {"text": "What places should I visit?", "intent": "destination_inquiry"},
        {"text": "How to travel to Kandy?", "intent": "transportation"},
        {"text": "Good hotels in Colombo", "intent": "accommodation"},
        {"text": "Sri Lankan cuisine", "intent": "food_inquiry"},
        {"text": "Buddhist culture", "intent": "culture_inquiry"},
        {"text": "Weather forecast", "intent": "weather_inquiry"},
        {"text": "Currency exchange", "intent": "currency_inquiry"},
        {"text": "Translate to Tamil", "intent": "translation_request"},
        {"text": "Show directions", "intent": "map_request"},
        {"text": "Hello there", "intent": "general_greeting"},
        {"text": "Need help", "intent": "help_request"}
    ]

    results = classifier.validate_classification(test_data)

    print("Model Validation Results:")
    print("=" * 30)
    print(f"Total samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Correct predictions: {results['correct_predictions']}")

    print("\nPer-intent accuracy:")
    for intent, accuracy in results['per_intent_accuracy'].items():
        print(f"  {intent}: {accuracy:.3f}")

    if results['misclassified_examples']:
        print(f"\nMisclassified examples: {len(results['misclassified_examples'])}")
        for example in results['misclassified_examples'][:3]:  # Show first 3
            print(f"  '{example['text']}' -> Expected: {example['expected']}, Got: {example['predicted']}")

if __name__ == "__main__":
    # Run tests
    test_intent_classifier()
    print("\n" + "="*60 + "\n")
    evaluate_intent_classifier()