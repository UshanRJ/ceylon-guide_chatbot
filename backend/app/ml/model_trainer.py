# backend/app/ml/model_trainer.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import joblib
import warnings
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, learning_curve, validation_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier
)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Advanced model training system with hyperparameter optimization,
    ensemble methods, and comprehensive evaluation metrics
    """

    def __init__(self):
        self.models_dir = Path("app/ml/models")
        self.training_data_dir = Path("data/training_data")
        print(f"Resolved path: {self.training_data_dir.resolve()}")
        print(f"Exists: {self.training_data_dir.exists()}")
        self.results_dir = Path("app/ml/results")
        self.experiments_dir = Path("app/ml/experiments")

        # Create directories
        for directory in [self.models_dir, self.results_dir, self.experiments_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize NLP tools
        self._initialize_nlp_tools()

        # Training configuration
        self.training_config = {
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'scoring_metrics': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            'hyperparameter_optimization': True,
            'ensemble_methods': True,
            'feature_selection': True,
            'text_preprocessing': True
        }

        # Model configurations
        self.model_configs = self._get_model_configurations()

        logger.info("ModelTrainer initialized with advanced ML capabilities")

    def _load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from JSON files in the training_data_dir"""
        all_data = []
        try:
            if not self.training_data_dir.exists():
                logger.warning(f"Training data directory not found: {self.training_data_dir}")
                return []

            json_files = list(self.training_data_dir.glob("*.json"))
            if not json_files:
                logger.warning(f"No JSON training data files found in {self.training_data_dir}")
                return []

            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        logger.warning(f"Skipping malformed training data file: {json_file}. Expected a list of dictionaries.")

            logger.info(f"Loaded {len(all_data)} samples from {len(json_files)} training data files.")
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            all_data = [] # Ensure all_data is empty on error
        return all_data

    def _initialize_nlp_tools(self):
        """Initialize NLP tools and download required data"""
        try:
            # Download NLTK data
            nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for item in nltk_downloads:
                try:
                    nltk.download(item, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download NLTK {item}: {e}")

            # Initialize spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Some features may be limited.")
                self.nlp = None

            # Initialize other NLP tools
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()

            try:
                self.stop_words = set(stopwords.words('english'))
            except Exception:
                self.stop_words = set()

        except Exception as e:
            logger.error(f"Error initializing NLP tools: {e}")

    def _get_model_configurations(self) -> Dict[str, Dict]:
        """Get comprehensive model configurations with hyperparameters"""
        return {
            'svm_linear': {
                'model': SVC,
                'params': {
                    'kernel': ['linear'],
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto'],
                    'probability': [True],
                    'random_state': [42],
                    'max_iter': [2000]
                },
                'description': 'Support Vector Machine with Linear Kernel'
            },
            'svm_rbf': {
                'model': SVC,
                'params': {
                    'kernel': ['rbf'],
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                    'probability': [True],
                    'random_state': [42],
                    'max_iter': [2000]
                },
                'description': 'Support Vector Machine with RBF Kernel'
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False],
                    'random_state': [42]
                },
                'description': 'Random Forest Classifier'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.8, 0.9, 1.0],
                    'random_state': [42]
                },
                'description': 'Gradient Boosting Classifier'
            },
            'naive_bayes_multinomial': {
                'model': MultinomialNB,
                'params': {
                    'alpha': [0.1, 0.5, 1.0],
                    'fit_prior': [True, False]
                },
                'description': 'Multinomial Naive Bayes'
            },
            'naive_bayes_complement': {
                'model': ComplementNB,
                'params': {
                    'alpha': [0.1, 0.5, 1.0],
                    'norm': [True, False]
                },
                'description': 'Complement Naive Bayes'
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                    'max_iter': [3000],
                    'random_state': [42]
                },
                'description': 'Logistic Regression'
            },
            'sgd_classifier': {
                'model': SGDClassifier,
                'params': {
                    'loss': ['hinge', 'log_loss',],
                    'penalty': ['l2'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['optimal'],
                    'max_iter': [2000],
                    'random_state': [42]
                },
                'description': 'Stochastic Gradient Descent Classifier'
            },
            'knn': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['brute'],
                    'metric': [ 'cosine', 'euclidean']
                },
                'description': 'K-Nearest Neighbors'
            },
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'random_state': [42]
                },
                'description': 'Decision Tree Classifier'
            },
            'mlp_neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,)],
                    'activation': ['relu'],
                    'solver': ['adam'],
                    'alpha': [0.0001, 0.001],
                    'learning_rate': ['adaptive'],
                    'max_iter': [2000],
                    'random_state': [42],
                    'early_stopping': [True],
                    'validation_fraction': [0.1]
                },
                'description': 'Multi-layer Perceptron Neural Network'
            }
        }

    def train_intent_classifier(self, training_data: List[Dict[str, Any]] = None,
                                experiment_name: str = None) -> Dict[str, Any]:
        """
        Train intent classification model with comprehensive evaluation and optimization
        """
        try:
            logger.info("Starting comprehensive intent classifier training")

            # Load and prepare training data
            if not training_data:
                training_data = self._load_training_data()

            if not training_data or len(training_data) < 10:
                return {"success": False, "error": "Insufficient training data"}

            # Create experiment directory
            experiment_name = experiment_name or f"intent_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_dir = self.experiments_dir / experiment_name
            experiment_dir.mkdir(exist_ok=True)

            # Prepare data
            X, y = self._prepare_training_data(training_data)

            # Data analysis
            data_analysis = self._analyze_training_data(X, y)
            logger.info(f"Training data analysis: {data_analysis}")

            # ADDED: Encode labels to integers for models that require it
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,  # Use encoded labels
                test_size=self.training_config['test_size'],
                random_state=self.training_config['random_state'],
                stratify=y_encoded  # Use encoded labels for stratification
            )

            # Text preprocessing and feature extraction
            feature_pipeline = self._create_feature_pipeline()
            X_train_features = feature_pipeline.fit_transform(X_train)
            X_test_features = feature_pipeline.transform(X_test)

            # Feature selection
            if self.training_config['feature_selection']:
                feature_selector = self._create_feature_selector()
                X_train_features = feature_selector.fit_transform(X_train_features, y_train)
                X_test_features = feature_selector.transform(X_test_features)
            else:
                feature_selector = None

            # Train multiple models
            model_results = {}
            best_model = None
            best_score = 0

            for model_name, config in self.model_configs.items():
                logger.info(f"Training {model_name}: {config['description']}")

                try:
                    # Hyperparameter optimization
                    if self.training_config['hyperparameter_optimization']:
                        model_result = self._train_with_hyperparameter_optimization(
                            config, X_train_features, y_train, X_test_features, y_test
                        )
                    else:
                        model_result = self._train_basic_model(
                            config, X_train_features, y_train, X_test_features, y_test
                        )

                    model_results[model_name] = model_result

                    # Track best model
                    if model_result['cv_score'] > best_score:
                        best_score = model_result['cv_score']
                        best_model = {
                            'name': model_name,
                            'model': model_result['model'],
                            'pipeline': feature_pipeline,
                            'feature_selector': feature_selector,
                            'label_encoder': self.label_encoder
                        }

                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    model_results[model_name] = {'error': str(e)}

            # Ensemble methods
            ensemble_results = {}
            if self.training_config['ensemble_methods'] and len([r for r in model_results.values() if 'model' in r]) >= 3:
                try:
                    ensemble_results = self._create_ensemble_models(
                        model_results, X_train_features, y_train, X_test_features, y_test
                    )

                    # Check if ensemble is better
                    for ensemble_name, ensemble_result in ensemble_results.items():
                        if ensemble_result['cv_score'] > best_score:
                            best_score = ensemble_result['cv_score']
                            best_model = {
                                'name': ensemble_name,
                                'model': ensemble_result['model'],
                                'pipeline': feature_pipeline,
                                'feature_selector': feature_selector,
                                'label_encoder': self.label_encoder
                            }
                except Exception as e:
                    logger.error(f"Error creating ensemble models: {e}")

            # Comprehensive evaluation of best model
            if best_model:
                evaluation_results = self._comprehensive_evaluation(
                    best_model, X_train_features, y_train, X_test_features, y_test
                )

                # Save best model
                model_path = self.models_dir / "intent_classifier.pkl"
                self._save_complete_model(best_model, model_path)

                # Save model metadata
                metadata = self._create_model_metadata(
                    best_model, evaluation_results, data_analysis, training_data
                )
                metadata_path = self.models_dir / "intent_classifier_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

                # Save experiment results
                experiment_results = {
                    'experiment_name': experiment_name,
                    'timestamp': datetime.now().isoformat(),
                    'best_model': best_model['name'],
                    'best_score': best_score,
                    'all_models': model_results,
                    'ensemble_models': ensemble_results,
                    'evaluation': evaluation_results,
                    'data_analysis': data_analysis,
                    'training_config': self.training_config
                }

                experiment_file = experiment_dir / "results.json"
                with open(experiment_file, 'w') as f:
                    json.dump(experiment_results, f, indent=2, default=str)

                # Generate detailed report
                self._generate_training_report(experiment_results, experiment_dir)

                logger.info(f"Training completed. Best model: {best_model['name']} with CV score: {best_score:.4f}")

                return {
                    "success": True,
                    "best_model": best_model['name'],
                    "accuracy": evaluation_results['test_accuracy'],
                    "cv_score": best_score,
                    "experiment_name": experiment_name,
                    "results": experiment_results
                }
            else:
                return {"success": False, "error": "No successful models trained"}

        except Exception as e:
            logger.error(f"Error in intent classifier training: {e}")
            return {"success": False, "error": str(e)}

    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Prepare training data for model training"""
        X = []
        y = []

        for item in training_data:
            if 'text' in item and 'intent' in item:
                X.append(str(item['text']))
                y.append(str(item['intent']))

        return X, y

    def _analyze_training_data(self, X: List[str], y: List[str]) -> Dict[str, Any]:
        """Analyze training data characteristics"""
        analysis = {
            'total_samples': len(X),
            'unique_intents': len(set(y)),
            'intent_distribution': dict(Counter(y)),
            'avg_text_length': np.mean([len(text) for text in X]),
            'text_length_std': np.std([len(text) for text in X]),
            'vocab_size': len(set(' '.join(X).split())),
            'class_balance': None
        }

        # Check class balance
        intent_counts = Counter(y)
        min_count = min(intent_counts.values())
        max_count = max(intent_counts.values())
        balance_ratio = min_count / max_count if max_count > 0 else 0

        if balance_ratio > 0.8:
            analysis['class_balance'] = 'balanced'
        elif balance_ratio > 0.5:
            analysis['class_balance'] = 'moderately_imbalanced'
        else:
            analysis['class_balance'] = 'highly_imbalanced'

        return analysis

    def _create_feature_pipeline(self) -> Pipeline:
        """Create comprehensive feature extraction pipeline"""
        steps = []

        # Text preprocessing
        if self.training_config['text_preprocessing']:
            steps.append(('preprocessor', TextPreprocessor(
                lowercase=True,
                remove_punctuation=True,
                remove_stopwords=True,
                lemmatize=True,
                stop_words=self.stop_words,
                lemmatizer=self.lemmatizer
            )))

        # Feature extraction - using both TF-IDF and Count features
        steps.append(('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                use_idf=True
            )),
            ('count', CountVectorizer(
                max_features=1000,
                ngram_range=(1, 1),
                min_df=1,
                max_df=0.95,
                binary=True
            ))
        ])))

        # Add feature scaling for better convergence
        steps.append(('scaler', StandardScaler(with_mean=False)))  # sparse-safe scaling

        return Pipeline(steps)

    def _create_feature_selector(self):
        """Create feature selector for dimensionality reduction"""
        return SelectKBest(score_func=chi2, k='all')

    def _train_with_hyperparameter_optimization(self, config: Dict, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train model with hyperparameter optimization using GridSearchCV"""
        try:
            # Create model
            model = config['model']()

            # Calculate appropriate number of CV folds based on smallest class size
            from collections import Counter
            class_counts = Counter(y_train)
            min_class_size = min(class_counts.values())

            # Use smaller number of folds if data is limited
            cv_folds = min(self.training_config['cv_folds'], min_class_size, 3)

            if cv_folds < 2:
                cv_folds = 2  # Minimum for cross-validation

            logger.info(f"Using {cv_folds} CV folds (min class size: {min_class_size})")

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model,
                config['params'],
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )

            # Fit grid search
            grid_search.fit(X_train, y_train)

            # Get best model
            best_model = grid_search.best_estimator_

            # Cross-validation score with same CV folds
            cv_scores = cross_val_score(
                best_model, X_train, y_train,
                cv=cv_folds,
                scoring='f1_macro'
            )

            # Test predictions
            y_pred = best_model.predict(X_test)

            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='macro')

            return {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'cv_folds_used': cv_folds,
                'grid_search_results': {
                    'best_score': grid_search.best_score_,
                    'total_fits': len(grid_search.cv_results_['mean_test_score'])
                }
            }

        except Exception as e:
            raise Exception(f"Hyperparameter optimization failed: {e}")

    def _train_basic_model(self, config: Dict, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train model with default parameters"""
        try:
            # Create model with default parameters
            model = config['model']()

            # Train model
            model.fit(X_train, y_train)

            # Calculate appropriate number of CV folds based on smallest class size
            from collections import Counter
            class_counts = Counter(y_train)
            min_class_size = min(class_counts.values())
            cv_folds = min(self.training_config['cv_folds'], min_class_size, 3)

            if cv_folds < 2:
                cv_folds = 2

            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='f1_macro'
            )

            # Test predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='macro')

            return {
                'model': model,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'cv_folds_used': cv_folds
            }

        except Exception as e:
            raise Exception(f"Basic training failed: {e}")


    def _create_ensemble_models(self, model_results: Dict, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Create ensemble models from trained individual models"""
        ensemble_results = {}

        try:
            # Get successful models
            successful_models = [(name, result) for name, result in model_results.items()
                                 if 'model' in result]

            if len(successful_models) < 3:
                return ensemble_results

            # Sort by performance
            successful_models.sort(key=lambda x: x[1]['cv_score'], reverse=True)

            # Take top models for ensemble
            top_models = successful_models[:5]  # Top 5 models

            # Calculate appropriate CV folds
            from collections import Counter
            class_counts = Counter(y_train)
            min_class_size = min(class_counts.values())
            cv_folds = min(3, min_class_size)

            if cv_folds < 2:
                cv_folds = 2

            # Voting Classifier (Hard Voting)
            voting_models = [(name, result['model']) for name, result in top_models]
            voting_classifier = VotingClassifier(estimators=voting_models, voting='hard')

            # Train and evaluate voting classifier
            voting_classifier.fit(X_train, y_train)
            cv_scores = cross_val_score(voting_classifier, X_train, y_train,
                                        cv=cv_folds, scoring='f1_macro')
            y_pred = voting_classifier.predict(X_test)

            ensemble_results['voting_hard'] = {
                'model': voting_classifier,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_f1': f1_score(y_test, y_pred, average='macro'),
                'component_models': [name for name, _ in top_models],
                'cv_folds_used': cv_folds
            }

            # Voting Classifier (Soft Voting) - for models with predict_proba
            prob_models = [(name, result['model']) for name, result in top_models
                           if hasattr(result['model'], 'predict_proba')]

            if len(prob_models) >= 3:
                soft_voting_classifier = VotingClassifier(estimators=prob_models, voting='soft')
                soft_voting_classifier.fit(X_train, y_train)
                cv_scores = cross_val_score(soft_voting_classifier, X_train, y_train,
                                            cv=cv_folds, scoring='f1_macro')
                y_pred = soft_voting_classifier.predict(X_test)

                ensemble_results['voting_soft'] = {
                    'model': soft_voting_classifier,
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_f1': f1_score(y_test, y_pred, average='macro'),
                    'component_models': [name for name, _ in prob_models],
                    'cv_folds_used': cv_folds
                }

            # Bagging Ensemble using best model
            best_model_name, best_model_result = top_models[0]
            bagging_classifier = BaggingClassifier(
                estimator=best_model_result['model'],
                n_estimators=10,
                random_state=42
            )

            bagging_classifier.fit(X_train, y_train)
            cv_scores = cross_val_score(bagging_classifier, X_train, y_train,
                                        cv=cv_folds, scoring='f1_macro')
            y_pred = bagging_classifier.predict(X_test)

            ensemble_results['bagging'] = {
                'model': bagging_classifier,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_f1': f1_score(y_test, y_pred, average='macro'),
                'base_model': best_model_name,
                'cv_folds_used': cv_folds
            }

            logger.info(f"Created {len(ensemble_results)} ensemble models")
            return ensemble_results

        except Exception as e:
            logger.error(f"Error creating ensemble models: {e}")
            return {}

    def _comprehensive_evaluation(self, best_model: Dict, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the best model"""
        try:
            model = best_model['model']
            label_encoder = best_model.get('label_encoder')

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Basic metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Convert back to string labels for reporting if label encoder exists
            if label_encoder:
                y_train_str = label_encoder.inverse_transform(y_train)
                y_test_str = label_encoder.inverse_transform(y_test)
                y_train_pred_str = label_encoder.inverse_transform(y_train_pred)
                y_test_pred_str = label_encoder.inverse_transform(y_test_pred)
            else:
                y_train_str = y_train
                y_test_str = y_test
                y_train_pred_str = y_train_pred
                y_test_pred_str = y_test_pred

            # Classification report
            test_report = classification_report(y_test_str, y_test_pred_str, output_dict=True)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)

            # Additional metrics
            f1_macro = f1_score(y_test, y_test_pred, average='macro')
            f1_micro = f1_score(y_test, y_test_pred, average='micro')
            f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

            # Calculate appropriate CV folds for evaluation
            from collections import Counter
            class_counts = Counter(y_train)
            min_class_size = min(class_counts.values())
            cv_folds = min(3, min_class_size)  # Use 3 or smaller for evaluation

            if cv_folds < 2:
                cv_folds = 2

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv_folds,
                                        scoring='f1_macro')

            # Learning curves (with appropriate CV)
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=cv_folds,
                train_sizes=np.linspace(0.1, 1.0, 5),  # Reduced from 10 to 5
                scoring='f1_macro'
            )

            evaluation = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'f1_weighted': f1_weighted,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_folds_used': cv_folds,
                'classification_report': test_report,
                'confusion_matrix': cm.tolist(),
                'learning_curve': {
                    'train_sizes': train_sizes.tolist(),
                    'train_scores_mean': train_scores.mean(axis=1).tolist(),
                    'train_scores_std': train_scores.std(axis=1).tolist(),
                    'val_scores_mean': val_scores.mean(axis=1).tolist(),
                    'val_scores_std': val_scores.std(axis=1).tolist()
                }
            }

            # Probability-based metrics (if available)
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)

                    # ROC AUC (for multi-class, use ovr)
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                    evaluation['roc_auc_macro'] = roc_auc

                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            # Cohen's Kappa
            evaluation['cohen_kappa'] = cohen_kappa_score(y_test, y_test_pred)

            # Matthews Correlation Coefficient
            try:
                evaluation['matthews_corrcoef'] = matthews_corrcoef(y_test, y_test_pred)
            except Exception:
                pass  # MCC may not work for multi-class

            return evaluation

        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            return {}

    def _save_complete_model(self, best_model: Dict, model_path: Path):
        """Save complete model with pipeline and feature selector"""
        try:
            complete_model = {
                'model': best_model['model'],
                'pipeline': best_model['pipeline'],
                'feature_selector': best_model['feature_selector'],
                'model_name': best_model['name'],
                'saved_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(model_path, 'wb') as f:
                pickle.dump(complete_model, f)

            logger.info(f"Complete model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error saving complete model: {e}")
            raise e

    def _create_model_metadata(self, best_model: Dict, evaluation: Dict,
                               data_analysis: Dict, training_data: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive model metadata"""
        return {
            'model_info': {
                'name': best_model['name'],
                'type': 'intent_classifier',
                'algorithm': str(type(best_model['model']).__name__),
                'version': '1.0',
                'created_date': datetime.now().isoformat()
            },
            'training_info': {
                'total_samples': len(training_data),
                'training_config': self.training_config,
                'data_analysis': data_analysis
            },
            'performance_metrics': {
                'accuracy': evaluation.get('test_accuracy', 0),
                'f1_macro': evaluation.get('f1_macro', 0),
                'cv_score': evaluation.get('cv_mean', 0),
                'cv_std': evaluation.get('cv_std', 0),
                'cohen_kappa': evaluation.get('cohen_kappa', 0)
            },
            'feature_info': {
                'pipeline_steps': str(best_model['pipeline'].steps) if best_model['pipeline'] else None,
                'feature_selector': str(best_model['feature_selector']) if best_model['feature_selector'] else None
            },
            'deployment_info': {
                'model_file': 'intent_classifier.pkl',
                'dependencies': {
                    'scikit-learn': '1.3.0',
                    'numpy': '1.24.0',
                    'nltk': '3.8',
                    'spacy': '3.7.0'
                }
            }
        }

    def _generate_training_report(self, results: Dict[str, Any], experiment_dir: Path):
        """Generate detailed training report"""
        try:
            report_lines = []

            # Header
            report_lines.extend([
                "=" * 80,
                "INTENT CLASSIFIER TRAINING REPORT",
                "=" * 80,
                f"Experiment: {results['experiment_name']}",
                f"Timestamp: {results['timestamp']}",
                f"Best Model: {results['best_model']}",
                f"Best Score: {results['best_score']:.4f}",
                ""
            ])

            # Data Analysis
            data_analysis = results['data_analysis']
            report_lines.extend([
                "DATA ANALYSIS",
                "-" * 40,
                f"Total Samples: {data_analysis['total_samples']}",
                f"Unique Intents: {data_analysis['unique_intents']}",
                f"Class Balance: {data_analysis['class_balance']}",
                f"Average Text Length: {data_analysis['avg_text_length']:.2f}",
                f"Vocabulary Size: {data_analysis['vocab_size']}",
                ""
            ])

            # Intent Distribution
            report_lines.append("INTENT DISTRIBUTION")
            report_lines.append("-" * 40)
            for intent, count in sorted(data_analysis['intent_distribution'].items()):
                percentage = (count / data_analysis['total_samples']) * 100
                report_lines.append(f"{intent}: {count} ({percentage:.1f}%)")
            report_lines.append("")

            # Model Performance Comparison
            report_lines.extend([
                "MODEL PERFORMANCE COMPARISON",
                "-" * 40,
                f"{'Model':<25} {'CV Score':<12} {'Test Acc':<12} {'Test F1':<12}"
            ])

            for model_name, result in results['all_models'].items():
                if 'error' not in result:
                    cv_score = result.get('cv_score', 0)
                    test_acc = result.get('test_accuracy', 0)
                    test_f1 = result.get('test_f1', 0)
                    report_lines.append(f"{model_name:<25} {cv_score:<12.4f} {test_acc:<12.4f} {test_f1:<12.4f}")
                else:
                    report_lines.append(f"{model_name:<25} ERROR: {result['error']}")

            report_lines.append("")

            # Ensemble Results
            if results['ensemble_models']:
                report_lines.extend([
                    "ENSEMBLE MODEL RESULTS",
                    "-" * 40
                ])
                for ensemble_name, result in results['ensemble_models'].items():
                    cv_score = result.get('cv_score', 0)
                    test_acc = result.get('test_accuracy', 0)
                    test_f1 = result.get('test_f1', 0)
                    report_lines.append(f"{ensemble_name:<25} {cv_score:<12.4f} {test_acc:<12.4f} {test_f1:<12.4f}")
                report_lines.append("")

            # Best Model Details
            evaluation = results['evaluation']
            report_lines.extend([
                "BEST MODEL EVALUATION",
                "-" * 40,
                f"Training Accuracy: {evaluation.get('train_accuracy', 0):.4f}",
                f"Test Accuracy: {evaluation.get('test_accuracy', 0):.4f}",
                f"F1 Macro: {evaluation.get('f1_macro', 0):.4f}",
                f"F1 Micro: {evaluation.get('f1_micro', 0):.4f}",
                f"F1 Weighted: {evaluation.get('f1_weighted', 0):.4f}",
                f"Cross-Validation Mean: {evaluation.get('cv_mean', 0):.4f}",
                f"Cross-Validation Std: {evaluation.get('cv_std', 0):.4f}",
                f"Cohen's Kappa: {evaluation.get('cohen_kappa', 0):.4f}",
                ""
            ])

            # Classification Report
            if 'classification_report' in evaluation:
                report_lines.extend([
                    "DETAILED CLASSIFICATION REPORT",
                    "-" * 40
                ])

                class_report = evaluation['classification_report']
                for intent, metrics in class_report.items():
                    if intent not in ['accuracy', 'macro avg', 'weighted avg']:
                        if isinstance(metrics, dict):
                            precision = metrics.get('precision', 0)
                            recall = metrics.get('recall', 0)
                            f1 = metrics.get('f1-score', 0)
                            support = metrics.get('support', 0)
                            report_lines.append(f"{intent:<20} P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f} S:{support}")

                report_lines.append("")

            # Training Configuration
            report_lines.extend([
                "TRAINING CONFIGURATION",
                "-" * 40
            ])
            for key, value in results['training_config'].items():
                report_lines.append(f"{key}: {value}")

            # Save report
            report_file = experiment_dir / "training_report.txt"
            with open(report_file, 'w') as f:
                f.write('\n'.join(report_lines))

            logger.info(f"Training report saved to {report_file}")

        except Exception as e:
            logger.error(f"Error generating training report: {e}")

    def train_entity_extraction_model(self) -> Dict[str, Any]:
        """Train custom entity extraction model (placeholder for advanced implementation)"""
        try:
            logger.info("Training entity extraction model")

            # For now, we're using rule-based entity extraction
            # In a more advanced implementation, you could train NER models here

            # Save entity patterns and rules
            entity_rules = {
                'version': '1.0',
                'created_date': datetime.now().isoformat(),
                'sri_lankan_locations': [
                    'colombo', 'kandy', 'galle', 'jaffna', 'negombo', 'ella', 'sigiriya',
                    'anuradhapura', 'polonnaruwa', 'trincomalee', 'batticaloa', 'matara',
                    'badulla', 'ratnapura', 'kurunegala', 'puttalam', 'kalutara', 'gampaha',
                    'nuwara eliya', 'haputale', 'bandarawela', 'tissamaharama', 'hambantota'
                ],
                'currency_patterns': {
                    'usd': [r'\$\d+(?:\.\d{2})?', r'\busd\b', r'\bdollars?\b'],
                    'eur': [r'€\d+(?:\.\d{2})?', r'\beur\b', r'\beuros?\b'],
                    'gbp': [r'£\d+(?:\.\d{2})?', r'\bgbp\b', r'\bpounds?\b'],
                    'lkr': [r'rs\.?\s*\d+(?:\.\d{2})?', r'\blkr\b', r'\brupees?\b'],
                    'inr': [r'₹\d+(?:\.\d{2})?', r'\binr\b', r'\bindian\s+rupees?\b']
                },
                'language_patterns': ['sinhala', 'tamil', 'english', 'sinhalese'],
                'time_patterns': {
                    'today': r'\btoday\b',
                    'tomorrow': r'\btomorrow\b',
                    'yesterday': r'\byesterday\b',
                    'now': r'\bnow\b|\bcurrent\b',
                    'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                    'month': r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
                },
                'tourism_entities': {
                    'attractions': [
                        'temple', 'fort', 'beach', 'mountain', 'waterfall', 'park', 'museum',
                        'palace', 'monastery', 'dagoba', 'stupa', 'rock', 'cave', 'garden'
                    ],
                    'activities': [
                        'safari', 'trekking', 'surfing', 'diving', 'whale watching', 'bird watching',
                        'tea plantation tour', 'cultural show', 'cooking class', 'meditation'
                    ],
                    'transportation_types': [
                        'bus', 'train', 'taxi', 'tuk tuk', 'car', 'van', 'motorbike', 'bicycle',
                        'boat', 'flight', 'helicopter'
                    ]
                }
            }

            rules_path = self.models_dir / "entity_rules.json"
            with open(rules_path, 'w') as f:
                json.dump(entity_rules, f, indent=2)

            # Create training metrics
            training_metrics = {
                'model_type': 'entity_extraction',
                'approach': 'rule_based',
                'rules_count': sum(len(v) if isinstance(v, list) else len(v.values()) if isinstance(v, dict) else 1
                                   for v in entity_rules.values() if v != entity_rules['version'] and v != entity_rules['created_date']),
                'categories': list(entity_rules.keys()),
                'performance_estimate': 0.85,  # Estimated based on rule coverage
                'last_updated': datetime.now().isoformat()
            }

            metrics_path = self.models_dir / "entity_extraction_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(training_metrics, f, indent=2)

            return {
                "success": True,
                "message": "Entity extraction rules updated successfully",
                "rules_count": training_metrics['rules_count'],
                "categories": training_metrics['categories'],
                "approach": "rule_based"
            }

        except Exception as e:
            logger.error(f"Error training entity extraction model: {e}")
            return {"success": False, "error": str(e)}

    def _load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from files"""
        training_data = []

        try:
            # Load base training data
            base_intents_path = self.training_data_dir / "intents.json"
            if base_intents_path.exists():
                with open(base_intents_path, 'r') as f:
                    base_data = json.load(f)
                    training_data.extend(base_data)

            # Load learned training data
            learned_intents_path = self.training_data_dir / "learned_intents.json"
            if learned_intents_path.exists():
                with open(learned_intents_path, 'r') as f:
                    learned_data = json.load(f)
                    training_data.extend(learned_data)

            # Generate synthetic training data if needed
            if len(training_data) < 100:
                synthetic_data = self._generate_synthetic_training_data()
                training_data.extend(synthetic_data)

            # Remove duplicates
            seen = set()
            unique_data = []
            for item in training_data:
                key = (item.get('text', ''), item.get('intent', ''))
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)

            logger.info(f"Loaded {len(unique_data)} unique training samples")
            return unique_data

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return []

    def _generate_synthetic_training_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data for better model performance"""
        synthetic_data = [
            # Destination inquiries
            {"text": "What places should I visit in Sri Lanka?", "intent": "destination_inquiry"},
            {"text": "Show me tourist attractions in Colombo", "intent": "destination_inquiry"},
            {"text": "Best destinations in Sri Lanka for families", "intent": "destination_inquiry"},
            {"text": "Where to go sightseeing in Kandy?", "intent": "destination_inquiry"},
            {"text": "Tourist spots near Galle", "intent": "destination_inquiry"},
            {"text": "What are the must-see places in Sri Lanka?", "intent": "destination_inquiry"},
            {"text": "Recommend some beautiful locations", "intent": "destination_inquiry"},
            {"text": "Top 10 places to visit", "intent": "destination_inquiry"},
            {"text": "Historic sites in Sri Lanka", "intent": "destination_inquiry"},
            {"text": "Natural attractions worth visiting", "intent": "destination_inquiry"},

            # Transportation
            {"text": "How to travel from Kandy to Ella?", "intent": "transportation"},
            {"text": "Bus routes in Sri Lanka", "intent": "transportation"},
            {"text": "Train schedule to Galle", "intent": "transportation"},
            {"text": "Best way to get around", "intent": "transportation"},
            {"text": "Transportation options available", "intent": "transportation"},
            {"text": "How much does a taxi cost?", "intent": "transportation"},
            {"text": "Is there a bus to Sigiriya?", "intent": "transportation"},
            {"text": "Train from Colombo to Kandy", "intent": "transportation"},
            {"text": "Tuk tuk prices in Sri Lanka", "intent": "transportation"},
            {"text": "Airport transfer options", "intent": "transportation"},

            # Accommodation
            {"text": "Hotels in Colombo", "intent": "accommodation"},
            {"text": "Where to stay in Kandy?", "intent": "accommodation"},
            {"text": "Budget accommodation options", "intent": "accommodation"},
            {"text": "Best resorts in Sri Lanka", "intent": "accommodation"},
            {"text": "Guesthouse recommendations", "intent": "accommodation"},
            {"text": "Luxury hotels near beach", "intent": "accommodation"},
            {"text": "Cheap places to stay", "intent": "accommodation"},
            {"text": "Booking accommodation in Ella", "intent": "accommodation"},
            {"text": "Homestay options", "intent": "accommodation"},
            {"text": "Hotels with good reviews", "intent": "accommodation"},

            # Food inquiries
            {"text": "What is Sri Lankan cuisine like?", "intent": "food_inquiry"},
            {"text": "Best restaurants in Colombo", "intent": "food_inquiry"},
            {"text": "Local food to try", "intent": "food_inquiry"},
            {"text": "Where to eat good curry?", "intent": "food_inquiry"},
            {"text": "Traditional Sri Lankan dishes", "intent": "food_inquiry"},
            {"text": "Vegetarian food options", "intent": "food_inquiry"},
            {"text": "Street food in Sri Lanka", "intent": "food_inquiry"},
            {"text": "What should I eat for breakfast?", "intent": "food_inquiry"},
            {"text": "Seafood restaurants near beach", "intent": "food_inquiry"},
            {"text": "Spicy food recommendations", "intent": "food_inquiry"},

            # Culture inquiries
            {"text": "Tell me about Sri Lankan culture", "intent": "culture_inquiry"},
            {"text": "What are the main festivals?", "intent": "culture_inquiry"},
            {"text": "Buddhist temples to visit", "intent": "culture_inquiry"},
            {"text": "Cultural shows in Kandy", "intent": "culture_inquiry"},
            {"text": "History of Sri Lanka", "intent": "culture_inquiry"},
            {"text": "Traditional dances", "intent": "culture_inquiry"},
            {"text": "Religious practices", "intent": "culture_inquiry"},
            {"text": "Local customs and traditions", "intent": "culture_inquiry"},
            {"text": "Art and handicrafts", "intent": "culture_inquiry"},
            {"text": "Museums worth visiting", "intent": "culture_inquiry"},

            # Weather inquiries
            {"text": "What's the weather like?", "intent": "weather_inquiry"},
            {"text": "Weather forecast for Colombo", "intent": "weather_inquiry"},
            {"text": "Is it raining in Kandy?", "intent": "weather_inquiry"},
            {"text": "Temperature today", "intent": "weather_inquiry"},
            {"text": "Climate in Sri Lanka", "intent": "weather_inquiry"},
            {"text": "Best time to visit", "intent": "weather_inquiry"},
            {"text": "Monsoon season", "intent": "weather_inquiry"},
            {"text": "Weather in hill country", "intent": "weather_inquiry"},
            {"text": "Sunny weather forecast", "intent": "weather_inquiry"},
            {"text": "When does it rain?", "intent": "weather_inquiry"},

            # Currency inquiries
            {"text": "Exchange rate USD to LKR", "intent": "currency_inquiry"},
            {"text": "Convert 100 dollars to rupees", "intent": "currency_inquiry"},
            {"text": "What's the current exchange rate?", "intent": "currency_inquiry"},
            {"text": "Currency conversion", "intent": "currency_inquiry"},
            {"text": "How much is 50 euros in LKR?", "intent": "currency_inquiry"},
            {"text": "Money exchange in Colombo", "intent": "currency_inquiry"},
            {"text": "Bank rates today", "intent": "currency_inquiry"},
            {"text": "Cost of living in Sri Lanka", "intent": "currency_inquiry"},
            {"text": "ATM availability", "intent": "currency_inquiry"},
            {"text": "Credit card acceptance", "intent": "currency_inquiry"},

            # Translation requests
            {"text": "Translate hello to Sinhala", "intent": "translation_request"},
            {"text": "How do you say thank you in Tamil?", "intent": "translation_request"},
            {"text": "What does Ayubowan mean?", "intent": "translation_request"},
            {"text": "Translate to English", "intent": "translation_request"},
            {"text": "Sinhala translation please", "intent": "translation_request"},
            {"text": "How to say goodbye in local language", "intent": "translation_request"},
            {"text": "Tamil phrases for tourists", "intent": "translation_request"},
            {"text": "Common Sinhala words", "intent": "translation_request"},
            {"text": "Language help needed", "intent": "translation_request"},
            {"text": "Translate this text", "intent": "translation_request"},

            # Map requests
            {"text": "Show me Sigiriya on map", "intent": "map_request"},
            {"text": "Directions to Galle Fort", "intent": "map_request"},
            {"text": "Where is Temple of Tooth?", "intent": "map_request"},
            {"text": "Map of Colombo", "intent": "map_request"},
            {"text": "Location of Ella Rock", "intent": "map_request"},
            {"text": "GPS coordinates needed", "intent": "map_request"},
            {"text": "How to reach this place?", "intent": "map_request"},
            {"text": "Distance between cities", "intent": "map_request"},
            {"text": "Route planning help", "intent": "map_request"},
            {"text": "Navigation assistance", "intent": "map_request"},

            # General greetings
            {"text": "Hello", "intent": "general_greeting"},
            {"text": "Hi there", "intent": "general_greeting"},
            {"text": "Good morning", "intent": "general_greeting"},
            {"text": "Thank you", "intent": "general_greeting"},
            {"text": "Thanks for your help", "intent": "general_greeting"},
            {"text": "Goodbye", "intent": "general_greeting"},
            {"text": "Nice to meet you", "intent": "general_greeting"},
            {"text": "Have a great day", "intent": "general_greeting"},
            {"text": "Appreciate your assistance", "intent": "general_greeting"},
            {"text": "See you later", "intent": "general_greeting"},

            # Help requests
            {"text": "Can you help me?", "intent": "help_request"},
            {"text": "I need assistance", "intent": "help_request"},
            {"text": "What can you do?", "intent": "help_request"},
            {"text": "How does this work?", "intent": "help_request"},
            {"text": "Please guide me", "intent": "help_request"},
            {"text": "I'm confused", "intent": "help_request"},
            {"text": "Need more information", "intent": "help_request"},
            {"text": "Can you explain?", "intent": "help_request"},
            {"text": "Help with planning", "intent": "help_request"},
            {"text": "Support needed", "intent": "help_request"}
        ]

        return synthetic_data

# Custom preprocessing classes for sklearn pipeline
class TextPreprocessor:
    """Custom text preprocessor for sklearn pipeline"""

    def __init__(self, lowercase=True, remove_punctuation=True,
                 remove_stopwords=True, lemmatize=True, stop_words=None, lemmatizer=None):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = stop_words or set()
        self.lemmatizer = lemmatizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

class FeatureUnion:
    """Simple feature union implementation"""

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for name, transformer in self.transformer_list:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        from scipy.sparse import hstack
        features = []
        for name, transformer in self.transformer_list:
            features.append(transformer.transform(X))
        return hstack(features)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Main execution functions
def main_training_workflow():
    """Main training workflow for development and testing"""
    trainer = ModelTrainer()

    # Train intent classifier
    logger.info("Starting main training workflow")

    try:
        # Load training data
        training_data = trainer._load_training_data()
        logger.info(f"Loaded {len(training_data)} training samples")

        # Train model
        results = trainer.train_intent_classifier(
            training_data=training_data,
            experiment_name="main_training_run"
        )

        if results['success']:
            logger.info(f"Training completed successfully. Best model: {results['best_model']}")
            logger.info(f"Test accuracy: {results['accuracy']:.4f}")

            # Train entity extraction
            entity_results = trainer.train_entity_extraction_model()
            logger.info(f"Entity extraction training: {entity_results['success']}")

            return {
                'intent_classifier': results,
                'entity_extraction': entity_results,
                'overall_success': True
            }
        else:
            logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
            return {
                'overall_success': False,
                'error': results.get('error', 'Training failed')
            }

    except Exception as e:
        logger.error(f"Main training workflow failed: {e}")
        return {
            'overall_success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Run main training workflow when script is executed directly
    results = main_training_workflow()

    if results['overall_success']:
        print("✅ Training completed successfully!")
        print(f"Intent Classifier: {results['intent_classifier']['best_model']}")
        print(f"Accuracy: {results['intent_classifier']['accuracy']:.4f}")
    else:
        print("❌ Training failed!")
        print(f"Error: {results['error']}")