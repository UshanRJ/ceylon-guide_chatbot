// frontend/src/services/chatService.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

class ChatService {
    constructor() {
        this.sessionId = null;
        this.conversationHistory = [];

        // Setup axios defaults
        axios.defaults.baseURL = API_BASE_URL;
        axios.defaults.timeout = 30000;

        // Add request interceptor
        axios.interceptors.request.use(
            (config) => {
                console.log('API Request:', config);
                return config;
            },
            (error) => {
                return Promise.reject(error);
            }
        );

        // Add response interceptor
        axios.interceptors.response.use(
            (response) => {
                return response;
            },
            (error) => {
                console.error('API Error:', error);
                return Promise.reject(error);
            }
        );
    }

    async createNewSession() {
        try {
            const response = await axios.get('/chat/session/new');
            if (response.data.success) {
                this.sessionId = response.data.session_id;
                this.conversationHistory = [];
                return response.data;
            }
            throw new Error('Failed to create session');
        } catch (error) {
            console.error('Error creating session:', error);
            throw error;
        }
    }

    async sendMessage(message) {
        try {
            if (!this.sessionId) {
                await this.createNewSession();
            }

            const response = await axios.post('/chat/send-message', {
                message: message,
                session_id: this.sessionId
            });

            if (response.data) {
                // Add to local conversation history
                this.conversationHistory.push({
                    type: 'user',
                    message: message,
                    timestamp: new Date().toISOString()
                });

                this.conversationHistory.push({
                    type: 'bot',
                    message: response.data.response,
                    timestamp: response.data.timestamp,
                    intent: response.data.intent,
                    confidence: response.data.confidence,
                    entities: response.data.entities,
                    tools_used: response.data.tools_used,
                    conversation_id: response.data.conversation_id
                });

                return response.data;
            }

            throw new Error('Invalid response from server');
        } catch (error) {
            console.error('Error sending message:', error);
            throw error;
        }
    }

    async getConversationHistory(limit = 20) {
        try {
            if (!this.sessionId) {
                return { conversations: [] };
            }

            const response = await axios.get(`/chat/conversation-history/${this.sessionId}?limit=${limit}`);
            return response.data;
        } catch (error) {
            console.error('Error fetching conversation history:', error);
            throw error;
        }
    }

    async submitFeedback(conversationId, rating, feedbackText = null, isHelpful = null) {
        try {
            const response = await axios.post('/chat/feedback', {
                conversation_id: conversationId,
                rating: rating,
                feedback_text: feedbackText,
                is_helpful: isHelpful
            });

            return response.data;
        } catch (error) {
            console.error('Error submitting feedback:', error);
            throw error;
        }
    }

    // Tool-specific methods
    async convertCurrency(amount, fromCurrency, toCurrency) {
        try {
            const response = await axios.post('/tools/currency/convert', {
                amount: amount,
                from_currency: fromCurrency,
                to_currency: toCurrency
            });

            return response.data;
        } catch (error) {
            console.error('Error converting currency:', error);
            throw error;
        }
    }

    async translateText(text, fromLanguage = 'english', toLanguage = 'sinhala') {
        try {
            const response = await axios.post('/tools/translate', {
                text: text,
                from_language: fromLanguage,
                to_language: toLanguage
            });

            return response.data;
        } catch (error) {
            console.error('Error translating text:', error);
            throw error;
        }
    }

    async getCurrentWeather(city) {
        try {
            const response = await axios.get(`/tools/weather/current?city=${encodeURIComponent(city)}`);
            return response.data;
        } catch (error) {
            console.error('Error getting weather:', error);
            throw error;
        }
    }

    async getWeatherForecast(city, days = 5) {
        try {
            const response = await axios.get(`/tools/weather/forecast?city=${encodeURIComponent(city)}&days=${days}`);
            return response.data;
        } catch (error) {
            console.error('Error getting weather forecast:', error);
            throw error;
        }
    }

    async getLocationInfo(location) {
        try {
            const response = await axios.get(`/tools/maps/location?location=${encodeURIComponent(location)}`);
            return response.data;
        } catch (error) {
            console.error('Error getting location info:', error);
            throw error;
        }
    }

    async getDirections(origin, destination) {
        try {
            const response = await axios.post('/tools/maps/directions', {
                origin: origin,
                destination: destination
            });

            return response.data;
        } catch (error) {
            console.error('Error getting directions:', error);
            throw error;
        }
    }

    async findNearbyPlaces(location, placeType = 'tourist_attraction') {
        try {
            const response = await axios.get(`/tools/maps/nearby?location=${encodeURIComponent(location)}&place_type=${placeType}`);
            return response.data;
        } catch (error) {
            console.error('Error finding nearby places:', error);
            throw error;
        }
    }

    async getToolsStatus() {
        try {
            const response = await axios.get('/tools/tools/status');
            return response.data;
        } catch (error) {
            console.error('Error getting tools status:', error);
            throw error;
        }
    }

    async getSupportedCurrencies() {
        try {
            const response = await axios.get('/tools/currency/supported');
            return response.data;
        } catch (error) {
            console.error('Error getting supported currencies:', error);
            throw error;
        }
    }

    async getSupportedLanguages() {
        try {
            const response = await axios.get('/tools/translate/languages');
            return response.data;
        } catch (error) {
            console.error('Error getting supported languages:', error);
            throw error;
        }
    }

    getSessionId() {
        return this.sessionId;
    }

    getConversationHistoryLocal() {
        return this.conversationHistory;
    }

    clearSession() {
        this.sessionId = null;
        this.conversationHistory = [];
    }
}

// Create singleton instance
const chatService = new ChatService();
export default chatService;