# README.md
# Sri Lanka Tourism Chatbot

A comprehensive AI-powered chatbot system for Sri Lankan tourism with Natural Language Processing, Machine Learning, and integrated tools for currency conversion, weather checking, translation, and maps.

## Features

### Core Functionality
- **Natural Language Processing**: Intent classification and entity extraction
- **Machine Learning**: Self-updating knowledge base with continuous learning
- **Persistent Storage**: PostgreSQL database for conversations and knowledge
- **Real-time Chat**: WebSocket support for instant messaging

### Integrated Tools
- **Currency Converter**: Real-time exchange rates with fallback data
- **Weather Checker**: Current weather and forecasts for Sri Lankan cities
- **Translator**: English ↔ Sinhala ↔ Tamil translation
- **Maps Integration**: Location information and directions with Google Maps

### Technology Stack
- **Backend**: Python, FastAPI, SQLAlchemy, spaCy, scikit-learn
- **Frontend**: React, styled-components, axios
- **Database**: PostgreSQL
- **Cache**: Redis
- **Deployment**: Docker & Docker Compose

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Optional: API keys for enhanced functionality

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sri_lanka_chatbot
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys (optional)
```

3. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

4. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup (Development)

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app/main.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

#### Database Setup
```bash
# Run PostgreSQL
docker run -d -p 5432:5432 -e POSTGRES_DB=chatbot_db -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin123 postgres:15

# Initialize with sample data
psql -h localhost -U admin -d chatbot_db -f data/knowledge_base/sri_lanka_tourism.sql
```

## API Documentation

### Chat Endpoints
- `POST /api/chat/send-message` - Send a message to the chatbot
- `GET /api/chat/conversation-history/{session_id}` - Get conversation history
- `POST /api/chat/feedback` - Submit feedback for a conversation
- `GET /api/chat/session/new` - Create a new chat session

### Tool Endpoints
- `POST /api/tools/currency/convert` - Convert currency
- `POST /api/tools/translate` - Translate text
- `GET /api/tools/weather/current` - Get current weather
- `GET /api/tools/weather/forecast` - Get weather forecast
- `GET /api/tools/maps/location` - Get location information
- `POST /api/tools/maps/directions` - Get directions between locations

## Architecture

### Backend Components
```
backend/
├── app/
│   ├── models/          # Database models and managers
│   ├── nlp/            # NLP components (intent, entities, response)
│   ├── ml/             # Machine learning and training
│   ├── tools/          # External API integrations
│   ├── api/            # FastAPI routes
│   └── utils/          # Helper functions and config
```

### Frontend Components
```
frontend/
├── src/
│   ├── components/     # React components
│   ├── services/       # API service layer
│   └── App.js         # Main application component
```

### Machine Learning Pipeline
1. **Intent Classification**: Categorizes user messages using SVM
2. **Entity Extraction**: Identifies locations, currencies, dates, etc.
3. **Knowledge Retrieval**: Searches relevant information from database
4. **Tool Integration**: Calls appropriate external APIs
5. **Response Generation**: Creates contextual responses
6. **Continuous Learning**: Updates model based on user feedback

## Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `GOOGLE_MAPS_API_KEY`: For maps and location services
- `GOOGLE_TRANSLATE_API_KEY`: For translation services
- `OPENWEATHERMAP_API_KEY`: For weather information
- `EXCHANGE_RATE_API_KEY`: For currency conversion

### API Keys Setup
The system works with fallback data if API keys are not provided, but for full functionality:

1. **Google Maps API**: Get from [Google Cloud Console](https://console.cloud.google.com/)
2. **OpenWeatherMap API**: Register at [OpenWeatherMap](https://openweathermap.org/api)
3. **Exchange Rate API**: Get from [ExchangeRate-API](https://exchangerate-api.com/)

## Usage Examples

### Basic Chat
```javascript
// Send a message
const response = await fetch('/api/chat/send-message', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "What are the best places to visit in Sri Lanka?",
    session_id: "your-session-id"
  })
});
```

### Tool Usage
```javascript
// Convert currency
const conversion = await fetch('/api/tools/currency/convert', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    amount: 100,
    from_currency: "USD",
    to_currency: "LKR"
  })
});

// Get weather
const weather = await fetch('/api/tools/weather/current?city=Colombo');

// Translate text
const translation = await fetch('/api/tools/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "Hello",
    from_language: "english",
    to_language: "sinhala"
  })
});
```

## Deployment

### Production Deployment
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up --build -d

# Scale services
docker-compose up --scale backend=3 --scale frontend=2
```

### Health Monitoring
- Backend health: `GET /health`
- Tools status: `GET /api/tools/tools/status`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues
1. **Database connection fails**: Check PostgreSQL is running and credentials are correct
2. **API tools not working**: Verify API keys are set correctly
3. **Frontend won't connect**: Ensure backend is running on port 8000
4. **Maps not loading**: Check Google Maps API key and browser console

### Logs
```bash
# View application logs
docker-compose logs -f backend
docker-compose logs -f frontend

# View database logs
docker-compose logs postgres
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Create an issue in the repository

---

**Made with ❤️ for Sri Lankan Tourism**