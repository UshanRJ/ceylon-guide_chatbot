# backend/app/api/tool_routes.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel

from backend.app.tools.currency_converter import CurrencyConverter
from backend.app.tools.weather_checker import WeatherChecker
from backend.app.tools.translator import Translator
from backend.app.tools.maps_integration import MapsIntegration

router = APIRouter(prefix="/tools", tags=["tools"])

# Initialize tools
currency_converter = CurrencyConverter()
weather_checker = WeatherChecker()
translator = Translator()
maps_integration = MapsIntegration()

# Request/Response Models
class CurrencyConvertRequest(BaseModel):
    amount: float
    from_currency: str
    to_currency: str

class TranslateRequest(BaseModel):
    text: str
    from_language: str = "english"
    to_language: str = "sinhala"

class DirectionsRequest(BaseModel):
    origin: str
    destination: str

# Currency Routes
@router.post("/currency/convert")
async def convert_currency(request: CurrencyConvertRequest) -> Dict[str, Any]:
    """Convert currency from one to another"""
    try:
        result = currency_converter.convert_currency(
            amount=request.amount,
            from_currency=request.from_currency,
            to_currency=request.to_currency
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/currency/supported")
async def get_supported_currencies() -> Dict[str, Any]:
    """Get list of supported currencies"""
    return {
        "success": True,
        "currencies": [
            {"code": "USD", "name": "US Dollar", "symbol": "$"},
            {"code": "EUR", "name": "Euro", "symbol": "€"},
            {"code": "GBP", "name": "British Pound", "symbol": "£"},
            {"code": "LKR", "name": "Sri Lankan Rupee", "symbol": "Rs"},
            {"code": "INR", "name": "Indian Rupee", "symbol": "₹"},
            {"code": "AUD", "name": "Australian Dollar", "symbol": "A$"},
            {"code": "CAD", "name": "Canadian Dollar", "symbol": "C$"},
            {"code": "SGD", "name": "Singapore Dollar", "symbol": "S$"},
            {"code": "JPY", "name": "Japanese Yen", "symbol": "¥"},
            {"code": "CNY", "name": "Chinese Yuan", "symbol": "¥"}
        ]
    }

# Weather Routes
@router.get("/weather/current")
async def get_current_weather(city: str) -> Dict[str, Any]:
    """Get current weather for a city"""
    try:
        result = weather_checker.get_current_weather(city)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/weather/forecast")
async def get_weather_forecast(city: str, days: int = 5) -> Dict[str, Any]:
    """Get weather forecast for a city"""
    try:
        result = weather_checker.get_forecast(city, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Translation Routes
@router.post("/translate")
async def translate_text(request: TranslateRequest) -> Dict[str, Any]:
    """Translate text between languages"""
    try:
        result = translator.translate(
            text=request.text,
            source_lang=request.from_language,
            target_lang=request.to_language
        )
        return {
            "success": True,
            "original_text": request.text,
            "translated_text": result,
            "from_language": request.from_language,
            "to_language": request.to_language
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/translate/languages")
async def get_supported_languages() -> Dict[str, Any]:
    """Get list of supported languages"""
    return {
        "success": True,
        "languages": [
            {"code": "english", "name": "English", "native": "English"},
            {"code": "sinhala", "name": "Sinhala", "native": "සිංහල"},
            {"code": "tamil", "name": "Tamil", "native": "தமிழ்"}
        ]
    }

# Maps Routes
@router.get("/maps/location")
async def get_location_info(location: str) -> Dict[str, Any]:
    """Get information about a location"""
    try:
        # Simulated response for now
        locations_db = {
            "sigiriya": {
                "location": "Sigiriya",
                "coordinates": {"lat": 7.957, "lng": 80.760},
                "country": "Sri Lanka",
                "description": "Ancient rock fortress and UNESCO World Heritage Site"
            },
            "colombo": {
                "location": "Colombo",
                "coordinates": {"lat": 6.927, "lng": 79.861},
                "country": "Sri Lanka",
                "description": "Capital and largest city of Sri Lanka"
            },
            "kandy": {
                "location": "Kandy",
                "coordinates": {"lat": 7.291, "lng": 80.636},
                "country": "Sri Lanka",
                "description": "Major city known for the Temple of the Sacred Tooth Relic"
            },
            "galle": {
                "location": "Galle",
                "coordinates": {"lat": 6.053, "lng": 80.220},
                "country": "Sri Lanka",
                "description": "Historic city with Dutch colonial architecture"
            }
        }

        location_key = location.lower().strip()
        if location_key in locations_db:
            return {
                "success": True,
                **locations_db[location_key]
            }

        # Default response for unknown locations
        return {
            "success": True,
            "location": location,
            "coordinates": {"lat": 6.927, "lng": 79.861},  # Default to Colombo
            "country": "Sri Lanka",
            "description": "Location in Sri Lanka"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/maps/directions")
async def get_directions(request: DirectionsRequest) -> Dict[str, Any]:
    """Get directions between two locations"""
    try:
        # This would integrate with a real maps API
        return {
            "success": True,
            "origin": request.origin,
            "destination": request.destination,
            "distance": "150 km",
            "duration": "3 hours",
            "route": "Via A1 Highway"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/maps/nearby")
async def find_nearby_places(
        location: str,
        place_type: str = "tourist_attraction"
) -> Dict[str, Any]:
    """Find nearby places of interest"""
    try:
        # Simulated response
        return {
            "success": True,
            "location": location,
            "places": [
                {
                    "name": "Nearby Temple",
                    "type": place_type,
                    "distance": "2 km",
                    "rating": 4.5
                },
                {
                    "name": "Local Market",
                    "type": "market",
                    "distance": "1.5 km",
                    "rating": 4.2
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Tools Status
@router.get("/status")
async def get_tools_status() -> Dict[str, Any]:
    """Get status of all tools"""
    return {
        "success": True,
        "tools": {
            "currency_converter": {"status": "active", "description": "Convert between currencies"},
            "weather": {"status": "active", "description": "Get weather information"},
            "translator": {"status": "active", "description": "Translate text between languages"},
            "maps": {"status": "active", "description": "Get location information and directions"}
        }
    }