import requests
from typing import Dict, Any, Optional
from datetime import datetime
from backend.app.utils.config import get_api_key

class WeatherChecker:
    def __init__(self):
        self.api_key = get_api_key('OPENWEATHERMAP_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.current_weather_url = "https://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "https://api.openweathermap.org/data/2.5/forecast"

        # Sri Lankan cities coordinates (fallback data)
        self.sri_lankan_cities = {
            'colombo': {'lat': 6.9271, 'lon': 79.8612},
            'kandy': {'lat': 7.2906, 'lon': 80.6337},
            'galle': {'lat': 6.0535, 'lon': 80.2210},
            'jaffna': {'lat': 9.6615, 'lon': 80.0255},
            'trincomalee': {'lat': 8.5874, 'lon': 81.2152},
            'negombo': {'lat': 7.2084, 'lon': 79.8358},
            'ella': {'lat': 6.8721, 'lon': 81.0465},
            'nuwara eliya': {'lat': 6.9497, 'lon': 80.7891},
            'anuradhapura': {'lat': 8.3114, 'lon': 80.4037},
            'matara': {'lat': 5.9549, 'lon': 80.5550}
        }

        # Fallback weather data for major cities
        self.fallback_weather = {
            'colombo': {
                'temperature': 28,
                'description': 'Partly cloudy',
                'humidity': 75,
                'wind_speed': 15
            },
            'kandy': {
                'temperature': 25,
                'description': 'Mostly cloudy',
                'humidity': 80,
                'wind_speed': 10
            },
            'galle': {
                'temperature': 29,
                'description': 'Sunny',
                'humidity': 70,
                'wind_speed': 20
            },
            'nuwara eliya': {
                'temperature': 18,
                'description': 'Cool and misty',
                'humidity': 85,
                'wind_speed': 5
            }
        }

    def get_current_weather(self, city: str) -> Dict[str, Any]:
        """Get current weather for a city"""
        try:
            city_lower = city.lower().strip()

            # Try API first if available
            if self.api_key:
                api_result = self._get_weather_from_api(city)
                if api_result['success']:
                    return api_result

            # Fallback to stored data
            if city_lower in self.fallback_weather:
                weather_data = self.fallback_weather[city_lower]
                return {
                    'success': True,
                    'city': city.title(),
                    'temperature': weather_data['temperature'],
                    'description': weather_data['description'],
                    'humidity': weather_data['humidity'],
                    'wind_speed': weather_data['wind_speed'],
                    'source': 'fallback_data',
                    'note': 'Weather data is approximate. For real-time data, please check a weather app.',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

            # General Sri Lanka weather info
            return {
                'success': True,
                'city': city.title(),
                'temperature': 27,
                'description': 'Tropical climate',
                'humidity': 75,
                'wind_speed': 12,
                'source': 'general_climate',
                'note': f'General weather info for Sri Lanka. {city} has a tropical climate with temperatures typically between 25-30°C.',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Weather check failed: {str(e)}'
            }

    def get_weather_forecast(self, city: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for a city"""
        try:
            if self.api_key:
                api_result = self._get_forecast_from_api(city, days)
                if api_result['success']:
                    return api_result

            # Fallback forecast
            city_lower = city.lower().strip()
            base_temp = 27
            if city_lower in self.fallback_weather:
                base_temp = self.fallback_weather[city_lower]['temperature']

            forecast_days = []
            for i in range(min(days, 5)):
                date = datetime.now()
                date = date.replace(day=date.day + i)

                # Simulate some weather variation
                temp_variation = (-2 + (i % 3)) if i % 2 == 0 else (1 + (i % 2))

                forecast_days.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'temperature_high': base_temp + temp_variation,
                    'temperature_low': base_temp + temp_variation - 5,
                    'description': 'Partly cloudy' if i % 2 == 0 else 'Sunny',
                    'humidity': 70 + (i * 3),
                    'rain_chance': 30 if i % 3 == 0 else 10
                })

            return {
                'success': True,
                'city': city.title(),
                'forecast': forecast_days,
                'source': 'fallback_forecast',
                'note': 'Weather forecast is approximate. For accurate forecasts, please use a dedicated weather service.'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Weather forecast failed: {str(e)}'
            }

    def _get_weather_from_api(self, city: str) -> Dict[str, Any]:
        """Get weather from OpenWeatherMap API"""
        try:
            params = {
                'q': f"{city},LK",  # LK is country code for Sri Lanka
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(self.current_weather_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()

                return {
                    'success': True,
                    'city': data['name'],
                    'temperature': round(data['main']['temp']),
                    'description': data['weather'][0]['description'].title(),
                    'humidity': data['main']['humidity'],
                    'wind_speed': round(data['wind']['speed'] * 3.6),  # Convert m/s to km/h
                    'pressure': data['main']['pressure'],
                    'feels_like': round(data['main']['feels_like']),
                    'source': 'openweathermap_api',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                return {'success': False, 'error': f'API returned status {response.status_code}'}

        except Exception as e:
            return {'success': False, 'error': f'API request failed: {str(e)}'}

    def _get_forecast_from_api(self, city: str, days: int) -> Dict[str, Any]:
        """Get forecast from OpenWeatherMap API"""
        try:
            params = {
                'q': f"{city},LK",
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }

            response = requests.get(self.forecast_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()

                # Process forecast data (group by day)
                daily_forecasts = {}
                for item in data['list']:
                    date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')

                    if date not in daily_forecasts:
                        daily_forecasts[date] = {
                            'date': date,
                            'temperatures': [],
                            'descriptions': [],
                            'humidity': [],
                            'rain_chance': 0
                        }

                    daily_forecasts[date]['temperatures'].append(item['main']['temp'])
                    daily_forecasts[date]['descriptions'].append(item['weather'][0]['description'])
                    daily_forecasts[date]['humidity'].append(item['main']['humidity'])

                    if 'rain' in item:
                        daily_forecasts[date]['rain_chance'] = max(daily_forecasts[date]['rain_chance'], 70)
                    elif 'clouds' in item['weather'][0]['description']:
                        daily_forecasts[date]['rain_chance'] = max(daily_forecasts[date]['rain_chance'], 30)

                # Process daily summaries
                forecast_days = []
                for date, day_data in list(daily_forecasts.items())[:days]:
                    temps = day_data['temperatures']
                    most_common_desc = max(set(day_data['descriptions']), key=day_data['descriptions'].count)

                    forecast_days.append({
                        'date': date,
                        'temperature_high': round(max(temps)),
                        'temperature_low': round(min(temps)),
                        'description': most_common_desc.title(),
                        'humidity': round(sum(day_data['humidity']) / len(day_data['humidity'])),
                        'rain_chance': day_data['rain_chance']
                    })

                return {
                    'success': True,
                    'city': data['city']['name'],
                    'forecast': forecast_days,
                    'source': 'openweathermap_api'
                }
            else:
                return {'success': False, 'error': f'Forecast API returned status {response.status_code}'}

        except Exception as e:
            return {'success': False, 'error': f'Forecast API request failed: {str(e)}'}

    def get_weather_alerts(self, city: str) -> Dict[str, Any]:
        """Get weather alerts for a city (basic implementation)"""
        try:
            # Get current weather first
            current_weather = self.get_current_weather(city)

            if not current_weather['success']:
                return current_weather

            alerts = []
            temp = current_weather['temperature']
            humidity = current_weather['humidity']

            # Generate basic alerts based on conditions
            if temp > 35:
                alerts.append({
                    'type': 'heat_warning',
                    'message': 'High temperature warning. Stay hydrated and avoid direct sunlight.',
                    'severity': 'moderate'
                })

            if temp < 15:
                alerts.append({
                    'type': 'cool_weather',
                    'message': 'Cool weather. Consider bringing warm clothing.',
                    'severity': 'low'
                })

            if humidity > 85:
                alerts.append({
                    'type': 'high_humidity',
                    'message': 'High humidity levels. Expect muggy conditions.',
                    'severity': 'low'
                })

            if 'rain' in current_weather['description'].lower():
                alerts.append({
                    'type': 'rain_alert',
                    'message': 'Rain expected. Carry an umbrella.',
                    'severity': 'moderate'
                })

            return {
                'success': True,
                'city': current_weather['city'],
                'alerts': alerts,
                'alert_count': len(alerts)
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Weather alerts failed: {str(e)}'
            }

