# backend/app/tools/maps_integration.py
import requests
from typing import Dict, Any, Optional, List
from backend.app.utils.config import get_api_key

class MapsIntegration:
    def __init__(self):
        self.api_key = get_api_key('GOOGLE_MAPS_API_KEY')
        self.geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.places_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        self.directions_url = "https://maps.googleapis.com/maps/api/directions/json"

        # Sri Lankan locations with coordinates (fallback)
        self.sri_lankan_locations = {
            'colombo': {'lat': 6.9271, 'lng': 79.8612, 'name': 'Colombo'},
            'kandy': {'lat': 7.2906, 'lng': 80.6337, 'name': 'Kandy'},
            'galle': {'lat': 6.0535, 'lng': 80.2210, 'name': 'Galle'},
            'ella': {'lat': 6.8721, 'lng': 81.0465, 'name': 'Ella'},
            'sigiriya': {'lat': 7.9568, 'lng': 80.7603, 'name': 'Sigiriya'},
            'negombo': {'lat': 7.2084, 'lng': 79.8358, 'name': 'Negombo'},
            'anuradhapura': {'lat': 8.3114, 'lng': 80.4037, 'name': 'Anuradhapura'},
            'trincomalee': {'lat': 8.5874, 'lng': 81.2152, 'name': 'Trincomalee'},
            'jaffna': {'lat': 9.6615, 'lng': 80.0255, 'name': 'Jaffna'},
            'mirissa': {'lat': 5.9487, 'lng': 80.4501, 'name': 'Mirissa'},
            'unawatuna': {'lat': 6.0108, 'lng': 80.2493, 'name': 'Unawatuna'},
            'nuwara eliya': {'lat': 6.9497, 'lng': 80.7891, 'name': 'Nuwara Eliya'}
        }

    def get_location_info(self, location_name: str) -> Dict[str, Any]:
        """Get location information including coordinates and details"""
        try:
            location_key = location_name.lower().strip()

            # Check if it's a known Sri Lankan location
            if location_key in self.sri_lankan_locations:
                location_data = self.sri_lankan_locations[location_key]
                return {
                    'success': True,
                    'location': location_data['name'],
                    'coordinates': {
                        'lat': location_data['lat'],
                        'lng': location_data['lng']
                    },
                    'country': 'Sri Lanka',
                    'source': 'local_database',
                    'map_url': self._generate_map_url(location_data['lat'], location_data['lng'], location_data['name'])
                }

            # Try Google Maps API if available
            if self.api_key:
                return self._get_location_from_api(location_name)

            # Fallback response
            return {
                'success': False,
                'error': f'Location "{location_name}" not found in our database. Please try a major Sri Lankan city.'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Location lookup failed: {str(e)}'
            }

    def get_directions(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get directions between two locations"""
        try:
            origin_info = self.get_location_info(origin)
            destination_info = self.get_location_info(destination)

            if not origin_info['success'] or not destination_info['success']:
                return {
                    'success': False,
                    'error': 'Could not find one or both locations'
                }

            # If API is available, get detailed directions
            if self.api_key:
                return self._get_directions_from_api(origin, destination)

            # Provide basic direction information
            origin_coords = origin_info['coordinates']
            dest_coords = destination_info['coordinates']

            # Calculate approximate distance (basic calculation)
            distance = self._calculate_distance(
                origin_coords['lat'], origin_coords['lng'],
                dest_coords['lat'], dest_coords['lng']
            )

            return {
                'success': True,
                'origin': origin_info['location'],
                'destination': destination_info['location'],
                'approximate_distance_km': round(distance, 1),
                'directions_summary': f"Travel from {origin_info['location']} to {destination_info['location']}",
                'map_url': self._generate_directions_url(origin_coords, dest_coords),
                'note': 'For detailed directions, please use Google Maps or similar navigation app.'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Directions lookup failed: {str(e)}'
            }

    def find_nearby_places(self, location: str, place_type: str = 'tourist_attraction') -> Dict[str, Any]:
        """Find nearby places of interest"""
        try:
            location_info = self.get_location_info(location)

            if not location_info['success']:
                return location_info

            # Provide basic nearby attractions for major Sri Lankan cities
            nearby_places = self._get_nearby_attractions(location.lower())

            if nearby_places:
                return {
                    'success': True,
                    'location': location_info['location'],
                    'nearby_places': nearby_places,
                    'place_type': place_type
                }
            else:
                return {
                    'success': False,
                    'error': f'No nearby {place_type} information available for {location}'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Nearby places lookup failed: {str(e)}'
            }

    def _get_location_from_api(self, location_name: str) -> Dict[str, Any]:
        """Get location info from Google Maps API"""
        try:
            params = {
                'address': location_name + ', Sri Lanka',
                'key': self.api_key
            }

            response = requests.get(self.geocoding_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    result = data['results'][0]
                    location = result['geometry']['location']

                    return {
                        'success': True,
                        'location': result['formatted_address'],
                        'coordinates': {
                            'lat': location['lat'],
                            'lng': location['lng']
                        },
                        'source': 'google_maps_api',
                        'map_url': self._generate_map_url(location['lat'], location['lng'], result['formatted_address'])
                    }

            return {
                'success': False,
                'error': 'Location not found via API'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'API request failed: {str(e)}'
            }

    def _get_directions_from_api(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get directions from Google Maps API"""
        try:
            params = {
                'origin': origin + ', Sri Lanka',
                'destination': destination + ', Sri Lanka',
                'key': self.api_key
            }

            response = requests.get(self.directions_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data['routes']:
                    route = data['routes'][0]
                    leg = route['legs'][0]

                    return {
                        'success': True,
                        'origin': leg['start_address'],
                        'destination': leg['end_address'],
                        'distance': leg['distance']['text'],
                        'duration': leg['duration']['text'],
                        'steps': [step['html_instructions'] for step in leg['steps'][:5]],  # First 5 steps
                        'source': 'google_maps_api'
                    }

            return {
                'success': False,
                'error': 'No route found via API'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Directions API request failed: {str(e)}'
            }

    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate approximate distance between two coordinates"""
        import math

        # Convert to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in kilometers

        return c * r

    def _generate_map_url(self, lat: float, lng: float, location_name: str) -> str:
        """Generate Google Maps URL"""
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}&query_place_id={location_name.replace(' ', '+')}"

    def _generate_directions_url(self, origin_coords: Dict, dest_coords: Dict) -> str:
        """Generate Google Maps directions URL"""
        return f"https://www.google.com/maps/dir/{origin_coords['lat']},{origin_coords['lng']}/{dest_coords['lat']},{dest_coords['lng']}"

    def _get_nearby_attractions(self, location: str) -> Optional[List[Dict[str, str]]]:
        """Get nearby attractions for major Sri Lankan locations"""
        attractions = {
            'colombo': [
                {'name': 'Gangaramaya Temple', 'type': 'Religious Site'},
                {'name': 'Galle Face Green', 'type': 'Park'},
                {'name': 'National Museum', 'type': 'Museum'},
                {'name': 'Independence Square', 'type': 'Historical Site'},
                {'name': 'Pettah Market', 'type': 'Shopping'}
            ],
            'kandy': [
                {'name': 'Temple of the Sacred Tooth Relic', 'type': 'Religious Site'},
                {'name': 'Kandy Lake', 'type': 'Natural Feature'},
                {'name': 'Royal Botanical Gardens', 'type': 'Garden'},
                {'name': 'Udawattakele Forest Reserve', 'type': 'Nature Reserve'},
                {'name': 'Kandy Cultural Centre', 'type': 'Cultural Site'}
            ],
            'galle': [
                {'name': 'Galle Fort', 'type': 'Historical Site'},
                {'name': 'Galle Lighthouse', 'type': 'Landmark'},
                {'name': 'Dutch Reformed Church', 'type': 'Religious Site'},
                {'name': 'Galle National Museum', 'type': 'Museum'},
                {'name': 'Jungle Beach', 'type': 'Beach'}
            ],
            'ella': [
                {'name': 'Nine Arch Bridge', 'type': 'Landmark'},
                {'name': 'Little Adams Peak', 'type': 'Mountain'},
                {'name': 'Ella Rock', 'type': 'Mountain'},
                {'name': 'Ravana Falls', 'type': 'Waterfall'},
                {'name': 'Lipton\'s Seat', 'type': 'Viewpoint'}
            ],
            'sigiriya': [
                {'name': 'Sigiriya Rock Fortress', 'type': 'Historical Site'},
                {'name': 'Pidurangala Rock', 'type': 'Mountain'},
                {'name': 'Sigiriya Museum', 'type': 'Museum'},
                {'name': 'Dambulla Cave Temple', 'type': 'Religious Site'},
                {'name': 'Minneriya National Park', 'type': 'National Park'}
            ]
        }

        return attractions.get(location.lower())