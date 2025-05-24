# backend/app/tools/currency_converter.py
import requests
from typing import Dict, Any, Optional
import os

from backend.app.utils.config import get_api_key

class CurrencyConverter:
    def __init__(self):
        self.api_key = get_api_key('EXCHANGE_RATE_API_KEY')
        self.base_url = "https://api.exchangerate-api.com/v4/latest"
        self.fallback_url = "https://api.fixer.io/latest"

        # Fallback rates (updated periodically)
        self.fallback_rates = {
            'USD': {'LKR': 320.50, 'EUR': 0.85, 'GBP': 0.73, 'INR': 83.25},
            'EUR': {'USD': 1.18, 'LKR': 378.20, 'GBP': 0.86, 'INR': 98.15},
            'GBP': {'USD': 1.37, 'LKR': 439.48, 'EUR': 1.16, 'INR': 114.05},
            'LKR': {'USD': 0.0031, 'EUR': 0.0026, 'GBP': 0.0023, 'INR': 0.26},
            'INR': {'USD': 0.012, 'EUR': 0.010, 'GBP': 0.0088, 'LKR': 3.85}
        }

    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Convert currency using live exchange rates"""
        try:
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()

            if from_currency == to_currency:
                return {
                    'success': True,
                    'amount': amount,
                    'from_currency': from_currency,
                    'to_currency': to_currency,
                    'converted_amount': amount,
                    'exchange_rate': 1.0,
                    'source': 'same_currency'
                }

            # Try to get live rates
            exchange_rate = self._get_exchange_rate(from_currency, to_currency)

            if exchange_rate:
                converted_amount = round(amount * exchange_rate, 2)
                return {
                    'success': True,
                    'amount': amount,
                    'from_currency': from_currency,
                    'to_currency': to_currency,
                    'converted_amount': converted_amount,
                    'exchange_rate': exchange_rate,
                    'source': 'live_api'
                }
            else:
                # Use fallback rates
                fallback_rate = self._get_fallback_rate(from_currency, to_currency)
                if fallback_rate:
                    converted_amount = round(amount * fallback_rate, 2)
                    return {
                        'success': True,
                        'amount': amount,
                        'from_currency': from_currency,
                        'to_currency': to_currency,
                        'converted_amount': converted_amount,
                        'exchange_rate': fallback_rate,
                        'source': 'fallback_rates',
                        'note': 'Using approximate rates. For exact rates, please check with your bank.'
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Currency conversion not available for {from_currency} to {to_currency}'
                    }

        except Exception as e:
            return {
                'success': False,
                'error': f'Currency conversion failed: {str(e)}'
            }

    def _get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get live exchange rate from API"""
        try:
            # Try primary API
            response = requests.get(f"{self.base_url}/{from_currency}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                rates = data.get('rates', {})
                return rates.get(to_currency)

            # Try fallback API if available
            if self.api_key:
                fallback_response = requests.get(
                    f"{self.fallback_url}?access_key={self.api_key}&base={from_currency}",
                    timeout=5
                )
                if fallback_response.status_code == 200:
                    fallback_data = fallback_response.json()
                    rates = fallback_data.get('rates', {})
                    return rates.get(to_currency)

            return None

        except Exception as e:
            print(f"Error fetching exchange rate: {e}")
            return None

    def _get_fallback_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get fallback exchange rate from stored rates"""
        try:
            if from_currency in self.fallback_rates:
                return self.fallback_rates[from_currency].get(to_currency)
            return None
        except Exception:
            return None

    def get_supported_currencies(self) -> Dict[str, Any]:
        """Get list of supported currencies"""
        return {
            'success': True,
            'currencies': {
                'USD': 'US Dollar',
                'EUR': 'Euro',
                'GBP': 'British Pound',
                'LKR': 'Sri Lankan Rupee',
                'INR': 'Indian Rupee',
                'AUD': 'Australian Dollar',
                'CAD': 'Canadian Dollar',
                'JPY': 'Japanese Yen',
                'SGD': 'Singapore Dollar',
                'MYR': 'Malaysian Ringgit'
            }
        }
