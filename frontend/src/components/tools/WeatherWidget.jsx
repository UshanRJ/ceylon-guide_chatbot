// frontend/src/components/tools/WeatherWidget.jsx
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiCloud, FiDroplet, FiWind, FiSearch } from 'react-icons/fi';
import chatService from '../../services/chatService';

const Container = styled.div`
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  color: ${props => props.theme.colors.gray[900]};
  margin-bottom: 1.5rem;
`;

const SearchForm = styled.form`
  display: flex;
  gap: 0.5rem;
  margin-bottom: 2rem;
`;

const SearchInput = styled.input`
  flex: 1;
  padding: 0.75rem 1rem;
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  font-size: 1rem;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const SearchButton = styled.button`
  padding: 0.75rem 1rem;
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    background: ${props => props.theme.colors.secondary};
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const WeatherCard = styled.div`
  background: ${props => props.theme.colors.gray[50]};
  border: 1px solid ${props => props.theme.colors.gray[200]};
  border-radius: 0.75rem;
  padding: 1.5rem;
  margin-bottom: 1rem;
`;

const WeatherHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const CityName = styled.h3`
  font-size: 1.25rem;
  color: ${props => props.theme.colors.gray[900]};
`;

const Temperature = styled.div`
  font-size: 2.5rem;
  font-weight: 700;
  color: ${props => props.theme.colors.primary};
`;

const WeatherDescription = styled.p`
  color: ${props => props.theme.colors.gray[600]};
  text-transform: capitalize;
  margin-bottom: 1rem;
`;

const WeatherDetails = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
`;

const DetailItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: ${props => props.theme.colors.gray[700]};
  
  svg {
    color: ${props => props.theme.colors.primary};
  }
`;

const ErrorMessage = styled.div`
  background: ${props => props.theme.colors.danger}10;
  color: ${props => props.theme.colors.danger};
  padding: 1rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
`;

const Loading = styled.div`
  text-align: center;
  color: ${props => props.theme.colors.gray[600]};
  padding: 2rem;
`;

const PresetCities = styled.div`
  margin-bottom: 1.5rem;
`;

const PresetButton = styled.button`
  padding: 0.5rem 1rem;
  margin: 0.25rem;
  background: ${props => props.theme.colors.gray[100]};
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const WeatherWidget = () => {
    const [city, setCity] = useState('');
    const [weatherData, setWeatherData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const popularCities = ['Colombo', 'Kandy', 'Galle', 'Negombo', 'Ella', 'Sigiriya'];

    const fetchWeather = async (cityName) => {
        setLoading(true);
        setError(null);

        try {
            const response = await chatService.getCurrentWeather(cityName);
            if (response.success) {
                setWeatherData(response);
            } else {
                setError(response.error || 'Failed to fetch weather data');
            }
        } catch (err) {
            setError('Failed to connect to weather service');
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (city.trim()) {
            fetchWeather(city);
        }
    };

    const handlePresetClick = (cityName) => {
        setCity(cityName);
        fetchWeather(cityName);
    };

    return (
        <Container>
            <Title>Weather Information</Title>

            <SearchForm onSubmit={handleSubmit}>
                <SearchInput
                    type="text"
                    value={city}
                    onChange={(e) => setCity(e.target.value)}
                    placeholder="Enter city name..."
                    disabled={loading}
                />
                <SearchButton type="submit" disabled={loading || !city.trim()}>
                    <FiSearch />
                    Search
                </SearchButton>
            </SearchForm>

            <PresetCities>
                <small style={{ color: '#6B7280', display: 'block', marginBottom: '0.5rem' }}>
                    Popular cities:
                </small>
                {popularCities.map((cityName) => (
                    <PresetButton key={cityName} onClick={() => handlePresetClick(cityName)}>
                        {cityName}
                    </PresetButton>
                ))}
            </PresetCities>

            {loading && <Loading>Loading weather data...</Loading>}

            {error && <ErrorMessage>{error}</ErrorMessage>}

            {weatherData && !loading && (
                <WeatherCard>
                    <WeatherHeader>
                        <CityName>{weatherData.location}</CityName>
                        <Temperature>{weatherData.temperature}°C</Temperature>
                    </WeatherHeader>

                    <WeatherDescription>{weatherData.description}</WeatherDescription>

                    <WeatherDetails>
                        <DetailItem>
                            <FiCloud />
                            <span>Feels like: {weatherData.feels_like}°C</span>
                        </DetailItem>
                        <DetailItem>
                            <FiDroplet />
                            <span>Humidity: {weatherData.humidity}%</span>
                        </DetailItem>
                        <DetailItem>
                            <FiWind />
                            <span>Wind: {weatherData.wind_speed} km/h</span>
                        </DetailItem>
                        <DetailItem>
                            <FiCloud />
                            <span>Pressure: {weatherData.pressure} hPa</span>
                        </DetailItem>
                    </WeatherDetails>
                </WeatherCard>
            )}
        </Container>
    );
};

export default WeatherWidget;